from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from doggy.core.runtime import RuntimeSettings
from doggy.core.status import StatusStore
from doggy.reaction.hub import DogCaught

log = logging.getLogger("doggy")

_SOOTHING_EXTS = {".mp3", ".wav", ".flac", ".ogg"}

# How long a slice we wait on the running player before re-reading config, and
# how long we idle when there is nothing to play. Small enough that turning the
# mode off (or an empty library) is noticed, and a running track stops, in ~1s.
_POLL_SECONDS = 0.5

# Sentinel from _await_exit: the volume changed mid-track and could NOT be
# applied live to the running stream (no PipeWire tools / node), so the caller
# re-spawns the track at the new volume as a fallback.
_REVOLUME = object()

Spawn = Callable[[Path, float], "subprocess.Popen | None"]
# Apply a volume [0,1] to a running player's audio stream in place; returns True
# if applied live (no restart needed), False if the caller must re-spawn.
SetVolume = Callable[["subprocess.Popen", float], bool]


class SoothingPlayer:
    """Loops the uploaded calm-audio library through the speaker on one daemon
    thread, and doubles as a hub Reaction: a confirmed catch cuts the current
    track instantly and holds playback until ``soothing_resume_seconds`` after
    the last catch.

    The hold is deliberately long (45s default). Escalation strikes -- the
    louder follow-up alarms fired while the animal stands its ground -- do NOT
    publish hub events, so a single catch's hold must be long enough to span a
    whole escalation sequence; otherwise soothing music would resume between
    strikes and play underneath the alarm.

    Threading: ``on_dog_caught`` runs on the detect thread (via SafeReaction) and
    only ever terminates the current subprocess and arms the hold -- never waits.
    The loop thread owns the subprocess lifecycle. A small lock guards the two
    fields they share (the running proc and the hold deadline).
    """

    def __init__(self, runtime: RuntimeSettings, library_dir: Path,
                 status: StatusStore, clock: Callable[[], float] = time.monotonic,
                 spawn: Spawn | None = None,
                 set_volume: SetVolume | None = None) -> None:
        self._runtime = runtime
        self._library_dir = Path(library_dir)
        self._status = status
        self._clock = clock
        self._spawn = spawn or self._spawn_player
        # Apply a live volume change to the running stream (True = applied, no
        # restart); default uses PipeWire, falls back to a re-spawn where absent.
        self._set_volume = set_volume or self._set_live_volume
        self._poll = _POLL_SECONDS
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Shared with the detect thread (on_dog_caught): the running player,
        # which proc a catch cut, and the hold deadline (monotonic clock).
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._interrupted: subprocess.Popen | None = None
        self._hold_until = 0.0
        # Loop-thread-only: last value pushed to status, to avoid update churn.
        self._current_track: str | None = None
        self._warned_no_player = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="soothing", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and let any current track be terminated. For clean
        shutdown (and test teardown); the daemon thread would also die with the
        process."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)

    # -- hub Reaction ------------------------------------------------------

    def on_dog_caught(self, event: DogCaught) -> None:
        # Runs on the detect thread: MUST stay non-blocking. Arm the hold, then
        # terminate() the current track (never wait() -- that would stall the
        # detect loop). The loop thread notices the exit and settles into the
        # hold, resuming with the next track once it clears.
        resume = self._runtime.get().soothing_resume_seconds
        with self._lock:
            # max(): a second catch during the hold extends it, never shortens.
            self._hold_until = max(self._hold_until, self._clock() + resume)
            proc = self._proc
            self._interrupted = proc  # remember which proc the catch cut
        if proc is not None:
            proc.terminate()

    # -- loop --------------------------------------------------------------

    def _run(self) -> None:
        index = 0
        fails_this_pass = 0
        while not self._stop.is_set():
            cfg = self._runtime.get()
            if not cfg.soothing_enabled or self._is_held():
                self._set_track(None)
                fails_this_pass = 0
                self._idle()
                continue
            tracks = self._list_tracks()
            if not tracks:
                self._set_track(None)
                fails_this_pass = 0
                self._idle()
                continue
            track = tracks[index % len(tracks)]
            index += 1
            if self._play(track):
                fails_this_pass += 1
                # A whole pass over the library failed (every file missing or the
                # player bailed): idle once before retrying so a broken library
                # can't spin the CPU.
                if fails_this_pass >= len(tracks):
                    fails_this_pass = 0
                    self._idle()
            else:
                fails_this_pass = 0
        self._set_track(None)

    def _play(self, track: Path) -> bool:
        """Play one track to its end at the live soothing volume. If the volume
        is changed while it plays, the same track is re-spawned at the new volume
        (from the start -- pw-play's --volume is fixed at spawn) so the change is
        heard immediately instead of only on the next track. Returns True only
        when it failed for track reasons (unplayable file / nonzero exit) so a
        fully broken library backs off; a track cut short on purpose (mode off,
        shutdown, or a catch) returns False."""
        while not self._stop.is_set():
            volume = self._runtime.get().soothing_volume
            proc = self._spawn(track, volume)
            if proc is None:
                log.info("soothing: could not start %s; skipping", track.name)
                return True
            with self._lock:
                self._proc = proc
                held = self._clock() < self._hold_until
            if held:
                # A catch armed the hold in the spawn window -- after the loop-top
                # hold check but before we registered proc here -- so on_dog_caught
                # saw a null proc and terminated nothing. Cut this track now; the
                # loop top settles into the hold. (Belt to _await_exit's check.)
                proc.terminate()
                self._wait_slice(proc)  # reap (bounded: one slice)
                with self._lock:
                    if self._proc is proc:
                        self._proc = None
                    if self._interrupted is proc:
                        self._interrupted = None
                self._set_track(None)
                return False
            self._set_track(track.name)
            # pw-play's --volume is overridden by the PipeWire session manager's
            # remembered per-stream volume, so assert the wanted volume on the
            # live stream once it registers. This is what makes the loudness
            # actually match the slider (afplay / no-PipeWire keeps --volume).
            self._set_volume(proc, volume)
            code = self._await_exit(proc, volume)
            with self._lock:
                if self._proc is proc:
                    self._proc = None
                interrupted = self._interrupted is proc
                if interrupted:
                    self._interrupted = None
            if code is _REVOLUME:
                # Volume changed: replay THIS track at the new volume. Keep the
                # status track name (unchanged), don't advance the library index.
                continue
            if code is None:
                # We stopped it (mode off / shutdown); the loop top will idle.
                self._set_track(None)
                return False
            if interrupted:
                # A catch cut it; the hold is armed. Resume the next track later.
                self._set_track(None)
                return False
            if code != 0:
                log.info("soothing: player exited %s for %s; skipping", code, track.name)
                return True
            return False
        return False

    def _await_exit(self, proc, volume: float):
        """Wait for the player to exit in poll-sized slices, re-reading config
        between them. Returns its exit code (including when a catch terminated
        it); None if we terminated it ourselves (mode off, shutdown, or a hold
        armed mid-flight); or the ``_REVOLUME`` sentinel if the volume changed
        (the caller re-spawns the same track at the new volume). ``volume`` is
        the value this track was spawned with."""
        while True:
            code = self._wait_slice(proc)
            if code is not None:
                return code
            cfg = self._runtime.get()
            if self._stop.is_set() or not cfg.soothing_enabled:
                proc.terminate()
                self._wait_slice(proc)  # reap (bounded: one slice)
                return None
            if self._is_held():
                # A catch armed the hold while this track was playing without its
                # terminate() reaching us; cut it so no calm track plays under the
                # armed hold. The loop top then settles into the hold.
                proc.terminate()
                self._wait_slice(proc)  # reap (bounded: one slice)
                return None
            if cfg.soothing_volume != volume:
                # The user moved the loudness slider. Prefer applying it live to
                # the running stream (no restart); only if that is unavailable
                # re-spawn the track at the new volume.
                if self._set_volume(proc, cfg.soothing_volume):
                    volume = cfg.soothing_volume
                else:
                    proc.terminate()
                    self._wait_slice(proc)  # reap (bounded: one slice)
                    return _REVOLUME

    def _wait_slice(self, proc) -> int | None:
        try:
            return proc.wait(timeout=self._poll)
        except subprocess.TimeoutExpired:
            return None

    def _idle(self) -> None:
        # Event.wait so stop() wakes us immediately instead of after a full poll.
        self._stop.wait(self._poll)

    def _is_held(self) -> bool:
        with self._lock:
            return self._clock() < self._hold_until

    def _set_track(self, name: str | None) -> None:
        if name != self._current_track:
            self._current_track = name
            self._status.update(soothing_track=name)

    def _list_tracks(self) -> list[Path]:
        d = self._library_dir
        if not d.is_dir():
            return []
        # Filter by audio extension: this also drops the upload's .upload.part
        # temp file (Path.glob does match dotfiles).
        return sorted(
            (p for p in d.glob("*") if p.is_file() and p.suffix.lower() in _SOOTHING_EXTS),
            key=lambda p: p.name,
        )

    def _spawn_player(self, path: Path, volume: float) -> subprocess.Popen | None:
        # Backend fallback pw-play -> pw-cat on the Pi (so a Bluetooth sink
        # works), afplay on macOS. Unlike CommandAlerter (pw-play -> paplay ->
        # aplay) we deliberately do NOT fall back to paplay/aplay: those decode
        # only PCM/WAV and the soothing library is mostly mp3, which pw-play and
        # pw-cat decode on-device. pw-play and pw-cat are the same PipeWire binary
        # but bare pw-cat needs an explicit --playback ("one of the playback/
        # record options must be provided"); pw-play defaults to playback.
        player = shutil.which("pw-play") or shutil.which("pw-cat")
        if player:
            cmd = [player, "--volume", str(volume)]
            if Path(player).name == "pw-cat":
                cmd.append("--playback")
            cmd.append(str(path))
        elif sys.platform == "darwin":
            cmd = ["afplay", "-v", str(volume), str(path)]
        else:
            if not self._warned_no_player:
                log.info("soothing: no audio player (pw-play/pw-cat/afplay) found; "
                         "soothing playback idle")
                self._warned_no_player = True
            return None
        try:
            return subprocess.Popen(cmd)
        except OSError:
            log.info("soothing: failed to launch %s", cmd[0])
            return None

    def _set_live_volume(self, proc: subprocess.Popen, volume: float) -> bool:
        """Set the running player's PipeWire stream volume in place, so a
        loudness change is heard without restarting the track, and so the wanted
        volume beats the session manager's remembered per-stream value. Best
        effort: returns False (the caller re-spawns / keeps --volume) when the
        PipeWire tools or the stream node are not found -- e.g. on macOS/afplay
        or a non-PipeWire host."""
        node = self._pw_stream_node(proc.pid)
        if node is None:
            return False
        level = f"{max(0.0, min(1.0, volume)):.3f}"
        try:
            done = subprocess.run(["wpctl", "set-volume", str(node), level],
                                  capture_output=True, timeout=3)
            return done.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _pw_stream_node(self, pid: int) -> int | None:
        """The PipeWire Stream/Output node id for a player process, retrying
        briefly because the node registers a beat after the process starts. The
        pid maps to a Client object; the audio node links to it via client.id."""
        for attempt in range(5):
            node = self._pw_stream_node_once(pid)
            if node is not None:
                return node
            if self._stop.wait(0.2):  # also lets shutdown cut the wait short
                return None
            _ = attempt
        return None

    @staticmethod
    def _pw_stream_node_once(pid: int) -> int | None:
        try:
            dump = subprocess.run(["pw-dump"], capture_output=True, timeout=3, text=True)
            if dump.returncode != 0:
                return None
            objs = json.loads(dump.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        client = None
        for o in objs:
            props = (o.get("info") or {}).get("props") or {}
            if props.get("application.process.id") == pid:
                client = o.get("id")
                break
        if client is None:
            return None
        for o in objs:
            if not str(o.get("type", "")).endswith("Node"):
                continue
            props = (o.get("info") or {}).get("props") or {}
            if (props.get("client.id") == client
                    and str(props.get("media.class", "")).startswith("Stream/Output")):
                return o.get("id")
        return None
