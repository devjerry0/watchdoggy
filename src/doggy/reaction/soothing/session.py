"""Playing ONE track to its end: the poll-slice wait loop that follows the
volume slider, honors mode/schedule changes within ~a second, and cuts the
track the moment a catch's hold is armed."""
from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Callable

from doggy.core.runtime import RuntimeSettings
from doggy.reaction.soothing.audio import SetVolume, Spawn
from doggy.reaction.soothing.state import PlaybackState

log = logging.getLogger("doggy")

# Sentinel from _await_exit: the volume changed mid-track and could NOT be
# applied live to the running stream (no PipeWire tools / node), so play()
# re-spawns the track at the new volume as a fallback.
_REVOLUME = object()


class TrackSession:
    """Plays single tracks for SoothingPlayer, sharing its stop event and
    cross-thread PlaybackState."""

    def __init__(self, runtime: RuntimeSettings, spawn: Spawn,
                 set_volume: SetVolume, state: PlaybackState,
                 stop: threading.Event, poll: float,
                 allowed: Callable[[object], bool],
                 set_track: Callable[[str | None], None]) -> None:
        self._runtime = runtime
        self._spawn = spawn
        self._set_volume = set_volume
        self._state = state
        self._stop = stop
        self._poll = poll
        # Whether the config still permits playing (mode on + inside schedule).
        self._allowed = allowed
        self._set_track = set_track

    def play(self, track: Path) -> bool:
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
            if self._state.register(proc):
                # A catch armed the hold in the spawn window -- after the
                # loop-top hold check but before we registered proc here -- so
                # on_dog_caught saw a null proc and terminated nothing. Cut this
                # track now; the loop top settles into the hold. (Belt to
                # _await_exit's check.)
                self._cut(proc)
                self._state.discard(proc)
                self._set_track(None)
                return False
            self._set_track(track.name)
            # pw-play's --volume is overridden by the PipeWire session manager's
            # remembered per-stream volume, so assert the wanted volume on the
            # live stream once it registers. This is what makes the loudness
            # actually match the slider (afplay / no-PipeWire keeps --volume).
            self._set_volume(proc, volume)
            code = self._await_exit(proc, volume)
            interrupted = self._state.release(proc)
            if code is _REVOLUME:
                # Volume changed: replay THIS track at the new volume. Keep the
                # status track name (unchanged); the caller's library index
                # doesn't advance because play() hasn't returned.
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
                log.info("soothing: player exited %s for %s; skipping",
                         code, track.name)
                return True
            return False
        return False

    def _await_exit(self, proc: subprocess.Popen, volume: float):
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
            if self._stop.is_set() or not self._allowed(cfg):
                # Mode off, shutting down, or the schedule window closed.
                self._cut(proc)
                return None
            if self._state.held():
                # A catch armed the hold while this track was playing without
                # its terminate() reaching us; cut it so no calm track plays
                # under the armed hold. The loop top then settles into the hold.
                self._cut(proc)
                return None
            if cfg.soothing_volume != volume:
                # The user moved the loudness slider. Prefer applying it live to
                # the running stream (no restart); only if that is unavailable
                # re-spawn the track at the new volume.
                if not self._set_volume(proc, cfg.soothing_volume):
                    self._cut(proc)
                    return _REVOLUME
                volume = cfg.soothing_volume

    def _cut(self, proc: subprocess.Popen) -> None:
        proc.terminate()
        self._wait_slice(proc)  # reap (bounded: one slice)

    def _wait_slice(self, proc: subprocess.Popen) -> int | None:
        try:
            return proc.wait(timeout=self._poll)
        except subprocess.TimeoutExpired:
            return None
