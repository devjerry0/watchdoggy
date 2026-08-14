from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from doggy.core.runtime import RuntimeSettings
from doggy.core.status import StatusStore
from doggy.decision.schedule import within_windows
from doggy.reaction.hub import DogCaught
from doggy.reaction.soothing.audio import (
    AudioBackend,
    SetVolume,
    Spawn,
    list_tracks,
)
from doggy.reaction.soothing.session import TrackSession
from doggy.reaction.soothing.state import PlaybackState

# How long a slice we wait on the running player before re-reading config, and
# how long we idle when there is nothing to play. Small enough that turning the
# mode off (or an empty library) is noticed, and a running track stops, in ~1s.
_POLL_SECONDS = 0.5


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

    Threading: ``on_dog_caught`` runs on the detect thread (via SafeReaction)
    and only ever terminates the current subprocess and arms the hold -- never
    waits. The loop thread owns the subprocess lifecycle (via TrackSession).
    PlaybackState is the monitor guarding what the two threads share.
    """

    def __init__(self, runtime: RuntimeSettings, library_dir: Path,
                 status: StatusStore, clock: Callable[[], float] = time.monotonic,
                 spawn: Spawn | None = None,
                 set_volume: SetVolume | None = None,
                 wall_clock: Callable[[], float] = time.time,
                 poll: float = _POLL_SECONDS) -> None:
        self._runtime = runtime
        self._library_dir = Path(library_dir)
        self._status = status
        self._clock = clock
        # Wall clock (not the monotonic `clock`) for the weekly schedule check.
        self._wall_clock = wall_clock
        self._poll = poll
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = PlaybackState(clock)
        # Default spawn/set_volume come from the platform backend; tests
        # inject fakes. set_volume=True means applied live, no restart needed.
        backend = AudioBackend(self._stop.wait)
        self._session = TrackSession(
            runtime=runtime,
            spawn=spawn or backend.spawn,
            set_volume=set_volume or backend.set_live_volume,
            state=self._state,
            stop=self._stop,
            poll=poll,
            allowed=self._allowed,
            set_track=self._set_track,
        )
        # Loop-thread-only: last value pushed to status, to avoid update churn.
        self._current_track: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="soothing",
                                        daemon=True)
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
        proc = self._state.arm_hold(self._clock() + resume)
        if proc is not None:
            proc.terminate()

    # -- loop --------------------------------------------------------------

    def _run(self) -> None:
        index = 0
        fails_this_pass = 0
        while not self._stop.is_set():
            cfg = self._runtime.get()
            if not self._allowed(cfg) or self._state.held():
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
            if not self._session.play(track):
                fails_this_pass = 0
                continue
            fails_this_pass += 1
            # A whole pass over the library failed (every file missing or the
            # player bailed): idle once before retrying so a broken library
            # can't spin the CPU.
            if fails_this_pass >= len(tracks):
                fails_this_pass = 0
                self._idle()
        self._set_track(None)

    # -- helpers -----------------------------------------------------------

    def _idle(self) -> None:
        # Event.wait so stop() wakes us immediately instead of after a full poll.
        self._stop.wait(self._poll)

    def _allowed(self, cfg) -> bool:
        """Whether soothing may play now: mode on, and inside a soothing
        window when the schedule is enabled (wall time)."""
        if not cfg.soothing_enabled:
            return False
        if not cfg.soothing_schedule_enabled:
            return True
        return within_windows(cfg.soothing_windows, self._wall_clock())

    def _set_track(self, name: str | None) -> None:
        if name != self._current_track:
            self._current_track = name
            self._status.update(soothing_track=name)

    def _list_tracks(self) -> list[Path]:
        return list_tracks(self._library_dir)
