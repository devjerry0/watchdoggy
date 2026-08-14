"""Shared playback state between the detect thread and the soothing loop.

One small monitor guards the three fields both threads touch: the running
player process, which process a catch cut, and the post-catch hold deadline
(monotonic clock). Callers terminate() processes OUTSIDE the lock."""
from __future__ import annotations

import subprocess
import threading
from typing import Callable


class PlaybackState:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._interrupted: subprocess.Popen | None = None
        self._hold_until = 0.0

    def arm_hold(self, deadline: float) -> subprocess.Popen | None:
        """Arm/extend the hold and mark the running proc as cut by a catch;
        returns that proc so the caller can terminate() it."""
        with self._lock:
            # max(): a second catch during the hold extends it, never shortens.
            self._hold_until = max(self._hold_until, deadline)
            self._interrupted = self._proc
            return self._proc

    def held(self) -> bool:
        with self._lock:
            return self._clock() < self._hold_until

    def register(self, proc: subprocess.Popen) -> bool:
        """Publish the new player proc; reports whether a hold was armed in
        the spawn window (the caller must cut the track itself)."""
        with self._lock:
            self._proc = proc
            return self._clock() < self._hold_until

    def discard(self, proc: subprocess.Popen) -> None:
        """Forget a proc that was cut before it ever counted as playing."""
        with self._lock:
            if self._proc is proc:
                self._proc = None
            if self._interrupted is proc:
                self._interrupted = None

    def release(self, proc: subprocess.Popen) -> bool:
        """Unpublish an exited proc; reports whether a catch cut it."""
        with self._lock:
            if self._proc is proc:
                self._proc = None
            interrupted = self._interrupted is proc
            if interrupted:
                self._interrupted = None
            return interrupted
