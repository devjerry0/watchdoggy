from __future__ import annotations

from collections import deque

from doggy.core.runtime import RuntimeSettings

_HOUR = 3600.0


class FireGate:
    """Decides whether a fire is allowed: master off switch, snooze, per-hour cap.

    Persistence is no longer its concern (that moved to ``Recorder``): the gate
    only answers ``allow`` and remembers fire timestamps for the rolling rate
    limit via ``note_fire``.
    """

    def __init__(self, runtime: RuntimeSettings) -> None:
        self._runtime = runtime
        self._fires: deque[float] = deque()
        self._snooze_until: float = 0.0

    def _prune(self, now: float) -> None:
        while self._fires and now - self._fires[0] >= _HOUR:
            self._fires.popleft()

    def fires_last_hour(self, now: float) -> int:
        self._prune(now)
        return len(self._fires)

    def snooze(self, seconds: float, now: float) -> None:
        self._snooze_until = now + seconds

    def cancel_snooze(self) -> None:
        self._snooze_until = 0.0

    def snooze_remaining(self, now: float) -> float:
        return max(0.0, self._snooze_until - now)

    def allow(self, now: float) -> bool:
        cfg = self._runtime.get()
        if not cfg.safety_enabled:
            return False
        if now < self._snooze_until:
            return False
        return self.fires_last_hour(now) < cfg.max_fires_per_hour

    def note_fire(self, now: float) -> None:
        self._fires.append(now)
