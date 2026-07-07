from __future__ import annotations

from collections import deque

import numpy as np

from doggy.events import EventRecord, EventStore
from doggy.core.runtime import RuntimeSettings

_HOUR = 3600.0


class SafetyGovernor:
    """Guards firing: master off switch, per-hour rate limit, event delegation.

    The single writer of the event log is the injected ``EventStore``; this
    class only decides whether a fire is allowed and delegates persistence.
    """

    def __init__(self, runtime: RuntimeSettings, event_store: EventStore) -> None:
        self._runtime = runtime
        self._store = event_store
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

    def allow_fire(self, now: float) -> bool:
        cfg = self._runtime.get()
        if not cfg.safety_enabled:
            return False
        if now < self._snooze_until:
            return False
        return self.fires_last_hour(now) < cfg.max_fires_per_hour

    def record_fire(
        self,
        frame: np.ndarray,
        confidence: float,
        latency_s: float,
        wall_time: float,
        now: float,
    ) -> EventRecord:
        self._fires.append(now)
        return self._store.add(frame, confidence, latency_s, wall_time, mono_ts=now)
