from __future__ import annotations

import random
from collections import deque
from enum import Enum

from doggy.detection import Detection
from doggy.state import RuntimeSettings


class TriggerState(str, Enum):
    IDLE = "IDLE"
    CONFIRMING = "CONFIRMING"
    COOLDOWN = "COOLDOWN"


class TriggerLogic:
    """Time-based confirmation + M-of-N window + jittered cooldown.

    update() returns True exactly on the transition into COOLDOWN (the fire edge).
    """

    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None) -> None:
        self._runtime = runtime
        self._rng = rng or random.Random()
        self.state = TriggerState.IDLE
        self._confirm_start: float = 0.0
        self._cooldown_until: float = 0.0
        self._window: deque[bool] = deque()

    def update(self, detections: list[Detection], now: float) -> bool:
        cfg = self._runtime.get()
        has_dog = any(
            d.label == "dog" and d.confidence >= cfg.confidence for d in detections
        )

        self._window.append(has_dog)
        while len(self._window) > cfg.window_n:
            self._window.popleft()
        m_of_n = sum(self._window) >= cfg.window_m

        if self.state is TriggerState.COOLDOWN:
            if now >= self._cooldown_until:
                self.state = TriggerState.IDLE
            else:
                return False

        if self.state is TriggerState.IDLE:
            if has_dog:
                self.state = TriggerState.CONFIRMING
                self._confirm_start = now
            return False

        # CONFIRMING
        if len(self._window) >= cfg.window_n and not m_of_n:
            self.state = TriggerState.IDLE
            return False
        if m_of_n and now - self._confirm_start >= cfg.confirm_seconds:
            self._cooldown_until = now + self._rng.uniform(
                cfg.cooldown_min_seconds, cfg.cooldown_max_seconds
            )
            self.state = TriggerState.COOLDOWN
            return True
        return False
