from __future__ import annotations

import random
from collections import deque
from enum import Enum

from doggy.detection import Detection, TARGET_LABEL
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
        self._confirm_max: float = 0.0
        # Confidence of the detection that caused the most recent fire; read by
        # the pipeline to log the event. Valid only immediately after update()
        # returns True.
        self.fire_confidence: float = 0.0
        # Time-to-react of the most recent fire: seconds from first sighting
        # (CONFIRMING start) to the fire edge. Valid only immediately after
        # update() returns True.
        self.fire_latency: float = 0.0

    def update(self, detections: list[Detection], now: float) -> bool:
        cfg = self._runtime.get()
        frame_confs = [
            d.confidence for d in detections
            if d.label == TARGET_LABEL and d.confidence >= cfg.confidence
        ]
        has_dog = bool(frame_confs)
        frame_max = max(frame_confs, default=0.0)

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
                self._confirm_max = frame_max
            return False

        # CONFIRMING. Track the peak confidence seen while confirming so the fire
        # reports the detection that actually triggered it -- not whatever is in
        # the frame on the fire edge, which can be empty after an M-of-N flicker.
        self._confirm_max = max(self._confirm_max, frame_max)
        if len(self._window) >= cfg.window_n and not m_of_n:
            self.state = TriggerState.IDLE
            return False
        if m_of_n and now - self._confirm_start >= cfg.confirm_seconds:
            self._cooldown_until = now + self._rng.uniform(
                cfg.cooldown_min_seconds, cfg.cooldown_max_seconds
            )
            self.state = TriggerState.COOLDOWN
            self.fire_confidence = self._confirm_max
            self.fire_latency = now - self._confirm_start
            return True
        return False
