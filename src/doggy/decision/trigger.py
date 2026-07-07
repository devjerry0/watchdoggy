from __future__ import annotations

import random
from collections import deque
from enum import Enum

from doggy.vision.detection import Detection
from doggy.core.runtime import RuntimeSettings


class TriggerState(str, Enum):
    IDLE = "IDLE"
    CONFIRMING = "CONFIRMING"
    COOLDOWN = "COOLDOWN"


class _State:
    """Internal FSM node. Stateless singleton; all mutable data lives on the
    owning TriggerLogic (passed in as ``ctx``). ``handle`` decides the fire edge
    (via ``ctx._fired``) and returns the state object for the next frame."""

    enum: TriggerState

    def handle(
        self,
        ctx: TriggerLogic,
        has_dog: bool,
        frame_max: float,
        m_of_n: bool,
        window_full: bool,
        now: float,
    ) -> _State:
        raise NotImplementedError


class _Idle(_State):
    enum = TriggerState.IDLE

    def handle(self, ctx, has_dog, frame_max, m_of_n, window_full, now):
        ctx._fired = False
        if has_dog:
            ctx._confirm_start = now
            ctx._confirm_max = frame_max
            return _CONFIRMING
        return _IDLE


class _Confirming(_State):
    enum = TriggerState.CONFIRMING

    def handle(self, ctx, has_dog, frame_max, m_of_n, window_full, now):
        ctx._fired = False
        cfg = ctx._cfg
        # Track the peak confidence seen while confirming so the fire reports the
        # detection that actually triggered it -- not whatever is in the frame on
        # the fire edge, which can be empty after an M-of-N flicker.
        ctx._confirm_max = max(ctx._confirm_max, frame_max)
        if window_full and not m_of_n:
            return _IDLE
        if m_of_n and now - ctx._confirm_start >= cfg.confirm_seconds:
            ctx._cooldown_until = now + ctx._rng.uniform(
                cfg.cooldown_min_seconds, cfg.cooldown_max_seconds
            )
            ctx.fire_confidence = ctx._confirm_max
            ctx.fire_latency = now - ctx._confirm_start
            ctx._fired = True
            return _COOLDOWN
        return _CONFIRMING


class _Cooldown(_State):
    enum = TriggerState.COOLDOWN

    def handle(self, ctx, has_dog, frame_max, m_of_n, window_full, now):
        if now >= ctx._cooldown_until:
            # Cooldown expired: fall through to Idle's handling of THIS SAME frame
            # (a dog present this frame enters CONFIRMING immediately).
            return _IDLE.handle(ctx, has_dog, frame_max, m_of_n, window_full, now)
        ctx._fired = False
        return _COOLDOWN


_IDLE = _Idle()
_CONFIRMING = _Confirming()
_COOLDOWN = _Cooldown()


class TriggerLogic:
    """Time-based confirmation + M-of-N window + jittered cooldown.

    update() returns True exactly on the transition into COOLDOWN (the fire edge).
    """

    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None) -> None:
        self._runtime = runtime
        self._rng = rng or random.Random()
        self._state_obj: _State = _IDLE
        self.state = TriggerState.IDLE
        self._confirm_start: float = 0.0
        self._cooldown_until: float = 0.0
        self._window: deque[bool] = deque()
        self._confirm_max: float = 0.0
        self._fired: bool = False
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
        self._cfg = cfg
        frame_confs = [
            d.confidence for d in detections
            if d.label in cfg.alert_labels and d.confidence >= cfg.confidence
        ]
        has_dog = bool(frame_confs)
        frame_max = max(frame_confs, default=0.0)

        self._window.append(has_dog)
        while len(self._window) > cfg.window_n:
            self._window.popleft()
        m_of_n = sum(self._window) >= cfg.window_m
        window_full = len(self._window) >= cfg.window_n

        self._state_obj = self._state_obj.handle(
            self, has_dog, frame_max, m_of_n, window_full, now
        )
        self.state = self._state_obj.enum
        return self._fired
