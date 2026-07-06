from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import TypedDict

import numpy as np

from doggy.config import TunableSettings

# Decimal places for the confidence value shown in the dashboard and event log.
CONFIDENCE_DECIMALS = 3


class FireEvent(TypedDict):
    """A recorded deterrent firing, shown in the dashboard event list."""

    ts: float
    confidence: float
    thumb: str


class RuntimeSettings:
    """Thread-safe holder for the live-tunable settings, swapped atomically."""

    def __init__(self, tunable: TunableSettings) -> None:
        self._lock = threading.Lock()
        self._tunable = tunable

    def get(self) -> TunableSettings:
        with self._lock:
            return self._tunable

    def update(self, tunable: TunableSettings) -> None:
        with self._lock:
            self._tunable = tunable


class FrameBuffer:
    """Holds only the most recent frame; setters overwrite (drop-oldest)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


@dataclass
class Status:
    state: str = "IDLE"
    fps: float = 0.0
    confidence: float = 0.0  # highest-confidence dog in the frame
    dogs: int = 0  # number of dogs detected in the frame
    fires_this_hour: int = 0
    last_fire_ts: float | None = None
    last_fire_thumb: str | None = None
    muted: bool = False
    temp_c: float | None = None
    detect_interval_effective: float = 0.0


_DEFAULT_MAX_EVENTS = 50  # recent fire events retained for the dashboard


class StatusStore:
    """Thread-safe snapshot of live status plus a bounded log of recent fires."""

    def __init__(self, max_events: int = _DEFAULT_MAX_EVENTS) -> None:
        self._lock = threading.Lock()
        self._status = Status()
        self._events: deque[FireEvent] = deque(maxlen=max_events)

    def update(self, **kwargs) -> None:
        with self._lock:
            self._status = replace(self._status, **kwargs)

    def snapshot(self) -> Status:
        with self._lock:
            return self._status

    def add_event(self, event: FireEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> list[FireEvent]:
        with self._lock:
            return list(self._events)
