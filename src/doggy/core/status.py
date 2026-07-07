from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace

import numpy as np

# Decimal places for the confidence value shown in the dashboard and event log.
CONFIDENCE_DECIMALS = 3


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
    confidence: float = 0.0  # highest-confidence candidate in the frame
    targets: int = 0  # watched animals in frame
    people: int = 0  # number of people detected (shown, never alerted on)
    # Debounced inventory items in the zone: [{"label": str, "count": int}].
    on_counter: list = field(default_factory=list)
    fires_this_hour: int = 0
    last_fire_ts: float | None = None
    last_fire_thumb: str | None = None
    muted: bool = False
    snoozed_until_seconds: float = 0.0  # snooze remaining, set each loop
    temp_c: float | None = None
    detect_interval_effective: float = 0.0
    # Power health from vcgencmd get_throttled; None = unreadable (non-Pi).
    undervolt_now: bool | None = None
    undervolt_since_boot: bool | None = None


class StatusStore:
    """Thread-safe snapshot of live status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = Status()

    def update(self, **kwargs) -> None:
        with self._lock:
            self._status = replace(self._status, **kwargs)

    def snapshot(self) -> Status:
        with self._lock:
            return self._status
