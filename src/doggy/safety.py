from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from doggy.state import RuntimeSettings

_HOUR = 3600.0


class SafetyGovernor:
    def __init__(self, runtime: RuntimeSettings, event_log_dir: Path) -> None:
        self._runtime = runtime
        self._dir = Path(event_log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fires: deque[float] = deque()

    def _prune(self, now: float) -> None:
        while self._fires and now - self._fires[0] >= _HOUR:
            self._fires.popleft()

    def fires_last_hour(self, now: float) -> int:
        self._prune(now)
        return len(self._fires)

    def allow_fire(self, now: float) -> bool:
        cfg = self._runtime.get()
        if not cfg.safety_enabled:
            return False
        return self.fires_last_hour(now) < cfg.max_fires_per_hour

    def record_fire(self, frame: np.ndarray, confidence: float, now: float) -> dict:
        self._fires.append(now)
        thumb_name = f"fire_{now:.3f}.jpg"
        cv2.imwrite(str(self._dir / thumb_name), frame)
        event = {"ts": now, "confidence": round(float(confidence), 3), "thumb": thumb_name}
        with (self._dir / "events.jsonl").open("a") as fh:
            fh.write(json.dumps(event) + "\n")
        return event
