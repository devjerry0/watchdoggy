from __future__ import annotations

import threading

from doggy.core.config import TunableSettings


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
