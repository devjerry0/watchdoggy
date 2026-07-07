from __future__ import annotations

import time
from typing import Callable


class Pacer:
    """Throttle a loop: `wait(interval)` sleeps only the time still needed to
    keep consecutive calls at least `interval` seconds apart. First call is free.
    Clock and sleep are injected for testing."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self, interval: float) -> None:
        if self._last is not None:
            remaining = interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()
