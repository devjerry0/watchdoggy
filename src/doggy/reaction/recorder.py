from __future__ import annotations

import numpy as np

from doggy.events.store import EventRecord, EventStore


class Recorder:
    """Persists a catch by delegating to the single-writer ``EventStore``.

    This is the primary effect that *produces* the ``EventRecord`` (its id names
    the thumbnail/clip), so it runs before the reactions fan out -- it is not a
    peer subscriber on the hub.
    """

    def __init__(self, store: EventStore) -> None:
        self._store = store

    def record(
        self,
        frame: np.ndarray,
        confidence: float,
        latency_s: float,
        wall_time: float,
        mono_ts: float,
    ) -> EventRecord:
        return self._store.add(frame, confidence, latency_s, wall_time, mono_ts=mono_ts)
