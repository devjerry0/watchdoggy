"""Retention policy: which events age out, which overflow the count cap,
and which clip files exceed their keep-newest budget. Pure decisions -- the
store does the actual file deletion and persistence."""
from __future__ import annotations

from typing import Callable

from doggy.events.record import EventRecord

SECONDS_PER_DAY = 86400


class RetentionPolicy:
    def __init__(self, max_events: int, max_age_days: int,
                 clip_retention: int) -> None:
        self._max_events = max_events
        self._max_age_days = max_age_days
        self._clip_retention = clip_retention

    def sweep(self, records: list[EventRecord],
              clock: Callable[[], float],
              ) -> tuple[list[EventRecord], list[EventRecord]]:
        """(survivors, dropped): the age cap first, then the count cap.
        The clock is only consulted when an age cap is configured."""
        survivors, expired = self._drop_expired(records, clock)
        survivors, overflow = self._drop_overflow(survivors)
        return survivors, expired + overflow

    def _drop_expired(self, records: list[EventRecord],
                      clock: Callable[[], float],
                      ) -> tuple[list[EventRecord], list[EventRecord]]:
        if self._max_age_days <= 0:
            return list(records), []
        now = clock()
        cutoff = self._max_age_days * SECONDS_PER_DAY
        expired = [r for r in records if _expired(r, now, cutoff)]
        kept = [r for r in records if not _expired(r, now, cutoff)]
        return kept, expired

    def _drop_overflow(self, records: list[EventRecord],
                       ) -> tuple[list[EventRecord], list[EventRecord]]:
        if self._max_events <= 0 or len(records) <= self._max_events:
            return records, []
        excess = len(records) - self._max_events
        return records[excess:], records[:excess]

    def stale_clips(self, records: list[EventRecord]) -> list[EventRecord]:
        """Oldest records whose clip files exceed the keep-newest budget."""
        if self._clip_retention <= 0:  # 0 = unlimited
            return []
        with_clips = [r for r in records if r.clip]
        excess = len(with_clips) - self._clip_retention
        return with_clips[:excess] if excess > 0 else []


def _expired(record: EventRecord, now: float, cutoff: float) -> bool:
    return record.wall_time is not None and now - record.wall_time > cutoff
