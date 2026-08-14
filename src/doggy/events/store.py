from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from doggy.core.status import CONFIDENCE_DECIMALS
from doggy.events import lab, report
from doggy.events.record import EventRecord, dump_line, load_records
from doggy.events.retention import RetentionPolicy

__all__ = ["EventStore", "EventRecord", "EVENTS_FILE"]

EVENTS_FILE = "events.jsonl"


class EventStore:
    """Disk-backed history of reaction events (JPEG + events.jsonl on the SD card).

    Source of truth for detected-dog events: each event writes a JPEG thumbnail
    and appends one JSON line. Old lines missing newer fields load with defaults.
    """

    def __init__(
        self,
        event_dir: Path,
        max_events: int = 500,
        max_age_days: int = 30,
        clip_retention: int = 10,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._dir = Path(event_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._retention = RetentionPolicy(max_events, max_age_days,
                                          clip_retention)
        self._clock = clock
        # The pipeline thread writes (add/prune/attach_clip) while the web thread
        # reads and mutates (list/delete/clear/stats); every public method that
        # touches _records or the jsonl takes this lock. RLock so add -> prune
        # (and any other) nesting is safe.
        self._lock = threading.RLock()
        # Records kept in memory ordered oldest -> newest.
        records, backfilled = load_records(self._jsonl, self._dir)
        self._records: list[EventRecord] = records
        # Persist any wall_time backfilled from thumbnail mtimes.
        if backfilled:
            self._rewrite()

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def _jsonl(self) -> Path:
        return self._dir / EVENTS_FILE

    def add(
        self,
        frame: np.ndarray,
        confidence: float,
        latency_s: float | None,
        wall_time: float | None,
        mono_ts: float,
    ) -> EventRecord:
        event_id = f"fire_{int(round((wall_time or mono_ts) * 1000))}"
        thumb = f"{event_id}.jpg"
        record = EventRecord(
            id=event_id,
            ts=mono_ts,
            wall_time=wall_time,
            confidence=round(float(confidence), CONFIDENCE_DECIMALS),
            latency_s=latency_s,
            thumb=thumb,
            clip=None,
        )
        cv2.imwrite(str(self._dir / thumb), frame)
        with self._lock:
            with self._jsonl.open("a") as fh:
                fh.write(dump_line(record))
            self._records.append(record)
            self.prune()
        return record

    def list(self, limit: int | None = None) -> list[EventRecord]:
        with self._lock:
            recent_first = list(reversed(self._records))
        if limit is not None:
            return recent_first[:limit]
        return recent_first

    def _update(self, id: str, mutate: Callable[[EventRecord], None]) -> None:
        """Apply a mutation to the record with this id (if any) and persist."""
        with self._lock:
            for record in self._records:
                if record.id == id:
                    mutate(record)
                    self._rewrite()
                    return

    def delete(self, id: str) -> bool:
        with self._lock:
            for i, record in enumerate(self._records):
                if record.id == id:
                    self._delete_files(record)
                    del self._records[i]
                    self._rewrite()
                    return True
            return False

    def clear(self) -> None:
        with self._lock:
            for record in self._records:
                self._delete_files(record)
            self._records = []
            self._rewrite()

    def attach_clip(self, id: str, clip_name: str) -> None:
        self._update(id, lambda r: setattr(r, "clip", clip_name))

    def attach_sound(self, id: str, sound: str) -> None:
        self._update(id, lambda r: setattr(r, "sound", sound))

    def bump_strikes(self, id: str) -> None:
        self._update(id, lambda r: setattr(r, "strikes", r.strikes + 1))

    def attach_outcome(
        self,
        id: str,
        clear_seconds: float | None,
        taken: list[str],
        wall_time: float,
    ) -> None:
        def stamp(record: EventRecord) -> None:
            record.clear_seconds = clear_seconds
            record.taken = list(taken)
            record.outcome_at = wall_time

        self._update(id, stamp)

    def stats(self) -> dict:
        today = datetime.fromtimestamp(self._clock()).date()
        with self._lock:
            records = list(self._records)
        return report.activity(records, today)

    def lab_stats(self) -> dict:
        now = self._clock()
        with self._lock:
            records = list(self._records)
        return lab.stats(records, now)

    def _delete_files(self, record: EventRecord) -> None:
        for name in (record.thumb, record.clip):
            if not name:
                continue
            path = self._dir / name
            if path.is_file():
                path.unlink()

    def prune(self) -> None:
        with self._lock:
            survivors, dropped = self._retention.sweep(self._records,
                                                       self._clock)
            for record in dropped:
                self._delete_files(record)
            self._records = survivors

            # Clips are far heavier than thumbnails: keep only the newest N of
            # them, deleting older clip files (but keeping the event + its
            # thumbnail). _records is oldest -> newest, so stale_clips is too.
            stale = self._retention.stale_clips(self._records)
            for record in stale:
                self._drop_clip_file(record)

            if dropped or stale:
                self._rewrite()

    def _drop_clip_file(self, record: EventRecord) -> None:
        path = self._dir / record.clip
        if path.is_file():
            path.unlink()
        record.clip = None

    def _rewrite(self) -> None:
        with self._jsonl.open("w") as fh:
            for record in self._records:
                fh.write(dump_line(record))
