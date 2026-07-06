from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from doggy.state import CONFIDENCE_DECIMALS

log = logging.getLogger("doggy")

EVENTS_FILE = "events.jsonl"


@dataclass
class EventRecord:
    """A single detected-dog reaction event, backed by a JPEG on the SD card."""

    id: str
    ts: float
    wall_time: float | None
    confidence: float
    latency_s: float | None
    thumb: str
    clip: str | None = None


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
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._dir = Path(event_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_events = max_events
        self._max_age_days = max_age_days
        self._clock = clock
        # Records kept in memory ordered oldest -> newest.
        self._records: list[EventRecord] = self._load()

    @property
    def _jsonl(self) -> Path:
        return self._dir / EVENTS_FILE

    def _load(self) -> list[EventRecord]:
        path = self._jsonl
        if not path.is_file():
            return []
        records: list[EventRecord] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # An abrupt power loss on the SD card can leave a torn/truncated final
            # line; a single bad line must not sink the whole history.
            try:
                obj = json.loads(line)
                thumb = obj["thumb"]
                records.append(
                    EventRecord(
                        id=obj.get("id") or Path(thumb).stem,
                        ts=obj["ts"],
                        wall_time=obj.get("wall_time"),
                        confidence=obj["confidence"],
                        latency_s=obj.get("latency_s"),
                        thumb=thumb,
                        clip=obj.get("clip"),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                log.warning("skipping malformed events.jsonl line: %r", line)
        return records

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
        with self._jsonl.open("a") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        self._records.append(record)
        self.prune()
        return record

    def list(self, limit: int | None = None) -> list[EventRecord]:
        recent_first = list(reversed(self._records))
        if limit is not None:
            return recent_first[:limit]
        return recent_first

    def _delete_files(self, record: EventRecord) -> None:
        for name in (record.thumb, record.clip):
            if not name:
                continue
            path = self._dir / name
            if path.is_file():
                path.unlink()

    def prune(self) -> None:
        survivors = self._records
        dropped: list[EventRecord] = []

        if self._max_age_days > 0:
            now = self._clock()
            cutoff = self._max_age_days * 86400
            kept: list[EventRecord] = []
            for record in survivors:
                if record.wall_time is not None and now - record.wall_time > cutoff:
                    dropped.append(record)
                else:
                    kept.append(record)
            survivors = kept

        if self._max_events > 0 and len(survivors) > self._max_events:
            excess = len(survivors) - self._max_events
            dropped.extend(survivors[:excess])
            survivors = survivors[excess:]

        if not dropped:
            return

        for record in dropped:
            self._delete_files(record)
        self._records = survivors
        self._rewrite()

    def _rewrite(self) -> None:
        with self._jsonl.open("w") as fh:
            for record in self._records:
                fh.write(json.dumps(asdict(record)) + "\n")
