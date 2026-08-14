"""EventRecord and its events.jsonl line format (parse, backfill, dump)."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("doggy")


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
    sound: str | None = None
    clear_seconds: float | None = None
    strikes: int = 1
    taken: list[str] = field(default_factory=list)
    outcome_at: float | None = None


def dump_line(record: EventRecord) -> str:
    return json.dumps(asdict(record)) + "\n"


def _wall_time_for(obj: dict, thumb: str,
                   event_dir: Path) -> tuple[float | None, bool]:
    wall_time = obj.get("wall_time")
    if wall_time is not None:
        return wall_time, False
    # Old events (pre-timestamp) have no wall_time, so they can't be
    # bucketed in stats or shown with a real date. Backfill from the
    # thumbnail's mtime -- the wall-clock moment the JPG was written,
    # i.e. when the catch happened -- so history stays usable.
    thumb_path = event_dir / thumb
    if not thumb_path.is_file():
        return None, False
    return thumb_path.stat().st_mtime, True


def _parse_line(line: str, event_dir: Path) -> tuple[EventRecord | None, bool]:
    # An abrupt power loss on the SD card can leave a torn/truncated final
    # line; a single bad line must not sink the whole history.
    try:
        obj = json.loads(line)
        thumb = obj["thumb"]
        wall_time, backfilled = _wall_time_for(obj, thumb, event_dir)
        record = EventRecord(
            id=obj.get("id") or Path(thumb).stem,
            ts=obj["ts"],
            wall_time=wall_time,
            confidence=obj["confidence"],
            latency_s=obj.get("latency_s"),
            thumb=thumb,
            clip=obj.get("clip"),
            sound=obj.get("sound"),
            clear_seconds=obj.get("clear_seconds"),
            strikes=int(obj.get("strikes") or 1),
            taken=list(obj.get("taken") or []),
            outcome_at=obj.get("outcome_at"),
        )
        return record, backfilled
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("skipping malformed events.jsonl line: %r", line)
        return None, False


def load_records(path: Path, event_dir: Path) -> tuple[list[EventRecord], bool]:
    """Read every parseable line; also report whether any wall_time was
    backfilled (the caller persists the repaired history once)."""
    if not path.is_file():
        return [], False
    records: list[EventRecord] = []
    any_backfilled = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        record, backfilled = _parse_line(line, event_dir)
        if record is None:
            continue
        records.append(record)
        any_backfilled = any_backfilled or backfilled
    return records, any_backfilled
