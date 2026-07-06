import json

import numpy as np

from doggy.events import EventStore


def _img():
    return np.zeros((8, 8, 3), np.uint8)


def test_add_writes_jpeg_and_jsonl(tmp_path):
    s = EventStore(tmp_path, max_events=10, max_age_days=0)
    r = s.add(_img(), confidence=0.8, latency_s=1.5, wall_time=1000.0, mono_ts=5.0)
    assert (tmp_path / r.thumb).is_file()
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["confidence"] == 0.8
    assert r.latency_s == 1.5 and r.wall_time == 1000.0 and r.id


def test_list_is_recent_first(tmp_path):
    s = EventStore(tmp_path, max_events=10, max_age_days=0)
    a = s.add(_img(), 0.5, None, 1.0, 1.0)
    b = s.add(_img(), 0.6, None, 2.0, 2.0)
    assert [e.id for e in s.list()] == [b.id, a.id]


def test_loads_existing_and_old_schema(tmp_path):
    # old line missing id/wall_time/latency_s/clip must still load
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"ts": 3.0, "confidence": 0.7, "thumb": "old.jpg"}) + "\n")
    s = EventStore(tmp_path, max_events=10, max_age_days=0)
    e = s.list()[0]
    assert e.thumb == "old.jpg" and e.wall_time is None and e.latency_s is None
    assert e.id == "old"  # stem of thumb


def test_prune_by_count_deletes_files(tmp_path):
    s = EventStore(tmp_path, max_events=2, max_age_days=0)
    r0 = s.add(_img(), 0.5, None, 1.0, 1.0)
    s.add(_img(), 0.5, None, 2.0, 2.0)
    s.add(_img(), 0.5, None, 3.0, 3.0)
    assert len(s.list()) == 2
    assert not (tmp_path / r0.thumb).exists()   # oldest file removed


def test_load_skips_malformed_lines(tmp_path):
    # A torn/truncated line (abrupt power loss on the SD card) or a line missing
    # required keys must not sink the whole history: valid events still load.
    good_a = json.dumps({"id": "fire_1", "ts": 1.0, "confidence": 0.5, "thumb": "fire_1.jpg"})
    good_b = json.dumps({"id": "fire_2", "ts": 2.0, "confidence": 0.6, "thumb": "fire_2.jpg"})
    (tmp_path / "events.jsonl").write_text(
        good_a + "\n"
        + "{not json\n"                                  # torn/garbage line
        + json.dumps({"ts": 3.0, "confidence": 0.7}) + "\n"  # missing "thumb"
        + "\n"                                            # blank line
        + good_b + "\n")
    s = EventStore(tmp_path, max_events=10, max_age_days=0)
    events = s.list()  # most-recent-first
    assert [e.id for e in events] == ["fire_2", "fire_1"]
    assert len(events) == 2


def test_prune_by_age(tmp_path):
    s = EventStore(tmp_path, max_events=100, max_age_days=1, clock=lambda: 1_000_000.0)
    old = s.add(_img(), 0.5, None, 1_000_000.0 - 2 * 86400, 1.0)  # 2 days old by wall_time
    fresh = s.add(_img(), 0.5, None, 1_000_000.0, 2.0)
    s.prune()
    ids = [e.id for e in s.list()]
    assert fresh.id in ids and old.id not in ids
