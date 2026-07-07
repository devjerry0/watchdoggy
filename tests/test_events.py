import itertools
import json
import random
import threading
import time

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


def test_delete_removes_record_and_file(tmp_path):
    s = EventStore(tmp_path, 10, 0)
    r = s.add(_img(), 0.5, None, 1.0, 1.0)
    assert s.delete(r.id) is True
    assert s.list() == [] and not (tmp_path / r.thumb).exists()
    assert s.delete("nope") is False


def test_clear_removes_all(tmp_path):
    s = EventStore(tmp_path, 10, 0)
    a = s.add(_img(), 0.5, None, 1.0, 1.0)
    b = s.add(_img(), 0.5, None, 2.0, 2.0)
    thumbs = [tmp_path / a.thumb, tmp_path / b.thumb]
    assert all(p.exists() for p in thumbs)
    s.clear()
    assert s.list() == []
    assert (tmp_path / "events.jsonl").read_text() == ""
    assert not any(p.exists() for p in thumbs)  # thumbnails removed from disk


def test_attach_clip_sets_and_persists(tmp_path):
    s = EventStore(tmp_path, 10, 0)
    r = s.add(_img(), 0.5, None, 1.0, 1.0)
    s.attach_clip(r.id, "clip.mp4")
    assert s.list()[0].clip == "clip.mp4"
    # A fresh store on the same dir must reload the clip from events.jsonl.
    reloaded = EventStore(tmp_path, 10, 0)
    assert reloaded.list()[0].clip == "clip.mp4"
    # Unknown id is a silent no-op: no raise, nothing changes.
    s.attach_clip("nope", "x.mp4")
    assert [e.clip for e in s.list()] == ["clip.mp4"]


def test_stats_counts_and_latency(tmp_path):
    # fixed "now" = 2026-07-06 18:00 UTC; two events today, one 3 days ago
    now = 1783360800.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    s.add(_img(), 0.5, 1.0, now - 3600, 1.0)          # today, 1h ago
    s.add(_img(), 0.5, 2.0, now - 7200, 2.0)          # today, 2h ago
    s.add(_img(), 0.5, 3.0, now - 3 * 86400, 3.0)     # 3 days ago
    st = s.stats()
    assert st["today"] == 2 and st["this_week"] == 3
    assert abs(st["avg_latency_s"] - 2.0) < 1e-9
    assert isinstance(st["busiest_hour"], int)
    assert len(st["per_day"]) == 7


def test_stats_busiest_hour_none_without_wall_time(tmp_path):
    s = EventStore(tmp_path, 100, 0)
    s.add(_img(), 0.5, 1.0, None, 1.0)
    assert s.stats()["busiest_hour"] is None


def test_concurrent_add_and_delete_is_safe(tmp_path):
    # The pipeline thread (add/prune) and the web thread (list/delete/stats)
    # share one EventStore. Without locking, a delete during prune raises
    # "list changed size during iteration" and the jsonl tears. Hammer it from
    # several threads and assert nothing blew up and the store stayed consistent.
    s = EventStore(tmp_path, max_events=50, max_age_days=30)
    base = time.time()
    counter = itertools.count()  # next() is atomic in CPython -> unique ids
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(400):
                roll = random.random()
                if roll < 0.5:
                    n = next(counter)
                    # wall_time near "now" so age-pruning doesn't nuke everything;
                    # unique n keeps event ids (and thumb files) distinct.
                    s.add(_img(), 0.5, 1.0, base + n, float(n))
                elif roll < 0.7:
                    ids = [e.id for e in s.list()]
                    if ids:
                        s.delete(random.choice(ids))
                elif roll < 0.85:
                    s.list()
                else:
                    s.stats()
        except BaseException as exc:  # noqa: BLE001 - report, don't crash the thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"threads raised: {errors!r}"
    # Internal consistency: memory, jsonl, and thumbnail files all agree.
    listed = s.list()
    lines = [ln for ln in (tmp_path / "events.jsonl").read_text().splitlines() if ln.strip()]
    assert len(listed) == len(lines)
    for e in listed:
        assert (tmp_path / e.thumb).is_file()
