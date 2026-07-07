import itertools
import json
import random
import threading
import time

import numpy as np
import pytest

from doggy.events.store import EventStore


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


def test_attach_sound_outcome_and_strikes_persist(tmp_path):
    s = EventStore(tmp_path, 100, 0)
    r = s.add(_img(), 0.9, 1.0, 1000.0, 1.0)
    s.attach_sound(r.id, "chirp.wav")
    s.bump_strikes(r.id)
    s.attach_outcome(r.id, clear_seconds=3.5, taken=["sandwich"], wall_time=1010.0)
    reloaded = EventStore(tmp_path, 100, 0).list()[0]
    assert reloaded.sound == "chirp.wav"
    assert reloaded.strikes == 2
    assert reloaded.clear_seconds == 3.5
    assert reloaded.taken == ["sandwich"]
    assert reloaded.outcome_at == 1010.0


def test_old_jsonl_lines_load_with_outcome_defaults(tmp_path):
    line = {"id": "fire_1", "ts": 1.0, "wall_time": 1000.0, "confidence": 0.9,
            "latency_s": 1.0, "thumb": "fire_1.jpg", "clip": None}
    (tmp_path / "events.jsonl").write_text(json.dumps(line) + "\n")
    r = EventStore(tmp_path, 100, 0).list()[0]
    assert r.sound is None and r.clear_seconds is None
    assert r.strikes == 1 and r.taken == [] and r.outcome_at is None


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


def _catch(s, sound, wall_time, mono_ts, clear=None, taken=(), outcome=True):
    r = s.add(_img(), 0.9, 1.0, wall_time, mono_ts)
    s.attach_sound(r.id, sound)
    if outcome:
        s.attach_outcome(r.id, clear_seconds=clear, taken=list(taken),
                         wall_time=wall_time + 30)
    return r


def test_lab_stats_per_sound_effectiveness(tmp_path):
    now = 1783360800.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    # Clears in time order: early quick escapes, later slow/never -> wearing off.
    for i, clear in enumerate([2.0, 3.0, 20.0, None, 30.0, None]):
        _catch(s, "chirp.wav", now - 3600 + i, float(i), clear=clear)
    st = s.lab_stats()
    assert len(st["sounds"]) == 1
    row = st["sounds"][0]
    assert row["sound"] == "chirp.wav"
    assert row["plays"] == 6 and row["completed"] == 6
    assert row["deterred_rate"] == pytest.approx(2 / 6)   # only the 2.0s and 3.0s clears
    assert row["avg_clear_s"] == pytest.approx((2.0 + 3.0 + 20.0 + 30.0) / 4)
    assert row["wearing_off"] is True
    assert st["thefts_this_week"] == 0                    # nothing taken


def test_lab_stats_wearing_off_scores_missing_clears_as_full_watch(tmp_path):
    now = 1783360800.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    # Early plays clear fast; the last three never clear (clear_seconds=None).
    # Only because a no-clear outcome scores STAYED_CLEAR_S (60s) does the
    # newer half (60/60/60) dwarf the older half (10/10/10); dropping the
    # Nones instead would leave nothing slow to compare and never flag.
    for i, clear in enumerate([10.0, 10.0, 10.0, None, None, None]):
        _catch(s, "hawk.wav", now - 3600 + i, float(i), clear=clear)
    row = s.lab_stats()["sounds"][0]
    assert row["completed"] == 6
    assert row["wearing_off"] is True


def test_lab_stats_counts_thefts_this_week_and_sorts_by_plays(tmp_path):
    # Fixture epoch 2030-01-01 UTC, far from the machine's real clock: the
    # theft-week window only matches these events if lab_stats derives "now"
    # from the injected clock (per real time they sit years outside it).
    now = 1893456000.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    _catch(s, "growl.mp3", now - 600, 1.0, clear=3.0, taken=["sandwich", "knife"])
    _catch(s, "growl.mp3", now - 300, 2.0, clear=2.0)
    # Ten days old: outside the theft week, still counted as a play.
    _catch(s, "chirp.wav", now - 10 * 86400, 3.0, clear=2.0, taken=["broccoli"])
    st = s.lab_stats()
    assert st["thefts_this_week"] == 2
    assert [r["sound"] for r in st["sounds"]] == ["growl.mp3", "chirp.wav"]


def test_lab_stats_pending_outcome_counts_as_play_only(tmp_path):
    now = 1783360800.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    _catch(s, "chirp.wav", now - 60, 1.0, outcome=False)
    row = s.lab_stats()["sounds"][0]
    assert row["plays"] == 1 and row["completed"] == 0
    assert row["deterred_rate"] is None and row["avg_clear_s"] is None
    assert row["wearing_off"] is False


def test_lab_stats_taken_event_is_not_deterred(tmp_path):
    now = 1783360800.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    # Left quickly, but with the sandwich: that's a failure, not a deterrence.
    _catch(s, "chirp.wav", now - 60, 1.0, clear=3.0, taken=["sandwich"])
    row = s.lab_stats()["sounds"][0]
    assert row["completed"] == 1
    assert row["deterred_rate"] == 0.0


def test_lab_stats_ignores_events_without_sound(tmp_path):
    now = 1783360800.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    s.add(_img(), 0.9, 1.0, now - 60, 1.0)   # no sound attached
    st = s.lab_stats()
    assert st["sounds"] == [] and st["thefts_this_week"] == 0


def test_report_card_quiet_week_when_no_events(tmp_path):
    now = 1893456000.0  # 2030-01-01 UTC: far from the real clock, like the lab tests
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    rc = s.stats()["report_card"]
    assert rc["grade"] == "A"
    assert rc["summary"] == "A quiet week."
    assert rc["attempts"] == 0 and rc["attempts_prev"] == 0
    assert rc["deterred_rate"] is None


def test_report_card_attempts_rose_with_weak_deterrence(tmp_path):
    now = 1893456000.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    # This week: 8 attempts, all completed, 4 deterred (quick, empty-handed).
    for i in range(4):
        _catch(s, "chirp.wav", now - 600 + i, float(i), clear=3.0)
    for i in range(4):
        _catch(s, "chirp.wav", now - 500 + i, float(4 + i), clear=30.0)  # too slow
    # Prev week: 3 attempts, 8 days ago.
    for i in range(3):
        _catch(s, "chirp.wav", now - 8 * 86400 + i, float(10 + i), clear=3.0)
    rc = s.stats()["report_card"]
    # 100 - min(40, 5*8) = 60; rose (8 > 3) -> 30; x 4/8 deterred -> 15 -> F.
    assert rc["grade"] == "F"
    assert rc["attempts"] == 8 and rc["attempts_prev"] == 3
    assert rc["deterred_rate"] == pytest.approx(0.5)
    assert rc["summary"] == "8 attempts, 4 of 8 deterred, up from 3 last week."


def test_report_card_attempts_fell_all_deterred(tmp_path):
    now = 1893456000.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    # This week: 5 attempts, all deterred. Prev week: 9.
    for i in range(5):
        _catch(s, "hawk.wav", now - 600 + i, float(i), clear=3.0)
    for i in range(9):
        _catch(s, "hawk.wav", now - 8 * 86400 + i, float(10 + i), clear=3.0)
    rc = s.stats()["report_card"]
    # 100 - min(40, 5*5) = 75; fell (5 < 9) -> +10 = 85; x 1.0 -> 85 -> mid-B.
    assert rc["grade"] == "B"
    assert rc["attempts"] == 5 and rc["attempts_prev"] == 9
    assert rc["deterred_rate"] == pytest.approx(1.0)
    assert rc["summary"] == "5 attempts, all deterred, down from 9 last week."


def test_report_card_no_outcome_clause_when_nothing_completed(tmp_path):
    now = 1893456000.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    # This week: 5 attempts still awaiting the outcome watcher. Prev week: 9.
    for i in range(5):
        _catch(s, "hawk.wav", now - 600 + i, float(i), outcome=False)
    for i in range(9):
        _catch(s, "hawk.wav", now - 8 * 86400 + i, float(10 + i), clear=3.0)
    rc = s.stats()["report_card"]
    # 100 - 25 = 75; fell -> 85; nothing completed -> no deterrence factor -> B.
    assert rc["grade"] == "B"
    assert rc["deterred_rate"] is None
    assert rc["summary"] == "5 attempts, down from 9 last week."


def test_report_card_equal_weeks_get_no_trend_adjustment(tmp_path):
    now = 1893456000.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    # This week: 6 attempts, all deterred. Prev week: also 6.
    for i in range(6):
        _catch(s, "hawk.wav", now - 600 + i, float(i), clear=3.0)
    for i in range(6):
        _catch(s, "hawk.wav", now - 8 * 86400 + i, float(10 + i), clear=3.0)
    rc = s.stats()["report_card"]
    # 100 - min(40, 5*6) = 70; equal weeks -> neither rose nor fell; x 1.0 = 70.
    # C band is [65, 80), thirds of 5: C+ at >=75, C- below 70, so 70 is mid-C.
    assert rc["grade"] == "C"
    assert rc["attempts"] == 6 and rc["attempts_prev"] == 6
    assert rc["deterred_rate"] == pytest.approx(1.0)
    assert "up from" not in rc["summary"] and "down from" not in rc["summary"]
    assert rc["summary"] == "6 attempts, all deterred."


def test_report_card_partial_outcomes_summary_counts_out_of_attempts(tmp_path):
    now = 1893456000.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    # This week: 5 attempts, only 2 completed (1 deterred, 1 too slow),
    # 3 still awaiting the outcome watcher. Prev week: 8.
    _catch(s, "hawk.wav", now - 600, 0.0, clear=3.0)
    _catch(s, "hawk.wav", now - 590, 1.0, clear=30.0)
    for i in range(3):
        _catch(s, "hawk.wav", now - 500 + i, float(2 + i), outcome=False)
    for i in range(8):
        _catch(s, "hawk.wav", now - 8 * 86400 + i, float(10 + i), clear=3.0)
    rc = s.stats()["report_card"]
    # 100 - min(40, 5*5) = 75; fell (5 < 8) -> +10 = 85; x 1/2 completed = 42.5 -> F.
    assert rc["grade"] == "F"
    assert rc["attempts"] == 5 and rc["attempts_prev"] == 8
    # Rate divides by COMPLETED (2); the summary counts out of ATTEMPTS (5).
    assert rc["deterred_rate"] == pytest.approx(0.5)
    assert "1 of 5 deterred" in rc["summary"]
    assert rc["summary"] == "5 attempts, 1 of 5 deterred, down from 8 last week."


def test_backfills_wall_time_from_thumb_mtime(tmp_path):
    import os
    (tmp_path / "fire_5.jpg").write_bytes(b"\xff\xd8\xff")
    mtime = 1_700_000_000.0
    os.utime(tmp_path / "fire_5.jpg", (mtime, mtime))
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"ts": 5.0, "confidence": 0.7, "thumb": "fire_5.jpg"}) + "\n")
    e = EventStore(tmp_path, 100, 0).list()[0]
    assert e.wall_time == mtime                       # backfilled from jpg mtime
    persisted = json.loads((tmp_path / "events.jsonl").read_text().strip())
    assert persisted["wall_time"] == mtime            # and persisted to disk


def test_no_backfill_when_thumb_missing(tmp_path):
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"ts": 5.0, "confidence": 0.7, "thumb": "gone.jpg"}) + "\n")
    assert EventStore(tmp_path, 100, 0).list()[0].wall_time is None
