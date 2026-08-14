"""EventStore stats: daily/weekly activity counts and the weekly report card."""
import pytest

from doggy.events.store import EventStore

from tests.events.conftest import _catch, _img


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
