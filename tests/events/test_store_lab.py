"""EventStore lab stats: per-sound deterrence effectiveness and wearing-off."""
import pytest

from doggy.events.store import EventStore

from tests.events.conftest import _catch, _img


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
