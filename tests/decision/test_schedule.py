from datetime import datetime

import pytest
from pydantic import ValidationError

from doggy.core.config import TunableSettings
from doggy.decision.schedule import armed_state


def _cfg(**kw):
    return TunableSettings(schedule_enabled=True, **kw)


# 2026-07-06 is a Monday; the window covers weeknights 21:00 -> 07:00.
WINDOW_NIGHT = {"days": [0, 1, 2, 3, 4], "start": "21:00", "end": "07:00"}


def test_inside_window_is_armed():
    armed, _ = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                           datetime(2026, 7, 6, 23, 0).timestamp())
    assert armed


def test_overnight_wrap_covers_early_morning():
    # Tuesday 03:00 belongs to Monday's 21:00-07:00 window.
    armed, _ = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                           datetime(2026, 7, 7, 3, 0).timestamp())
    assert armed


def test_outside_window_is_off_duty_with_countdown():
    armed, nxt = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                             datetime(2026, 7, 6, 12, 0).timestamp())
    assert not armed
    assert nxt == pytest.approx(9 * 3600)   # 12:00 -> 21:00


def test_schedule_disabled_means_always_armed():
    armed, nxt = armed_state(TunableSettings(), 0.0)
    assert armed and nxt is None


def test_enabled_but_no_windows_means_always_armed():
    armed, nxt = armed_state(_cfg(armed_windows=[]), 0.0)
    assert armed and nxt is None


def test_bad_window_times_rejected():
    with pytest.raises(ValidationError):
        TunableSettings(armed_windows=[{"days": [0], "start": "25:00", "end": "07:00"}])


def test_empty_days_rejected():
    with pytest.raises(ValidationError):
        TunableSettings(armed_windows=[{"days": [], "start": "21:00", "end": "07:00"}])


def test_out_of_range_day_rejected():
    with pytest.raises(ValidationError):
        TunableSettings(armed_windows=[{"days": [7], "start": "21:00", "end": "07:00"}])


def test_end_boundary_is_exclusive_off_duty():
    # 07:00 sharp is the flip out of the overnight window: off duty.
    armed, nxt = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                             datetime(2026, 7, 7, 7, 0).timestamp())
    assert not armed
    # Tuesday 07:00 -> Tuesday 21:00 is the next arming.
    assert nxt == pytest.approx(14 * 3600)


def test_start_boundary_is_inclusive_armed():
    armed, nxt = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                             datetime(2026, 7, 6, 21, 0).timestamp())
    assert armed
    # Monday 21:00 -> Tuesday 07:00 is the next flip to off duty.
    assert nxt == pytest.approx(10 * 3600)


def test_countdown_while_armed_points_at_window_end():
    armed, nxt = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                             datetime(2026, 7, 6, 23, 0).timestamp())
    assert armed
    assert nxt == pytest.approx(8 * 3600)   # Mon 23:00 -> Tue 07:00


def test_daytime_only_non_wrapping_window():
    # A same-day window (no midnight wrap): 09:00-17:00 on Saturday (day 5).
    day = {"days": [5], "start": "09:00", "end": "17:00"}
    inside, _ = armed_state(_cfg(armed_windows=[day]),
                            datetime(2026, 7, 11, 12, 0).timestamp())   # Sat noon
    outside, nxt = armed_state(_cfg(armed_windows=[day]),
                               datetime(2026, 7, 11, 8, 0).timestamp())  # Sat 08:00
    assert inside
    assert not outside
    assert nxt == pytest.approx(3600)   # 08:00 -> 09:00


def test_weekend_gap_is_off_duty():
    # Saturday is not in the weeknight window, and Friday's window ended 07:00.
    armed, _ = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                           datetime(2026, 7, 11, 12, 0).timestamp())   # Saturday
    assert not armed


def test_sunday_night_wraps_into_monday_morning():
    # Sunday=6; a 21:00-07:00 window belongs to its start day, so Monday
    # 2026-07-06 03:00 is covered by Sunday 2026-07-05's night. This pins the
    # week-wrap: Monday's weekday 0 must map back to day 6, not day -1.
    sunday_night = {"days": [6], "start": "21:00", "end": "07:00"}
    armed, _ = armed_state(_cfg(armed_windows=[sunday_night]),
                           datetime(2026, 7, 6, 3, 0).timestamp())
    assert armed
