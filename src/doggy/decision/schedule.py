from __future__ import annotations

from datetime import datetime, timedelta
from itertools import product
from typing import Iterator

from doggy.core.config import ArmedWindow, TunableSettings

# How far ahead _seconds_to_flip looks for the next arm/disarm boundary. A week
# plus a day comfortably covers any weekly pattern (the widest gap between a
# window on one weekday and the next is under 7 days).
_LOOKAHEAD_DAYS = 8


def _to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def _window_active(w: ArmedWindow, dt: datetime) -> bool:
    minutes = dt.hour * 60 + dt.minute
    start = _to_minutes(w.start)
    end = _to_minutes(w.end)
    if end > start:
        return dt.weekday() in w.days and start <= minutes < end
    # Overnight wrap: the window belongs to its START day. Either we are on the
    # start day at/after start, or on the following day before end.
    if dt.weekday() in w.days and minutes >= start:
        return True
    return (dt.weekday() - 1) % 7 in w.days and minutes < end


def _seconds_to_flip(windows: tuple[ArmedWindow, ...], dt: datetime, armed: bool) -> float | None:
    """Seconds until the armed/off-duty state next flips, or None if it never
    does within the lookahead (e.g. windows that cover the whole week).

    Evaluated at boundaries only: the active state can only change at a window's
    start or end minute, so we test each such minute in the next few days and
    take the earliest one whose active-state differs from now.
    """
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    flips = (cand for cand in _boundaries(windows, midnight)
             if cand > dt
             and any(_window_active(x, cand) for x in windows) != armed)
    best = min(flips, default=None)
    if best is None:
        return None
    return (best - dt).total_seconds()


def _boundaries(windows: tuple[ArmedWindow, ...],
                midnight: datetime) -> Iterator[datetime]:
    """Every window start/end minute over the lookahead days."""
    for day_offset, w in product(range(_LOOKAHEAD_DAYS + 1), windows):
        day = midnight + timedelta(days=day_offset)
        yield day + timedelta(minutes=_to_minutes(w.start))
        yield day + timedelta(minutes=_to_minutes(w.end))


def within_windows(windows: tuple[ArmedWindow, ...], wall_now: float) -> bool:
    """True if any weekly window covers this moment (wall-clock, local time).
    Empty windows -> False; each caller decides what "no windows" means."""
    if not windows:
        return False
    dt = datetime.fromtimestamp(wall_now)
    return any(_window_active(w, dt) for w in windows)


def armed_state(cfg: TunableSettings, wall_now: float) -> tuple[bool, float | None]:
    """Whether the appliance should react right now, and the seconds until that
    changes. When the schedule is off (or has no windows) it is always armed and
    there is no countdown."""
    if not cfg.schedule_enabled or not cfg.armed_windows:
        return True, None
    armed = within_windows(cfg.armed_windows, wall_now)
    dt = datetime.fromtimestamp(wall_now)
    return armed, _seconds_to_flip(cfg.armed_windows, dt, armed)
