"""Per-sound deterrence analytics for the dashboard's lab card. Pure
functions over a records snapshot, like `report`."""
from __future__ import annotations

from datetime import datetime, timedelta

from doggy.events.record import EventRecord
from doggy.events.report import DAYS_PER_WEEK, STAYED_CLEAR_S, deterred

# Wearing-off detection: need this many completed plays before the trend is
# trusted, and the newer half must clear this much slower than the older half.
MIN_COMPLETED_FOR_TREND = 6
WEARING_OFF_RATIO = 1.5


def _wearing_off(completed: list[EventRecord]) -> bool:
    """True when a sound's recent clears run much slower than its early ones.

    Compares the average effective clear of the newer half of completed events
    against the older half (events arrive oldest -> newest). An odd count puts
    the extra event in the first half: the older, larger sample makes the
    steadier baseline for judging the newer events.
    """
    if len(completed) < MIN_COMPLETED_FOR_TREND:
        return False
    effective = [
        r.clear_seconds if r.clear_seconds is not None else STAYED_CLEAR_S
        for r in completed
    ]
    mid = (len(effective) + 1) // 2
    first = sum(effective[:mid]) / mid
    second = sum(effective[mid:]) / (len(effective) - mid)
    return second >= WEARING_OFF_RATIO * first


def _sound_row(sound: str, plays: list[EventRecord]) -> dict:
    completed = [r for r in plays if r.outcome_at is not None]
    dtr = [r for r in completed if deterred(r)]
    clears = [r.clear_seconds for r in plays if r.clear_seconds is not None]
    return {
        "sound": sound,
        "plays": len(plays),
        "completed": len(completed),
        "deterred_rate": len(dtr) / len(completed) if completed else None,
        "avg_clear_s": sum(clears) / len(clears) if clears else None,
        "wearing_off": _wearing_off(completed),
    }


def stats(records: list[EventRecord], now: float) -> dict:
    """Per-sound deterrence effectiveness.

    A play is any event with that sound; it completes once the outcome
    watcher stamps ``outcome_at``. Deterred means the target left within
    DETERRED_WITHIN_S seconds without taking anything.
    """
    # Same Pi-local calendar semantics as report.activity(): last 7 local days.
    week = {datetime.fromtimestamp(now).date() - timedelta(days=n)
            for n in range(DAYS_PER_WEEK)}
    thefts = sum(
        len(r.taken) for r in records
        if r.wall_time is not None and datetime.fromtimestamp(r.wall_time).date() in week
    )

    by_sound: dict[str, list[EventRecord]] = {}
    for r in records:  # records arrive oldest -> newest, so groups stay in time order
        if r.sound:
            by_sound.setdefault(r.sound, []).append(r)

    sounds = [_sound_row(sound, plays) for sound, plays in by_sound.items()]
    sounds.sort(key=lambda s: s["plays"], reverse=True)
    return {"sounds": sounds, "thefts_this_week": thefts}
