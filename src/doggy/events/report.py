"""Pure analytics over event records: the dashboard's activity summary and
the weekly report card. No I/O, no locks -- the store hands these functions
a snapshot and a clock reading. Per-sound analytics live in `lab`."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta

from doggy.events.record import EventRecord

# A sound "deterred" the target when it left within this many seconds and took nothing.
DETERRED_WITHIN_S = 15.0
# Effective clear time scored when the target never left: the outcome watcher gives
# up at MAX_WATCH_SECONDS (60s), so a no-clear outcome counts as the full watch.
STAYED_CLEAR_S = 60.0

# Report-card scoring: each attempt costs ATTEMPT_PENALTY points (capped),
# a worse week than last costs WORSE_WEEK_PENALTY, a better one earns
# BETTER_WEEK_BONUS, and the whole score scales by the deterred rate.
ATTEMPT_PENALTY = 5.0
ATTEMPT_PENALTY_CAP = 40.0
WORSE_WEEK_PENALTY = 30.0
BETTER_WEEK_BONUS = 10.0

# Report-card bands, best first: letter, band floor, band ceiling.
GRADE_BANDS = (("A", 90.0, 100.0), ("B", 80.0, 90.0), ("C", 65.0, 80.0), ("D", 50.0, 65.0))

DAYS_PER_WEEK = 7


def deterred(record: EventRecord) -> bool:
    """A completed event where the target left quickly and took nothing."""
    return (
        record.outcome_at is not None
        and record.clear_seconds is not None
        and record.clear_seconds <= DETERRED_WITHIN_S
        and not record.taken
    )


def _grade(score: float) -> str:
    """Letter for a 0-100 score; top third of a band earns '+', bottom third '-'."""
    for letter, lo, hi in GRADE_BANDS:
        if score < lo:
            continue
        third = (hi - lo) / 3
        if score >= hi - third:
            return letter + "+"
        if score < lo + third:
            return letter + "-"
        return letter
    return "F"


def _split_weeks(records: list[EventRecord],
                 today: date) -> tuple[list[EventRecord], int]:
    """This week's events + last week's attempt count (the trend baseline)."""
    this_week = {today - timedelta(days=n) for n in range(DAYS_PER_WEEK)}
    prev_week = {today - timedelta(days=n)
                 for n in range(DAYS_PER_WEEK, 2 * DAYS_PER_WEEK)}
    week: list[EventRecord] = []
    attempts_prev = 0
    for r in records:
        if r.wall_time is None:
            continue
        day = datetime.fromtimestamp(r.wall_time).date()
        if day in this_week:
            week.append(r)
        attempts_prev += day in prev_week
    return week, attempts_prev


def _trend_adjustment(attempts: int, attempts_prev: int) -> float:
    if attempts > attempts_prev:
        return -WORSE_WEEK_PENALTY
    if attempts < attempts_prev:
        return BETTER_WEEK_BONUS
    return 0.0


def _trend_note(attempts: int, attempts_prev: int) -> str | None:
    if attempts > attempts_prev:
        return f"up from {attempts_prev} last week"
    if attempts < attempts_prev:
        return f"down from {attempts_prev} last week"
    return None


def _summary(attempts: int, attempts_prev: int, completed: int,
             deterred_count: int) -> str:
    parts = [f"{attempts} attempts"]
    if completed:
        parts.append("all deterred" if deterred_count == attempts
                     else f"{deterred_count} of {attempts} deterred")
    note = _trend_note(attempts, attempts_prev)
    if note:
        parts.append(note)
    return ", ".join(parts) + "."


def report_card(records: list[EventRecord], today: date) -> dict:
    """Weekly letter grade: fewer attempts and quick, empty-handed exits score high."""
    week, attempts_prev = _split_weeks(records, today)
    attempts = len(week)
    if attempts == 0 and attempts_prev == 0:
        return {"grade": "A", "attempts": 0, "attempts_prev": 0,
                "deterred_rate": None, "summary": "A quiet week."}

    completed = [r for r in week if r.outcome_at is not None]
    dtr = [r for r in completed if deterred(r)]
    deterred_rate = len(dtr) / len(completed) if completed else None

    score = 100.0 - min(ATTEMPT_PENALTY_CAP, ATTEMPT_PENALTY * attempts)
    score += _trend_adjustment(attempts, attempts_prev)
    if deterred_rate is not None:
        score *= deterred_rate
    score = max(0.0, min(100.0, score))

    return {
        "grade": _grade(score),
        "attempts": attempts,
        "attempts_prev": attempts_prev,
        "deterred_rate": deterred_rate,
        "summary": _summary(attempts, attempts_prev, len(completed), len(dtr)),
    }


def activity(records: list[EventRecord], today: date) -> dict:
    """Activity summary for the dashboard, bucketed by local wall-clock time."""
    # Last 7 calendar days, oldest -> newest, with today last.
    days = [today - timedelta(days=n) for n in range(DAYS_PER_WEEK - 1, -1, -1)]
    counts: dict = {day: 0 for day in days}
    hours: list[int] = []
    for record in records:
        if record.wall_time is None:
            continue
        dt = datetime.fromtimestamp(record.wall_time)
        hours.append(dt.hour)
        if dt.date() in counts:
            counts[dt.date()] += 1

    latencies = [r.latency_s for r in records if r.latency_s is not None]
    return {
        "today": counts[today],
        "this_week": sum(counts.values()),
        "per_day": [{"day": day.isoformat(), "count": counts[day]} for day in days],
        "busiest_hour": Counter(hours).most_common(1)[0][0] if hours else None,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "report_card": report_card(records, today),
    }
