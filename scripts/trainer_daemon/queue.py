"""The daemon's side of the job queue: reading requests, writing result
overlays, and the auto rules that synthesize jobs when the queue is empty."""
from __future__ import annotations

import json
import time

from trainer_daemon.billing import credit_refusal
from trainer_daemon.env import DATASET_DIR, JOBS_DIR, log, settings
from trainer_daemon.update import update_due

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24 * 3600
# Schedules run at fixed hours, so "a week since the last pass" must allow
# for the pass having finished a little after the hour it started.
SCHEDULE_SLACK = 12 * SECONDS_PER_HOUR
# Training runs on the pass right after a labeling pass; labels older than
# this mean that pass was missed (e.g. refused for credits) -- wait for the
# next one.
FRESH_LABELS_SECONDS = 12 * SECONDS_PER_HOUR

# Auto jobs pause after this many consecutive failures of a kind, waiting
# 1h, then doubling up to 24h. Born from the Sep 2026 outage: a full Volume
# failed every run in 90s and the daemon re-queued one per tick for six
# days (~$13 of credits, silently). Manual jobs from the web UI bypass this
# entirely -- they never pass through synthesize_job.
BACKOFF_AFTER = 3
BACKOFF_BASE_SECONDS = SECONDS_PER_HOUR
BACKOFF_CAP_SECONDS = SECONDS_PER_DAY
# A single failure this expensive arms the backoff on its own: three
# back-to-back 10h hang-failures would otherwise burn ~the whole monthly
# budget before the third strike. (Duration uses requested_at, so a manual
# job that sat queued while the timer was down can trip this early -- cheap
# false positive, manual retries bypass the backoff anyway.)
LONG_FAILURE_SECONDS = 4 * SECONDS_PER_HOUR


def failure_backoff(existing: list[dict], kind: str, now: float) -> float | None:
    """Seconds until auto jobs of `kind` may run again, or None when clear.
    Counts the newest streak of failed results; any success resets it.
    Refusals (credit gate) carry status "refused" and never count."""
    finished = sorted((j for j in existing if j.get("kind") == kind
                       and j.get("status") in ("done", "failed")),
                      key=lambda j: j.get("updated_at", 0.0), reverse=True)
    streak = 0
    for job in finished:
        if job.get("status") != "failed":
            break
        streak += 1
    if streak:
        newest = finished[0]
        duration = newest.get("updated_at", 0.0) - newest.get("requested_at", 0.0)
        if newest.get("requested_at") and duration >= LONG_FAILURE_SECONDS:
            streak = max(streak, BACKOFF_AFTER)
    if streak < BACKOFF_AFTER:
        return None
    wait = min(BACKOFF_BASE_SECONDS * 2 ** min(streak - BACKOFF_AFTER, 5),
               BACKOFF_CAP_SECONDS)
    # Clamp a future-reading stamp (Pi clock regressed after the failure was
    # written): never let skew stretch the window beyond `wait` from now.
    newest_at = min(finished[0].get("updated_at", 0.0), now)
    remaining = newest_at + wait - now
    return remaining if remaining > 0 else None


def jobs() -> list[dict]:
    found = []
    for path in sorted(JOBS_DIR.glob("job_*.json")):
        if path.name.endswith(".result.json"):
            continue
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        result_path = path.with_name(f"{path.stem}.result.json")
        if result_path.is_file():
            job.update(json.loads(result_path.read_text()))
        found.append(job)
    return found


def write_result(job_id: str, status: str, detail: str,
                 extra: dict | None = None) -> None:
    payload = {"status": status, "detail": detail, "updated_at": time.time()}
    payload.update(extra or {})
    (JOBS_DIR / f"{job_id}.result.json").write_text(json.dumps(payload))


def sidecar_stats() -> tuple[int, int, float]:
    """(missing_prelabels, labeled_count, newest_labeled_at)."""
    missing, labeled, newest = 0, 0, 0.0
    for sidecar in DATASET_DIR.glob("sample_*.json"):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if "prelabels" not in meta:
            missing += 1
        if not meta.get("human_label"):
            continue
        labeled += 1
        newest = max(newest, meta.get("labeled_at", 0.0))
    return missing, labeled, newest


def labels_since(when: float) -> int:
    count = 0
    for sidecar in DATASET_DIR.glob("sample_*.json"):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("human_label") and meta.get("labeled_at", 0.0) > when:
            count += 1
    return count


def last_nightly_slot(hour: int) -> float:
    """The most recent occurrence of the nightly hour, as a timestamp."""
    slot = time.localtime()
    seconds_today = slot.tm_hour * 3600 + slot.tm_min * 60 + slot.tm_sec
    slot_offset = hour * 3600
    if seconds_today >= slot_offset:
        return time.time() - (seconds_today - slot_offset)
    return time.time() - seconds_today - SECONDS_PER_DAY + slot_offset


def synthesize_job(existing: list[dict]) -> dict | None:
    def newest_done(kind: str) -> float:
        return max((j.get("updated_at", 0.0) for j in existing
                    if j.get("kind") == kind and j.get("status") == "done"),
                   default=0.0)

    conf = settings()
    now = time.time()
    # Out of Modal credits: don't manufacture cloud jobs that would only
    # be refused every pass; self-updates still run (GitHub, not Modal).
    if credit_refusal("train") is not None:
        return _update_job(now)
    missing, _, _ = sidecar_stats()
    last_prelabel = newest_done("prelabel")
    label_wait = failure_backoff(existing, "prelabel", now)
    # Labeling at most every label_interval_hours (weekly, user-decided), at
    # the configured hour; SCHEDULE_SLACK lets a pass that finished at 02:40
    # come due again at 02:00 a week later. Labeling goes FIRST.
    label_due = (missing > 0
                 and last_prelabel < last_nightly_slot(conf["nightly_prelabel_hour"])
                 and now - last_prelabel >= conf["label_interval_hours"]
                 * SECONDS_PER_HOUR - SCHEDULE_SLACK)
    if label_due:
        if label_wait is None:
            return queue_auto("prelabel",
                              f"weekly: {missing} new frames to label")
        log(f"weekly labeling due but backing off after repeated failures "
            f"({label_wait / SECONDS_PER_HOUR:.1f}h left)")
    last_train = newest_done("train")
    new_labels = labels_since(last_train)
    # Training only on the pass right after a labeling pass (user-decided:
    # every run trains on fresh auto-labels), so it lands on every other
    # weekly labeling day. Exceptions so it can never deadlock: nothing
    # awaits labels, or the labeler is stuck in failure backoff.
    just_labeled = (last_prelabel > last_train
                    and now - last_prelabel < FRESH_LABELS_SECONDS)
    if ((just_labeled or missing == 0 or label_wait is not None)
            and now - last_train >= conf["train_interval_hours"]
            * SECONDS_PER_HOUR - SCHEDULE_SLACK
            and new_labels >= conf["min_new_labels"]):
        wait = failure_backoff(existing, "train", now)
        if wait is None:
            return queue_auto("train",
                              f"auto: {new_labels} new labels since last run")
        log(f"auto-train due but backing off after repeated failures "
            f"({wait / SECONDS_PER_HOUR:.1f}h left); fix the cause or queue "
            f"a manual run from /training")
    return _update_job(now)


def _update_job(now: float) -> dict | None:
    # Lowest priority: only checked when no training/prelabeling is due.
    versions = update_due(now)
    if versions is None:
        return None
    current, latest = versions
    return queue_auto("update", f"auto: {latest} released (installed {current})")


def queue_auto(kind: str, why: str) -> dict:
    job = {"id": f"job_{int(time.time() * 1000)}", "kind": kind,
           "status": "queued", "requested_at": time.time(),
           "updated_at": time.time(), "detail": why, "auto": True}
    (JOBS_DIR / f"{job['id']}.json").write_text(json.dumps(job))
    log(f"auto-queued {kind}: {why}")
    return job
