"""The daemon's side of the job queue: reading requests, writing result
overlays, and the auto rules that synthesize jobs when the queue is empty."""
from __future__ import annotations

import json
import time

from trainer_daemon.env import DATASET_DIR, JOBS_DIR, log, settings

AUTO_PRELABEL_MIN = 10
PRELABEL_COOLDOWN = 6 * 3600.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24 * 3600


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
    last_train = newest_done("train")
    new_labels = labels_since(last_train)
    if (now - last_train >= conf["train_interval_hours"] * SECONDS_PER_HOUR
            and new_labels >= conf["min_new_labels"]):
        return queue_auto("train",
                          f"auto: {new_labels} new labels since last run")
    missing, _, _ = sidecar_stats()
    last_prelabel = newest_done("prelabel")
    # Nightly: after the configured hour, prelabel EVERY new frame once, so
    # the morning's label queue is already stocked with ·x boxes.
    if missing > 0 and last_prelabel < last_nightly_slot(conf["nightly_prelabel_hour"]):
        return queue_auto("prelabel",
                          f"nightly: {missing} new frames to prelabel")
    if missing >= AUTO_PRELABEL_MIN and now - last_prelabel >= PRELABEL_COOLDOWN:
        return queue_auto("prelabel", f"auto: {missing} frames lack prelabels")
    return None


def queue_auto(kind: str, why: str) -> dict:
    job = {"id": f"job_{int(time.time() * 1000)}", "kind": kind,
           "status": "queued", "requested_at": time.time(),
           "updated_at": time.time(), "detail": why, "auto": True}
    (JOBS_DIR / f"{job['id']}.json").write_text(json.dumps(job))
    log(f"auto-queued {kind}: {why}")
    return job
