"""One daemon pass: reap stale runs, pick (or synthesize) a job, run it."""
from __future__ import annotations

import time

from trainer_daemon.cloud import write_billing
from trainer_daemon.env import JOBS_DIR, STALE_RUNNING, log
from trainer_daemon.queue import jobs, synthesize_job, write_result
from trainer_daemon.runs import RUNNERS


def _reap_running() -> bool:
    """Handle any 'running' job: True means one is still live (pass ends)."""
    for job in jobs():
        if job.get("status") != "running":
            continue
        if time.time() - job.get("updated_at", 0.0) < STALE_RUNNING:
            log(f"{job['id']} still running; nothing to do")
            return True
        write_result(job["id"], "failed", "stale: gave up after 3h")
    return False


def _next_job() -> dict | None:
    queued = [j for j in jobs() if j.get("status") == "queued"]
    job = min(queued, key=lambda j: j.get("requested_at", 0.0), default=None)
    if job is not None:
        return job
    return synthesize_job(jobs())


def main() -> int:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    write_billing()  # keep the training page's budget card fresh
    if _reap_running():
        return 0

    job = _next_job()
    if job is None:
        log("nothing to do")
        return 0

    log(f"running {job['kind']} job {job['id']}")
    write_result(job["id"], "running", "")
    try:
        detail = RUNNERS[job["kind"]](job)
    except Exception as exc:
        log(f"job {job['id']} FAILED: {exc}")
        write_result(job["id"], "failed", str(exc)[:300])
        return 1
    extra = {"summary": job["_summary"]} if "_summary" in job else None
    write_result(job["id"], "done", detail, extra)
    log(f"job {job['id']} done: {detail}")
    return 0
