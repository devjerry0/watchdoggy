"""One daemon pass: reap stale runs, pick (or synthesize) a job, run it."""
from __future__ import annotations

import subprocess
import time

from trainer_daemon.billing import credit_refusal, write_billing
from trainer_daemon.clock import sync_clock
from trainer_daemon.env import JOBS_DIR, STALE_RUNNING, log
from trainer_daemon.queue import jobs, synthesize_job, write_result
from trainer_daemon.runs import RUNNERS


def _pipeline_alive() -> bool:
    """Whether a cloud-pipeline child (`modal run .../modal_pipeline.py`) is
    live for the trainer user. Systemd oneshot passes never overlap, so a
    'running' overlay with no live child is a crash/power-loss leftover --
    reap it now instead of idling out the full STALE_RUNNING window. (A
    hand-run 'update' job has no modal child and may be reaped early; its
    final write_result overwrites the overlay, so no state is lost.)"""
    probe = subprocess.run(
        ["pgrep", "-u", "trainer", "-f", "modal_pipeline.py"],
        capture_output=True)
    return probe.returncode == 0


def _reap_running() -> bool:
    """Handle any 'running' job: True means one is still live (pass ends)."""
    for job in jobs():
        if job.get("status") != "running":
            continue
        if not _pipeline_alive():
            write_result(job["id"], "failed",
                         "stale: no live pipeline process (daemon died mid-run)")
            continue
        if time.time() - job.get("updated_at", 0.0) < STALE_RUNNING:
            log(f"{job['id']} still running; nothing to do")
            return True
        write_result(job["id"], "failed",
                     f"stale: gave up after {STALE_RUNNING / 3600:.0f}h")
    return False


def _next_job() -> dict | None:
    queued = [j for j in jobs() if j.get("status") == "queued"]
    job = min(queued, key=lambda j: j.get("requested_at", 0.0), default=None)
    if job is not None:
        return job
    return synthesize_job(jobs())


def main() -> int:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    sync_clock()  # first: everything below stamps times
    write_billing()  # keep the training page's budget card fresh
    if _reap_running():
        return 0

    job = _next_job()
    if job is None:
        log("nothing to do")
        return 0

    refusal = credit_refusal(job["kind"])
    if refusal is not None:
        log(f"job {job['id']} {refusal}")
        # "refused" (not "failed"): no-credits must never masquerade as
        # "cloud broken" -- it would arm the failure backoff and banner.
        write_result(job["id"], "refused", refusal)
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
