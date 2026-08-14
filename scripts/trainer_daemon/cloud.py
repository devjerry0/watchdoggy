"""Talking to Modal: kicking off cloud runs (with auto GPU/batch tiering)
and reading workspace billing for per-run cost attribution."""
from __future__ import annotations

import json
import os
import subprocess
import time

from trainer_daemon.env import (
    BATCH_TIERS,
    DATASET_DIR,
    DOGGY_ROOT,
    GPU_TIERS,
    JOBS_DIR,
    MODAL,
    MODAL_SUBPROCESS_TIMEOUT,
    PIPELINE,
    log,
    settings,
)
from trainer_daemon.queue import sidecar_stats

BILLING_CLI_TIMEOUT_S = 60


def billing_summary() -> dict | None:
    """Workspace spend this month, straight from Modal. The workspace runs
    only this appliance, so before/after deltas attribute cost per run."""
    try:
        proc = subprocess.run([str(MODAL), "billing", "summary", "--json"],
                              capture_output=True,
                              timeout=BILLING_CLI_TIMEOUT_S, check=True)
        summary = json.loads(proc.stdout)
        return {"metered_cost": float(summary.get("metered_cost", 0)),
                "billed_cost": float(summary.get("billed_cost", 0)),
                "credits_used": -float(summary.get("adjustments", {})
                                       .get("credits", 0)),
                "fetched_at": time.time()}
    except Exception as exc:
        log(f"WARNING: billing summary unavailable: {exc}")
        return None


def write_billing() -> dict | None:
    summary = billing_summary()
    if summary:
        (JOBS_DIR / "billing.json").write_text(json.dumps(summary))
    return summary


def run_cost(before: dict | None) -> float | None:
    after = write_billing()
    if not (before and after):
        return None
    return round(max(0.0, after["metered_cost"] - before["metered_cost"]), 2)


def gpu() -> str:
    chosen = settings().get("gpu", "auto")
    if chosen != "auto":
        return chosen
    _, labeled, _ = sidecar_stats()
    return next(tier for floor, tier in GPU_TIERS if labeled >= floor)


def batch(job: dict) -> int:
    chosen = {**settings(), **(job.get("params") or {})}.get("batch", "auto")
    if chosen != "auto":
        return int(chosen)
    _, labeled, _ = sidecar_stats()
    return next(size for floor, size in BATCH_TIERS if labeled >= floor)


def modal_run(entrypoint: str, arguments: list[str], job_id: str) -> None:
    # The full cloud-run output streams into the job's log file, which the
    # training page tails live via /api/training/log/{job_id}.
    chosen_gpu = gpu()
    log(f"cloud GPU: {chosen_gpu}")
    with open(JOBS_DIR / f"{job_id}.log", "ab") as log_file:
        subprocess.run([str(MODAL), "run", f"{PIPELINE}::{entrypoint}",
                        "--dataset-dir", str(DATASET_DIR)] + arguments,
                       check=True, cwd=DOGGY_ROOT,
                       timeout=MODAL_SUBPROCESS_TIMEOUT,
                       stdout=log_file, stderr=subprocess.STDOUT,
                       env={**os.environ, "DOGGY_TRAIN_GPU": chosen_gpu})
