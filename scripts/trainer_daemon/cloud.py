"""Talking to Modal: kicking off cloud runs (auto GPU, auto batch tiering).
Money lives in trainer_daemon.billing."""
from __future__ import annotations

import os
import subprocess

from trainer_daemon.env import (
    AUTO_GPU,
    AUTO_LABEL_GPU,
    BATCH_TIERS,
    DATASET_DIR,
    DOGGY_ROOT,
    JOBS_DIR,
    MODAL,
    MODAL_SUBPROCESS_TIMEOUT,
    PIPELINE,
    log,
    settings,
)
from trainer_daemon.queue import sidecar_stats


def gpu() -> str:
    chosen = settings().get("gpu", "auto")
    return AUTO_GPU if chosen == "auto" else chosen


def batch(job: dict) -> int:
    chosen = {**settings(), **(job.get("params") or {})}.get("batch", "auto")
    if chosen != "auto":
        return int(chosen)
    _, labeled, _ = sidecar_stats()
    return next(size for floor, size in BATCH_TIERS if labeled >= floor)


def modal_run(entrypoint: str, arguments: list[str], job_id: str) -> None:
    # The full cloud-run output streams into the job's log file, which the
    # training page tails live via /api/training/log/{job_id}.
    # Training and labeling are different workloads with different best
    # GPUs: the pipeline reads DOGGY_TRAIN_GPU for the train container and
    # DOGGY_LABEL_GPU for the prelabel container.
    chosen_gpu = gpu()
    labeling = entrypoint == "kickoff_prelabels"
    log(f"cloud GPU: {AUTO_LABEL_GPU if labeling else chosen_gpu} "
        f"({'labeling' if labeling else 'training'})")
    with open(JOBS_DIR / f"{job_id}.log", "ab") as log_file:
        subprocess.run([str(MODAL), "run", f"{PIPELINE}::{entrypoint}",
                        "--dataset-dir", str(DATASET_DIR)] + arguments,
                       check=True, cwd=DOGGY_ROOT,
                       timeout=MODAL_SUBPROCESS_TIMEOUT,
                       stdout=log_file, stderr=subprocess.STDOUT,
                       env={**os.environ, "DOGGY_TRAIN_GPU": chosen_gpu,
                            "DOGGY_LABEL_GPU": AUTO_LABEL_GPU})
