"""Appliance paths, recipe defaults, and the two local side doors the
daemon uses: stdout logging (journald) and the detector's HTTPS API."""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

DOGGY_ROOT = Path("/home/doggy/doggy")
JOBS_DIR = DOGGY_ROOT / "jobs"
DATASET_DIR = DOGGY_ROOT / "dataset"
DEPLOYED_BUNDLE = DOGGY_ROOT / "models/kitchen_ncnn_model"
STAGING_BUNDLE = Path.home() / "staging_ncnn_model"
MODAL = Path.home() / "modal-env/bin/modal"
PIPELINE = DOGGY_ROOT / "scripts/modal_pipeline.py"
LOCAL_API = "https://localhost:8443"
LOCAL_API_TIMEOUT_S = 15

# ONE real deadline: every cloud job gets the same 10h ceiling
# (CLOUD_JOB_CEILING in scripts/modal_pipeline.py -- keep these in sync).
# The wrappers below are not extra timeouts, just the wait layers that must
# OUTLAST that ceiling or they'd kill the wait and strand the result:
# subprocess +30min, then systemd TimeoutStartSec / stale-reap +1h
# (setup-pi-trainer.sh writes the systemd value).
CLOUD_JOB_CEILING = 10 * 3600
# A nightly job runs TWO cloud phases back to back (prelabel, then
# consensus), each under the ceiling: the wrapper must outlast both.
MODAL_SUBPROCESS_TIMEOUT = 2 * CLOUD_JOB_CEILING + 1800
STALE_RUNNING = float(2 * CLOUD_JOB_CEILING + 3600)

# Recipe + schedule defaults; the training page's settings file overrides.
# Schedule (user-decided 2026-10-11): label weekly, train every other
# labeling pass (~twice a month).
SETTINGS_DEFAULTS = {"epochs": 80, "batch": "auto", "freeze": 10,
                     "augment": "auto", "train_interval_hours": 336,
                     "label_interval_hours": 168,
                     "min_new_labels": 1000, "nightly_prelabel_hour": 2,
                     "gpu": "auto", "auto_update": True,
                     "monthly_credits": 30}
# "auto" GPU. Benchmarked 2026-10-11 (yolo26n training at the pipeline's
# settings, cpu=8): L40S 416-435 img/s vs L4 180, A100-40GB 184, H100
# 445-478; more CPU changed nothing. L40S is ~2.3x faster than L4 AND ~20%
# cheaper per trained image ($1.60 vs $2.02 per million incl. cpu/memory).
AUTO_GPU = "L40S"
# Labeling GPU, measured separately (yolo26x inference on real mirror
# frames, cpu=4, one frame per call): T4 21 frames/s, L4 38, L40S 81.5 --
# L40S matches L4 per frame ($0.0075 vs $0.0077 per 1k) at 2x the speed;
# T4 is the priciest per frame. Same card as training today, but the two
# workloads differ, so they are chosen independently.
AUTO_LABEL_GPU = "L40S"
# auto batch. Benchmarked 2026-10-11 on L40S: batch 32 422 img/s, 64 468,
# 128 532, 256 540 -- 128 is the ceiling (+26%, -21% cost per image). Tiny
# first-day datasets keep 16 so an epoch still has ~30+ optimizer steps.
BATCH_TIERS = ((800, 128), (0, 16))


def settings() -> dict:
    merged = dict(SETTINGS_DEFAULTS)
    path = JOBS_DIR / "trainer-settings.json"
    if path.is_file():
        try:
            merged.update(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return merged


def log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def api_post(path: str, payload: dict,
             timeout: float = LOCAL_API_TIMEOUT_S) -> dict:
    context = ssl.create_default_context()  # household CA: skip verification
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(
        f"{LOCAL_API}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, context=context,
                                timeout=timeout) as resp:
        return json.loads(resp.read())
