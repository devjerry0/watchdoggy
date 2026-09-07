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
MODAL_SUBPROCESS_TIMEOUT = CLOUD_JOB_CEILING + 1800
STALE_RUNNING = float(CLOUD_JOB_CEILING + 3600)

# Recipe + schedule defaults; the training page's settings file overrides.
SETTINGS_DEFAULTS = {"epochs": 80, "batch": "auto", "freeze": 10,
                     "augment": True, "train_interval_hours": 48,
                     "min_new_labels": 5, "nightly_prelabel_hour": 2,
                     "gpu": "auto", "auto_update": True,
                     "monthly_credits": 30}
# auto tiers by labeled-frame count. The nano model is dataloader-bound on
# small sets -- a bigger GPU only pays once epochs are long enough; batch
# grows with data but stays small enough for ~30+ optimizer steps/epoch.
GPU_TIERS = ((2500, "A10G"), (0, "L4"))
BATCH_TIERS = ((2500, 64), (800, 32), (0, 16))


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
