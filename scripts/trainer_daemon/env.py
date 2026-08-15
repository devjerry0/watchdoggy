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

# Timeout stack, outermost to innermost: systemd (4h) > this daemon's
# subprocess (3.5h) > the Modal function (3h). Each layer outlasts the one
# inside it, so the innermost real deadline is the one that fires.
STALE_RUNNING = 4 * 3600.0
MODAL_SUBPROCESS_TIMEOUT = int(3.5 * 3600)

# Recipe + schedule defaults; the training page's settings file overrides.
SETTINGS_DEFAULTS = {"epochs": 200, "batch": "auto", "freeze": 10,
                     "augment": True, "train_interval_hours": 48,
                     "min_new_labels": 5, "nightly_prelabel_hour": 2,
                     "gpu": "auto", "auto_update": True}
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
