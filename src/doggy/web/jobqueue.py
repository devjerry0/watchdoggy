"""The web side of the cross-user training job queue (files under jobs_dir).

The trainer daemon -- a separate user with the one firewall exception --
consumes job_*.json requests and, unable to edit the web's files, reports
progress in sibling *.result.json files that overlay the original on read.
Trainer settings (recipe knobs + auto-schedule) live beside the jobs in
trainer-settings.json; bounds keep a fat-fingered form from queueing a
10-hour GPU burn."""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import HTTPException

SETTINGS_DEFAULTS = {"epochs": 200, "batch": "auto", "freeze": 10,
                     "augment": True,
                     "train_interval_hours": 48, "min_new_labels": 5,
                     "nightly_prelabel_hour": 2,
                     "monthly_credits": 30,  # Modal starter free tier
                     "gpu": "auto"}  # auto = tier by dataset size (daemon)
_GPU_CHOICES = {"auto", "T4", "L4", "A10G", "A100"}
_BATCH_CHOICES = {"auto", 8, 16, 32, 64}
_INT_BOUNDS = {"epochs": (10, 500), "freeze": (0, 23),
               "train_interval_hours": (6, 336), "min_new_labels": (1, 100),
               "nightly_prelabel_hour": (0, 23), "monthly_credits": (0, 100000)}


def list_jobs(jobs_dir: Path) -> list[dict]:
    """Every job, newest first, with the daemon's result overlay applied."""
    if not jobs_dir.is_dir():
        return []
    jobs = []
    for path in sorted(jobs_dir.glob("job_*.json"), reverse=True):
        if path.name.endswith(".result.json"):
            continue
        job = _read_json(path)
        if job is None:
            continue
        result = _read_json(path.with_name(f"{path.stem}.result.json"))
        job.update(result or {})
        jobs.append(job)
    return jobs


def queue_job(jobs_dir: Path, kind: str, params: dict) -> dict:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job = {"id": f"job_{int(time.time() * 1000)}", "kind": kind,
           "status": "queued", "requested_at": time.time(),
           "updated_at": time.time(), "detail": "", "auto": False,
           "params": params}
    (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
    return job


def settings_path(jobs_dir: Path) -> Path:
    return jobs_dir / "trainer-settings.json"


def trainer_settings(jobs_dir: Path) -> dict:
    merged = dict(SETTINGS_DEFAULTS)
    saved = _read_json(settings_path(jobs_dir))
    merged.update(saved or {})
    return merged


def save_trainer_settings(jobs_dir: Path, updates: dict) -> dict:
    merged = {**trainer_settings(jobs_dir), **updates}
    jobs_dir.mkdir(parents=True, exist_ok=True)
    settings_path(jobs_dir).write_text(json.dumps(merged))
    return merged


def validated(body: dict) -> dict:
    """The subset of a settings body that is known and in bounds (422 on
    anything out of range); unknown keys are dropped."""
    clean = {}
    for key, (low, high) in _INT_BOUNDS.items():
        if key not in body:
            continue
        value = body[key]
        if not isinstance(value, int) or not low <= value <= high:
            raise HTTPException(status_code=422,
                                detail=f"{key} must be {low}..{high}")
        clean[key] = value
    if "augment" in body:
        if not isinstance(body["augment"], bool):
            raise HTTPException(status_code=422, detail="augment must be bool")
        clean["augment"] = body["augment"]
    if "gpu" in body:
        if body["gpu"] not in _GPU_CHOICES:
            raise HTTPException(status_code=422,
                                detail=f"gpu must be one of {sorted(_GPU_CHOICES)}")
        clean["gpu"] = body["gpu"]
    if "batch" in body:
        if body["batch"] not in _BATCH_CHOICES:
            raise HTTPException(status_code=422,
                                detail="batch must be auto, 8, 16, 32, or 64")
        clean["batch"] = body["batch"]
    return clean


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
