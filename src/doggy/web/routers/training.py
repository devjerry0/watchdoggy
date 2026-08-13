from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from doggy.core.config import Settings

# Job kinds the training page can request. "train" = full cloud pipeline run
# (prelabel + build + fine-tune + eval + gated deploy); "prelabel" = fresh
# big-model boxes only, so the label page is stocked before a labeling
# session. The trainer daemon -- a separate user with the one firewall
# exception -- consumes these files and writes its results back into them.
_KINDS = {"train", "prelabel"}
_HISTORY_SHOWN = 10

# Trainer settings: recipe knobs for cloud runs + the auto-schedule. The web
# writes trainer-settings.json; the daemon reads it each pass. Bounds keep a
# fat-fingered form from queueing a 10-hour GPU burn.
SETTINGS_DEFAULTS = {"epochs": 200, "batch": 16, "freeze": 10, "augment": True,
                     "train_interval_hours": 48, "min_new_labels": 5,
                     "nightly_prelabel_hour": 2}
_INT_BOUNDS = {"epochs": (10, 500), "batch": (4, 64), "freeze": (0, 23),
               "train_interval_hours": (6, 336), "min_new_labels": (1, 100),
               "nightly_prelabel_hour": (0, 23)}


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    def _jobs() -> list[dict]:
        jobs_dir = Path(settings.jobs_dir)
        if not jobs_dir.is_dir():
            return []
        jobs = []
        for path in sorted(jobs_dir.glob("job_*.json"), reverse=True):
            if path.name.endswith(".result.json"):
                continue
            try:
                job = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            # The trainer daemon runs as a different user and can't edit the
            # web's job files; it reports progress in a sibling result file
            # that overlays the original on read.
            result_path = path.with_name(f"{path.stem}.result.json")
            if result_path.is_file():
                try:
                    job.update(json.loads(result_path.read_text()))
                except (OSError, ValueError):
                    pass
            jobs.append(job)
        return jobs

    def _dataset_counts() -> tuple[int, int]:
        unlabeled = 0
        missing_prelabels = 0
        dataset_dir = Path(settings.dataset_dir)
        if not dataset_dir.is_dir():
            return 0, 0
        for sidecar in dataset_dir.glob("sample_*.json"):
            try:
                meta = json.loads(sidecar.read_text())
            except (OSError, ValueError):
                continue
            if not meta.get("human_label"):
                unlabeled += 1
            if "prelabels" not in meta:
                missing_prelabels += 1
        return unlabeled, missing_prelabels

    def _settings_path() -> Path:
        return Path(settings.jobs_dir) / "trainer-settings.json"

    def _trainer_settings() -> dict:
        merged = dict(SETTINGS_DEFAULTS)
        if _settings_path().is_file():
            try:
                merged.update(json.loads(_settings_path().read_text()))
            except (OSError, ValueError):
                pass
        return merged

    def _validated(body: dict) -> dict:
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
        return clean

    @router.get("/api/training/status")
    def api_status() -> dict:
        jobs = _jobs()
        unlabeled, missing_prelabels = _dataset_counts()
        last_train = next((j for j in jobs
                           if j.get("kind") == "train"
                           and j.get("status") == "done"), None)
        trainer = _trainer_settings()
        next_auto = None
        if last_train:
            next_auto = (last_train.get("updated_at", 0)
                         + trainer["train_interval_hours"] * 3600.0)
        model_path = Path(settings.model_path)
        deployed_at = (model_path.stat().st_mtime if model_path.exists() else None)
        return {"unlabeled": unlabeled,
                "missing_prelabels": missing_prelabels,
                "jobs": jobs[:_HISTORY_SHOWN],
                "last_train": last_train,
                "next_auto_train": next_auto,
                "settings": trainer,
                "model": {"path": str(settings.model_path),
                          "deployed_at": deployed_at,
                          "rollback_available":
                              model_path.with_name(model_path.name + ".prev").exists()}}

    @router.get("/api/training/settings")
    def api_get_settings() -> dict:
        return _trainer_settings()

    @router.post("/api/training/settings")
    def api_save_settings(body: dict) -> dict:
        merged = {**_trainer_settings(), **_validated(body)}
        Path(settings.jobs_dir).mkdir(parents=True, exist_ok=True)
        _settings_path().write_text(json.dumps(merged))
        return {"ok": True, "settings": merged}

    @router.post("/api/training/request")
    def api_request(body: dict) -> dict:
        kind = body.get("kind")
        if kind not in _KINDS:
            raise HTTPException(status_code=422,
                                detail="kind must be train or prelabel")
        params = _validated(body.get("params") or {})
        pending = [j for j in _jobs()
                   if j.get("kind") == kind
                   and j.get("status") in ("queued", "running")]
        if pending:
            return {"ok": True, "job": pending[0], "already_pending": True}
        jobs_dir = Path(settings.jobs_dir)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job = {"id": f"job_{int(time.time() * 1000)}", "kind": kind,
               "status": "queued", "requested_at": time.time(),
               "updated_at": time.time(), "detail": "", "auto": False,
               "params": params}
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        return {"ok": True, "job": job, "already_pending": False}

    @router.get("/api/training/log/{job_id}", response_class=PlainTextResponse)
    def api_log(job_id: str) -> str:
        """Tail of a job's cloud-run log (written live by the daemon)."""
        safe = Path(job_id).name  # traversal guard
        path = Path(settings.jobs_dir) / f"{safe}.log"
        if not safe.startswith("job_") or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-200:])

    @router.get("/api/training/report/{job_id}", response_class=PlainTextResponse)
    def api_report(job_id: str) -> str:
        """The full markdown report a train job produced (written by the
        trainer daemon next to the job file)."""
        safe = Path(job_id).name  # traversal guard
        path = Path(settings.jobs_dir) / f"{safe}.report.md"
        if not safe.startswith("job_") or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return path.read_text()

    return router
