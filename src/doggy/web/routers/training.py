from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from doggy.core.config import Settings
from doggy.web import jobqueue
from doggy.web.sidecar_index import SidecarIndex

# Job kinds the training page can request. "train" = full cloud pipeline run
# (prelabel + build + fine-tune + eval + gated deploy); "prelabel" = fresh
# big-model boxes only, so the label page is stocked before a labeling
# session; "update" = self-update from the latest GitHub release (health-
# gated install with rollback, run by the trainer daemon).
_KINDS = {"train", "prelabel", "update"}
_HISTORY_SHOWN = 10
_LOG_TAIL_LINES = 200
SECONDS_PER_HOUR = 3600.0


def _dataset_counts(index: SidecarIndex) -> tuple[int, int]:
    unlabeled = 0
    missing_prelabels = 0
    for _, meta in index.snapshot():
        unlabeled += not meta.get("human_label")
        missing_prelabels += "prelabels" not in meta
    return unlabeled, missing_prelabels


def _model_summary(model_path: Path) -> dict:
    deployed_at = model_path.stat().st_mtime if model_path.exists() else None
    return {"path": str(model_path),
            "deployed_at": deployed_at,
            "rollback_available":
                model_path.with_name(model_path.name + ".prev").exists()}


def _billing(jobs_dir: Path) -> dict | None:
    path = jobs_dir / "billing.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _software(jobs_dir: Path) -> dict:
    """Installed release + the trainer's last GitHub check (both optional:
    the release file appears with the first self-update install)."""
    release_file = Path(".release")
    installed = None
    if release_file.is_file():
        installed = release_file.read_text().strip() or None
    check = None
    check_path = jobs_dir / "update-check.json"
    if check_path.is_file():
        try:
            check = json.loads(check_path.read_text())
        except (OSError, ValueError):
            check = None
    return {"installed": installed, "check": check}


def build_router(settings: Settings, index: SidecarIndex) -> APIRouter:
    router = APIRouter()
    # Dataset tallies recomputed only when the index generation moves; the
    # status endpoint is polled every few seconds by the training page.
    memo: dict = {"gen": -1, "counts": (0, 0)}

    def _counts() -> tuple[int, int]:
        index.snapshot()
        if memo["gen"] != index.generation:
            memo["counts"] = _dataset_counts(index)
            memo["gen"] = index.generation
        return memo["counts"]

    @router.get("/api/training/status")
    def api_status() -> JSONResponse:
        jobs = jobqueue.list_jobs(Path(settings.jobs_dir))
        unlabeled, missing_prelabels = _counts()
        last_train = next((j for j in jobs
                           if j.get("kind") == "train"
                           and j.get("status") == "done"), None)
        trainer = jobqueue.trainer_settings(Path(settings.jobs_dir))
        next_auto = None
        if last_train:
            next_auto = (last_train.get("updated_at", 0)
                         + trainer["train_interval_hours"] * SECONDS_PER_HOUR)
        # Direct JSONResponse: the jobs carry fat _summary blobs and the
        # jsonable_encoder pass over them per poll is pure overhead.
        return JSONResponse(
            {"unlabeled": unlabeled,
             "missing_prelabels": missing_prelabels,
             "jobs": jobs[:_HISTORY_SHOWN],
             "last_train": last_train,
             "next_auto_train": next_auto,
             "settings": trainer,
             "billing": _billing(Path(settings.jobs_dir)),
             "model": _model_summary(Path(settings.model_path)),
             "software": _software(Path(settings.jobs_dir))})

    @router.get("/api/training/settings")
    def api_get_settings() -> dict:
        return jobqueue.trainer_settings(Path(settings.jobs_dir))

    @router.post("/api/training/settings")
    def api_save_settings(body: dict) -> dict:
        merged = jobqueue.save_trainer_settings(Path(settings.jobs_dir),
                                                jobqueue.validated(body))
        return {"ok": True, "settings": merged}

    @router.post("/api/training/request")
    def api_request(body: dict) -> dict:
        kind = body.get("kind")
        if kind not in _KINDS:
            raise HTTPException(status_code=422,
                                detail="kind must be train or prelabel")
        params = jobqueue.validated(body.get("params") or {})
        pending = [j for j in jobqueue.list_jobs(Path(settings.jobs_dir))
                   if j.get("kind") == kind
                   and j.get("status") in ("queued", "running")]
        if pending:
            return {"ok": True, "job": pending[0], "already_pending": True}
        job = jobqueue.queue_job(Path(settings.jobs_dir), kind, params)
        return {"ok": True, "job": job, "already_pending": False}

    @router.get("/api/training/log/{job_id}", response_class=PlainTextResponse)
    def api_log(job_id: str) -> str:
        """Tail of a job's cloud-run log (written live by the daemon)."""
        safe = Path(job_id).name  # traversal guard
        path = Path(settings.jobs_dir) / f"{safe}.log"
        if not safe.startswith("job_") or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-_LOG_TAIL_LINES:])

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
