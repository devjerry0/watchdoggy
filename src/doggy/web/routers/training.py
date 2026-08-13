from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from doggy.core.config import Settings

# Job kinds the review page can request. "train" = full cloud pipeline run
# (prelabel + build + fine-tune + eval + gated deploy); "prelabel" = fresh
# big-model boxes only, so the review page is stocked before a labeling
# session. The trainer daemon -- a separate user with the one firewall
# exception -- consumes these files and writes its results back into them.
_KINDS = {"train", "prelabel"}
_HISTORY_SHOWN = 10
AUTO_TRAIN_INTERVAL_SECONDS = 48 * 3600.0


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

    @router.get("/api/training/status")
    def api_status() -> dict:
        jobs = _jobs()
        unlabeled, missing_prelabels = _dataset_counts()
        last_train = next((j for j in jobs
                           if j.get("kind") == "train"
                           and j.get("status") == "done"), None)
        next_auto = None
        if last_train:
            next_auto = last_train.get("updated_at", 0) + AUTO_TRAIN_INTERVAL_SECONDS
        return {"unlabeled": unlabeled,
                "missing_prelabels": missing_prelabels,
                "jobs": jobs[:_HISTORY_SHOWN],
                "last_train": last_train,
                "next_auto_train": next_auto}

    @router.post("/api/training/request")
    def api_request(body: dict) -> dict:
        kind = body.get("kind")
        if kind not in _KINDS:
            raise HTTPException(status_code=422,
                                detail="kind must be train or prelabel")
        pending = [j for j in _jobs()
                   if j.get("kind") == kind
                   and j.get("status") in ("queued", "running")]
        if pending:
            return {"ok": True, "job": pending[0], "already_pending": True}
        jobs_dir = Path(settings.jobs_dir)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job = {"id": f"job_{int(time.time() * 1000)}", "kind": kind,
               "status": "queued", "requested_at": time.time(),
               "updated_at": time.time(), "detail": "", "auto": False}
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        return {"ok": True, "job": job, "already_pending": False}

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
