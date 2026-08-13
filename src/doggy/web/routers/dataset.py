from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status
from fastapi.responses import FileResponse

from doggy.core.config import Settings
from doggy.events.store import EventStore
from doggy.reaction.dataset import DatasetCapture

# Human verdicts the review page can attach to a sample. "dog" = a real dog is
# present (a person may be too); "person" = person but no dog (the false-alarm
# class); "empty" = nothing of interest (pure background negative); "skip"
# parks a frame (unclear/blurry) without pretending it was judged. "no_dog" is
# the legacy coarse verdict from the catch log's one-tap button, kept valid.
_VERDICTS = {"dog", "person", "empty", "no_dog", "skip"}
# Review order: highest training signal first, newest first within a class.
_REASON_PRIORITY = {"fire": 0, "suppressed": 1, "borderline": 2,
                    "person_activity": 3, "periodic": 4}


def build_router(settings: Settings, capture: DatasetCapture,
                 event_store: EventStore) -> APIRouter:
    router = APIRouter()

    def _sidecars() -> list[Path]:
        d = Path(settings.dataset_dir)
        return sorted(d.glob("sample_*.json")) if d.is_dir() else []

    @router.get("/api/dataset")
    def api_dataset() -> dict:
        return capture.stats()

    @router.get("/api/dataset/next")
    def api_next_unlabeled() -> dict:
        """The next frame to review: unlabeled, highest-signal reason first."""
        pending = []
        labeled = 0
        for side in _sidecars():
            try:
                meta = json.loads(side.read_text())
            except (OSError, ValueError):
                continue
            if meta.get("human_label"):
                labeled += 1
                continue
            prio = min((_REASON_PRIORITY.get(r, 9) for r in meta.get("reasons", [])),
                       default=9)
            pending.append((prio, -meta.get("wall_time", 0), side.stem, meta))
        pending.sort()
        if not pending:
            return {"remaining": 0, "labeled": labeled, "sample": None}
        _, _, stem, meta = pending[0]
        return {"remaining": len(pending), "labeled": labeled,
                "sample": {"name": stem, "image": f"/dataset/{stem}.jpg",
                           "reasons": meta.get("reasons", []),
                           "detections": meta.get("detections", {})}}

    @router.post("/api/dataset/label")
    def api_label(body: dict) -> dict:
        name = Path(str(body.get("name", ""))).name  # traversal guard
        verdict = body.get("verdict")
        if verdict not in _VERDICTS:
            raise HTTPException(status_code=422,
                                detail="verdict must be dog, person, empty, or skip")
        side = Path(settings.dataset_dir) / f"{name}.json"
        if not name.startswith("sample_") or not side.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        meta = json.loads(side.read_text())
        meta["human_label"] = verdict
        meta["labeled_at"] = time.time()
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.get("/dataset/{name}")
    def dataset_image(name: str) -> FileResponse:
        # Path(name).name strips any directory components -> no path traversal.
        safe = Path(name).name
        path = Path(settings.dataset_dir) / safe
        if not safe.startswith("sample_") or not safe.endswith(".jpg") or not path.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        return FileResponse(path)

    @router.post("/api/dataset/mark/{event_id}")
    def api_mark_false_positive(event_id: str) -> dict:
        """The catch log's "Not a dog" button: copy that event's raw frame into
        the dataset labeled as a user-confirmed false positive. Works even with
        capture off -- an explicit user label is always worth keeping."""
        record = next((r for r in event_store.list()
                       if r.id == Path(event_id).name), None)
        if record is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        src = Path(settings.event_log_dir) / record.thumb
        if not src.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="snapshot missing")
        out = Path(settings.dataset_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"sample_{int(time.time() * 1000)}"
        shutil.copyfile(src, out / f"{stem}.jpg")
        (out / f"{stem}.json").write_text(json.dumps({
            "wall_time": time.time(),
            "reasons": ["user_marked_fp"],
            "event_id": record.id,
            "event_confidence": record.confidence,
            # The tap IS the verdict: this frame needs no second review.
            "human_label": "no_dog",
            "labeled_at": time.time(),
        }))
        return {"ok": True}

    @router.post("/api/dataset/clear")
    def api_clear_dataset() -> dict:
        d = Path(settings.dataset_dir)
        if d.is_dir():
            for p in d.glob("sample_*"):
                if p.is_file():
                    p.unlink()
        return {"ok": True}

    return router
