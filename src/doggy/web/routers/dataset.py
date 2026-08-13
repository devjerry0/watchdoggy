from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status

from doggy.core.config import Settings
from doggy.events.store import EventStore
from doggy.reaction.dataset import DatasetCapture


def build_router(settings: Settings, capture: DatasetCapture,
                 event_store: EventStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/dataset")
    def api_dataset() -> dict:
        return capture.stats()

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
