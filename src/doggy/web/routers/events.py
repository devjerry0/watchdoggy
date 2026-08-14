from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status
from fastapi.responses import FileResponse, Response

from doggy.core.config import Settings
from doggy.events.store import EventRecord, EventStore


def _age_seconds(record: EventRecord) -> float:
    """Prefer ``wall_time`` (Unix epoch): the monotonic ``ts`` is not
    comparable across restarts, so fall back to it only when the clock was
    never set (``wall_time is None``)."""
    if record.wall_time:
        return max(0.0, time.time() - record.wall_time)
    return max(0.0, time.monotonic() - record.ts)


def _event_dict(record: EventRecord) -> dict:
    """Serialize an EventRecord for the API, computing a live age."""
    age = _age_seconds(record)
    return {
        "id": record.id,
        "ts": record.ts,
        "wall_time": record.wall_time,
        "confidence": record.confidence,
        "latency_s": record.latency_s,
        "thumb": record.thumb,
        "clip": record.clip,
        "sound": record.sound,
        "clear_seconds": record.clear_seconds,
        "strikes": record.strikes,
        "taken": record.taken,
        "age_seconds": age,
    }


def build_router(settings: Settings, event_store: EventStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/events")
    def api_events(limit: int | None = None) -> dict:
        return {"events": [_event_dict(r) for r in event_store.list(limit=limit)]}

    @router.delete("/api/events/{id}")
    def api_delete_event(id: str) -> dict:
        # Path(id).name strips any directory components → no path traversal.
        if not event_store.delete(Path(id).name):
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="not found")
        return {"ok": True}

    @router.post("/api/events/clear")
    def api_clear_events() -> dict:
        event_store.clear()
        return {"ok": True}

    @router.get("/api/export")
    def api_export() -> Response:
        # Buffer the whole zip in RAM: media is already compressed (ZIP_STORED),
        # so this is tens of MB at worst -- fine on the Pi, no streaming needed.
        records = event_store.list()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            jsonl = Path(settings.event_log_dir) / "events.jsonl"
            if jsonl.is_file():
                z.write(jsonl, "events.jsonl")
            # No lock is held while reading files: a record whose thumb/clip was
            # concurrently deleted (delete/prune) fails is_file and is skipped.
            media = (name for r in records for name in (r.thumb, r.clip) if name)
            for name in media:
                p = Path(settings.event_log_dir) / name
                if p.is_file():
                    z.write(p, name)
        return Response(buf.getvalue(), media_type="application/zip", headers={
            "Content-Disposition": "attachment; filename=watchdoggy-export.zip"})

    @router.get("/api/stats")
    def api_stats() -> dict:
        return event_store.stats()

    @router.get("/api/lab")
    def api_lab() -> dict:
        return event_store.lab_stats()

    @router.get("/clips/{name}")
    def clip(name: str) -> FileResponse:
        # Path(name).name strips any directory components → no path traversal.
        path = Path(settings.event_log_dir) / Path(name).name
        if not path.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="not found")
        return FileResponse(path)

    @router.get("/events/{name}")
    def event_thumb(name: str) -> FileResponse:
        # Path(name).name strips any directory components → no path traversal.
        path = Path(settings.event_log_dir) / Path(name).name
        if not path.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="not found")
        return FileResponse(path)

    return router
