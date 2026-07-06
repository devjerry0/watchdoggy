from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import cv2
from fastapi import FastAPI, HTTPException
from fastapi import status as http_status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError

from doggy.alerter import Alerter
from doggy.config import Settings, TunableSettings
from doggy.events import EventRecord, EventStore
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore

_STATIC = Path(__file__).parent / "static"
# Min interval between streamed JPEG frames (~10 FPS) so the MJPEG encode loop
# never starves the detect loop.
_MJPEG_FRAME_INTERVAL_SECONDS = 0.1
# Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY -> _CONTENT (0.47); accept either
# (prefer the new name so current Starlette doesn't emit a deprecation warning).
_HTTP_422 = getattr(http_status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or getattr(
    http_status, "HTTP_422_UNPROCESSABLE_ENTITY", 422
)


def _event_dict(record: EventRecord) -> dict:
    """Serialize an EventRecord for the API, computing a live age.

    ``age_seconds`` prefers ``wall_time`` (Unix epoch): the monotonic ``ts`` is
    not comparable across restarts, so fall back to it only when the clock was
    never set (``wall_time is None``).
    """
    if record.wall_time:
        age = max(0.0, time.time() - record.wall_time)
    else:
        age = max(0.0, time.monotonic() - record.ts)
    return {
        "id": record.id,
        "ts": record.ts,
        "wall_time": record.wall_time,
        "confidence": record.confidence,
        "latency_s": record.latency_s,
        "thumb": record.thumb,
        "clip": record.clip,
        "age_seconds": age,
    }


def _write_env(tunable: TunableSettings, path: Path = Path(".env")) -> None:
    """Persist the tunable settings into .env in place: update existing keys,
    append missing ones, and preserve comments and non-tunable (structural) keys.
    """
    def _fmt(v: object) -> str:
        if isinstance(v, (str, int, float, bool)) or isinstance(v, Path):
            return str(v)
        return json.dumps(v)   # lists/tuples -> JSON so pydantic-settings can re-parse
    updates = {f"DOGGY_{k.upper()}": _fmt(v) for k, v in tunable.model_dump().items()}
    lines: list[str] = []
    seen: set[str] = set()
    if path.exists():
        for raw in path.read_text().splitlines():
            stripped = raw.strip()
            if "=" in stripped and not stripped.startswith("#"):
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    lines.append(f"{key}={updates[key]}")
                    seen.add(key)
                    continue
            lines.append(raw)
    for key, val in updates.items():
        if key not in seen:
            lines.append(f"{key}={val}")
    path.write_text("\n".join(lines) + "\n")


def create_app(settings: Settings, runtime: RuntimeSettings,
               annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter,
               event_store: EventStore,
               save_env: Callable[[TunableSettings], None] = _write_env) -> FastAPI:
    app = FastAPI(title="doggy")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/status")
    def api_status() -> dict:
        return {
            **asdict(status.snapshot()),
            "settings": runtime.get().model_dump(mode="json"),
            "events": [_event_dict(r) for r in event_store.list(limit=10)],
        }

    @app.get("/api/events")
    def api_events(limit: int | None = None) -> dict:
        return {"events": [_event_dict(r) for r in event_store.list(limit=limit)]}

    @app.delete("/api/events/{id}")
    def api_delete_event(id: str) -> dict:
        # Path(id).name strips any directory components → no path traversal.
        if not event_store.delete(Path(id).name):
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="not found")
        return {"ok": True}

    @app.post("/api/events/clear")
    def api_clear_events() -> dict:
        event_store.clear()
        return {"ok": True}

    @app.get("/api/stats")
    def api_stats() -> dict:
        return event_store.stats()

    @app.get("/clips/{name}")
    def clip(name: str) -> FileResponse:
        # Path(name).name strips any directory components → no path traversal.
        path = Path(settings.event_log_dir) / Path(name).name
        if not path.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="not found")
        return FileResponse(path)

    @app.patch("/api/settings")
    def api_patch(patch: dict) -> dict:
        merged = {**runtime.get().model_dump(), **patch}
        try:
            updated = TunableSettings(**merged)
        except ValidationError as exc:
            raise HTTPException(status_code=_HTTP_422, detail=str(exc)) from exc
        runtime.update(updated)
        return updated.model_dump(mode="json")

    @app.post("/api/test-sound")
    def api_test_sound() -> dict:
        alerter.alert()
        return {"ok": True}

    @app.post("/api/settings/save")
    def api_save() -> dict:
        save_env(runtime.get())
        return {"ok": True}

    @app.get("/events/{name}")
    def event_thumb(name: str) -> FileResponse:
        # Path(name).name strips any directory components → no path traversal.
        path = Path(settings.event_log_dir) / Path(name).name
        if not path.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="not found")
        return FileResponse(path)

    @app.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        def gen():
            while True:
                frame = annotated_buffer.get()
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + buf.tobytes() + b"\r\n")
                time.sleep(_MJPEG_FRAME_INTERVAL_SECONDS)

        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def serve(settings: Settings, runtime: RuntimeSettings,
          annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter,
          event_store: EventStore) -> None:
    import uvicorn

    app = create_app(settings, runtime, annotated_buffer, status, alerter, event_store)
    uvicorn.run(app, host=settings.web_host, port=settings.web_port, log_level="warning")
