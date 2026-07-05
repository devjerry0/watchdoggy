from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError

from doggy.alerter import Alerter
from doggy.config import Settings, TunableSettings
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore

_STATIC = Path(__file__).parent / "static"


def _write_env(tunable: TunableSettings, path: Path = Path(".env")) -> None:
    lines = [f"DOGGY_{k.upper()}={v}" for k, v in tunable.model_dump().items()]
    path.write_text("\n".join(lines) + "\n")


def create_app(settings: Settings, runtime: RuntimeSettings,
               annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter,
               save_env: Callable[[TunableSettings], None] = _write_env) -> FastAPI:
    app = FastAPI(title="doggy")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/status")
    def api_status() -> dict:
        snap = status.snapshot()
        return {
            "state": snap.state, "fps": snap.fps, "confidence": snap.confidence,
            "fires_this_hour": snap.fires_this_hour, "muted": snap.muted,
            "last_fire_ts": snap.last_fire_ts, "last_fire_thumb": snap.last_fire_thumb,
            "settings": runtime.get().model_dump(mode="json"),
            "events": status.events(),
        }

    @app.patch("/api/settings")
    def api_patch(patch: dict) -> dict:
        merged = {**runtime.get().model_dump(), **patch}
        try:
            updated = TunableSettings(**merged)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
                time.sleep(0.1)  # throttle to ~10 FPS so it never starves detection

        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def serve(settings: Settings, runtime: RuntimeSettings,
          annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter) -> None:
    import uvicorn

    app = create_app(settings, runtime, annotated_buffer, status, alerter)
    uvicorn.run(app, host=settings.web_host, port=settings.web_port, log_level="warning")
