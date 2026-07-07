from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from doggy.reaction.sound import Alerter
from doggy.core.config import Settings, TunableSettings
from doggy.events.store import EventStore
from doggy.decision.gate import FireGate
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.web.envfile import write_env as _write_env
from doggy.web.routers import events, snooze, sounds
from doggy.web.routers import settings as settings_router
from doggy.web.routers import status as status_router


def create_app(settings: Settings, runtime: RuntimeSettings,
               annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter,
               event_store: EventStore, gate: FireGate,
               save_env: Callable[[TunableSettings], None] = _write_env) -> FastAPI:
    app = FastAPI(title="doggy")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/ca.pem")
    def ca_cert() -> FileResponse:
        # Public material: lets each device trust the home CA once, after
        # which the dashboard shows a normal padlock (no warnings).
        if settings.ca_cert and Path(settings.ca_cert).is_file():
            return FileResponse(settings.ca_cert, media_type="application/x-pem-file",
                                filename="watchdoggy-ca.pem")
        raise HTTPException(status_code=404, detail="not set up")

    app.include_router(
        status_router.build_router(runtime, annotated_buffer, status, event_store))
    app.include_router(settings_router.build_router(runtime, save_env))
    app.include_router(events.build_router(settings, event_store))
    app.include_router(sounds.build_router(settings, runtime, alerter))
    app.include_router(snooze.build_router(gate))

    return app


def serve(settings: Settings, runtime: RuntimeSettings,
          annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter,
          event_store: EventStore, gate: FireGate) -> None:
    import uvicorn

    app = create_app(settings, runtime, annotated_buffer, status, alerter, event_store, gate)
    # TLS is all-or-nothing: with only one of cert/key set we serve plain http.
    kwargs: dict = {}
    if settings.ssl_cert and settings.ssl_key:
        kwargs = {"ssl_certfile": str(settings.ssl_cert),
                  "ssl_keyfile": str(settings.ssl_key)}
    uvicorn.run(app, host=settings.web_host, port=settings.web_port,
                log_level="warning", **kwargs)
