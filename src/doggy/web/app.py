from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from doggy.reaction.dataset import DatasetCapture
from doggy.reaction.sound import Alerter
from doggy.core.config import Settings, TunableSettings
from doggy.events.store import EventStore
from doggy.decision.gate import FireGate
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.web.door import create_door_app
from doggy.web.envfile import write_env as _write_env
from doggy.web.sidecar_index import SidecarIndex
from doggy.web.routers import dataset as dataset_router
from doggy.web.routers import events, snooze, soothing, sounds, speaker, talk
from doggy.web.routers import settings as settings_router
from doggy.web.routers import status as status_router
from doggy.web.routers import training as training_router


def create_app(settings: Settings, runtime: RuntimeSettings,
               annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter,
               event_store: EventStore, gate: FireGate,
               dataset: DatasetCapture | None = None,
               save_env: Callable[[TunableSettings], None] = _write_env) -> FastAPI:
    # `dataset` is unused since the web layer reads sidecars via the shared
    # SidecarIndex; the parameter stays so the composition root's (and the
    # tests') call shape is stable.
    _ = dataset
    # One index for every dataset/training view: per-request re-parsing of
    # thousands of sidecars is what made those pages take ~10s on the Pi.
    sidecar_index = SidecarIndex(settings.dataset_dir)
    app = FastAPI(title="doggy")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/label")
    @app.get("/review")  # legacy alias: old bookmarks keep working
    def label_page() -> FileResponse:
        # The labeling tool: verdicts, hand-drawn boxes, history rail.
        return FileResponse(Path(__file__).parent / "static" / "label.html")

    @app.get("/training")
    def training_page() -> FileResponse:
        # The training console: recipe + schedule, run logs, job history,
        # and full reports from the trainer's cloud runs.
        return FileResponse(Path(__file__).parent / "static" / "training.html")

    @app.get("/ca.pem")
    def ca_cert() -> FileResponse:
        # Public material: lets each device trust the home CA once, after
        # which the dashboard shows a normal padlock (no warnings).
        if settings.ca_cert and Path(settings.ca_cert).is_file():
            return FileResponse(settings.ca_cert, media_type="application/x-pem-file",
                                filename="watchdoggy-ca.pem")
        raise HTTPException(status_code=404, detail="not set up")

    @app.get("/ping")
    def ping() -> Response:
        # Cross-origin trust probe target for the onboarding door (web/door.py):
        # an http door page fetches this over https to test whether the CA is
        # trusted, so it needs the same CORS header the door's own /ping carries.
        return Response(status_code=204, headers={"Access-Control-Allow-Origin": "*"})

    app.include_router(
        status_router.build_router(runtime, annotated_buffer, status, event_store))
    app.include_router(settings_router.build_router(runtime, save_env))
    app.include_router(events.build_router(settings, event_store))
    app.include_router(sounds.build_router(settings, runtime, alerter))
    app.include_router(soothing.build_router(settings))
    app.include_router(snooze.build_router(gate))
    app.include_router(talk.build_router(runtime))
    app.include_router(speaker.build_router())
    app.include_router(dataset_router.build_router(settings, event_store,
                                                   sidecar_index))
    app.include_router(training_router.build_router(settings, sidecar_index))

    return app


def serve(settings: Settings, runtime: RuntimeSettings,
          annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter,
          event_store: EventStore, gate: FireGate,
          dataset: DatasetCapture | None = None) -> None:
    import logging
    import threading

    import uvicorn

    app = create_app(settings, runtime, annotated_buffer, status, alerter, event_store,
                     gate, dataset)
    # TLS is all-or-nothing: with only one of cert/key set we serve plain http.
    if not (settings.ssl_cert and settings.ssl_key):
        uvicorn.run(app, host=settings.web_host, port=settings.web_port,
                    log_level="warning")
        return

    # TLS configured: the dashboard moves to https on ssl_port, and the plain
    # web_port keeps serving the onboarding door so old bookmarks still land
    # somewhere useful. The door is intentionally unauthenticated -- it only
    # serves public material and a trust probe (see web/door.py).
    door = create_door_app(settings)
    threading.Thread(
        target=lambda: uvicorn.run(door, host=settings.web_host,
                                   port=settings.web_port, log_level="warning"),
        daemon=True,
    ).start()
    logging.getLogger("doggy").info(
        "dashboard https on port %d; onboarding door http on port %d",
        settings.ssl_port, settings.web_port)
    uvicorn.run(app, host=settings.web_host, port=settings.ssl_port,
                log_level="warning", ssl_certfile=str(settings.ssl_cert),
                ssl_keyfile=str(settings.ssl_key))
