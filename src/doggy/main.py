from __future__ import annotations

import logging
import signal
import threading

from doggy.alerter import build_alerter
from doggy.camera import build_camera
from doggy.config import load_settings
from doggy.detector import build_detector
from doggy.events import EventStore
from doggy.pipeline import Pipeline
from doggy.safety import SafetyGovernor
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore


def main() -> None:
    settings = load_settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("doggy")

    runtime = RuntimeSettings(settings.tunable())
    status = StatusStore()
    raw_buffer = FrameBuffer()
    annotated_buffer = FrameBuffer()
    # Single writer of the event log; the web endpoints read the same dir.
    event_store = EventStore(
        settings.event_log_dir,
        settings.event_retention_max,
        settings.event_retention_days,
    )
    safety = SafetyGovernor(runtime, event_store)

    detector = build_detector(settings, runtime)   # loads model now (fail fast)
    camera = build_camera(settings)
    alerter = build_alerter(settings, runtime)

    pipeline = Pipeline(
        settings=settings, detector=detector, camera=camera, alerter=alerter,
        runtime=runtime, status=status, raw_buffer=raw_buffer,
        annotated_buffer=annotated_buffer, safety=safety,
    )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    if settings.web_enabled:
        from doggy.web import serve
        threading.Thread(
            target=serve,
            args=(settings, runtime, annotated_buffer, status, alerter, event_store),
            daemon=True,
        ).start()
        log.info("dashboard at http://%s:%s", settings.web_host, settings.web_port)

    log.info("doggy starting")
    pipeline.run(stop)
    log.info("doggy stopped")


if __name__ == "__main__":
    main()
