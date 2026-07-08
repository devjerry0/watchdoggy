from __future__ import annotations

import logging
import signal
import threading

from doggy.reaction.sound import SoundReaction, build_alerter
from doggy.reaction.hub import ReactionHub, SafeReaction
from doggy.reaction.clips import ClipBuffer, ClipService
from doggy.reaction.outcome import OutcomeWatcher
from doggy.reaction.recorder import Recorder
from doggy.reaction.soothing import SoothingPlayer
from doggy.vision.camera import build_camera
from doggy.core.config import load_settings
from doggy.vision.analysis import DetectionAnalyzer
from doggy.vision.detector import build_detector
from doggy.vision.filters.base import FilterChain
from doggy.vision.filters.person import PersonSuppressionFilter
from doggy.vision.filters.zone import ZoneInclusionFilter
from doggy.events.store import EventStore
from doggy.pipeline import Pipeline
from doggy.decision.gate import FireGate
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore


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
        settings.clip_retention,
    )
    gate = FireGate(runtime)
    recorder = Recorder(event_store)

    detector = build_detector(settings, runtime)   # loads model now (fail fast)
    analyzer = DetectionAnalyzer(
        detector, FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()]))
    camera = build_camera(settings)
    alerter = build_alerter(settings, runtime)
    clip_service = ClipService(
        event_store, event_store.dir, ClipBuffer(settings.clip_window_seconds), runtime)
    soothing = SoothingPlayer(runtime, settings.soothing_dir, status)
    outcome = OutcomeWatcher(event_store, gate, alerter, runtime)
    # Reactions fan out on a catch; each is wrapped so one failure can't stop the
    # others or kill the detect loop. ClipService registers its pending clip here;
    # OutcomeWatcher opens the incident it measures per-frame; SoothingPlayer (last)
    # cuts its music and holds it after the catch.
    hub = ReactionHub(
        [SafeReaction(SoundReaction(alerter, event_store)), SafeReaction(clip_service),
         SafeReaction(outcome), SafeReaction(soothing)])

    pipeline = Pipeline(
        settings=settings, analyzer=analyzer, camera=camera,
        runtime=runtime, status=status, raw_buffer=raw_buffer,
        annotated_buffer=annotated_buffer, gate=gate, recorder=recorder, hub=hub,
        clip_service=clip_service, outcome=outcome,
    )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    if settings.web_enabled:
        from doggy.web import serve
        threading.Thread(
            target=serve,
            args=(settings, runtime, annotated_buffer, status, alerter, event_store, gate),
            daemon=True,
        ).start()
        log.info("dashboard at http://%s:%s", settings.web_host, settings.web_port)

    log.info("doggy starting")
    soothing.start()
    pipeline.run(stop)
    log.info("doggy stopped")


if __name__ == "__main__":
    main()
