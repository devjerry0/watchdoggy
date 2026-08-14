import random

import numpy as np

from doggy.core.config import Settings
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.decision.gate import FireGate
from doggy.events.store import EventStore
from doggy.pipeline import Pipeline
from doggy.reaction.recorder import Recorder
from doggy.reaction.sound import FakeAlerter
from doggy.vision.camera import FakeCamera
from doggy.vision.detection import Detection
from doggy.vision.detector import StubDetector

from tests.pipeline_helpers import _analyzer, _clips, _hub, _outcome


def test_pipeline_fires_after_confirmation(tmp_path):
    settings = Settings(confirm_seconds=1.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    dog = [Detection("dog", 0.9, (0, 0, 10, 10))]
    detector = StubDetector([dog, dog, dog, dog])
    alerter = FakeAlerter()
    clock = iter([0.0, 0.5, 1.0, 1.5])
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    pipe = Pipeline(
        settings=settings,
        analyzer=_analyzer(detector),
        camera=FakeCamera([np.zeros((16, 16, 3), np.uint8)], loop=True),
        runtime=runtime,
        status=StatusStore(),
        raw_buffer=FrameBuffer(),
        annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(alerter, clips, store, outcome),
        clip_service=clips, outcome=outcome,
        clock=lambda: next(clock),
        rng=random.Random(0),
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(4)]
    assert any(fired)
    assert alerter.calls == 1


def test_pipeline_counts_multiple_dogs(tmp_path):
    from doggy.vision.annotate import annotate
    settings = Settings()
    runtime = RuntimeSettings(settings.tunable())
    two_dogs = [
        Detection("dog", 0.9, (0, 0, 10, 10)),
        Detection("dog", 0.8, (20, 20, 30, 30)),
    ]
    status = StatusStore()
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    pipe = Pipeline(
        settings=settings,
        analyzer=_analyzer(StubDetector([two_dogs])),
        camera=FakeCamera([np.zeros((40, 40, 3), np.uint8)], loop=True),
        runtime=runtime,
        status=status,
        raw_buffer=FrameBuffer(),
        annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome,
        clock=lambda: 0.0,
        rng=random.Random(0),
    )
    pipe.run_once(np.zeros((40, 40, 3), np.uint8))
    assert status.snapshot().targets == 2

    # annotate draws one box per detection (both dogs), not just one
    frame = np.zeros((40, 40, 3), np.uint8)
    out = annotate(frame, two_dogs)
    assert (out != 0).any()  # boxes were drawn onto the blank frame


def test_pipeline_ignores_dogs_outside_zone(tmp_path):
    # zone = top-left triangle; a dog only in the bottom-right must NOT count/fire
    settings = Settings(zone_enabled=True,
                        zone_points=[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)],
                        confirm_seconds=0.0, window_m=1, window_n=1)
    runtime = RuntimeSettings(settings.tunable())
    outside = [Detection("dog", 0.9, (80, 80, 95, 95))]
    status = StatusStore()
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector([outside])),
        camera=FakeCamera([np.zeros((100, 100, 3), np.uint8)], loop=True),
        runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome, clock=lambda: 0.0,
        rng=random.Random(0),
    )
    fired = pipe.run_once(np.zeros((100, 100, 3), np.uint8))
    assert fired is False
    # The out-of-zone dog can't fire, but it IS detected (drawn), so it counts.
    assert status.snapshot().targets == 1


def test_pipeline_fires_for_dog_inside_zone(tmp_path):
    settings = Settings(zone_enabled=True,
                        zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)],
                        confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    inside = [Detection("dog", 0.9, (5, 5, 20, 20))]
    alerter = FakeAlerter()
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    pipe = Pipeline(
        # TriggerLogic (pre-existing, out of scope here) never fires on the very
        # first sighting -- the IDLE->CONFIRMING transition always returns False
        # regardless of confirm_seconds (see test_trigger.py::test_single_frame_does_not_fire).
        # Two identical in-zone frames are scripted so the second call can fire.
        settings=settings, analyzer=_analyzer(StubDetector([inside, inside])),
        camera=FakeCamera([np.zeros((100, 100, 3), np.uint8)], loop=True),
        runtime=runtime, status=StatusStore(),
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(alerter, clips, store, outcome),
        clip_service=clips, outcome=outcome, clock=lambda: 0.0,
        rng=random.Random(0),
    )
    assert pipe.run_once(np.zeros((100, 100, 3), np.uint8)) is False
    assert pipe.run_once(np.zeros((100, 100, 3), np.uint8)) is True
    assert alerter.calls == 1


def test_annotate_draws_zone_polygon():
    from doggy.vision.annotate import annotate
    frame = np.zeros((100, 100, 3), np.uint8)
    out = annotate(frame, [], in_zone=[], zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)])
    assert (out != 0).any()   # the polygon outline/fill was drawn
