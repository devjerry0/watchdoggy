import random

import numpy as np

from doggy.alerter import FakeAlerter
from doggy.camera import FakeCamera
from doggy.config import Settings
from doggy.detection import Detection
from doggy.detector import StubDetector
from doggy.events import EventStore
from doggy.pipeline import Pipeline
from doggy.safety import SafetyGovernor
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore


def test_pipeline_fires_after_confirmation(tmp_path):
    settings = Settings(confirm_seconds=1.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    dog = [Detection("dog", 0.9, (0, 0, 10, 10))]
    detector = StubDetector([dog, dog, dog, dog])
    alerter = FakeAlerter()
    clock = iter([0.0, 0.5, 1.0, 1.5])
    pipe = Pipeline(
        settings=settings,
        detector=detector,
        camera=FakeCamera([np.zeros((16, 16, 3), np.uint8)], loop=True),
        alerter=alerter,
        runtime=runtime,
        status=StatusStore(),
        raw_buffer=FrameBuffer(),
        annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, EventStore(tmp_path, 10, 0)),
        clock=lambda: next(clock),
        rng=random.Random(0),
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(4)]
    assert any(fired)
    assert alerter.calls == 1


def test_pipeline_counts_multiple_dogs(tmp_path):
    from doggy.pipeline import annotate
    settings = Settings()
    runtime = RuntimeSettings(settings.tunable())
    two_dogs = [
        Detection("dog", 0.9, (0, 0, 10, 10)),
        Detection("dog", 0.8, (20, 20, 30, 30)),
    ]
    status = StatusStore()
    pipe = Pipeline(
        settings=settings,
        detector=StubDetector([two_dogs]),
        camera=FakeCamera([np.zeros((40, 40, 3), np.uint8)], loop=True),
        alerter=FakeAlerter(),
        runtime=runtime,
        status=status,
        raw_buffer=FrameBuffer(),
        annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, EventStore(tmp_path, 10, 0)),
        clock=lambda: 0.0,
        rng=random.Random(0),
    )
    pipe.run_once(np.zeros((40, 40, 3), np.uint8))
    assert status.snapshot().dogs == 2

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
    pipe = Pipeline(
        settings=settings, detector=StubDetector([outside]),
        camera=FakeCamera([np.zeros((100, 100, 3), np.uint8)], loop=True),
        alerter=FakeAlerter(), runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, EventStore(tmp_path, 10, 0)), clock=lambda: 0.0,
        rng=random.Random(0),
    )
    fired = pipe.run_once(np.zeros((100, 100, 3), np.uint8))
    assert fired is False
    assert status.snapshot().dogs == 0

def test_pipeline_fires_for_dog_inside_zone(tmp_path):
    settings = Settings(zone_enabled=True,
                        zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)],
                        confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    inside = [Detection("dog", 0.9, (5, 5, 20, 20))]
    alerter = FakeAlerter()
    pipe = Pipeline(
        # TriggerLogic (pre-existing, out of scope here) never fires on the very
        # first sighting -- the IDLE->CONFIRMING transition always returns False
        # regardless of confirm_seconds (see test_trigger.py::test_single_frame_does_not_fire).
        # Two identical in-zone frames are scripted so the second call can fire.
        settings=settings, detector=StubDetector([inside, inside]),
        camera=FakeCamera([np.zeros((100, 100, 3), np.uint8)], loop=True),
        alerter=alerter, runtime=runtime, status=StatusStore(),
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, EventStore(tmp_path, 10, 0)), clock=lambda: 0.0,
        rng=random.Random(0),
    )
    assert pipe.run_once(np.zeros((100, 100, 3), np.uint8)) is False
    assert pipe.run_once(np.zeros((100, 100, 3), np.uint8)) is True
    assert alerter.calls == 1

def test_pipeline_records_trigger_confidence_not_empty_fire_frame(tmp_path):
    # The fire edge can land on a frame with no current detection (flicker); the
    # recorded event confidence must reflect the detection that triggered (0.9),
    # not the empty fire-edge frame -- which used to log "conf 0".
    settings = Settings(confirm_seconds=1.0, window_m=2, window_n=3,
                        cooldown_min_seconds=5, cooldown_max_seconds=5, confidence=0.5)
    runtime = RuntimeSettings(settings.tunable())
    dog = [Detection("dog", 0.9, (0, 0, 10, 10))]
    none: list[Detection] = []
    status = StatusStore()
    clock = iter([0.0, 0.5, 1.0])
    pipe = Pipeline(
        settings=settings, detector=StubDetector([dog, dog, none]),
        camera=FakeCamera([np.zeros((16, 16, 3), np.uint8)], loop=True),
        alerter=FakeAlerter(), runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, EventStore(tmp_path, 10, 0)),
        clock=lambda: next(clock), rng=random.Random(0),
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(3)]
    assert fired[2] is True                       # fires on the empty 3rd frame
    assert status.events()[-1]["confidence"] == 0.9   # not 0.0


def test_pipeline_suppresses_person_misclassified_as_dog(tmp_path):
    # a "dog" box coincident with a person box is a misclassified human -> no fire
    settings = Settings(confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5,
                        person_suppression_enabled=True, person_iou_threshold=0.85,
                        confidence=0.5)
    runtime = RuntimeSettings(settings.tunable())
    both = [Detection("dog", 0.9, (0, 0, 100, 180)),
            Detection("person", 0.9, (2, 2, 98, 178))]
    status = StatusStore()
    pipe = Pipeline(
        settings=settings, detector=StubDetector([both, both]),
        camera=FakeCamera([np.zeros((200, 200, 3), np.uint8)], loop=True),
        alerter=FakeAlerter(), runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, EventStore(tmp_path, 10, 0)), clock=lambda: 0.0,
        rng=random.Random(0),
    )
    frame = np.zeros((200, 200, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(2)]
    assert not any(fired)
    assert status.snapshot().dogs == 0
    assert status.snapshot().people == 1


def test_pipeline_real_dog_near_person_still_fires(tmp_path):
    # dog has its own distinct box that only clips the person -> low IoU -> fires
    settings = Settings(confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5,
                        person_suppression_enabled=True, person_iou_threshold=0.85,
                        confidence=0.5)
    runtime = RuntimeSettings(settings.tunable())
    both = [Detection("dog", 0.9, (150, 150, 190, 190)),
            Detection("person", 0.9, (0, 0, 100, 200))]
    alerter = FakeAlerter()
    status = StatusStore()
    pipe = Pipeline(
        settings=settings, detector=StubDetector([both, both]),
        camera=FakeCamera([np.zeros((200, 200, 3), np.uint8)], loop=True),
        alerter=alerter, runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, EventStore(tmp_path, 10, 0)), clock=lambda: 0.0,
        rng=random.Random(0),
    )
    frame = np.zeros((200, 200, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(2)]  # first sighting never fires
    assert fired[1] is True
    assert status.snapshot().dogs == 1
    assert alerter.calls == 1


def test_annotate_draws_zone_polygon():
    from doggy.pipeline import annotate
    frame = np.zeros((100, 100, 3), np.uint8)
    out = annotate(frame, [], in_zone=[], zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)])
    assert (out != 0).any()   # the polygon outline/fill was drawn
