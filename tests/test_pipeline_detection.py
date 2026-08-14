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
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    clock = iter([0.0, 0.5, 1.0])
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector([dog, dog, none])),
        camera=FakeCamera([np.zeros((16, 16, 3), np.uint8)], loop=True),
        runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome,
        clock=lambda: next(clock), rng=random.Random(0),
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(3)]
    assert fired[2] is True                       # fires on the empty 3rd frame
    assert store.list()[0].confidence == 0.9   # not 0.0


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
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector([both, both])),
        camera=FakeCamera([np.zeros((200, 200, 3), np.uint8)], loop=True),
        runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome, clock=lambda: 0.0,
        rng=random.Random(0),
    )
    frame = np.zeros((200, 200, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(2)]
    assert not any(fired)
    assert status.snapshot().targets == 0
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
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector([both, both])),
        camera=FakeCamera([np.zeros((200, 200, 3), np.uint8)], loop=True),
        runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(alerter, clips, store, outcome),
        clip_service=clips, outcome=outcome, clock=lambda: 0.0,
        rng=random.Random(0),
    )
    frame = np.zeros((200, 200, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(2)]  # first sighting never fires
    assert fired[1] is True
    assert status.snapshot().targets == 1
    assert alerter.calls == 1


def test_pipeline_monitor_mode_counts_targets_without_firing(tmp_path):
    # Monitor mode (empty alert set): the dog shows as "in view" every frame, but
    # nothing can ever fire and the trigger never even enters CONFIRMING -- the
    # timing config here would fire on the 2nd frame if "dog" were alertable.
    settings = Settings(target_labels=("dog",), alert_labels=(),
                        confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    dog = [Detection("dog", 0.9, (0, 0, 10, 10))]
    status = StatusStore()
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    clock = iter([0.0, 1.0, 2.0])
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector([dog, dog, dog])),
        camera=FakeCamera([np.zeros((16, 16, 3), np.uint8)], loop=True),
        runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome,
        clock=lambda: next(clock), rng=random.Random(0),
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    for _ in range(3):
        assert pipe.run_once(frame) is False
        assert status.snapshot().targets == 1
        assert status.snapshot().state == "IDLE"   # never CONFIRMING


def test_pipeline_reports_counter_inventory(tmp_path):
    # A cup seen twice becomes "on the counter" (2-of-5 debounce); it is
    # inventory only, so it must never count as a target or fire. The timing
    # config would fire on the 2nd frame if the cup were a candidate (same
    # firing-capable setup as the monitor-mode test above).
    settings = Settings(confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    cup = [Detection("cup", 0.5, (10, 10, 20, 20))]
    status = StatusStore()
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    clock = iter([0.0, 1.0])
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector([cup, cup])),
        camera=FakeCamera([np.zeros((40, 40, 3), np.uint8)], loop=True),
        runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome, clock=lambda: next(clock),
        rng=random.Random(0),
    )
    frame = np.zeros((40, 40, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(2)]
    assert not any(fired)
    assert status.snapshot().on_counter == [{"label": "cup", "count": 1}]
    assert status.snapshot().targets == 0
    assert status.snapshot().confidence == 0.0    # candidates stayed empty
