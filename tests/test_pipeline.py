import random

import numpy as np
import pytest

from doggy.reaction.sound import FakeAlerter, SoundReaction
from doggy.vision.camera import FakeCamera
from doggy.core.config import Settings
from doggy.vision.analysis import DetectionAnalyzer
from doggy.vision.detection import Detection
from doggy.vision.detector import StubDetector
from doggy.vision.filters.base import FilterChain
from doggy.vision.filters.person import PersonSuppressionFilter
from doggy.vision.filters.zone import ZoneInclusionFilter
from doggy.events.store import EventStore
from doggy.pipeline import Pipeline
from doggy.decision.gate import FireGate
from doggy.reaction.hub import ReactionHub, SafeReaction
from doggy.reaction.clips import ClipBuffer, ClipService
from doggy.reaction.outcome import OutcomeWatcher
from doggy.reaction.recorder import Recorder
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore


def _analyzer(detector):
    return DetectionAnalyzer(
        detector, FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()]))


def _clips(store, settings, runtime):
    return ClipService(store, store.dir, ClipBuffer(settings.clip_window_seconds), runtime)


def _outcome(store, runtime):
    return OutcomeWatcher(store, FireGate(runtime), FakeAlerter(), runtime)


def _hub(alerter, clip_service, store, outcome):
    return ReactionHub(
        [SafeReaction(SoundReaction(alerter, store)), SafeReaction(clip_service),
         SafeReaction(outcome)])


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


def test_pipeline_finalizes_clip_after_postroll(tmp_path):
    # End-to-end deferred clip path: fire -> pending -> finalize -> attach_clip.
    # A fire lands on the 2nd run_once (TriggerLogic never fires on first sighting).
    # fire_ts = 1.0, post-roll end = 1.1; the 3rd run_once's clock (2.0) is past
    # that, so the pending clip is sliced, encoded, and attached to the event.
    settings = Settings(clips_enabled=True, clip_window_seconds=10,
                        clip_preroll_seconds=0.1, clip_postroll_seconds=0.1, clip_fps=6,
                        confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5, confidence=0.5)
    runtime = RuntimeSettings(settings.tunable())
    dog = [Detection("dog", 0.9, (0, 0, 10, 10))]
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    clock = iter([0.0, 1.0, 2.0])
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector([dog, dog, dog])),
        camera=FakeCamera([np.zeros((16, 16, 3), np.uint8)], loop=True),
        runtime=runtime, status=StatusStore(),
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome,
        clock=lambda: next(clock), rng=random.Random(0),
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(3)]
    assert fired[1] is True                     # fired on the 2nd sighting

    rec = store.list()[0]
    assert rec.clip is not None                 # clip was attached to the event
    assert (tmp_path / rec.clip).is_file()       # and the clip file exists on disk


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


def test_pipeline_attaches_outcome_after_dog_leaves(tmp_path):
    # End-to-end wiring: a fire, then the zone stays empty past the clear
    # debounce, so the OutcomeWatcher attaches the outcome to the stored event.
    settings = Settings(confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    dog = [Detection("dog", 0.9, (0, 0, 10, 10))]
    none: list[Detection] = []
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    clock = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector([dog, dog, none, none, none])),
        camera=FakeCamera([np.zeros((16, 16, 3), np.uint8)], loop=True),
        runtime=runtime, status=StatusStore(),
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome,
        clock=lambda: next(clock), rng=random.Random(0),
    )
    frame = np.zeros((16, 16, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(5)]
    assert fired[1] is True                    # fired on the 2nd sighting (mono 1.0)
    rec = store.list()[0]
    assert rec.clear_seconds is not None       # gone at 2.0; debounce met at 4.0
    assert rec.outcome_at is not None


def test_pipeline_fire_frame_inventory_not_in_before_snapshot(tmp_path):
    # Theft attribution: the "before" inventory is snapshotted when the fire
    # publishes DogCaught, and run_once feeds the fire frame to the watcher
    # only AFTER the fire block -- so a sighting on the fire frame itself must
    # not promote an item into "before". The banana (seen on the two frames
    # before any dog) is genuinely before and counts as taken; the sandwich
    # reaches the tracker's 2-of-5 bar only WITH the fire frame, so it must
    # stay out of "before" and out of `taken`.
    settings = Settings(confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    banana = Detection("banana", 0.8, (20, 20, 30, 30))
    sandwich = Detection("sandwich", 0.8, (30, 30, 40, 40))
    script = [
        [banana],              # t=0.0  banana's 1st sighting
        [banana, sandwich],    # t=1.0  banana at 2-of-5; sandwich's 1st sighting
        [dog],                 # t=2.0  first dog sighting: arms, never fires
        [dog, sandwich],       # t=3.0  FIRE; sandwich's 2nd sighting rides the fire frame
        [],                    # t=4.0  everything gone; clear debounce starts
        [],                    # t=4.5  0.5s clear: not yet
        [],                    # t=6.1  2.1s clear: finalize; both foods aged out
    ]
    store = EventStore(tmp_path, 10, 0)
    clips = _clips(store, settings, runtime)
    outcome = _outcome(store, runtime)
    clock = iter([0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 6.1])
    pipe = Pipeline(
        settings=settings, analyzer=_analyzer(StubDetector(script)),
        camera=FakeCamera([np.zeros((100, 100, 3), np.uint8)], loop=True),
        runtime=runtime, status=StatusStore(),
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        gate=FireGate(runtime), recorder=Recorder(store), hub=_hub(FakeAlerter(), clips, store, outcome),
        clip_service=clips, outcome=outcome,
        clock=lambda: next(clock), rng=random.Random(0),
    )
    frame = np.zeros((100, 100, 3), np.uint8)
    fired = [pipe.run_once(frame) for _ in range(7)]
    assert fired[3] is True                          # fired on the 2nd dog sighting (mono 3.0)
    rec = store.list()[0]
    assert rec.taken == ["banana"]                   # sandwich only sighted pre-fire once
    assert rec.clear_seconds == pytest.approx(1.0)   # gone at 4.0, fire at 3.0
    assert rec.outcome_at is not None


def test_annotate_draws_zone_polygon():
    from doggy.vision.annotate import annotate
    frame = np.zeros((100, 100, 3), np.uint8)
    out = annotate(frame, [], in_zone=[], zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)])
    assert (out != 0).any()   # the polygon outline/fill was drawn
