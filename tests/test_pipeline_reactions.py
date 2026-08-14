import random

import numpy as np
import pytest

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
