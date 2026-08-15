import numpy as np

from doggy.reaction.capture_policy import (
    DEDUP_HAMMING_FLOOR,
    CapturePolicy,
    dhash,
    hamming,
)
from doggy.vision.analysis import FrameAnalysis
from doggy.vision.detection import Detection


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


def _analysis(conf=None):
    targets = [] if conf is None else [Detection("dog", conf, (10, 10, 40, 40))]
    return FrameAnalysis(shape=(120, 160, 3), people=[], targets=targets,
                         candidates=list(targets), suppressed=[], lowconf=[])


def test_dhash_stable_and_sensitive():
    f = _frame(1)
    assert dhash(f) == dhash(f.copy())
    assert hamming(dhash(f), dhash(_frame(2))) >= DEDUP_HAMMING_FLOOR


def test_duplicate_suppresses_near_identical_saves():
    policy = CapturePolicy()
    frame = _frame(3)
    assert not policy.duplicate(frame, ["periodic"], _analysis())
    policy.mark_saved(["periodic"], 0.0, frame)
    # The identical scene again: suppressed as a near-duplicate...
    assert policy.duplicate(frame.copy(), ["person_activity"], _analysis())
    # ...but a genuinely different scene passes.
    assert not policy.duplicate(_frame(4), ["person_activity"], _analysis())


def test_fires_and_dog_frames_are_exempt_from_dedup():
    policy = CapturePolicy()
    frame = _frame(5)
    policy.mark_saved(["periodic"], 0.0, frame)
    assert not policy.duplicate(frame.copy(), ["fire"], _analysis())
    assert not policy.duplicate(frame.copy(), ["fire_context"], _analysis())
    # A frame with any dog detection is never thinned: on a fixed camera
    # distinct dog moments hash nearly alike, and dogs are the scarce class.
    assert not policy.duplicate(frame.copy(), ["borderline"], _analysis(0.5))


def test_flicker_fires_on_band_crossings_not_steady_scores():
    policy = CapturePolicy()
    # A steady borderline score enters the band once: borderline triggers,
    # but flicker does not.
    reasons = policy.reasons_due(_analysis(0.5), 0.0)
    assert "borderline" in reasons and "flicker" not in reasons
    # Oscillating in and out of the band inside the window = flicker.
    policy.reasons_due(_analysis(0.9), 1.0)   # leaves the band (crossing 2)
    reasons = policy.reasons_due(_analysis(0.5), 2.0)  # re-enters (crossing 3)
    assert "flicker" in reasons


def test_flicker_crossings_age_out():
    policy = CapturePolicy()
    policy.reasons_due(_analysis(0.5), 0.0)
    policy.reasons_due(_analysis(0.9), 1.0)
    # Long after the window, one fresh crossing is not enough.
    reasons = policy.reasons_due(_analysis(0.5), 500.0)
    assert "flicker" not in reasons
