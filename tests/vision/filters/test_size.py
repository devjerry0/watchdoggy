from doggy.core.config import TunableSettings
from doggy.vision.analysis import FrameAnalysis
from doggy.vision.detection import Detection
from doggy.vision.filters.size import OversizeFilter, is_oversize

# 480x640 frame (h, w) like the Pi camera.
SHAPE = (480, 640, 3)


def _analysis(candidates):
    return FrameAnalysis(shape=SHAPE, people=[], targets=list(candidates),
                         candidates=list(candidates))


def dog(box, c=0.9):
    return Detection("dog", c, box)


# Geometry from the calibration set (2026-08-13, 58 real triggering frames).


def test_bent_over_torso_is_oversize():
    # The classic FP: a=0.32, h=0.60 of frame (person hunched at dishwasher).
    box = (100, 190, 500, 480)  # 400x290 -> area 0.377, height 0.60
    assert is_oversize(box, SHAPE, 0.16)


def test_medium_hunched_person_caught_by_joint_rule():
    # a=0.128, h=0.51 -- under the flat ceiling but big AND tall.
    box = (200, 235, 456, 480)  # 256x245 -> area 0.204? recalc: use exact target
    # 0.128 * 640*480 = 39321 px; h 0.51*480=245 -> w=160
    box = (200, 235, 360, 480)
    assert is_oversize(box, SHAPE, 0.16)


def test_hind_legs_standing_dog_is_kept():
    # Real catch: tall (h=0.54) but small area (0.077) -- must survive.
    # 0.077 * 307200 = 23654 px; h = 0.54*480 = 259 -> w = 91
    box = (300, 221, 391, 480)
    assert not is_oversize(box, SHAPE, 0.16)


def test_biggest_real_dog_is_kept():
    # Largest true dog in the calibration set: a=0.111, h=0.33.
    # h = 158 px, w = 0.111*307200/158 = 216
    box = (200, 322, 416, 480)
    assert not is_oversize(box, SHAPE, 0.16)


def test_filter_narrows_candidates_records_suppressed_keeps_targets():
    big = dog((100, 190, 500, 480))
    small = dog((10, 400, 100, 470))
    a = _analysis([big, small])
    OversizeFilter().apply(a, TunableSettings())
    assert a.candidates == [small]
    assert big in a.suppressed
    assert a.targets == [big, small]  # still drawn


def test_disabled_passes_everything():
    big = dog((100, 190, 500, 480))
    a = _analysis([big])
    OversizeFilter().apply(a, TunableSettings(oversize_suppression_enabled=False))
    assert a.candidates == [big] and a.suppressed == []


def test_ceiling_is_tunable():
    # A modest box (a~0.09, h=0.33) passes at the default ceiling but is
    # suppressed when the user tightens the ceiling below it.
    box = dog((200, 322, 375, 480))  # 175x158 -> 0.090
    a = _analysis([box])
    OversizeFilter().apply(a, TunableSettings(oversize_area_ceiling=0.05))
    assert a.candidates == []
