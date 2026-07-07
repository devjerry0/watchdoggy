import numpy as np

from doggy.core.config import TunableSettings
from doggy.vision.analysis import DetectionAnalyzer
from doggy.vision.detection import Detection
from doggy.vision.detector import StubDetector
from doggy.vision.filters.base import FilterChain
from doggy.vision.filters.person import (
    PersonSuppressionFilter,
    iou,
    suppress_targets_overlapping_people,
)
from doggy.vision.filters.zone import ZoneInclusionFilter


def dog(box, c=0.9):
    return Detection("dog", c, box)


def person(box, c=0.9):
    return Detection("person", c, box)


def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # a & b each area 100, intersection (5,0,10,10)=50, union=150
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9


def test_coincident_dog_and_person_is_suppressed():
    # "dog" box ~= person box (one human, double-labeled) -> removed
    dogs = [dog((0, 0, 100, 200))]
    people = [person((2, 2, 98, 198))]
    assert suppress_targets_overlapping_people(dogs, people, 0.85) == []


def test_real_dog_behind_person_is_kept():
    # dog has its own small distinct box that only clips the person -> low IoU -> kept
    dogs = [dog((150, 150, 190, 190))]
    people = [person((0, 0, 100, 200))]
    assert suppress_targets_overlapping_people(dogs, people, 0.85) == dogs


def test_no_people_keeps_all_dogs():
    dogs = [dog((0, 0, 10, 10))]
    assert suppress_targets_overlapping_people(dogs, [], 0.85) == dogs


def test_only_the_coincident_dog_is_removed():
    coincident = dog((0, 0, 100, 200))
    real = dog((300, 300, 340, 340))
    people = [person((0, 0, 100, 200))]
    assert suppress_targets_overlapping_people([coincident, real], people, 0.85) == [real]


def test_reseed_after_suppression_keeps_detect_only_class_out_of_candidates():
    # The alert-class dog is coincident with a person -> suppressed. The bird is
    # detect-only (in target_labels, not alert_labels). The suppression filter
    # reseeds candidates from the survivors; that reseed must go through the
    # alert set, or the bird would resurrect as a candidate and could fire.
    cfg = TunableSettings(target_labels=["dog", "bird"], alert_labels=["dog"],
                          person_suppression_enabled=True, person_iou_threshold=0.85)
    detections = [
        Detection("dog", 0.9, (0, 0, 100, 200)),
        Detection("person", 0.9, (2, 2, 98, 198)),   # coincident with the "dog"
        Detection("bird", 0.9, (300, 300, 340, 340)),  # elsewhere in frame
    ]
    chain = FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()])
    analyzer = DetectionAnalyzer(StubDetector([detections]), chain)
    a = analyzer.analyze(np.zeros((400, 400, 3), np.uint8), cfg)
    assert [d.label for d in a.targets] == ["bird"]   # dog suppressed, bird drawn
    assert a.candidates == []                          # bird never becomes a candidate
