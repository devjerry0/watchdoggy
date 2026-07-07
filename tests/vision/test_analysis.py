import numpy as np

from doggy.vision.analysis import DetectionAnalyzer, FrameAnalysis
from doggy.vision.detection import Detection
from doggy.vision.detector import StubDetector
from doggy.vision.filters.base import FilterChain
from doggy.vision.filters.person import PersonSuppressionFilter
from doggy.vision.filters.zone import ZoneInclusionFilter


def _cfg(**over):
    from doggy.core.config import TunableSettings
    base = dict(person_suppression_enabled=True, person_iou_threshold=0.85,
                zone_enabled=True, zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)])
    base.update(over)
    return TunableSettings(**base)


def test_analyzer_splits_suppresses_and_zones():
    dog_in = Detection("dog", 0.9, (5, 5, 20, 20))
    dog_out = Detection("dog", 0.9, (80, 80, 95, 95))
    fake_person = Detection("person", 0.9, (30, 30, 60, 90))
    fake_dog = Detection("dog", 0.9, (31, 31, 59, 89))   # coincident with person
    det = StubDetector([[dog_in, dog_out, fake_person, fake_dog]])
    chain = FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()])
    a = DetectionAnalyzer(det, chain).analyze(np.zeros((100, 100, 3), np.uint8), _cfg())
    assert a.people == [fake_person]
    assert fake_dog not in a.dogs                 # suppressed
    assert a.candidates == [dog_in]               # zone kept only the in-zone dog
    assert dog_out in a.dogs                      # still drawn, just not a candidate


def test_chain_respects_enable_flags():
    fake_person = Detection("person", 0.9, (30, 30, 60, 90))
    fake_dog = Detection("dog", 0.9, (31, 31, 59, 89))
    det = StubDetector([[fake_person, fake_dog]])
    chain = FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()])
    cfg = _cfg(person_suppression_enabled=False, zone_enabled=False)
    a = DetectionAnalyzer(det, chain).analyze(np.zeros((100, 100, 3), np.uint8), cfg)
    assert a.dogs == [fake_dog] and a.candidates == [fake_dog]


def test_chain_runs_filters_in_order():
    calls = []
    class F:
        def __init__(self, tag): self.tag = tag
        def apply(self, analysis, cfg): calls.append(self.tag)
    FilterChain([F("a"), F("b")]).run(
        FrameAnalysis((1, 1, 3), [], [], []), _cfg())
    assert calls == ["a", "b"]
