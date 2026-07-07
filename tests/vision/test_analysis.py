import numpy as np

from doggy.core.config import TunableSettings
from doggy.vision.analysis import DetectionAnalyzer, FrameAnalysis, InventoryTracker
from doggy.vision.detection import Detection
from doggy.vision.detector import StubDetector
from doggy.vision.filters.base import FilterChain
from doggy.vision.filters.person import PersonSuppressionFilter
from doggy.vision.filters.zone import ZoneInclusionFilter


def _cfg(**over):
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
    assert fake_dog not in a.targets              # suppressed
    assert a.candidates == [dog_in]               # zone kept only the in-zone dog
    assert dog_out in a.targets                   # still drawn, just not a candidate


def test_chain_respects_enable_flags():
    fake_person = Detection("person", 0.9, (30, 30, 60, 90))
    fake_dog = Detection("dog", 0.9, (31, 31, 59, 89))
    det = StubDetector([[fake_person, fake_dog]])
    chain = FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()])
    cfg = _cfg(person_suppression_enabled=False, zone_enabled=False)
    a = DetectionAnalyzer(det, chain).analyze(np.zeros((100, 100, 3), np.uint8), cfg)
    assert a.targets == [fake_dog] and a.candidates == [fake_dog]


def test_analyzer_honors_target_labels():
    cat = Detection("cat", 0.9, (0, 0, 10, 10))
    dog = Detection("dog", 0.9, (20, 20, 30, 30))
    analyzer = DetectionAnalyzer(StubDetector([[cat, dog]]), FilterChain([]))
    cfg = TunableSettings(target_labels=["cat"], alert_labels=["cat"])
    analysis = analyzer.analyze(np.zeros((100, 100, 3), np.uint8), cfg)
    assert analysis.targets == [cat]


def test_detect_only_class_never_becomes_candidate():
    bird = Detection("bird", 0.9, (0, 0, 10, 10))
    dog = Detection("dog", 0.9, (20, 20, 30, 30))
    analyzer = DetectionAnalyzer(StubDetector([[bird, dog]]), FilterChain([]))
    cfg = TunableSettings(target_labels=["dog", "bird"], alert_labels=["dog"])
    analysis = analyzer.analyze(np.zeros((100, 100, 3), np.uint8), cfg)
    assert analysis.targets == [bird, dog]   # both drawn
    assert analysis.candidates == [dog]      # only the dog can fire


def test_analyzer_collects_inventory_separately():
    snack = Detection("sandwich", 0.5, (0, 0, 10, 10))
    dog = Detection("dog", 0.9, (20, 20, 30, 30))
    analyzer = DetectionAnalyzer(StubDetector([[snack, dog]]), FilterChain([]))
    analysis = analyzer.analyze(np.zeros((100, 100, 3), np.uint8), TunableSettings())
    assert analysis.inventory == [snack]
    assert analysis.targets == [dog]
    assert analysis.candidates == [dog]           # inventory never becomes a candidate


def test_inventory_tracker_debounces_two_of_five():
    t = InventoryTracker()
    assert t.update(["cup"]) == []                # seen once: not yet present
    assert t.update(["cup", "cup"]) == [{"label": "cup", "count": 2}]
    for _ in range(4):
        t.update([])                              # gone 4 frames
    assert t.update([]) == []


def test_inventory_tracker_labels_applies_debounce_without_consuming():
    t = InventoryTracker()
    t.update(["cup"])
    t.update(["cup", "banana"])                   # cup: 2 of 5; banana: 1 of 5
    assert t.labels() == {"cup"}
    assert t.labels() == {"cup"}                  # pure read: window unchanged


def test_chain_runs_filters_in_order():
    calls = []
    class F:
        def __init__(self, tag): self.tag = tag
        def apply(self, analysis, cfg): calls.append(self.tag)
    FilterChain([F("a"), F("b")]).run(
        FrameAnalysis((1, 1, 3), [], [], []), _cfg())
    assert calls == ["a", "b"]
