import json

import numpy as np

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.events.store import EventStore
from doggy.reaction.dataset import DatasetCapture
from doggy.reaction.hub import DogCaught
from doggy.vision.analysis import FrameAnalysis
from doggy.vision.detection import Detection


def _img():
    return np.zeros((8, 8, 3), np.uint8)


def _analysis(targets=(), people=(), suppressed=(), lowconf=()):
    return FrameAnalysis(shape=(8, 8, 3), people=list(people),
                         targets=list(targets), candidates=list(targets),
                         suppressed=list(suppressed), lowconf=list(lowconf))


def _cfg(**kw):
    return TunableSettings(dataset_enabled=True, **kw)


def _capture(tmp_path, cap=10**9, enabled=True, wall=None):
    rt = RuntimeSettings(TunableSettings(dataset_enabled=enabled))
    wall_state = {"t": 1_700_000_000.0}

    def wall_clock():
        wall_state["t"] += 1.0  # unique stems per save
        return wall_state["t"]

    return DatasetCapture(tmp_path, cap, rt, wall_clock=wall or wall_clock)


def _samples(tmp_path):
    return sorted(tmp_path.glob("sample_*.json"))


def test_disabled_saves_nothing(tmp_path):
    c = _capture(tmp_path)
    cfg = TunableSettings()  # dataset_enabled defaults False
    c.on_frame(_img(), _analysis(suppressed=[Detection("dog", 0.9, (0, 0, 4, 4))]),
               10.0, cfg)
    assert _samples(tmp_path) == []


def test_suppressed_detection_saves_frame_with_sidecar(tmp_path):
    c = _capture(tmp_path)
    sup = Detection("dog", 0.88, (0, 0, 4, 4))
    c.on_frame(_img(), _analysis(suppressed=[sup]), 10.0, _cfg())
    sides = _samples(tmp_path)
    assert len(sides) == 1
    side = json.loads(sides[0].read_text())
    assert "suppressed" in side["reasons"]
    assert side["detections"]["suppressed"][0]["label"] == "dog"
    assert sides[0].with_suffix(".jpg").is_file()


def test_borderline_confidence_saves(tmp_path):
    c = _capture(tmp_path)
    c.on_frame(_img(), _analysis(targets=[Detection("dog", 0.5, (0, 0, 4, 4))]),
               10.0, _cfg())
    side = json.loads(_samples(tmp_path)[0].read_text())
    assert "borderline" in side["reasons"]


def test_confident_detection_alone_does_not_save(tmp_path):
    c = _capture(tmp_path)
    c.on_frame(_img(), _analysis(targets=[Detection("dog", 0.95, (0, 0, 4, 4))]),
               10.0, _cfg())
    # 0.95 is above the borderline band and nothing was suppressed; the first
    # frame does trigger the hourly periodic sample, so exactly one with only
    # that reason.
    sides = _samples(tmp_path)
    assert len(sides) == 1
    assert json.loads(sides[0].read_text())["reasons"] == ["periodic"]


def test_cooldown_limits_borderline_rate(tmp_path):
    c = _capture(tmp_path)
    d = Detection("dog", 0.5, (0, 0, 4, 4))
    for i in range(5):
        c.on_frame(_img(), _analysis(targets=[d]), 10.0 + i, _cfg())  # 1s apart
    borderline = [s for s in _samples(tmp_path)
                  if "borderline" in json.loads(s.read_text())["reasons"]]
    assert len(borderline) == 1  # 10s cooldown swallowed the rest
    c.on_frame(_img(), _analysis(targets=[d]), 25.0, _cfg())  # past cooldown
    borderline = [s for s in _samples(tmp_path)
                  if "borderline" in json.loads(s.read_text())["reasons"]]
    assert len(borderline) == 2


def test_fire_event_saves_via_hub(tmp_path):
    c = _capture(tmp_path)
    store = EventStore(tmp_path / "events", 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1_700_000_500.0, 10.0)
    c.on_dog_caught(DogCaught(r, _img(), 10.0))
    sides = _samples(tmp_path)
    assert len(sides) == 1
    side = json.loads(sides[0].read_text())
    assert side["reasons"] == ["fire"] and side["event_id"] == r.id


def test_fire_ignored_when_disabled(tmp_path):
    c = _capture(tmp_path, enabled=False)
    store = EventStore(tmp_path / "events", 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1_700_000_500.0, 10.0)
    c.on_dog_caught(DogCaught(r, _img(), 10.0))
    assert _samples(tmp_path) == []


def test_prune_deletes_oldest_past_cap(tmp_path):
    c = _capture(tmp_path, cap=1)  # everything over 1 byte -> keep pruning
    d = Detection("dog", 0.5, (0, 0, 4, 4))
    c.on_frame(_img(), _analysis(targets=[d]), 10.0, _cfg())
    c.on_frame(_img(), _analysis(targets=[d]), 25.0, _cfg())
    # With a 1-byte cap every save prunes all OLDER samples; only sidecars
    # without their jpg would be a bug, so assert pairs stay consistent.
    jpgs = sorted(tmp_path.glob("sample_*.jpg"))
    sides = _samples(tmp_path)
    assert len(jpgs) <= 1 and len(sides) <= 1
    for s in sides:
        assert s.with_suffix(".jpg").is_file() or not jpgs


def test_person_activity_sampling_during_and_after_person(tmp_path):
    # A cooking session: person visible, then bent out of sight (no boxes).
    # Samples flow every PERSON_ACTIVITY_SECONDS while a person is in frame or
    # was seen within the last minute -- including the head-only frames where
    # the model sees nothing at all (prime hard negatives).
    c = _capture(tmp_path)
    person = Detection("person", 0.9, (0, 0, 4, 6))
    c.on_frame(_img(), _analysis(people=[person]), 100.0, _cfg())    # session start
    sides = _samples(tmp_path)
    assert any("person_activity" in json.loads(x.read_text())["reasons"] for x in sides)
    n0 = len(sides)
    c.on_frame(_img(), _analysis(people=[person]), 110.0, _cfg())    # cooldown holds
    assert len(_samples(tmp_path)) == n0
    c.on_frame(_img(), _analysis(people=[person]), 180.0, _cfg())    # still cooking
    # Person bends out of sight; 45s after last sighting (still "recent") and
    # past the 120s cooldown -> the head-only-phase sample is due.
    c.on_frame(_img(), _analysis(), 225.0, _cfg())
    tail = [x for x in _samples(tmp_path)
            if "person_activity" in json.loads(x.read_text())["reasons"]]
    assert len(tail) == 2
    # Long after the person left (>60s since last sighting): no more sampling.
    c.on_frame(_img(), _analysis(), 500.0, _cfg())
    tail = [x for x in _samples(tmp_path)
            if "person_activity" in json.loads(x.read_text())["reasons"]]
    assert len(tail) == 2


def test_stats_counts_and_reasons(tmp_path):
    c = _capture(tmp_path)
    c.on_frame(_img(), _analysis(suppressed=[Detection("dog", 0.9, (0, 0, 4, 4))]),
               10.0, _cfg())
    st = c.stats()
    assert st["samples"] == 1
    assert st["by_reason"].get("suppressed") == 1
    assert st["bytes"] > 0 and st["cap_bytes"] == 10**9


def test_lowconf_detection_triggers_borderline_capture(tmp_path):
    # The confusion-zone harvest: a 0.4 "dog" below the alarm threshold still
    # produces a borderline sample (this path was dead before lowconf existed).
    c = _capture(tmp_path)
    weak = Detection("dog", 0.4, (0, 0, 4, 4))
    c.on_frame(_img(), _analysis(lowconf=[weak]), 10.0, _cfg())
    sides = _samples(tmp_path)
    assert any("borderline" in json.loads(x.read_text())["reasons"] for x in sides)
    side = next(json.loads(x.read_text()) for x in sides
                if "borderline" in json.loads(x.read_text())["reasons"])
    assert side["detections"]["lowconf"][0]["confidence"] == 0.4


def test_lowconf_person_counts_for_person_activity(tmp_path):
    # A bent person scoring 0.35 (below threshold) still marks the kitchen as
    # occupied for the session sampler.
    c = _capture(tmp_path)
    bent = Detection("person", 0.35, (0, 4, 6, 8))
    c.on_frame(_img(), _analysis(lowconf=[bent]), 100.0, _cfg())
    sides = _samples(tmp_path)
    assert any("person_activity" in json.loads(x.read_text())["reasons"] for x in sides)


def test_fire_saves_approach_context_from_raw_ring(tmp_path):
    # Frames flow through on_frame for 12s before a fire; the fire then saves
    # the raw approach sequence (spaced, capped) as fire_context samples.
    c = _capture(tmp_path)
    for i in range(12):  # one frame per second for 12s
        c.on_frame(_img(), _analysis(), 100.0 + i, _cfg())
    store = EventStore(tmp_path / "events", 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1_700_000_500.0, 112.0)
    c.on_dog_caught(DogCaught(r, _img(), 112.0))
    sides = [json.loads(x.read_text()) for x in _samples(tmp_path)]
    ctx = [m for m in sides if m["reasons"] == ["fire_context"]]
    fire = [m for m in sides if m["reasons"] == ["fire"]]
    assert len(fire) == 1
    assert 4 <= len(ctx) <= 6          # 12s window / 2s spacing, capped at 6
    assert all(m["event_id"] == r.id for m in ctx)
    # Every context sample has its jpg pair.
    for side in _samples(tmp_path):
        assert side.with_suffix(".jpg").is_file()


def test_no_ring_growth_when_disabled(tmp_path):
    c = _capture(tmp_path, enabled=False)
    cfg = TunableSettings()  # dataset_enabled False
    for i in range(5):
        c.on_frame(_img(), _analysis(), 100.0 + i, cfg)
    assert c._ring.slice_timed(0, 1000) == []
