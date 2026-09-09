import json

import numpy as np

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.events.store import EventStore
from doggy.reaction.dataset import DatasetCapture
from doggy.reaction.hub import DogCaught
from doggy.vision.analysis import FrameAnalysis
from doggy.vision.detection import Detection


def _img(seed=0):
    # Mid-gray base (not black: harvest triggers skip lights-off frames) with
    # per-seed noise: near-duplicate suppression drops repeats of the same
    # scene, so tests that expect a SECOND save must pass a fresh seed.
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, (8, 8, 3)).astype(np.uint8)


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

    # prune_interval=0: these tests assert per-save pruning semantics.
    return DatasetCapture(tmp_path, cap, rt, wall_clock=wall or wall_clock,
                          prune_interval=0.0)


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
    c.on_frame(_img(1), _analysis(targets=[d]), 25.0, _cfg())  # past cooldown
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


def test_prune_never_deletes_labeled_frames(tmp_path):
    import json as _json
    c = _capture(tmp_path, cap=10**9)
    d = Detection("dog", 0.5, (0, 0, 4, 4))
    c.on_frame(_img(1), _analysis(targets=[d]), 10.0, _cfg())
    labeled = sorted(_samples(tmp_path))[0]
    meta = _json.loads(labeled.read_text())
    meta["human_label"] = "dog"
    labeled.write_text(_json.dumps(meta))
    # Shrink the cap to nothing: every subsequent save prunes hard, and the
    # old blind oldest-first prune would have eaten the labeled frame first.
    c._cap = 1
    c.on_frame(_img(2), _analysis(targets=[d]), 25.0, _cfg())
    c.on_frame(_img(3), _analysis(targets=[d]), 40.0, _cfg())
    survivors = {s.stem for s in _samples(tmp_path)}
    assert labeled.stem in survivors  # the labeled frame outlives the cap


def test_prune_runs_once_per_interval_not_per_save(tmp_path, monkeypatch):
    # Pruning stats every sample_* file (~95k on the Pi): per save it was a
    # 2-4s CPU burst felt as an FPS dip every few minutes.
    from doggy.reaction import dataset as ds
    calls = []
    monkeypatch.setattr(ds, "prune", lambda d, cap: calls.append(cap))
    mono = {"t": 100.0}
    rt = RuntimeSettings(TunableSettings(dataset_enabled=True))
    c = DatasetCapture(tmp_path, 10**9, rt, clock=lambda: mono["t"],
                       wall_clock=lambda: 1_700_000_000.0 + mono["t"],
                       prune_interval=600.0)
    d = Detection("dog", 0.5, (0, 0, 4, 4))
    c.on_frame(_img(1), _analysis(targets=[d]), 10.0, _cfg())
    mono["t"] += 30.0
    c.on_frame(_img(2), _analysis(targets=[d]), 25.0, _cfg())
    assert len(calls) == 1                      # second save inside the interval
    mono["t"] += 600.0
    c.on_frame(_img(3), _analysis(targets=[d]), 40.0, _cfg())
    assert len(calls) == 2                      # interval elapsed -> prunes again
