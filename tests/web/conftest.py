import json

import numpy as np
from fastapi.testclient import TestClient

from doggy.core.config import Settings
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.decision.gate import FireGate
from doggy.events.store import EventStore
from doggy.reaction.sound import FakeAlerter
from doggy.web import create_app

# Shared test doubles for tests/web/test_api_*.py, test_dataset_*.py and
# test_training_*.py. Split out of the (formerly) oversized test_api.py /
# test_dataset.py / test_training.py so no single test file grows past the
# 200-line cap; every helper below is byte-identical to its original.


# -- api ---------------------------------------------------------------

def client(tmp_path, saved=None):
    settings = Settings(event_log_dir=tmp_path)
    runtime = RuntimeSettings(settings.tunable())
    alerter = FakeAlerter()
    store = EventStore(tmp_path, 100, 0)
    gate = FireGate(runtime)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), alerter, store, gate,
                     save_env=lambda t: saved.update(t.model_dump()) if saved is not None else None)
    return TestClient(app), runtime, alerter


def _app_with_events(tmp_path):
    settings = Settings(event_log_dir=tmp_path)
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path, 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(), store,
                     FireGate(runtime))
    return TestClient(app)


def _seeded_store(tmp_path, n=2):
    store = EventStore(tmp_path, 100, 0)
    ids = []
    for i in range(n):
        r = store.add(np.zeros((8, 8, 3), np.uint8), 0.8, 1.0, 1000.0 + i, float(i))
        ids.append(r.id)
    return store, ids


def _app_with_store(tmp_path, store):
    settings = Settings(event_log_dir=tmp_path)
    runtime = RuntimeSettings(settings.tunable())
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(), store,
                     FireGate(runtime))
    return TestClient(app)


def _sounds_client(tmp_path):
    sounds = tmp_path / "sounds"
    sounds.mkdir()
    settings = Settings(event_log_dir=tmp_path, clips_dir=sounds)
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path, 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, FireGate(runtime))
    return TestClient(app), sounds, runtime


def _serve_deps(s):
    runtime = RuntimeSettings(s.tunable())
    return (runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
            EventStore(s.event_log_dir, 100, 0), FireGate(runtime))


# -- dataset -------------------------------------------------------------

def _dataset_client(tmp_path):
    settings = Settings(event_log_dir=tmp_path / "events",
                        dataset_dir=tmp_path / "dataset")
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path / "events", 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, FireGate(runtime))
    return TestClient(app), store, tmp_path / "dataset"


def _seed_sample(dataset_dir, stem, reasons, wall_time, label=None):
    import numpy as np
    import cv2
    import json as _json
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dataset_dir / f"{stem}.jpg"), np.zeros((8, 8, 3), np.uint8))
    meta = {"wall_time": wall_time, "reasons": reasons,
            "detections": {"targets": [{"label": "dog", "confidence": 0.8,
                                        "box": [0, 0, 4, 4]}]}}
    if label:
        meta["human_label"] = label
    (dataset_dir / f"{stem}.json").write_text(_json.dumps(meta))


def _seed_full(ddir, stem, verdict=None, prelabel_dogs=None, hand=None,
               nano_dog_conf=None, wall=100.0):
    import json as _json
    ddir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import cv2
    cv2.imwrite(str(ddir / f"{stem}.jpg"), np.zeros((8, 8, 3), np.uint8))
    meta = {"wall_time": wall, "reasons": ["fire"]}
    if verdict:
        meta["human_label"] = verdict
        meta["labeled_at"] = wall + 1
    if prelabel_dogs is not None:
        meta["prelabels"] = {"model": "yolo26x", "boxes": [
            {"label": "dog", "box": [0, 0, 4, 4]} for _ in range(prelabel_dogs)]}
    if hand:
        meta["human_boxes"] = hand
    if nano_dog_conf is not None:
        meta["detections"] = {"targets": [
            {"label": "dog", "confidence": nano_dog_conf, "box": [0, 0, 4, 4]}]}
    (ddir / f"{stem}.json").write_text(_json.dumps(meta))


# -- training --------------------------------------------------------------

def _training_client(tmp_path):
    settings = Settings(event_log_dir=tmp_path / "events",
                        dataset_dir=tmp_path / "dataset",
                        jobs_dir=tmp_path / "jobs")
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path / "events", 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, FireGate(runtime))
    return TestClient(app), tmp_path


def _seed_sidecar(tmp_path, stem, labeled, prelabeled):
    d = tmp_path / "dataset"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"wall_time": 1.0, "reasons": ["fire"]}
    if labeled:
        meta["human_label"] = "dog"
    if prelabeled:
        meta["prelabels"] = {"model": "yolo26x", "boxes": []}
    (d / f"{stem}.json").write_text(json.dumps(meta))
