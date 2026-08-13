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


def _client(tmp_path):
    settings = Settings(event_log_dir=tmp_path / "events",
                        dataset_dir=tmp_path / "dataset")
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path / "events", 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, FireGate(runtime))
    return TestClient(app), store, tmp_path / "dataset"


def test_stats_empty(tmp_path):
    c, _, _ = _client(tmp_path)
    body = c.get("/api/dataset").json()
    assert body["samples"] == 0 and body["by_reason"] == {}


def test_mark_event_as_false_positive(tmp_path):
    c, store, dataset_dir = _client(tmp_path)
    r = store.add(np.zeros((8, 8, 3), np.uint8), 0.9, 1.0, 1_700_000_000.0, 1.0)
    assert c.post(f"/api/dataset/mark/{r.id}").json() == {"ok": True}
    sides = list(dataset_dir.glob("sample_*.json"))
    assert len(sides) == 1
    side = json.loads(sides[0].read_text())
    assert side["reasons"] == ["user_marked_fp"]
    assert side["event_id"] == r.id
    assert sides[0].with_suffix(".jpg").is_file()
    # And it shows up in stats.
    assert c.get("/api/dataset").json()["by_reason"]["user_marked_fp"] == 1


def test_mark_unknown_event_404(tmp_path):
    c, _, _ = _client(tmp_path)
    assert c.post("/api/dataset/mark/nope").status_code == 404


def test_mark_traversal_guarded(tmp_path):
    c, _, _ = _client(tmp_path)
    assert c.post("/api/dataset/mark/..%2F..%2Fetc").status_code == 404


def test_clear_removes_samples(tmp_path):
    c, store, dataset_dir = _client(tmp_path)
    r = store.add(np.zeros((8, 8, 3), np.uint8), 0.9, 1.0, 1_700_000_000.0, 1.0)
    c.post(f"/api/dataset/mark/{r.id}")
    assert c.post("/api/dataset/clear").json() == {"ok": True}
    assert list(dataset_dir.glob("sample_*")) == []


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


def test_next_serves_highest_signal_first_and_skips_labeled(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["periodic"], 100)
    _seed_sample(ddir, "sample_2", ["fire"], 50)          # older but higher signal
    _seed_sample(ddir, "sample_3", ["fire"], 60, label="dog")  # already labeled
    d = c.get("/api/dataset/next").json()
    assert d["remaining"] == 2 and d["labeled"] == 1
    assert d["sample"]["name"] == "sample_2"
    assert d["sample"]["image"] == "/dataset/sample_2.jpg"
    assert d["sample"]["detections"]["targets"][0]["label"] == "dog"


def test_label_writes_sidecar_and_next_advances(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    assert c.post("/api/dataset/label",
                  json={"name": "sample_1", "verdict": "no_dog"}).json() == {"ok": True}
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert meta["human_label"] == "no_dog" and "labeled_at" in meta
    d = c.get("/api/dataset/next").json()
    assert d["sample"] is None and d["labeled"] == 1


def test_label_rejects_bad_verdict_and_unknown_name(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    assert c.post("/api/dataset/label",
                  json={"name": "sample_1", "verdict": "cat"}).status_code == 422
    assert c.post("/api/dataset/label",
                  json={"name": "nope", "verdict": "dog"}).status_code == 404


def test_dataset_image_served_and_guarded(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    assert c.get("/dataset/sample_1.jpg").status_code == 200
    assert c.get("/dataset/..%2Fevents%2Fx.jpg").status_code == 404
    assert c.get("/dataset/sample_1.json").status_code == 404  # jpgs only


def test_review_page_served(tmp_path):
    c, _, _ = _client(tmp_path)
    html = c.get("/review").text
    assert "Person, no dog" in html and "Nothing here" in html and "Skip" in html


def test_mark_fp_is_prelabeled_no_dog(tmp_path):
    import numpy as np
    c, store, ddir = _client(tmp_path)
    r = store.add(np.zeros((8, 8, 3), np.uint8), 0.9, 1.0, 1_700_000_000.0, 1.0)
    c.post(f"/api/dataset/mark/{r.id}")
    side = json.loads(next(iter(ddir.glob("sample_*.json"))).read_text())
    assert side["human_label"] == "no_dog"
    # And it never shows up in the review queue.
    assert c.get("/api/dataset/next").json()["sample"] is None


def test_rich_verdicts_person_and_empty(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    _seed_sample(ddir, "sample_2", ["periodic"], 90)
    assert c.post("/api/dataset/label",
                  json={"name": "sample_1", "verdict": "person"}).json() == {"ok": True}
    assert c.post("/api/dataset/label",
                  json={"name": "sample_2", "verdict": "empty"}).json() == {"ok": True}
    assert json.loads((ddir / "sample_1.json").read_text())["human_label"] == "person"
    assert json.loads((ddir / "sample_2.json").read_text())["human_label"] == "empty"
    assert c.get("/api/dataset/next").json()["sample"] is None
