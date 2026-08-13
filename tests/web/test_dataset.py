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
