from fastapi.testclient import TestClient

from doggy.alerter import FakeAlerter
from doggy.config import Settings
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore
from doggy.web import create_app


def client(saved=None):
    settings = Settings()
    runtime = RuntimeSettings(settings.tunable())
    alerter = FakeAlerter()
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), alerter,
                     save_env=lambda t: saved.update(t.model_dump()) if saved is not None else None)
    return TestClient(app), runtime, alerter


def test_status_returns_settings_and_state():
    c, _, _ = client()
    r = c.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "IDLE"
    assert body["settings"]["confidence"] == 0.55


def test_patch_updates_runtime():
    c, runtime, _ = client()
    r = c.patch("/api/settings", json={"confidence": 0.8})
    assert r.status_code == 200
    assert runtime.get().confidence == 0.8
    assert c.get("/api/status").json()["settings"]["confidence"] == 0.8


def test_patch_rejects_invalid():
    c, _, _ = client()
    r = c.patch("/api/settings", json={"window_m": 9, "window_n": 3})
    assert r.status_code == 422


def test_test_sound_triggers_alerter():
    c, _, alerter = client()
    assert c.post("/api/test-sound").status_code == 200
    assert alerter.calls == 1


def test_save_persists(tmp_path):
    saved = {}
    c, _, _ = client(saved=saved)
    c.patch("/api/settings", json={"confidence": 0.65})
    assert c.post("/api/settings/save").status_code == 200
    assert saved["confidence"] == 0.65
