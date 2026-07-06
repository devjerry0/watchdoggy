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


def test_save_persists():
    saved = {}
    c, _, _ = client(saved=saved)
    c.patch("/api/settings", json={"confidence": 0.65})
    assert c.post("/api/settings/save").status_code == 200
    assert saved["confidence"] == 0.65


def test_write_env_preserves_structural_keys(tmp_path):
    from doggy.web import _write_env
    from doggy.config import TunableSettings
    env = tmp_path / ".env"
    env.write_text("DOGGY_CAMERA_INDEX=1\nDOGGY_CONFIDENCE=0.55\n# comment\n")
    _write_env(TunableSettings(confidence=0.7), path=env)
    text = env.read_text()
    assert "DOGGY_CAMERA_INDEX=1" in text
    assert "DOGGY_CONFIDENCE=0.7" in text
    assert "# comment" in text


def _app_with_events(tmp_path):
    from doggy.alerter import FakeAlerter
    from doggy.config import Settings
    from doggy.state import FrameBuffer, RuntimeSettings, StatusStore

    settings = Settings(event_log_dir=tmp_path)
    runtime = RuntimeSettings(settings.tunable())
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter())
    return TestClient(app)


def test_events_route_serves_thumbnail(tmp_path):
    (tmp_path / "fire_1.jpg").write_bytes(b"\xff\xd8\xff")
    c = _app_with_events(tmp_path)
    r = c.get("/events/fire_1.jpg")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8\xff"


def test_events_route_404_for_missing(tmp_path):
    c = _app_with_events(tmp_path)
    assert c.get("/events/nope.jpg").status_code == 404


def test_write_env_roundtrips_zone_points(tmp_path, monkeypatch):
    from doggy.web import _write_env
    from doggy.config import Settings, TunableSettings
    env = tmp_path / ".env"
    env.write_text("DOGGY_CAMERA_INDEX=0\n")
    _write_env(TunableSettings(zone_enabled=True,
                               zone_points=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.1)]), env)
    text = env.read_text()
    assert "DOGGY_ZONE_POINTS=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.1]]" in text
    assert "DOGGY_CAMERA_INDEX=0" in text            # structural key preserved
    # and it re-parses:
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=str(env))
    assert s.zone_points == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.1)]


def test_index_has_zone_controls():
    from fastapi.testclient import TestClient
    from doggy.web import create_app
    from doggy.config import Settings
    from doggy.state import FrameBuffer, RuntimeSettings, StatusStore
    from doggy.alerter import FakeAlerter
    s = Settings()
    app = create_app(s, RuntimeSettings(s.tunable()), FrameBuffer(), StatusStore(), FakeAlerter())
    html = TestClient(app).get("/").text
    assert "Finish zone" in html and "Clear zone" in html
    assert "detect_interval_seconds" in html


def test_index_has_temp_readout():
    from fastapi.testclient import TestClient
    from doggy.web import create_app
    from doggy.config import Settings
    from doggy.state import FrameBuffer, RuntimeSettings, StatusStore
    from doggy.alerter import FakeAlerter
    s = Settings()
    app = create_app(s, RuntimeSettings(s.tunable()), FrameBuffer(), StatusStore(), FakeAlerter())
    html = TestClient(app).get("/").text
    assert 'id="temp"' in html
    assert "COOLING" in html
