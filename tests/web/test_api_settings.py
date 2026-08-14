from doggy.core.config import Settings
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.decision.gate import FireGate
from doggy.events.store import EventStore
from doggy.reaction.sound import FakeAlerter
from doggy.web import create_app
from fastapi.testclient import TestClient

from .conftest import client


def test_status_returns_settings_and_state(tmp_path):
    c, _, _ = client(tmp_path)
    r = c.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "IDLE"
    assert body["settings"]["confidence"] == 0.55


def test_patch_updates_runtime(tmp_path):
    c, runtime, _ = client(tmp_path)
    r = c.patch("/api/settings", json={"confidence": 0.8})
    assert r.status_code == 200
    assert runtime.get().confidence == 0.8
    assert c.get("/api/status").json()["settings"]["confidence"] == 0.8


def test_patch_rejects_invalid(tmp_path):
    c, _, _ = client(tmp_path)
    r = c.patch("/api/settings", json={"window_m": 9, "window_n": 3})
    assert r.status_code == 422


def test_status_exposes_armed_fields(tmp_path):
    c, _, _ = client(tmp_path)
    body = c.get("/api/status").json()
    assert body["armed"] is True             # default: no schedule -> always armed
    assert body["next_change_seconds"] is None
    assert body["settings"]["schedule_enabled"] is False
    assert body["settings"]["armed_windows"] == []


def test_patch_armed_windows_round_trips(tmp_path):
    c, runtime, _ = client(tmp_path)
    win = [{"days": [0, 1, 2, 3, 4], "start": "21:00", "end": "07:00"}]
    r = c.patch("/api/settings", json={"schedule_enabled": True, "armed_windows": win})
    assert r.status_code == 200
    assert runtime.get().schedule_enabled is True
    assert c.get("/api/status").json()["settings"]["armed_windows"] == win


def test_patch_rejects_bad_armed_window(tmp_path):
    c, _, _ = client(tmp_path)
    bad = [{"days": [0], "start": "24:61", "end": "07:00"}]
    assert c.patch("/api/settings", json={"armed_windows": bad}).status_code == 422


def test_index_has_schedule_controls(tmp_path):
    c, _, _ = client(tmp_path)
    html = c.get("/").text
    assert "On a schedule" in html
    assert "Add times" in html
    assert 'id="schedule_enabled"' in html


def test_index_has_notify_toggle(tmp_path):
    c, _, _ = client(tmp_path)
    html = c.get("/").text
    assert "Notify this device" in html
    assert 'id="notify_enabled"' in html


def test_test_sound_triggers_alerter(tmp_path):
    c, _, alerter = client(tmp_path)
    assert c.post("/api/test-sound").status_code == 200
    assert alerter.calls == 1


def test_save_persists(tmp_path):
    saved = {}
    c, _, _ = client(tmp_path, saved=saved)
    c.patch("/api/settings", json={"confidence": 0.65})
    assert c.post("/api/settings/save").status_code == 200
    assert saved["confidence"] == 0.65


def test_write_env_preserves_structural_keys(tmp_path):
    from doggy.web.envfile import _write_env
    from doggy.core.config import TunableSettings
    env = tmp_path / ".env"
    env.write_text("DOGGY_CAMERA_INDEX=1\nDOGGY_CONFIDENCE=0.55\n# comment\n")
    _write_env(TunableSettings(confidence=0.7), path=env)
    text = env.read_text()
    assert "DOGGY_CAMERA_INDEX=1" in text
    assert "DOGGY_CONFIDENCE=0.7" in text
    assert "# comment" in text


def test_write_env_roundtrips_zone_points(tmp_path, monkeypatch):
    from doggy.web.envfile import _write_env
    from doggy.core.config import Settings, TunableSettings
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


def test_index_has_zone_controls(tmp_path):
    s = Settings(event_log_dir=tmp_path)
    store = EventStore(tmp_path, 100, 0)
    runtime = RuntimeSettings(s.tunable())
    app = create_app(s, runtime, FrameBuffer(), StatusStore(),
                     FakeAlerter(), store, FireGate(runtime))
    html = TestClient(app).get("/").text
    assert "Save area" in html and "Clear area" in html
    assert "detect_interval_seconds" in html


def test_index_has_temp_readout(tmp_path):
    s = Settings(event_log_dir=tmp_path)
    store = EventStore(tmp_path, 100, 0)
    runtime = RuntimeSettings(s.tunable())
    app = create_app(s, runtime, FrameBuffer(), StatusStore(),
                     FakeAlerter(), store, FireGate(runtime))
    html = TestClient(app).get("/").text
    assert 'id="temp"' in html
    assert "Temperature" in html


def test_index_has_value_feature_sections(tmp_path):
    s = Settings(event_log_dir=tmp_path)
    store = EventStore(tmp_path, 100, 0)
    runtime = RuntimeSettings(s.tunable())
    app = create_app(s, runtime, FrameBuffer(), StatusStore(),
                     FakeAlerter(), store, FireGate(runtime))
    html = TestClient(app).get("/").text
    # New value features
    assert "Snooze" in html
    assert "Activity" in html or "today" in html
    assert "Save video clips" in html
    assert 'id="selected_sound"' in html and "<select" in html
    assert "/api/events" in html and "/api/snooze" in html and "/api/sounds" in html
    # Existing controls must still be present
    assert "Save area" in html and "Clear area" in html
    assert "Temperature" in html and "detect_interval_seconds" in html
