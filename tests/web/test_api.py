import numpy as np
from fastapi.testclient import TestClient

from doggy.reaction.sound import FakeAlerter
from doggy.core.config import Settings
from doggy.events.store import EventStore
from doggy.decision.gate import FireGate
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.web import create_app


def client(tmp_path, saved=None):
    settings = Settings(event_log_dir=tmp_path)
    runtime = RuntimeSettings(settings.tunable())
    alerter = FakeAlerter()
    store = EventStore(tmp_path, 100, 0)
    gate = FireGate(runtime)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), alerter, store, gate,
                     save_env=lambda t: saved.update(t.model_dump()) if saved is not None else None)
    return TestClient(app), runtime, alerter


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


def _app_with_events(tmp_path):
    settings = Settings(event_log_dir=tmp_path)
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path, 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(), store,
                     FireGate(runtime))
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


def test_events_list_and_delete(tmp_path):
    store, ids = _seeded_store(tmp_path, 2)
    c = _app_with_store(tmp_path, store)
    r = c.get("/api/events").json()
    assert len(r["events"]) == 2 and "age_seconds" in r["events"][0]
    assert c.delete(f"/api/events/{ids[0]}").status_code == 200
    assert len(c.get("/api/events").json()["events"]) == 1
    assert c.delete("/api/events/nope").status_code == 404


def test_stats_endpoint(tmp_path):
    store, _ = _seeded_store(tmp_path, 0)
    c = _app_with_store(tmp_path, store)
    assert "today" in c.get("/api/stats").json()


def test_clear(tmp_path):
    store, _ = _seeded_store(tmp_path, 2)
    c = _app_with_store(tmp_path, store)
    c.post("/api/events/clear")
    assert c.get("/api/events").json()["events"] == []


def test_events_age_prefers_wall_time(tmp_path):
    store, _ = _seeded_store(tmp_path, 1)
    c = _app_with_store(tmp_path, store)
    ev = c.get("/api/events").json()["events"][0]
    # wall_time is a real (old) epoch -> age comes from the wall clock and is large.
    assert ev["wall_time"] is not None and ev["age_seconds"] > 0


def test_events_limit(tmp_path):
    store, _ = _seeded_store(tmp_path, 3)
    c = _app_with_store(tmp_path, store)
    assert len(c.get("/api/events", params={"limit": 1}).json()["events"]) == 1


def test_lab_endpoint_shape(tmp_path):
    import time
    store, ids = _seeded_store(tmp_path, 1)
    store.attach_sound(ids[0], "chirp.wav")
    store.attach_outcome(ids[0], clear_seconds=4.0, taken=[], wall_time=time.time())
    c = _app_with_store(tmp_path, store)
    body = c.get("/api/lab").json()
    assert "thefts_this_week" in body
    assert len(body["sounds"]) == 1
    row = body["sounds"][0]
    assert set(row) == {"sound", "plays", "completed", "deterred_rate",
                        "avg_clear_s", "wearing_off"}
    assert row["sound"] == "chirp.wav" and row["plays"] == 1


def test_index_has_deterrence_card(tmp_path):
    store, _ = _seeded_store(tmp_path, 0)
    c = _app_with_store(tmp_path, store)
    html = c.get("/").text
    assert "Deterrence" in html
    assert "/api/lab" in html


def test_clips_route_serves_and_404(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"data")
    store, _ = _seeded_store(tmp_path, 0)
    c = _app_with_store(tmp_path, store)
    assert c.get("/clips/clip.mp4").content == b"data"
    assert c.get("/clips/missing.mp4").status_code == 404


def _sounds_client(tmp_path):
    sounds = tmp_path / "sounds"
    sounds.mkdir()
    settings = Settings(event_log_dir=tmp_path, clips_dir=sounds)
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path, 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, FireGate(runtime))
    return TestClient(app), sounds, runtime


def test_sounds_lists_files_and_selected(tmp_path):
    c, sounds, _ = _sounds_client(tmp_path)
    (sounds / "bark.wav").write_bytes(b"RIFF")
    (sounds / "growl.mp3").write_bytes(b"ID3")
    (sounds / "notes.txt").write_bytes(b"nope")  # non-audio: excluded
    body = c.get("/api/sounds").json()
    assert body["sounds"] == ["bark.wav", "growl.mp3"]
    assert body["selected"] == "random"


def test_sounds_reflects_selected(tmp_path):
    c, sounds, runtime = _sounds_client(tmp_path)
    (sounds / "bark.wav").write_bytes(b"RIFF")
    c.patch("/api/settings", json={"selected_sound": "bark.wav"})
    assert c.get("/api/sounds").json()["selected"] == "bark.wav"


def test_upload_saves_wav(tmp_path):
    c, sounds, _ = _sounds_client(tmp_path)
    r = c.post("/api/sounds", files={"file": ("new.wav", b"RIFFDATA", "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "name": "new.wav"}
    assert (sounds / "new.wav").read_bytes() == b"RIFFDATA"


def test_upload_sanitizes_path_traversal(tmp_path):
    c, sounds, _ = _sounds_client(tmp_path)
    r = c.post("/api/sounds", files={"file": ("../evil.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 200
    assert r.json()["name"] == "evil.wav"
    assert (sounds / "evil.wav").is_file()


def test_upload_rejects_non_audio(tmp_path):
    c, _, _ = _sounds_client(tmp_path)
    r = c.post("/api/sounds", files={"file": ("x.txt", b"x", "text/plain")})
    assert r.status_code == 422


def test_snooze_endpoint_blocks_then_cancel_re_allows(tmp_path):
    import time
    settings = Settings(event_log_dir=tmp_path)
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path, 100, 0)
    gate = FireGate(runtime)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, gate)
    c = TestClient(app)
    assert gate.allow(now=time.monotonic()) is True
    assert c.post("/api/snooze", json={"minutes": 5}).json() == {"ok": True}
    assert gate.allow(now=time.monotonic()) is False  # snoozed
    assert c.post("/api/snooze/cancel").json() == {"ok": True}
    assert gate.allow(now=time.monotonic()) is True   # re-armed
