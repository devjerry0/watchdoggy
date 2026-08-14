import io
import zipfile

from doggy.core.config import Settings
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.decision.gate import FireGate
from doggy.events.store import EventStore
from doggy.reaction.sound import FakeAlerter
from doggy.web import create_app, serve
from fastapi.testclient import TestClient

from .conftest import _app_with_store, _seeded_store, _serve_deps


def test_dashboard_ping_is_204_with_cors(tmp_path):
    s = Settings(event_log_dir=tmp_path)
    c = TestClient(create_app(s, *_serve_deps(s)))
    r = c.get("/ping")
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == "*"


def test_serve_runs_door_and_dashboard_when_tls(monkeypatch, tmp_path):
    import threading

    runs = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: runs.append(kw))

    class FakeThread:  # run the door synchronously so both uvicorn.run calls land
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(threading, "Thread", FakeThread)
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    s = Settings(event_log_dir=tmp_path, ssl_cert=cert, ssl_key=key)
    serve(s, *_serve_deps(s))

    assert len(runs) == 2
    door, dashboard = runs  # door starts first (in the daemon thread), then the dashboard
    assert door["port"] == s.web_port
    assert "ssl_certfile" not in door and "ssl_keyfile" not in door
    assert dashboard["port"] == s.ssl_port
    assert dashboard["ssl_certfile"] == str(cert) and dashboard["ssl_keyfile"] == str(key)


def test_serve_passes_ssl_when_configured(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(kw))
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    s = Settings(event_log_dir=tmp_path, ssl_cert=cert, ssl_key=key)
    serve(s, *_serve_deps(s))
    assert calls["ssl_certfile"] == str(cert) and calls["ssl_keyfile"] == str(key)


def test_serve_plain_http_without_ssl(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(kw))
    s = Settings(event_log_dir=tmp_path)
    serve(s, *_serve_deps(s))
    # Exactly today's call: no ssl kwargs sneak in when TLS is not configured.
    assert calls == {"host": s.web_host, "port": s.web_port, "log_level": "warning"}


def test_serve_ignores_partial_ssl_config(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(kw))
    cert = tmp_path / "c.pem"
    cert.write_text("x")
    s = Settings(event_log_dir=tmp_path, ssl_cert=cert)
    serve(s, *_serve_deps(s))
    assert "ssl_certfile" not in calls and "ssl_keyfile" not in calls


def test_ca_pem_served_when_configured(tmp_path):
    pem = b"-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"
    ca = tmp_path / "ca.pem"
    ca.write_bytes(pem)
    s = Settings(event_log_dir=tmp_path, ca_cert=ca)
    c = TestClient(create_app(s, *_serve_deps(s)))
    r = c.get("/ca.pem")
    assert r.status_code == 200
    assert r.content == pem
    assert r.headers["content-type"].startswith("application/x-pem-file")
    assert "watchdoggy-ca.pem" in r.headers["content-disposition"]


def test_ca_pem_404_when_not_configured(tmp_path):
    s = Settings(event_log_dir=tmp_path)
    c = TestClient(create_app(s, *_serve_deps(s)))
    assert c.get("/ca.pem").status_code == 404


def test_ca_pem_404_when_file_missing(tmp_path):
    s = Settings(event_log_dir=tmp_path, ca_cert=tmp_path / "gone.pem")
    c = TestClient(create_app(s, *_serve_deps(s)))
    assert c.get("/ca.pem").status_code == 404


def test_export_returns_zip_with_events(tmp_path):
    store, _ = _seeded_store(tmp_path, 1)
    c = _app_with_store(tmp_path, store)
    r = c.get("/api/export")
    assert r.status_code == 200
    assert "watchdoggy-export.zip" in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert "events.jsonl" in names
    assert any(n.endswith(".jpg") for n in names)


def test_export_omits_deleted_thumb(tmp_path):
    store, _ = _seeded_store(tmp_path, 1)
    thumb = store.list()[0].thumb
    (tmp_path / thumb).unlink()  # file vanished from disk (concurrent delete/prune)
    c = _app_with_store(tmp_path, store)
    r = c.get("/api/export")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "events.jsonl" in names
    assert thumb not in names  # missing file skipped, no crash


def test_export_includes_clip(tmp_path):
    store, ids = _seeded_store(tmp_path, 1)
    (tmp_path / "fire_0.webp").write_bytes(b"RIFFWEBPDATA")
    store.attach_clip(ids[0], "fire_0.webp")
    c = _app_with_store(tmp_path, store)
    r = c.get("/api/export")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "fire_0.webp" in names  # clip file rides along in the export zip


def test_index_has_kiosk_and_export(tmp_path):
    store, _ = _seeded_store(tmp_path, 0)
    c = _app_with_store(tmp_path, store)
    html = c.get("/").text
    assert "Fullscreen" in html
    assert "Export all" in html
    assert "/api/export" in html


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
