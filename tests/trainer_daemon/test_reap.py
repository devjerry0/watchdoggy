"""Stale-run reaping: a crashed pass (power loss, kill) must be reaped on
the NEXT tick via the process-liveness probe, not idle out the 13h window.

The daemon lives in scripts/ (it ships to the Pi, not the wheel), so the
tests add scripts/ to the path themselves."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from trainer_daemon import daemon, env  # noqa: E402
from trainer_daemon import queue as q  # noqa: E402


def _seed_running(tmp_path, monkeypatch, age_seconds):
    monkeypatch.setattr(env, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(q, "JOBS_DIR", tmp_path)
    (tmp_path / "job_1000.json").write_text(json.dumps(
        {"id": "job_1000", "kind": "train", "status": "queued",
         "requested_at": time.time() - age_seconds - 60}))
    (tmp_path / "job_1000.result.json").write_text(json.dumps(
        {"status": "running", "detail": "",
         "updated_at": time.time() - age_seconds}))


def _result(tmp_path):
    return json.loads((tmp_path / "job_1000.result.json").read_text())


def test_dead_process_reaps_immediately(tmp_path, monkeypatch):
    _seed_running(tmp_path, monkeypatch, age_seconds=120)
    monkeypatch.setattr(daemon, "_pipeline_alive", lambda: False)
    assert daemon._reap_running() is False
    assert _result(tmp_path)["status"] == "failed"
    assert "no live pipeline process" in _result(tmp_path)["detail"]


def test_live_fresh_run_is_left_alone(tmp_path, monkeypatch):
    _seed_running(tmp_path, monkeypatch, age_seconds=120)
    monkeypatch.setattr(daemon, "_pipeline_alive", lambda: True)
    assert daemon._reap_running() is True
    assert _result(tmp_path)["status"] == "running"


def test_live_but_ancient_run_hits_the_stale_ceiling(tmp_path, monkeypatch):
    _seed_running(tmp_path, monkeypatch, age_seconds=env.STALE_RUNNING + 600)
    monkeypatch.setattr(daemon, "_pipeline_alive", lambda: True)
    assert daemon._reap_running() is False
    assert _result(tmp_path)["status"] == "failed"
    assert "gave up" in _result(tmp_path)["detail"]


def test_runtime_confidence_prefers_settings_json_then_env(tmp_path, monkeypatch):
    from trainer_daemon import runs
    monkeypatch.setattr(runs, "DOGGY_ROOT", tmp_path)
    assert runs.runtime_confidence() == runs.FALLBACK_FIRE_CONF
    (tmp_path / ".env").write_text("DOGGY_CONFIDENCE=0.55\n")
    assert runs.runtime_confidence() == 0.55
    (tmp_path / "settings.json").write_text(json.dumps({"version": 1, "tunables": {"confidence": 0.6}}))
    assert runs.runtime_confidence() == 0.6           # JSON wins over a stale .env line
