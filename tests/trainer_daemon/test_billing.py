"""The credit gate: cloud jobs must not start once Modal credits are spent.

The daemon lives in scripts/ (it ships to the Pi, not the wheel), so the
tests add scripts/ to the path themselves."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from trainer_daemon import billing, env  # noqa: E402


def _seed(tmp_path, monkeypatch, summary, budget=30):
    monkeypatch.setattr(env, "JOBS_DIR", tmp_path)
    if summary is not None:
        (tmp_path / "billing.json").write_text(json.dumps(summary))
    (tmp_path / "trainer-settings.json").write_text(
        json.dumps({"monthly_credits": budget}))


def test_refuses_cloud_kinds_when_credits_spent(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch,
          {"metered_cost": 30.0, "billed_cost": 0.0, "credits_used": 30.0})
    assert "no Modal credits left" in billing.credit_refusal("train")
    assert "no Modal credits left" in billing.credit_refusal("prelabel")


def test_update_jobs_never_refused(tmp_path, monkeypatch):
    # Self-updates come from GitHub, not Modal: they must run even broke.
    _seed(tmp_path, monkeypatch,
          {"metered_cost": 99.0, "billed_cost": 5.0, "credits_used": 30.0})
    assert billing.credit_refusal("update") is None


def test_allows_while_credits_remain(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch,
          {"metered_cost": 4.2, "billed_cost": 0.0, "credits_used": 4.2})
    assert billing.credit_refusal("train") is None


def test_cash_billing_refuses_even_under_budget(tmp_path, monkeypatch):
    # billed_cost > 0 means Modal is charging real money right now; the
    # configured budget being higher changes nothing.
    _seed(tmp_path, monkeypatch,
          {"metered_cost": 10.0, "billed_cost": 0.5, "credits_used": 9.5},
          budget=100)
    assert "billed beyond credits" in billing.credit_refusal("train")


def test_missing_billing_fails_open(tmp_path, monkeypatch):
    # A cost guard, not a lock: unknown billing must not kill autonomy.
    _seed(tmp_path, monkeypatch, summary=None)
    assert billing.credit_refusal("train") is None


def test_corrupt_billing_fails_open(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "JOBS_DIR", tmp_path)
    (tmp_path / "billing.json").write_text("{nope")
    assert billing.credit_refusal("train") is None


def test_zero_budget_means_no_cloud(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch,
          {"metered_cost": 0.0, "billed_cost": 0.0, "credits_used": 0.0},
          budget=0)
    assert billing.credit_refusal("train") is not None


def test_augment_setting_maps_legacy_booleans_to_modes():
    from trainer_daemon import runs
    assert runs.augment_mode(True) == "auto"
    assert runs.augment_mode(False) == "off"
    assert runs.augment_mode("one") == "one"


def test_compare_flag_reaches_the_cloud_run(monkeypatch):
    from trainer_daemon import runs
    monkeypatch.setattr(runs, "batch", lambda job: 128)
    monkeypatch.setattr(runs, "runtime_confidence", lambda: 0.6)
    assert "--compare-init" in runs.recipe_arguments({"params": {"compare_init": True}})
    assert "--compare-init" not in runs.recipe_arguments({"params": {}})


def _fake_modal(monkeypatch, tmp_path, fail_first: bool, write_call: bool):
    import subprocess
    from trainer_daemon import cloud
    monkeypatch.setattr(cloud, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(cloud, "gpu", lambda: "L40S")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if "--call-file" in cmd and write_call:
            (tmp_path / "job_1.call").write_text("fc-123")
        if fail_first and len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(cloud.subprocess, "run", run)
    return cloud, calls


def test_dropped_connection_collects_the_result_by_call_id(monkeypatch, tmp_path):
    cloud, calls = _fake_modal(monkeypatch, tmp_path, fail_first=True, write_call=True)
    cloud.modal_run("kickoff", ["--out-dir", "/tmp/x"], "job_1")
    assert "--detach" in calls[0]
    assert calls[1][1:3] == ["run", f"{cloud.PIPELINE}::collect"]
    assert calls[1][calls[1].index("--call-id") + 1] == "fc-123"
    assert calls[1][calls[1].index("--kind") + 1] == "train"
    assert not (tmp_path / "job_1.call").exists()   # cleaned up after success


def test_failure_before_the_cloud_call_still_fails(monkeypatch, tmp_path):
    import subprocess
    import pytest
    cloud, calls = _fake_modal(monkeypatch, tmp_path, fail_first=True, write_call=False)
    with pytest.raises(subprocess.CalledProcessError):
        cloud.modal_run("kickoff_prelabels", ["--out-dir", "/tmp/x"], "job_1")
    assert len(calls) == 1                           # no collect attempt


def test_hung_client_timeout_also_collects(monkeypatch, tmp_path):
    import subprocess
    from trainer_daemon import cloud
    monkeypatch.setattr(cloud, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(cloud, "gpu", lambda: "L40S")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            (tmp_path / "job_2.call").write_text("fc-9")
            raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(cloud.subprocess, "run", run)
    cloud.modal_run("kickoff_prelabels", ["--out-dir", "/tmp/y"], "job_2")
    assert calls[1][calls[1].index("--kind") + 1] == "label"
