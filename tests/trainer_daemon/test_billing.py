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
