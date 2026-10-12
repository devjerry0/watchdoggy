"""The labeler job on the Pi: its own recipe, the report, and the one
labeling pass a swap queues.

The daemon lives in scripts/ (it ships to the Pi, not the wheel), so the
tests add scripts/ to the path themselves."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import modal_pipeline  # noqa: E402
from trainer_daemon import daemon, env, runs  # noqa: E402
from trainer_daemon import queue as q  # noqa: E402


def _results(decision: str) -> dict:
    return {"run_name": "r1", "labeler": "yolo26x", "decision": decision,
            "reason": "25 exam errors vs the active labeler's 30",
            "promoted": decision == "swap", "gate": None, "replay": None,
            "checkpoints": {}, "epochs": 100, "epochs_requested": 100,
            "batch": 16, "gpu": "L40S", "images_per_second": 120,
            "dataset": {}, "timings": {"total": 1.0},
            "report_md": f"# Labeler run r1\\n\\nDecision: {decision}\\n"}


def _wire(tmp_path, monkeypatch, decision="no swap"):
    """Point the daemon at tmp_path with saved small-model settings (batch
    auto, 120 epochs) and a fake cloud run returning `decision`."""
    for module in (env, q, runs):
        monkeypatch.setattr(module, "JOBS_DIR", tmp_path)
    (tmp_path / "trainer-settings.json").write_text(
        json.dumps({"batch": "auto", "epochs": 120}))
    monkeypatch.setattr(runs, "DEPLOYED_BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(runs, "billing_summary", lambda: None)
    monkeypatch.setattr(runs, "run_cost", lambda before: 1.25)

    def no_batch(job):
        raise AssertionError("the labeler never uses the small model's batch")

    monkeypatch.setattr(runs, "batch", no_batch)
    calls = []

    def fake_modal_run(entrypoint, arguments, job_id):
        calls.append((entrypoint, arguments))
        out = Path(arguments[arguments.index("--out-dir") + 1])
        modal_pipeline._write_labeler_results(out, _results(decision))

    monkeypatch.setattr(runs, "modal_run", fake_modal_run)
    return calls


def _arg(arguments, flag):
    return arguments[arguments.index(flag) + 1]


def _queued_prelabels(tmp_path):
    return [j for j in q.jobs() if j["kind"] == "prelabel"]


def test_labeler_job_runs_its_own_recipe(tmp_path, monkeypatch):
    calls = _wire(tmp_path, monkeypatch)
    (tmp_path / "bundle").mkdir()
    detail = runs.RUNNERS["labeler"]({"id": "job_1", "kind": "labeler"})
    entrypoint, arguments = calls[0]
    assert entrypoint == "kickoff_labeler"
    assert _arg(arguments, "--epochs") == "100"
    assert _arg(arguments, "--auto-sample") == "8000"
    assert _arg(arguments, "--deployed-dir") == str(tmp_path / "bundle")
    assert "--batch" not in arguments  # the cloud trains at LABELER_BATCH
    assert (tmp_path / "job_1.report.md").read_text().startswith("# Labeler run r1")
    assert "25 exam errors vs the active labeler's 30" in detail
    assert detail.endswith("($1.25)")


def test_job_params_set_epochs_and_sample(tmp_path, monkeypatch):
    calls = _wire(tmp_path, monkeypatch)
    runs.run_labeler_job({"id": "job_1", "kind": "labeler",
                          "params": {"epochs": 3, "auto_sample": 500}})
    arguments = calls[0][1]
    assert _arg(arguments, "--epochs") == "3"
    assert _arg(arguments, "--auto-sample") == "500"
    assert "--deployed-dir" not in arguments  # no bundle on this Pi


def test_swap_queues_exactly_one_labeling_pass(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, decision="swap")
    detail = runs.run_labeler_job({"id": "job_1", "kind": "labeler"})
    queued = _queued_prelabels(tmp_path)
    assert len(queued) == 1 and queued[0]["status"] == "queued"
    assert detail.startswith("SWAPPED")


def test_swap_reuses_a_pending_labeling_pass(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, decision="swap")
    pending = q.queue_auto("prelabel", "weekly: 10 new frames to label")
    runs.run_labeler_job({"id": "job_1", "kind": "labeler"})
    assert [j["id"] for j in _queued_prelabels(tmp_path)] == [pending["id"]]


@pytest.mark.parametrize("decision", ["no swap", "inconclusive"])
def test_kept_labeler_queues_nothing(tmp_path, monkeypatch, decision):
    _wire(tmp_path, monkeypatch, decision=decision)
    detail = runs.run_labeler_job({"id": "job_1", "kind": "labeler"})
    assert _queued_prelabels(tmp_path) == []
    assert detail.startswith("kept the active labeler")


def test_failed_run_fails_the_job_and_queues_nothing(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, decision="failed")
    with pytest.raises(RuntimeError, match="25 exam errors"):
        runs.run_labeler_job({"id": "job_1", "kind": "labeler"})
    assert _queued_prelabels(tmp_path) == []


def test_refused_labeler_job_runs_nothing(tmp_path, monkeypatch):
    calls = _wire(tmp_path, monkeypatch, decision="swap")
    (tmp_path / "billing.json").write_text(json.dumps(
        {"metered_cost": 30.0, "billed_cost": 0.0, "credits_used": 30.0}))
    (tmp_path / "job_1.json").write_text(json.dumps(
        {"id": "job_1", "kind": "labeler", "status": "queued",
         "requested_at": 1.0}))
    monkeypatch.setattr(daemon, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(daemon, "sync_clock", lambda: None)
    monkeypatch.setattr(daemon, "write_billing", lambda: None)
    assert daemon.main() == 0
    assert q.jobs()[0]["status"] == "refused"
    assert calls == [] and _queued_prelabels(tmp_path) == []
