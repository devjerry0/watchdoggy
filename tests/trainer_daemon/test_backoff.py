"""Failure backoff: a broken cloud must stop the daemon from re-queueing a
doomed auto job every 30-minute tick (the Sep 2026 six-day, ~$13 streak).

The daemon lives in scripts/ (it ships to the Pi, not the wheel), so the
tests add scripts/ to the path themselves."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import json  # noqa: E402

from trainer_daemon import env  # noqa: E402
from trainer_daemon import queue as q  # noqa: E402
from trainer_daemon.queue import (BACKOFF_AFTER, BACKOFF_BASE_SECONDS,
                                  BACKOFF_CAP_SECONDS, LONG_FAILURE_SECONDS,
                                  failure_backoff)  # noqa: E402

NOW = 1_788_700_000.0


def _job(kind, status, age_seconds, ran_seconds=60.0):
    return {"kind": kind, "status": status, "updated_at": NOW - age_seconds,
            "requested_at": NOW - age_seconds - ran_seconds}


def test_clear_below_threshold():
    existing = [_job("train", "failed", 60 * i)
                for i in range(BACKOFF_AFTER - 1)]
    assert failure_backoff(existing, "train", NOW) is None


def test_backs_off_at_threshold():
    existing = [_job("train", "failed", 60 * (i + 1))
                for i in range(BACKOFF_AFTER)]
    wait = failure_backoff(existing, "train", NOW)
    assert wait is not None
    assert 0 < wait <= BACKOFF_BASE_SECONDS


def test_success_resets_streak():
    existing = ([_job("train", "failed", 60)]
                + [_job("train", "done", 120)]
                + [_job("train", "failed", 60 * (i + 3)) for i in range(10)])
    assert failure_backoff(existing, "train", NOW) is None


def test_streak_grows_the_wait_up_to_the_cap():
    existing = [_job("train", "failed", 60 * (i + 1)) for i in range(50)]
    wait = failure_backoff(existing, "train", NOW)
    assert wait is not None
    assert wait <= BACKOFF_CAP_SECONDS


def test_expires_after_the_window():
    existing = [_job("train", "failed",
                     BACKOFF_CAP_SECONDS + 3600 * (i + 1))
                for i in range(50)]
    assert failure_backoff(existing, "train", NOW) is None


def test_kinds_are_independent():
    existing = [_job("prelabel", "failed", 60 * (i + 1)) for i in range(10)]
    assert failure_backoff(existing, "train", NOW) is None
    assert failure_backoff(existing, "prelabel", NOW) is not None


def test_queued_and_running_jobs_do_not_break_the_streak():
    existing = ([_job("train", "queued", 10), _job("train", "running", 20)]
                + [_job("train", "failed", 60 * (i + 1))
                   for i in range(BACKOFF_AFTER)])
    assert failure_backoff(existing, "train", NOW) is not None


def test_refused_results_never_count_toward_the_streak():
    # Credit-gate refusals carry status "refused": "no credits" must not
    # masquerade as "cloud broken".
    existing = [_job("train", "refused", 60 * (i + 1)) for i in range(10)]
    assert failure_backoff(existing, "train", NOW) is None


def test_single_expensive_hang_failure_arms_backoff_alone():
    # One 12h-hang failure costs ~$10; it must not take three strikes.
    existing = [_job("train", "failed", 60,
                     ran_seconds=LONG_FAILURE_SECONDS + 600)]
    assert failure_backoff(existing, "train", NOW) is not None


def test_future_stamp_never_stretches_the_window_beyond_wait():
    # Pi clock regressed after the failure was written: stamp reads future.
    existing = [_job("train", "failed", -3600 * (i + 1))
                for i in range(BACKOFF_AFTER)]
    wait = failure_backoff(existing, "train", NOW)
    assert wait is not None and wait <= BACKOFF_CAP_SECONDS


def _seed_queue_env(tmp_path, monkeypatch, failures, labels=1):
    """Point queue's module globals at a temp jobs+dataset tree with a due
    training window and `failures` consecutive failed train results."""
    jobs_dir = tmp_path / "jobs"
    ds = tmp_path / "ds"
    jobs_dir.mkdir()
    ds.mkdir()
    monkeypatch.setattr(env, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(q, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(q, "DATASET_DIR", ds)
    monkeypatch.setattr(q, "update_due", lambda now: None)
    (jobs_dir / "trainer-settings.json").write_text(
        json.dumps({"min_new_labels": 1}))
    for i in range(labels):
        (ds / f"sample_{i}.json").write_text(json.dumps(
            {"human_label": "dog", "labeled_at": NOW,
             "prelabels": []}))
    import time
    for i in range(failures):
        jid = f"job_{1000 + i}"
        (jobs_dir / f"{jid}.json").write_text(json.dumps(
            {"id": jid, "kind": "train", "status": "queued",
             "requested_at": time.time() - 400 - i}))
        (jobs_dir / f"{jid}.result.json").write_text(json.dumps(
            {"status": "failed", "detail": "boom",
             "updated_at": time.time() - 300 - i}))
    return jobs_dir


def test_synthesize_suppresses_auto_train_during_backoff(tmp_path, monkeypatch):
    # THE money path: a due auto-train with 3 fresh failures must queue
    # nothing at all (the Sep 2026 storm re-queued one every 30 minutes).
    jobs_dir = _seed_queue_env(tmp_path, monkeypatch, failures=BACKOFF_AFTER)
    before = sorted(jobs_dir.glob("job_*.json"))
    assert q.synthesize_job(q.jobs()) is None
    assert sorted(jobs_dir.glob("job_*.json")) == before


def test_synthesize_queues_auto_train_below_threshold(tmp_path, monkeypatch):
    jobs_dir = _seed_queue_env(tmp_path, monkeypatch,
                               failures=BACKOFF_AFTER - 1)
    job = q.synthesize_job(q.jobs())
    assert job is not None and job["kind"] == "train"
    assert (jobs_dir / f"{job['id']}.json").is_file()


def test_synthesize_queues_again_once_backoff_expires(tmp_path, monkeypatch):
    import time
    jobs_dir = _seed_queue_env(tmp_path, monkeypatch, failures=0)
    for i in range(BACKOFF_AFTER):
        jid = f"job_{2000 + i}"
        (jobs_dir / f"{jid}.json").write_text(json.dumps(
            {"id": jid, "kind": "train", "status": "queued",
             "requested_at": time.time() - 200_000 - i}))
        (jobs_dir / f"{jid}.result.json").write_text(json.dumps(
            {"status": "failed", "detail": "boom",
             "updated_at": time.time() - 172_800 - i}))  # 2 days ago
    job = q.synthesize_job(q.jobs())
    assert job is not None and job["kind"] == "train"
