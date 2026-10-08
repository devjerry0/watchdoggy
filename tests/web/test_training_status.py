import json

from .conftest import _seed_sidecar
from .conftest import _training_client as _client


def test_status_counts_unlabeled_and_missing_prelabels(tmp_path):
    c, root = _client(tmp_path)
    _seed_sidecar(root, "sample_1", labeled=False, prelabeled=False)
    _seed_sidecar(root, "sample_2", labeled=True, prelabeled=False)
    _seed_sidecar(root, "sample_3", labeled=False, prelabeled=True)
    d = c.get("/api/training/status").json()
    assert d["unlabeled"] == 2
    assert d["missing_prelabels"] == 2
    assert d["jobs"] == [] and d["last_train"] is None


def test_request_creates_job_file_and_dedupes(tmp_path):
    c, root = _client(tmp_path)
    first = c.post("/api/training/request", json={"kind": "train"}).json()
    assert first["ok"] and not first["already_pending"]
    files = list((root / "jobs").glob("job_*.json"))
    assert len(files) == 1
    job = json.loads(files[0].read_text())
    assert job["kind"] == "train" and job["status"] == "queued"
    # Same kind queued again -> no second file, flagged as pending.
    second = c.post("/api/training/request", json={"kind": "train"}).json()
    assert second["already_pending"]
    assert len(list((root / "jobs").glob("job_*.json"))) == 1
    # A different kind queues alongside.
    other = c.post("/api/training/request", json={"kind": "prelabel"}).json()
    assert not other["already_pending"]
    assert len(list((root / "jobs").glob("job_*.json"))) == 2


def test_request_rejects_unknown_kind(tmp_path):
    c, _ = _client(tmp_path)
    assert c.post("/api/training/request",
                  json={"kind": "mine-bitcoin"}).status_code == 422


def test_status_reports_software_and_update_check(tmp_path):
    c, root = _client(tmp_path)
    # Nothing installed / never checked: the keys exist but are empty.
    d = c.get("/api/training/status").json()
    assert d["software"] == {"installed": None, "check": None}
    # The trainer's daily check leaves a stamp beside the jobs.
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    (jobs / "update-check.json").write_text(json.dumps(
        {"checked_at": 1700000000.0, "latest": "v1.1.0", "installed": "v1.0.0"}))
    d = c.get("/api/training/status").json()
    assert d["software"]["check"]["latest"] == "v1.1.0"


def test_status_reports_last_done_train_and_next_auto(tmp_path):
    c, root = _client(tmp_path)
    jobs = root / "jobs"
    jobs.mkdir()
    (jobs / "job_1.json").write_text(json.dumps(
        {"id": "job_1", "kind": "train", "status": "done",
         "requested_at": 100.0, "updated_at": 200.0, "detail": "deployed"}))
    (jobs / "job_2.json").write_text(json.dumps(
        {"id": "job_2", "kind": "prelabel", "status": "done",
         "requested_at": 300.0, "updated_at": 400.0, "detail": ""}))
    d = c.get("/api/training/status").json()
    assert d["last_train"]["id"] == "job_1"
    assert d["next_auto_train"] == 200.0 + 48 * 3600
    # done jobs don't block new requests
    again = c.post("/api/training/request", json={"kind": "train"}).json()
    assert not again["already_pending"]


def test_status_surfaces_deploy_job_pushed_out_of_history_window(tmp_path):
    # The model card reads the last DEPLOYED train job; nightly prelabel and
    # update jobs must not push it out of reach of the 10-job history slice.
    c, root = _client(tmp_path)
    jobs = root / "jobs"
    jobs.mkdir()
    (jobs / "job_01.json").write_text(json.dumps(
        {"id": "job_01", "kind": "train", "status": "done",
         "requested_at": 100.0, "updated_at": 200.0,
         "detail": "DEPLOYED new model (held-out 20/24 catches, 1 FP)"}))
    for i in range(2, 14):
        (jobs / f"job_{i:02d}.json").write_text(json.dumps(
            {"id": f"job_{i:02d}", "kind": "prelabel", "status": "done",
             "requested_at": 100.0 * i, "updated_at": 100.0 * i + 50,
             "detail": "auto-labeled 5"}))
    d = c.get("/api/training/status").json()
    assert "job_01" not in [j["id"] for j in d["jobs"]]
    assert d["last_deploy"]["id"] == "job_01"


def test_result_file_overlays_job_status(tmp_path):
    c, root = _client(tmp_path)
    jobs = root / "jobs"
    jobs.mkdir()
    (jobs / "job_9.json").write_text(json.dumps(
        {"id": "job_9", "kind": "train", "status": "queued",
         "requested_at": 100.0, "updated_at": 100.0, "detail": ""}))
    (jobs / "job_9.result.json").write_text(json.dumps(
        {"status": "done", "updated_at": 500.0, "detail": "deployed 22/23"}))
    d = c.get("/api/training/status").json()
    assert len(d["jobs"]) == 1
    assert d["jobs"][0]["status"] == "done"
    assert d["jobs"][0]["detail"] == "deployed 22/23"
    assert d["last_train"]["updated_at"] == 500.0
    # a completed job no longer dedupes new requests
    assert not c.post("/api/training/request",
                      json={"kind": "train"}).json()["already_pending"]


def test_report_endpoint_serves_markdown_and_guards(tmp_path):
    c, root = _client(tmp_path)
    jobs = root / "jobs"
    jobs.mkdir()
    (jobs / "job_7.report.md").write_text("# Training run\nall good\n")
    r = c.get("/api/training/report/job_7")
    assert r.status_code == 200 and "all good" in r.text
    assert c.get("/api/training/report/job_404").status_code == 404
    assert c.get("/api/training/report/..%2Fsecrets").status_code == 404


def test_pages_served_with_menu(tmp_path):
    c, _ = _client(tmp_path)
    for route in ("/label", "/review"):
        html = c.get(route).text
        assert "Person only" in html and "person there too" in html
    training = c.get("/training").text
    assert "Recipe" in training and "Live model" in training
    assert "Improvement" in training
    for html in (c.get("/label").text, c.get("/training").text, c.get("/").text):
        assert 'href="/label"' in html and 'href="/training"' in html


def test_log_endpoint_tails_and_guards(tmp_path):
    c, root = _client(tmp_path)
    jobs = root / "jobs"
    jobs.mkdir()
    (jobs / "job_5.log").write_text("\n".join(f"line{i}" for i in range(300)))
    r = c.get("/api/training/log/job_5")
    assert r.status_code == 200
    assert "line299" in r.text and "line50" not in r.text  # last 200 only
    assert c.get("/api/training/log/job_none").status_code == 404
    assert c.get("/api/training/log/..%2F.env").status_code == 404


def test_status_includes_billing_when_daemon_wrote_it(tmp_path):
    c, root = _client(tmp_path)
    jobs = root / "jobs"
    jobs.mkdir()
    (jobs / "billing.json").write_text(json.dumps(
        {"metered_cost": 2.72, "billed_cost": 0.0, "credits_used": 2.72,
         "fetched_at": 1000.0}))
    d = c.get("/api/training/status").json()
    assert d["billing"]["metered_cost"] == 2.72
    assert d["settings"]["monthly_credits"] == 30
    assert c.post("/api/training/settings",
                  json={"monthly_credits": 100}).json()["settings"]["monthly_credits"] == 100


def _seed_job(root, job_id, kind, status):
    jobs = root / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f"{job_id}.json").write_text(json.dumps(
        {"id": job_id, "kind": kind, "status": "queued",
         "requested_at": 1.0, "updated_at": 1.0}))
    if status != "queued":
        (jobs / f"{job_id}.result.json").write_text(json.dumps(
            {"status": status, "detail": "", "updated_at": 2.0}))


def test_failure_streak_is_per_kind_not_pooled(tmp_path):
    # 3 failed trains behind a NEWER prelabel success: the daemon is backing
    # off train, so the banner field must still report 3 (list is newest
    # first by filename; higher ids are newer).
    c, root = _client(tmp_path)
    _seed_job(root, "job_1001", "train", "failed")
    _seed_job(root, "job_1002", "train", "failed")
    _seed_job(root, "job_1003", "train", "failed")
    _seed_job(root, "job_1004", "prelabel", "done")
    d = c.get("/api/training/status").json()
    assert d["cloud_failure_streak"] == 3
    assert d["cloud_backoff_after"] == 3


def test_interleaved_sub_threshold_failures_do_not_pool(tmp_path):
    # 2 failed trains + 2 failed prelabels interleaved: neither kind reaches
    # the daemon's threshold, so the streak must read 2, not 4.
    c, root = _client(tmp_path)
    _seed_job(root, "job_1001", "train", "failed")
    _seed_job(root, "job_1002", "prelabel", "failed")
    _seed_job(root, "job_1003", "train", "failed")
    _seed_job(root, "job_1004", "prelabel", "failed")
    d = c.get("/api/training/status").json()
    assert d["cloud_failure_streak"] == 2


def test_refused_and_running_jobs_leave_the_streak_alone(tmp_path):
    c, root = _client(tmp_path)
    _seed_job(root, "job_1001", "train", "failed")
    _seed_job(root, "job_1002", "train", "failed")
    _seed_job(root, "job_1003", "train", "failed")
    _seed_job(root, "job_1004", "train", "refused")
    _seed_job(root, "job_1005", "train", "running")
    d = c.get("/api/training/status").json()
    assert d["cloud_failure_streak"] == 3
