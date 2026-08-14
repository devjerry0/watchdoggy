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
