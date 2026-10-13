import json

from .conftest import _dataset_client as _client
from .conftest import _seed_full


def _box(x=1.0):
    return {"label": "dog", "box": [x, 1.0, 5.0, 5.0]}


def test_batch_applies_all_three_kinds_in_one_request(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1")                       # unlabeled
    _seed_full(ddir, "sample_2")                       # unlabeled
    _seed_full(ddir, "sample_3", verdict="empty")      # human-labeled
    d = c.post("/api/dataset/apply-cloud-results", json={
        "model": "yolo26x",
        "prelabels": {"sample_1": [_box()], "sample_2": []},
        "auto_verdicts": {"sample_1": "dog"},
        "disputes": {"sample_3": {"model_says": "dog", "nano_conf": 0.9}},
    }).json()
    assert d == {"ok": True, "prelabels": 2, "auto_verdicts": 1,
                 "disputes": 1, "clear_auto": 0, "missing": 0}
    meta1 = json.loads((ddir / "sample_1.json").read_text())
    assert meta1["prelabels"]["boxes"] == [_box()]
    assert meta1["auto_label"]["verdict"] == "dog"
    meta3 = json.loads((ddir / "sample_3.json").read_text())
    assert meta3["disputed"]["model_says"] == "dog"


def test_batch_respects_human_wins_and_settled_disputes(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1", verdict="person")     # human label stands
    _seed_full(ddir, "sample_2", verdict="empty")
    meta2 = json.loads((ddir / "sample_2.json").read_text())
    meta2["dispute_settled_at"] = 123.0                # already arbitrated
    (ddir / "sample_2.json").write_text(json.dumps(meta2))
    d = c.post("/api/dataset/apply-cloud-results", json={
        "auto_verdicts": {"sample_1": "dog"},
        "disputes": {"sample_2": {"model_says": "dog", "nano_conf": 0.8}},
    }).json()
    assert d["auto_verdicts"] == 0 and d["disputes"] == 0
    assert "auto_label" not in json.loads((ddir / "sample_1.json").read_text())
    assert "disputed" not in json.loads((ddir / "sample_2.json").read_text())


def test_batch_validates_everything_before_writing_anything(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1")
    bad = c.post("/api/dataset/apply-cloud-results", json={
        "prelabels": {"sample_1": [_box()],
                      "sample_2": [{"label": "cat", "box": [1, 1, 2, 2]}]},
    })
    assert bad.status_code == 422
    # The valid frame was NOT partially applied.
    assert "prelabels" not in json.loads((ddir / "sample_1.json").read_text())
    assert c.post("/api/dataset/apply-cloud-results", json={
        "auto_verdicts": {"sample_1": "maybe"}}).status_code == 422


def test_batch_counts_missing_and_guards_traversal(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1")
    d = c.post("/api/dataset/apply-cloud-results", json={
        "prelabels": {"sample_1": [], "sample_404": [],
                      "../../etc/passwd": []},
    }).json()
    assert d["prelabels"] == 1 and d["missing"] == 2


def test_batch_skips_an_empty_sidecar_and_applies_the_rest(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1")
    _seed_full(ddir, "sample_2")
    (ddir / "sample_1.json").write_text("")            # power cut mid-write
    d = c.post("/api/dataset/apply-cloud-results", json={
        "auto_verdicts": {"sample_1": "dog", "sample_2": "empty"}}).json()
    assert d["auto_verdicts"] == 1 and d["missing"] == 1
    assert json.loads((ddir / "sample_2.json").read_text())["auto_label"]["verdict"] == "empty"


def test_batch_clears_auto_labels_the_jury_no_longer_vouches_for(tmp_path):
    c, _, ddir = _client(tmp_path)
    for stem in ("sample_1", "sample_2", "sample_3", "sample_4"):
        _seed_full(ddir, stem)
    c.post("/api/dataset/apply-cloud-results", json={
        "auto_verdicts": {"sample_1": "dog", "sample_2": "dog"}})
    meta2 = json.loads((ddir / "sample_2.json").read_text())
    meta2["human_label"] = "person"                    # a human label stands
    (ddir / "sample_2.json").write_text(json.dumps(meta2))
    (ddir / "sample_3.json").write_text("")            # power cut mid-write
    d = c.post("/api/dataset/apply-cloud-results", json={
        # sample_4 has no auto-label: re-applying a clear is harmless
        "clear_auto": ["sample_1", "sample_2", "sample_3", "sample_4"]}).json()
    assert d["clear_auto"] == 1 and d["missing"] == 1
    assert "auto_label" not in json.loads((ddir / "sample_1.json").read_text())
    assert json.loads((ddir / "sample_2.json").read_text()) == meta2
    unlabeled = c.get("/api/dataset/frames?filter=unlabeled").json()["frames"]
    assert {f["name"] for f in unlabeled} == {"sample_1", "sample_4"}  # index upserted
    assert c.post("/api/dataset/apply-cloud-results", json={
        "clear_auto": "sample_1"}).status_code == 422
