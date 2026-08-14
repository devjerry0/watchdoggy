import json

from .conftest import _dataset_client as _client
from .conftest import _seed_full, _seed_sample


def test_label_with_hand_boxes_persists_and_lists(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    r = c.post("/api/dataset/label", json={
        "name": "sample_1", "verdict": "dog",
        "boxes": [{"label": "dog", "box": [10, 20, 110, 120]},
                  {"label": "person", "box": [200.4, 5, 260, 400]}]})
    assert r.json() == {"ok": True}
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert meta["human_label"] == "dog"
    assert meta["human_boxes"] == [{"label": "dog", "box": [10, 20, 110, 120]},
                                   {"label": "person", "box": [200.4, 5, 260, 400]}]
    row = c.get("/api/dataset/labeled").json()["labeled"][0]
    assert row["human_boxes"][0]["label"] == "dog"


def test_label_rejects_malformed_boxes(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    bad = [
        [{"label": "cat", "box": [0, 0, 5, 5]}],       # unknown class
        [{"label": "dog", "box": [5, 5, 5, 9]}],       # zero width
        [{"label": "dog", "box": [0, 0, 5]}],          # wrong arity
        [{"label": "dog", "box": ["a", 0, 5, 5]}],     # non-numeric
        "boxes-as-string",                              # not a list
    ]
    for boxes in bad:
        r = c.post("/api/dataset/label",
                   json={"name": "sample_1", "verdict": "dog", "boxes": boxes})
        assert r.status_code == 422, boxes
    assert "human_boxes" not in json.loads((ddir / "sample_1.json").read_text())


def test_undo_keeps_hand_boxes(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    c.post("/api/dataset/label", json={
        "name": "sample_1", "verdict": "dog",
        "boxes": [{"label": "dog", "box": [1, 2, 3, 4]}]})
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "clear"})
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert "human_label" not in meta          # back in the queue...
    assert meta["human_boxes"]                # ...but the drawing work survives
    nxt = c.get("/api/dataset/next").json()["sample"]
    assert nxt["name"] == "sample_1" and nxt["human_boxes"]


def test_prelabels_merge_into_sidecar_and_payloads(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    r = c.post("/api/dataset/prelabels", json={
        "name": "sample_1", "model": "yolo26x",
        "boxes": [{"label": "dog", "box": [5, 5, 50, 60]}]})
    assert r.json() == {"ok": True}
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert meta["prelabels"]["model"] == "yolo26x"
    assert meta["prelabels"]["boxes"][0]["label"] == "dog"
    assert meta["reasons"] == ["fire"]  # merge, not replace
    nxt = c.get("/api/dataset/next").json()["sample"]
    assert nxt["prelabels"]["boxes"][0]["box"] == [5, 5, 50, 60]
    # unknown sample and malformed boxes are rejected
    assert c.post("/api/dataset/prelabels", json={
        "name": "sample_9", "boxes": []}).status_code == 404
    assert c.post("/api/dataset/prelabels", json={
        "name": "sample_1", "boxes": [{"label": "cat", "box": [0, 0, 1, 1]}]
    }).status_code == 422


def test_sample_detail_and_guards(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1", verdict="dog", prelabel_dogs=1)
    d = c.get("/api/dataset/sample/sample_1").json()
    assert d["verdict"] == "dog"
    assert d["prelabels"]["boxes"][0]["label"] == "dog"
    assert c.get("/api/dataset/sample/sample_none").status_code == 404
    assert c.get("/api/dataset/sample/..%2Fx").status_code == 404


def test_thumbnail_generated_cached_and_guarded(tmp_path):
    import numpy as np
    import cv2
    c, _, ddir = _client(tmp_path)
    ddir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ddir / "sample_1.jpg"), np.zeros((480, 640, 3), np.uint8))
    r = c.get("/dataset/thumb/sample_1.jpg")
    assert r.status_code == 200
    thumb = ddir / "thumbs" / "sample_1.jpg"
    assert thumb.is_file()
    small = cv2.imread(str(thumb))
    assert small.shape[1] == 160  # downscaled, not the full frame
    assert c.get("/dataset/thumb/sample_1.jpg").status_code == 200  # cache hit
    assert c.get("/dataset/thumb/..%2F.env").status_code == 404
    assert c.get("/dataset/thumb/notasample.jpg").status_code == 404
    # pruned source -> stale thumb removed on next request
    (ddir / "sample_1.jpg").unlink()
    assert c.get("/dataset/thumb/sample_1.jpg").status_code == 404
    assert not thumb.is_file()


def test_conflicting_verdict_clears_hand_boxes(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1", prelabel_dogs=1)
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "dog",
           "boxes": [{"label": "dog", "box": [1, 1, 5, 5]}]})
    # Compatible re-tap keeps the boxes.
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "dog"})
    assert json.loads((ddir / "sample_1.json").read_text())["human_boxes"]
    # A contradicting tap (person on a dog-boxed frame) clears them.
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "person"})
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert meta["human_label"] == "person" and "human_boxes" not in meta
