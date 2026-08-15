import json

from .conftest import _dataset_client as _client
from .conftest import _seed_full


def test_frames_filters_and_counts(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1")                                  # unlabeled, no prelabels
    _seed_full(ddir, "sample_2", verdict="dog", prelabel_dogs=1)  # fine dog
    _seed_full(ddir, "sample_3", verdict="dog", prelabel_dogs=0)  # NEEDS BOXES
    _seed_full(ddir, "sample_4", verdict="person", prelabel_dogs=0)
    _seed_full(ddir, "sample_5", verdict="dog", prelabel_dogs=0,
               hand=[{"label": "dog", "box": [1, 1, 3, 3]}])      # saved by hand boxes
    _seed_full(ddir, "sample_6", verdict="dog", nano_dog_conf=0.9)  # nano fallback saves it
    d = c.get("/api/dataset/frames?filter=unlabeled").json()
    assert [f["name"] for f in d["frames"]] == ["sample_1"]
    assert d["counts"]["unlabeled"] == 1
    assert d["counts"]["dog"] == 4
    assert d["counts"]["needs_boxes"] == 1
    assert d["counts"]["no_prelabels"] == 2  # samples 1 and 6 lack the key
    needs = c.get("/api/dataset/frames?filter=needs_boxes").json()["frames"]
    assert [f["name"] for f in needs] == ["sample_3"]
    dogs = c.get("/api/dataset/frames?filter=dog").json()["frames"]
    assert {f["name"] for f in dogs} == {"sample_2", "sample_3", "sample_5", "sample_6"}
    assert [f for f in dogs if f["name"] == "sample_5"][0]["hand_boxes"] is True
    assert c.get("/api/dataset/frames?filter=bogus").status_code == 422


def test_autolabel_lifecycle(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1")                       # unlabeled
    _seed_full(ddir, "sample_2", verdict="dog", prelabel_dogs=1)
    r = c.post("/api/dataset/autolabel",
               json={"name": "sample_1", "verdict": "dog"})
    assert r.json() == {"ok": True}
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert meta["auto_label"]["verdict"] == "dog"
    assert "human_label" not in meta
    # off the human queue, visible under the auto chip, counted as dog
    d = c.get("/api/dataset/frames?filter=unlabeled").json()
    assert d["counts"]["unlabeled"] == 0 and d["counts"]["auto"] == 1
    auto = c.get("/api/dataset/frames?filter=auto").json()["frames"]
    assert auto[0]["name"] == "sample_1" and auto[0]["auto"] is True
    assert auto[0]["verdict"] == "dog"
    # human verdict overrules and clears the auto label
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "person"})
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert meta["human_label"] == "person" and "auto_label" not in meta
    # autolabel never touches a human-labeled frame
    r = c.post("/api/dataset/autolabel",
               json={"name": "sample_2", "verdict": "empty"})
    assert r.json()["skipped"] == "human label wins"
    # validation
    assert c.post("/api/dataset/autolabel",
                  json={"name": "sample_1", "verdict": "dog_mixed"}).status_code == 422
    assert c.post("/api/dataset/autolabel",
                  json={"name": "sample_404", "verdict": "dog"}).status_code == 404


def test_dispute_lifecycle(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1", verdict="person", prelabel_dogs=1)
    r = c.post("/api/dataset/dispute",
               json={"name": "sample_1", "model_says": "dog", "nano_conf": 0.91})
    assert r.json() == {"ok": True}
    d = c.get("/api/dataset/frames?filter=disputed").json()
    assert d["counts"]["disputed"] == 1
    assert d["frames"][0]["disputed"] is True
    detail = c.get("/api/dataset/sample/sample_1").json()
    assert detail["disputed"]["model_says"] == "dog"
    # re-judging settles the dispute
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "dog"})
    assert c.get("/api/dataset/frames?filter=disputed").json()["counts"]["disputed"] == 0
    assert c.post("/api/dataset/dispute",
                  json={"name": "sample_404", "model_says": "dog"}).status_code == 404
    assert c.post("/api/dataset/dispute",
                  json={"name": "sample_1"}).status_code == 422


def test_settled_disputes_are_never_reopened(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1", verdict="person", prelabel_dogs=1)
    c.post("/api/dataset/dispute",
           json={"name": "sample_1", "model_says": "dog", "nano_conf": 0.9})
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "person"})
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert "disputed" not in meta and meta["dispute_settled_at"] > 0
    # the machine may not ask the same question twice
    r = c.post("/api/dataset/dispute",
               json={"name": "sample_1", "model_says": "dog", "nano_conf": 0.95})
    assert r.json()["skipped"] == "human already arbitrated"
    assert "disputed" not in json.loads((ddir / "sample_1.json").read_text())


def test_exam_filter_is_stable_hash_val_split(tmp_path):
    import hashlib
    c, _, ddir = _client(tmp_path)
    val_stem = train_stem = None
    for i in range(50):
        stem = f"sample_{1000 + i}"
        in_val = int(hashlib.sha1(stem.encode()).hexdigest(), 16) % 4 == 0
        if in_val and not val_stem:
            val_stem = stem
        if not in_val and not train_stem:
            train_stem = stem
        if val_stem and train_stem:
            break
    _seed_full(ddir, val_stem, verdict="dog", prelabel_dogs=1)
    _seed_full(ddir, train_stem, verdict="dog", prelabel_dogs=1)
    _seed_full(ddir, "sample_9999" if int(hashlib.sha1(b"sample_9999").hexdigest(), 16) % 4 == 0 else val_stem + "x")  # unlabeled, never exam
    d = c.get("/api/dataset/frames?filter=exam").json()
    assert d["counts"]["exam"] == 1
    assert d["frames"][0]["name"] == val_stem


def test_user_marked_fp_frames_are_always_exam(tmp_path):
    # Escaped failures are permanent exam members regardless of the hash
    # split (mirrors kitchen_training.build._split).
    import hashlib as h
    c, _, ddir = _client(tmp_path)
    # Find a stem the hash split would put in TRAIN.
    stem = next(f"sample_{2000 + i}" for i in range(50)
                if int(h.sha1(f"sample_{2000 + i}".encode()).hexdigest(), 16) % 4)
    _seed_full(ddir, stem, verdict="no_dog", reasons=["user_marked_fp"])
    d = c.get("/api/dataset/frames?filter=exam").json()
    assert d["counts"]["exam"] == 1
