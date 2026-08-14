import json

import numpy as np
from fastapi.testclient import TestClient

from doggy.core.config import Settings
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.decision.gate import FireGate
from doggy.events.store import EventStore
from doggy.reaction.sound import FakeAlerter
from doggy.web import create_app


def _client(tmp_path):
    settings = Settings(event_log_dir=tmp_path / "events",
                        dataset_dir=tmp_path / "dataset")
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path / "events", 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, FireGate(runtime))
    return TestClient(app), store, tmp_path / "dataset"


def test_stats_empty(tmp_path):
    c, _, _ = _client(tmp_path)
    body = c.get("/api/dataset").json()
    assert body["samples"] == 0 and body["by_reason"] == {}


def test_mark_event_as_false_positive(tmp_path):
    c, store, dataset_dir = _client(tmp_path)
    r = store.add(np.zeros((8, 8, 3), np.uint8), 0.9, 1.0, 1_700_000_000.0, 1.0)
    assert c.post(f"/api/dataset/mark/{r.id}").json() == {"ok": True}
    sides = list(dataset_dir.glob("sample_*.json"))
    assert len(sides) == 1
    side = json.loads(sides[0].read_text())
    assert side["reasons"] == ["user_marked_fp"]
    assert side["event_id"] == r.id
    assert sides[0].with_suffix(".jpg").is_file()
    # And it shows up in stats.
    assert c.get("/api/dataset").json()["by_reason"]["user_marked_fp"] == 1


def test_mark_unknown_event_404(tmp_path):
    c, _, _ = _client(tmp_path)
    assert c.post("/api/dataset/mark/nope").status_code == 404


def test_mark_traversal_guarded(tmp_path):
    c, _, _ = _client(tmp_path)
    assert c.post("/api/dataset/mark/..%2F..%2Fetc").status_code == 404


def test_clear_removes_samples(tmp_path):
    c, store, dataset_dir = _client(tmp_path)
    r = store.add(np.zeros((8, 8, 3), np.uint8), 0.9, 1.0, 1_700_000_000.0, 1.0)
    c.post(f"/api/dataset/mark/{r.id}")
    assert c.post("/api/dataset/clear").json() == {"ok": True}
    assert list(dataset_dir.glob("sample_*")) == []


def _seed_sample(dataset_dir, stem, reasons, wall_time, label=None):
    import numpy as np
    import cv2
    import json as _json
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dataset_dir / f"{stem}.jpg"), np.zeros((8, 8, 3), np.uint8))
    meta = {"wall_time": wall_time, "reasons": reasons,
            "detections": {"targets": [{"label": "dog", "confidence": 0.8,
                                        "box": [0, 0, 4, 4]}]}}
    if label:
        meta["human_label"] = label
    (dataset_dir / f"{stem}.json").write_text(_json.dumps(meta))


def test_next_serves_highest_signal_first_and_skips_labeled(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["periodic"], 100)
    _seed_sample(ddir, "sample_2", ["fire"], 50)          # older but higher signal
    _seed_sample(ddir, "sample_3", ["fire"], 60, label="dog")  # already labeled
    d = c.get("/api/dataset/next").json()
    assert d["remaining"] == 2 and d["labeled"] == 1
    assert d["sample"]["name"] == "sample_2"
    assert d["sample"]["image"] == "/dataset/sample_2.jpg"
    assert d["sample"]["detections"]["targets"][0]["label"] == "dog"


def test_label_writes_sidecar_and_next_advances(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    assert c.post("/api/dataset/label",
                  json={"name": "sample_1", "verdict": "no_dog"}).json() == {"ok": True}
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert meta["human_label"] == "no_dog" and "labeled_at" in meta
    d = c.get("/api/dataset/next").json()
    assert d["sample"] is None and d["labeled"] == 1


def test_label_rejects_bad_verdict_and_unknown_name(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    assert c.post("/api/dataset/label",
                  json={"name": "sample_1", "verdict": "cat"}).status_code == 422
    assert c.post("/api/dataset/label",
                  json={"name": "nope", "verdict": "dog"}).status_code == 404


def test_dataset_image_served_and_guarded(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    assert c.get("/dataset/sample_1.jpg").status_code == 200
    assert c.get("/dataset/..%2Fevents%2Fx.jpg").status_code == 404
    assert c.get("/dataset/sample_1.json").status_code == 404  # jpgs only


def test_review_page_served(tmp_path):
    c, _, _ = _client(tmp_path)
    html = c.get("/review").text
    assert "Person only" in html and "Skip" in html
    # The dog button must SAY that a person in the frame doesn't change the
    # verdict -- its absence caused repeated labeling confusion.
    assert "person there too" in html


def test_mark_fp_is_prelabeled_no_dog(tmp_path):
    import numpy as np
    c, store, ddir = _client(tmp_path)
    r = store.add(np.zeros((8, 8, 3), np.uint8), 0.9, 1.0, 1_700_000_000.0, 1.0)
    c.post(f"/api/dataset/mark/{r.id}")
    side = json.loads(next(iter(ddir.glob("sample_*.json"))).read_text())
    assert side["human_label"] == "no_dog"
    # And it never shows up in the review queue.
    assert c.get("/api/dataset/next").json()["sample"] is None


def test_rich_verdicts_person_and_empty(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    _seed_sample(ddir, "sample_2", ["periodic"], 90)
    assert c.post("/api/dataset/label",
                  json={"name": "sample_1", "verdict": "person"}).json() == {"ok": True}
    assert c.post("/api/dataset/label",
                  json={"name": "sample_2", "verdict": "empty"}).json() == {"ok": True}
    assert json.loads((ddir / "sample_1.json").read_text())["human_label"] == "person"
    assert json.loads((ddir / "sample_2.json").read_text())["human_label"] == "empty"
    assert c.get("/api/dataset/next").json()["sample"] is None


def test_dog_mixed_verdict_flags_frame_for_box_surgery(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    assert c.post("/api/dataset/label",
                  json={"name": "sample_1", "verdict": "dog_mixed"}).json() == {"ok": True}
    assert json.loads((ddir / "sample_1.json").read_text())["human_label"] == "dog_mixed"


def test_labeled_listing_newest_first_with_verdicts(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    _seed_sample(ddir, "sample_2", ["fire"], 90)
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "dog"})
    c.post("/api/dataset/label", json={"name": "sample_2", "verdict": "person"})
    d = c.get("/api/dataset/labeled").json()["labeled"]
    assert [r["name"] for r in d] == ["sample_2", "sample_1"]  # newest labeled first
    assert d[0]["verdict"] == "person" and d[1]["verdict"] == "dog"
    assert d[0]["image"] == "/dataset/sample_2.jpg"


def test_clear_verdict_returns_frame_to_queue(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "dog"})
    assert c.get("/api/dataset/next").json()["sample"] is None
    assert c.post("/api/dataset/label",
                  json={"name": "sample_1", "verdict": "clear"}).json() == {"ok": True}
    meta = json.loads((ddir / "sample_1.json").read_text())
    assert "human_label" not in meta and "labeled_at" not in meta
    assert c.get("/api/dataset/next").json()["sample"]["name"] == "sample_1"


def test_relabel_overwrites(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_sample(ddir, "sample_1", ["fire"], 100)
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "dog"})
    c.post("/api/dataset/label", json={"name": "sample_1", "verdict": "person"})
    assert json.loads((ddir / "sample_1.json").read_text())["human_label"] == "person"


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


def _seed_full(ddir, stem, verdict=None, prelabel_dogs=None, hand=None,
               nano_dog_conf=None, wall=100.0):
    import json as _json
    ddir.mkdir(parents=True, exist_ok=True)
    import numpy as np, cv2
    cv2.imwrite(str(ddir / f"{stem}.jpg"), np.zeros((8, 8, 3), np.uint8))
    meta = {"wall_time": wall, "reasons": ["fire"]}
    if verdict:
        meta["human_label"] = verdict
        meta["labeled_at"] = wall + 1
    if prelabel_dogs is not None:
        meta["prelabels"] = {"model": "yolo26x", "boxes": [
            {"label": "dog", "box": [0, 0, 4, 4]} for _ in range(prelabel_dogs)]}
    if hand:
        meta["human_boxes"] = hand
    if nano_dog_conf is not None:
        meta["detections"] = {"targets": [
            {"label": "dog", "confidence": nano_dog_conf, "box": [0, 0, 4, 4]}]}
    (ddir / f"{stem}.json").write_text(_json.dumps(meta))


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


def test_sample_detail_and_guards(tmp_path):
    c, _, ddir = _client(tmp_path)
    _seed_full(ddir, "sample_1", verdict="dog", prelabel_dogs=1)
    d = c.get("/api/dataset/sample/sample_1").json()
    assert d["verdict"] == "dog"
    assert d["prelabels"]["boxes"][0]["label"] == "dog"
    assert c.get("/api/dataset/sample/sample_none").status_code == 404
    assert c.get("/api/dataset/sample/..%2Fx").status_code == 404


def test_thumbnail_generated_cached_and_guarded(tmp_path):
    import numpy as np, cv2
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
        if in_val and not val_stem: val_stem = stem
        if not in_val and not train_stem: train_stem = stem
        if val_stem and train_stem: break
    _seed_full(ddir, val_stem, verdict="dog", prelabel_dogs=1)
    _seed_full(ddir, train_stem, verdict="dog", prelabel_dogs=1)
    _seed_full(ddir, "sample_9999" if int(hashlib.sha1(b"sample_9999").hexdigest(), 16) % 4 == 0 else val_stem + "x")  # unlabeled, never exam
    d = c.get("/api/dataset/frames?filter=exam").json()
    assert d["counts"]["exam"] == 1
    assert d["frames"][0]["name"] == val_stem


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
