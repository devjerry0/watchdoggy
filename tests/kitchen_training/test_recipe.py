"""The cost recipe (2026-10-11): augmentation modes, warm start from the
deployed model's weights, and champion promotion on deploy."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import importlib  # noqa: E402

# The package re-exports a build() function that shadows the module name.
build = importlib.import_module("kitchen_training.build")
pipeline = importlib.import_module("kitchen_training.pipeline")
training = importlib.import_module("kitchen_training.training")


def test_auto_keeps_all_three_copies_while_the_dataset_is_small():
    small, big = build.AUTO_ALL_VARIANTS_BELOW - 1, build.AUTO_ALL_VARIANTS_BELOW
    assert build.augment_mode("auto", small) == "all"
    assert build.augment_mode("auto", big) == "off"
    # Legacy settings files: True meant "augment" -> auto, False -> off.
    assert build.augment_mode(True, big) == "off"
    assert build.augment_mode(False, small) == "off"
    # Explicit choices are honored at any size.
    assert build.augment_mode("all", big) == "all"
    assert build.augment_mode("one", big) == "one"
    assert build.augment_mode("off", small) == "off"


def test_one_copy_is_stable_and_spread_over_the_variants():
    stems = [f"sample_{i}" for i in range(3000)]
    picks = [build.variant_tags(s, "one") for s in stems]
    assert all(len(p) == 1 for p in picks)
    assert picks == [build.variant_tags(s, "one") for s in stems]  # rebuilds match
    spread = Counter(p[0] for p in picks)
    assert set(spread) == set(build.TRAIN_VARIANTS)
    assert min(spread.values()) > 800  # ~1000 each
    assert build.variant_tags("sample_1", "all") == tuple(build.TRAIN_VARIANTS)
    assert build.variant_tags("sample_1", "off") == ()


def test_training_starts_from_the_champion_when_it_exists(tmp_path, monkeypatch):
    champion, base = tmp_path / "champion.pt", tmp_path / "yolo26n.pt"
    monkeypatch.setattr(training, "CHAMPION_MODEL", champion)
    monkeypatch.setattr(training, "BASE_MODEL", base)
    assert training.start_weights() == base  # no champion yet: fresh run
    champion.write_bytes(b"weights")
    assert training.start_weights() == champion


def test_promote_champion_replaces_it_atomically(tmp_path, monkeypatch):
    champion = tmp_path / "models" / "champion.pt"
    monkeypatch.setattr(pipeline, "CHAMPION_MODEL", champion)
    best = tmp_path / "best.pt"
    best.write_bytes(b"new weights")
    pipeline.promote_champion(best)
    assert champion.read_bytes() == b"new weights"
    assert not champion.with_suffix(".pt.tmp").exists()


class _RecordingModel:
    """Captures predict() kwargs; returns one empty result."""

    def __init__(self):
        self.kwargs = None

    def predict(self, source, **kwargs):
        self.kwargs = kwargs

        class _Boxes(list):
            pass

        class _Result:
            boxes = _Boxes()
            names = {0: "person", 1: "dog"}
        return [_Result()]


def test_bundles_are_scored_at_their_own_export_shape():
    # Forcing imgsz=640 on a 640x480 bundle squared every frame and dropped
    # the exam from 135 to 109 caught; the Pi never passes imgsz.
    evaluation = importlib.import_module("kitchen_training.evaluation")
    consensus = importlib.import_module("kitchen_training.consensus")
    model = _RecordingModel()
    evaluation._max_dog_conf(model, "frame.jpg", "cpu")
    assert "imgsz" not in model.kwargs
    model = _RecordingModel()
    consensus._nano_scores(model, "frame.jpg")
    assert "imgsz" not in model.kwargs


def test_export_matches_the_camera_frame():
    config = importlib.import_module("kitchen_training.config")
    assert config.EXPORT_SHAPE == (480, 640)  # (h, w) of the C922's 640x480


def test_frames_are_read_ahead_in_order(monkeypatch):
    # Volume reads are slow one at a time; read_frames fetches ahead in
    # parallel but must hand frames back in the caller's order (the build
    # relies on it: humans first, near-duplicate pruning by order).
    import threading
    import time as _time
    dataset = importlib.import_module("kitchen_training.dataset")
    monkeypatch.setattr(dataset, "READ_CHUNK", 8)
    in_flight, peak, lock = [0], [0], threading.Lock()

    def fake_read(stem):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        _time.sleep(0.01)
        with lock:
            in_flight[0] -= 1
        return stem.encode(), f"img-{stem}"

    monkeypatch.setattr(dataset, "_read_frame", fake_read)
    stems = [f"s{i}" for i in range(50)]
    out = list(dataset.read_frames(stems))
    assert [s for s, _, _ in out] == stems
    assert [img for _, _, img in out] == [f"img-{s}" for s in stems]
    assert peak[0] > 1                      # really parallel
    assert list(dataset.read_frames([])) == []


def test_mirror_files_live_in_day_folders_on_the_volume(monkeypatch, tmp_path):
    # One flat folder of ~192k files made the v1 volume ~100x slower, and v2
    # caps a folder at 262,144 files: the cloud mirror shards by the first 5
    # timestamp digits (~1.16 days); a local pull stays flat.
    from mirror_sync import mirror_key, shard_of
    config = importlib.import_module("kitchen_training.config")
    assert shard_of("sample_1791739648418.jpg") == "17917"
    assert shard_of(".labeled-index.json") == ""
    assert mirror_key("sample_1791739648418.json") == "pipeline/mirror/17917/sample_1791739648418.json"
    assert mirror_key(".labeled-index.json") == "pipeline/mirror/.labeled-index.json"
    monkeypatch.setattr(config, "DATASET_MIRROR", tmp_path)
    monkeypatch.setattr(config, "MIRROR_SHARDED", True)
    assert config.mirror_path("sample_1791739648418.jpg") == tmp_path / "17917" / "sample_1791739648418.jpg"
    (tmp_path / "17917").mkdir()
    (tmp_path / "17917" / "sample_1791739648418.jpg").write_bytes(b"x")
    assert [p.name for p in config.mirror_glob("sample_*.jpg")] == ["sample_1791739648418.jpg"]
    monkeypatch.setattr(config, "MIRROR_SHARDED", False)
    assert config.mirror_path("sample_1791739648418.jpg") == tmp_path / "sample_1791739648418.jpg"


def test_empty_or_corrupt_frames_read_as_none(monkeypatch, tmp_path):
    # A 0-byte jpg from a power cut crashed labeling at frame 46,000
    # (imdecode asserts on an empty buffer); it must come back as None.
    dataset = importlib.import_module("kitchen_training.dataset")
    config = importlib.import_module("kitchen_training.config")
    monkeypatch.setattr(config, "DATASET_MIRROR", tmp_path)
    monkeypatch.setattr(config, "MIRROR_SHARDED", False)
    (tmp_path / "sample_1.jpg").write_bytes(b"")
    (tmp_path / "sample_2.jpg").write_bytes(b"not a jpeg")
    assert dataset._read_frame("sample_1") == (b"", None)
    assert dataset._read_frame("sample_2")[1] is None
    assert dataset._read_frame("sample_missing") == (b"", None)


def test_labeling_runs_in_batches_and_skips_unreadable(monkeypatch, tmp_path, capsys):
    dataset = importlib.import_module("kitchen_training.dataset")
    calls = []

    class FakeModel:
        def predict(self, images, **kwargs):
            calls.append(len(images))
            return [object() for _ in images]

    frames = [(f"s{i}", b"x", None if i == 5 else f"img{i}") for i in range(40)]
    monkeypatch.setattr(dataset, "read_frames", lambda stems: iter(frames))
    monkeypatch.setattr(dataset, "_boxes", lambda p: {"dogs": [], "people": [], "shape": [480, 640]})
    monkeypatch.setattr(dataset, "_load_cache", lambda: {})
    monkeypatch.setattr(dataset, "_save_cache", lambda cache: None)
    import types, sys
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda path: FakeModel()))
    cache = dataset.prelabel([s for s, _, _ in frames])
    assert calls == [16, 16, 7]                       # 39 readable frames in batches
    assert cache["s5"]["unreadable"] is True and len(cache) == 40
    assert "prelabels ready (40 frames, 40 newly computed, 40 frames in" in capsys.readouterr().out


def test_boxes_the_pi_lacks_are_resent_even_when_cached(monkeypatch, tmp_path):
    import json
    cache = {"sample_new": {"dogs": [[1, 1, 5, 5]], "people": []},
             "sample_lost": {"dogs": [], "people": [[2, 2, 6, 6]]},
             "sample_had": {"dogs": [], "people": []},
             "sample_old_index": {"dogs": [], "people": []}}
    (tmp_path / "cache.json").write_text(json.dumps(cache))
    (tmp_path / ".labeled-index.json").write_text(json.dumps({
        "sample_lost": [None, None, 0, 0, 0],      # cached, Pi never got boxes
        "sample_had": [None, None, 0, 0, 1],
        "sample_old_index": [None, None, 0, 0]}))  # pre-field index: leave it
    monkeypatch.setattr(pipeline, "PRELABEL_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(pipeline, "DATASET_MIRROR", tmp_path)
    sent = pipeline.fresh_prelabels({"sample_lost", "sample_had", "sample_old_index"})
    assert sorted(sent) == ["sample_lost", "sample_new"]
    assert sent["sample_lost"] == [{"label": "person", "box": [2, 2, 6, 6]}]


def test_boxes_from_another_labeler_are_resent(monkeypatch, tmp_path):
    # A rollback, or a relabel whose results never reached the Pi, leaves
    # sidecars holding another labeler's boxes: the cached boxes go again.
    import json
    stems = ("sample_stock", "sample_kitchen", "sample_old_index")
    (tmp_path / "cache.json").write_text(json.dumps(
        {stem: {"dogs": [], "people": []} for stem in stems}))
    (tmp_path / ".labeled-index.json").write_text(json.dumps({
        "sample_stock": [None, None, 0, 0, 1, "yolo26x"],
        "sample_kitchen": [None, None, 0, 0, 1, "yolo26x-kitchen-abcd1234"],
        "sample_old_index": [None, None, 0, 0, 1]}))  # pre-tag index: leave it
    monkeypatch.setattr(pipeline, "PRELABEL_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(pipeline, "DATASET_MIRROR", tmp_path)
    monkeypatch.setattr(pipeline, "LABELER_TAG", "yolo26x")
    assert sorted(pipeline.fresh_prelabels(set(stems))) == ["sample_kitchen"]
    monkeypatch.setattr(pipeline, "LABELER_TAG", "yolo26x-kitchen-abcd1234")
    assert sorted(pipeline.fresh_prelabels(set(stems))) == ["sample_stock"]


def test_labeling_results_name_the_labeler(monkeypatch, tmp_path):
    (tmp_path / "cache.json").write_text("{}")
    monkeypatch.setattr(pipeline, "PRELABEL_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(pipeline, "DATASET_MIRROR", tmp_path)
    monkeypatch.setattr(pipeline, "LABELER_TAG", "yolo26x-kitchen-abcd1234")
    monkeypatch.setattr(pipeline, "_mirror_stems", lambda: [])
    monkeypatch.setattr(pipeline, "prelabel", lambda stems: {})
    assert pipeline.prelabel_phase("run")["labeler"] == "yolo26x-kitchen-abcd1234"


def _legacy_jury_id(consensus, deployed):
    """The jury id exactly as computed before labeler swaps existed."""
    import hashlib
    digest = hashlib.sha256(repr((
        consensus.AUTO_DOG_NANO_CONF, "solo dog -> dispute",
        consensus.SOLO_DOG_DISPUTE_CONF, consensus.AUTO_CLEAR_NANO_CONF,
        consensus.AUTO_PERSON_NANO_CONF,
        consensus.AUDIT_DOG_NANO_CONF)).encode())
    for weights in sorted(deployed.glob("**/*.bin")):
        digest.update(weights.read_bytes())
    return digest.hexdigest()[:16]


def _labeler_fixture(tmp_path):
    stock = tmp_path / "models" / "yolo26x.pt"
    stock.parent.mkdir()
    stock.write_bytes(b"stock weights")
    deployed = tmp_path / "deployed_ncnn_model"
    deployed.mkdir()
    (deployed / "model.ncnn.bin").write_bytes(b"nano weights")
    return stock, deployed


def test_stock_labeler_keeps_the_legacy_cache_and_jury_id(monkeypatch, tmp_path):
    # Deploying labeler swaps changes nothing while only stock yolo26x
    # exists: the same box cache (no relabel) and the same jury id (the
    # jury keeps its memory).
    config = importlib.import_module("kitchen_training.config")
    consensus = importlib.import_module("kitchen_training.consensus")
    stock, deployed = _labeler_fixture(tmp_path)
    assert config.labeler_identity(stock, tmp_path) == (
        stock, tmp_path / "prelabel-cache.json", "yolo26x", None)
    monkeypatch.setattr(consensus, "LABELER_DIGEST", None)
    assert consensus._jury_id(deployed) == _legacy_jury_id(consensus, deployed)


def test_a_labeler_swap_moves_the_cache_and_jury_and_rollback_restores_them(
        monkeypatch, tmp_path):
    import hashlib
    config = importlib.import_module("kitchen_training.config")
    consensus = importlib.import_module("kitchen_training.consensus")
    stock, deployed = _labeler_fixture(tmp_path)
    stock_identity = config.labeler_identity(stock, tmp_path)
    legacy = _legacy_jury_id(consensus, deployed)
    labeler = stock.with_name("labeler.pt")
    labeler.write_bytes(b"kitchen weights")
    sha = hashlib.sha256(b"kitchen weights").hexdigest()
    weights, cache, tag, digest = config.labeler_identity(stock, tmp_path)
    assert (weights, cache, tag, digest) == (
        labeler, tmp_path / f"prelabel-cache-{sha[:12]}.json",
        f"yolo26x-kitchen-{sha[:8]}", sha)
    monkeypatch.setattr(consensus, "LABELER_DIGEST", digest)
    assert consensus._jury_id(deployed) != legacy
    labeler.write_bytes(b"retrained kitchen weights")  # a second swap
    assert config.labeler_identity(stock, tmp_path)[3] == hashlib.sha256(
        b"retrained kitchen weights").hexdigest()
    labeler.unlink()  # rollback
    assert config.labeler_identity(stock, tmp_path) == stock_identity
    monkeypatch.setattr(consensus, "LABELER_DIGEST", None)
    assert consensus._jury_id(deployed) == legacy


def test_index_records_which_sidecars_have_boxes(tmp_path):
    import json
    from mirror_sync import _write_index

    class FakeVolume:
        def batch_upload(self, force=False):
            class Up:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def put_file(self, *a): pass
            return Up()

    (tmp_path / "sample_1.json").write_text(json.dumps(
        {"prelabels": {"model": "yolo26x", "boxes": []}}))
    (tmp_path / "sample_2.json").write_text(json.dumps({"human_label": "dog"}))
    (tmp_path / "sample_3.json").write_text("")    # power cut mid-write
    _write_index(FakeVolume(), tmp_path)
    index = json.loads((tmp_path / ".labeled-index.json").read_text())
    assert index == {"sample_1": [None, None, 0, 0, 1, "yolo26x"],
                     "sample_2": ["dog", None, 0, 0, 0, None]}


def test_uploads_are_sized_by_bytes():
    from mirror_sync import UPLOAD_CHUNK_MAX_FILES, upload_chunks
    jpgs = {f"sample_{i}.jpg": 100_000 for i in range(200)}      # ~19 MB
    assert [len(c) for c in upload_chunks(list(jpgs), jpgs)] == [83, 83, 34]
    sidecars = {f"sample_{i}.json": 2_000 for i in range(5000)}
    assert [len(c) for c in upload_chunks(list(sidecars), sidecars)] == \
        [UPLOAD_CHUNK_MAX_FILES, UPLOAD_CHUNK_MAX_FILES, 1000]
    huge = {"sample_big.jpg": 20 << 20, "sample_a.json": 10}
    assert list(upload_chunks(list(huge), huge)) == [["sample_big.jpg"], ["sample_a.json"]]


def test_a_dog_only_the_small_model_sees_is_disputed_not_auto_labeled(monkeypatch, tmp_path):
    import sys, types
    consensus = importlib.import_module("kitchen_training.consensus")
    no_x = {"dogs": [], "people": []}
    assert consensus.consensus_verdict(no_x, 0.95, 0.0) is None
    assert consensus.consensus_verdict({"dogs": [[1, 1, 5, 5]], "people": []}, 0.5, 0.0) == "dog"
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda *a, **k: None))
    monkeypatch.setattr(consensus, "_entries", lambda stems: {s: [None, None, 0, 0] for s in stems})
    monkeypatch.setattr(consensus, "read_frames", lambda stems: iter((s, b"", "img") for s in stems))
    scores = {"sure": (0.95, 0.0), "unsure": (0.5, 0.0)}
    monkeypatch.setattr(consensus, "_nano_scores", lambda nano, image, it=iter(scores.values()): next(it))
    out = consensus.judge_stems(list(scores), {"sure": no_x, "unsure": no_x}, tmp_path)
    assert out["auto_verdicts"] == {}
    assert out["disputes"] == {"sure": {"model_says": "dog", "nano_conf": 0.95}}
    assert sorted(out["declined"]) == ["sure", "unsure"]
    # The disputed frame is also declined: progress counts it once.
    assert out["judged"] == 2 and out["clear_auto"] == []


def _jury_volume(monkeypatch, tmp_path, index, record=..., active=None):
    """A volume with a shipped index, a deployed nano, and (unless left
    out) a labeler-of-record. Returns (consensus, deployed dir, cache)."""
    import json
    consensus = importlib.import_module("kitchen_training.consensus")
    _, deployed = _labeler_fixture(tmp_path)
    (tmp_path / ".labeled-index.json").write_text(json.dumps(index))
    monkeypatch.setattr(consensus, "DATASET_MIRROR", tmp_path)
    monkeypatch.setattr(consensus, "JURY_MEMORY", tmp_path / "jury-no-action.json")
    monkeypatch.setattr(consensus, "JURY_LABELER", tmp_path / "jury-labeler.json")
    monkeypatch.setattr(consensus, "PENDING_REJUDGE", tmp_path / "pending.json")
    monkeypatch.setattr(consensus, "LABELER_DIGEST", active)
    if record is not ...:
        consensus.JURY_LABELER.write_text(json.dumps({"labeler": record}))
    return consensus, deployed, {stem: {"dogs": [], "people": []} for stem in index}


_JURY_INDEX = {"sample_auto": [None, "dog", 0, 0, 1, "yolo26x"],
               "sample_new": [None, None, 0, 0, 0, None],
               "sample_hand": ["empty", None, 0, 0, 1, "yolo26x"]}


def test_first_pass_after_deploy_rejudges_nothing_and_records_stock(monkeypatch, tmp_path):
    import json
    consensus, deployed, cache = _jury_volume(monkeypatch, tmp_path, _JURY_INDEX)
    plan = consensus.jury_plan(list(_JURY_INDEX), cache, deployed)
    assert plan["todo"] == ["sample_new"]
    consensus.finish_jury(plan, [])
    assert json.loads(consensus.JURY_LABELER.read_text()) == {"labeler": None}


def test_auto_labels_are_rejudged_only_when_the_labeler_changed(monkeypatch, tmp_path):
    consensus, deployed, cache = _jury_volume(
        monkeypatch, tmp_path, _JURY_INDEX, record="aaa", active="aaa")
    assert consensus.jury_plan(list(_JURY_INDEX), cache, deployed)["todo"] == ["sample_new"]
    monkeypatch.setattr(consensus, "LABELER_DIGEST", "bbb")     # a swap
    plan = consensus.jury_plan(list(_JURY_INDEX), cache, deployed)
    assert plan["todo"] == ["sample_auto", "sample_new"]        # hand label: audit path only
    assert plan["rejudge"] == 1
    monkeypatch.setattr(consensus, "LABELER_DIGEST", None)      # rollback to stock
    assert consensus.jury_plan(list(_JURY_INDEX), cache, deployed)["todo"] == [
        "sample_auto", "sample_new"]


def test_a_new_labeler_clears_auto_labels_the_jury_no_longer_vouches_for(
        monkeypatch, tmp_path, capsys):
    import json
    import sys
    import types
    index = {"dog_lost": [None, "dog", 0, 0, 1, "yolo26x"],       # X now sees no dog
             "empty_now_dog": [None, "empty", 0, 0, 1, "yolo26x"],
             "dog_kept": [None, "dog", 0, 0, 1, "yolo26x"],
             "solo_dog": [None, "dog", 0, 0, 1, "yolo26x"],      # nano alone: dispute
             "fresh": [None, None, 0, 0, 1, "yolo26x"]}
    consensus, deployed, cache = _jury_volume(
        monkeypatch, tmp_path, index, record=None, active="bbb")
    x_dog = {"dogs": [[1, 1, 5, 5]], "people": []}
    cache.update(empty_now_dog=x_dog, dog_kept=x_dog)
    scores = {"dog_lost": (0.5, 0.0), "empty_now_dog": (0.6, 0.0),
              "dog_kept": (0.9, 0.0), "solo_dog": (0.95, 0.0), "fresh": (0.5, 0.0)}
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda *a, **k: None))
    monkeypatch.setattr(consensus, "read_frames", lambda stems: iter((s, b"", s) for s in stems))
    monkeypatch.setattr(consensus, "_nano_scores", lambda nano, image: scores[image])
    plan = consensus.jury_plan(list(index), cache, deployed)
    assert plan["rejudge"] == 4
    part = consensus.judge_stems(plan["todo"], cache, deployed)
    assert part["auto_verdicts"] == {"empty_now_dog": "dog"}
    assert sorted(part["clear_auto"]) == ["dog_lost", "solo_dog"]
    assert part["disputes"] == {"solo_dog": {"model_says": "dog", "nano_conf": 0.95}}
    assert part["judged"] == 5
    verdicts, disputes, clears = consensus.finish_jury(plan, [part])
    assert verdicts == {"empty_now_dog": "dog"} and clears == ["dog_lost", "solo_dog"]
    assert "re-judged 4 auto-labels: 1 kept, 1 changed, 2 cleared" in capsys.readouterr().out
    assert json.loads(consensus.JURY_LABELER.read_text()) == {"labeler": "bbb"}
    assert consensus.jury_plan(list(index), cache, deployed)["rejudge"] == 0


def test_rejudge_results_are_resent_until_the_index_shows_them(monkeypatch, tmp_path):
    import json
    index = {"lost": [None, "dog", 0, 0, 1, "yolo26x"],      # clear never arrived
             "done": [None, None, 0, 0, 1, "yolo26x"],       # the Pi cleared it
             "human": ["dog", "dog", 0, 0, 1, "yolo26x"],    # a human label stands
             "relabeled": [None, "dog", 0, 0, 1, "yolo26x"],
             "vouched": [None, "dog", 0, 0, 1, "yolo26x"],   # a rollback re-judged it
             "lost_change": [None, "empty", 0, 0, 1, "yolo26x"],  # change never arrived
             "applied_change": [None, "dog", 0, 0, 1, "yolo26x"]}
    consensus, _, _ = _jury_volume(monkeypatch, tmp_path, index)
    consensus.PENDING_REJUDGE.write_text(json.dumps(
        {"done": None, "gone": None, "human": None, "lost": None,
         "relabeled": None, "vouched": None, "lost_change": "dog",
         "applied_change": "dog"}))
    part = {"auto_verdicts": {"relabeled": "empty", "fresh": "person"},
            "disputes": {}, "declined": ["new_clear"], "clear_auto": ["new_clear"],
            "changed": ["relabeled"], "judged": 4}
    verdicts, _, clears = consensus.finish_jury(
        {"jury": "j", "no_action": [],
         "todo": ["new_clear", "relabeled", "vouched", "fresh"]}, [part])
    assert clears == ["lost", "new_clear"]
    assert verdicts == {"relabeled": "empty", "fresh": "person", "lost_change": "dog"}
    assert json.loads(consensus.PENDING_REJUDGE.read_text()) == {
        "lost": None, "lost_change": "dog", "new_clear": None, "relabeled": "empty"}


def test_a_pass_that_fails_after_judging_keeps_the_labeler_of_record(monkeypatch, tmp_path):
    import json
    import pytest
    consensus, deployed, cache = _jury_volume(
        monkeypatch, tmp_path, _JURY_INDEX, record="aaa", active="bbb")
    plan = consensus.jury_plan(list(_JURY_INDEX), cache, deployed)
    monkeypatch.setattr(consensus, "JURY_MEMORY", tmp_path / "missing" / "memory.json")
    with pytest.raises(OSError):
        consensus.finish_jury(plan, [])
    assert json.loads(consensus.JURY_LABELER.read_text()) == {"labeler": "aaa"}
    assert "sample_auto" in consensus.jury_plan(list(_JURY_INDEX), cache, deployed)["todo"]


def test_live_small_model_box_trains_only_with_a_human_label():
    fusion = importlib.import_module("kitchen_training.fusion")
    no_x = {"shape": [480, 640], "dogs": [], "people": []}
    live = {"detections": {"targets": [{"label": "dog", "confidence": 0.9, "box": [64, 48, 128, 96]}]}}
    assert fusion.fuse("s", "dog", live, no_x) == (None, "drop")           # auto label
    lines, source = fusion.fuse("s", "dog", {**live, "human_label": "dog"}, no_x)
    assert source == "fallback" and lines[0].startswith("1 ")


def test_phase_timing_helpers(capsys):
    from kitchen_training.config import duration, phase, rate
    assert (duration(42), duration(600), duration(3 * 3600)) == ("42s", "10.0 min", "3.00 h")
    assert rate(3000, 60) == "3000 frames in 60s (50 frames/s)"
    timings = {}
    with phase("build", timings):
        pass
    assert "build" in timings and "[train] build took 0s" in capsys.readouterr().out


def test_report_scores_mid_run_checkpoints(tmp_path):
    weights = tmp_path / "train" / "weights"
    weights.mkdir(parents=True)
    for name in ("epoch0", "epoch80", "epoch40", "last", "best"):
        (weights / f"{name}.pt").write_bytes(b"")
    assert list(pipeline.checkpoints(tmp_path)) == ["epoch40", "epoch80", "last"]
    assert training.CHECKPOINT_EVERY == 40


def test_ncnn_threads_capped_at_the_container_allowance(monkeypatch):
    import sys, types
    from kitchen_training.config import cap_ncnn_threads

    class Opt:
        num_threads = 21

    class Net:
        def __init__(self):
            self.opt = Opt()

    fake = types.SimpleNamespace(Net=Net)
    monkeypatch.setitem(sys.modules, "ncnn", fake)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    cap_ncnn_threads()
    assert fake.Net is Net                       # no allowance known: untouched
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    cap_ncnn_threads()
    cap_ncnn_threads()                           # idempotent
    assert fake.Net().opt.num_threads == 8 and fake.Net.__mro__[1] is Net


def test_training_trains_only_the_head_ncnn_deploys(monkeypatch, tmp_path):
    """NCNN cannot run YOLO26's end-to-end head and exports the classic one,
    while YOLO26's loss phases the classic head out over the epochs: the Oct 11
    120-epoch run starved it and lost the gate. Training must switch the
    end-to-end branch off so the trainer uses the classic loss."""
    import sys, types
    seen = {}

    class FakeModel:
        def __init__(self, path):
            self.model = types.SimpleNamespace(yaml={"end2end": True})

        def train(self, **kwargs):
            seen["end2end"] = self.model.yaml["end2end"]
            raise SystemExit  # stop before the save-dir lookup

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    import pytest
    with pytest.raises(SystemExit):
        training._train_local(tmp_path, epochs=1, batch=8, freeze=10, init=tmp_path / "w.pt")
    assert seen["end2end"] is False


def test_a_classic_only_model_uses_the_classic_loss_and_keeps_every_classic_weight():
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import intersect_dicts
    full = DetectionModel("yolo26n.yaml", nc=80, verbose=False)
    cfg = dict(full.yaml, end2end=False)
    classic = DetectionModel(cfg, nc=80, verbose=False)
    classic.args = get_cfg()
    assert isinstance(classic.init_criterion(), v8DetectionLoss)
    kept = intersect_dicts(full.state_dict(), classic.state_dict())
    assert len(kept) == len(classic.state_dict())
    assert all("one2one" in k for k in full.state_dict() if k not in kept)


def test_pt_models_are_scored_with_the_deployed_head(monkeypatch, tmp_path):
    import sys, types
    evaluation = importlib.import_module("kitchen_training.evaluation")
    heads = []

    class FakeModel:
        def __init__(self, path):
            self.model = types.SimpleNamespace(end2end=True)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    monkeypatch.setattr(evaluation, "eval_frames", lambda run_dir: [])
    monkeypatch.setattr(evaluation, "dog_confs", lambda model, frames, dev: heads.append(model.model.end2end) or {})
    monkeypatch.setattr(evaluation, "_val_map50", lambda model, run_dir: None)
    monkeypatch.setattr(evaluation, "curve", lambda frames, confs: {"0.7": {"heldout": {"caught": 0, "missed": 0, "false_fires": 0}}})
    evaluation.evaluate(tmp_path, {"fine-tune": tmp_path / "a.pt", "baseline": tmp_path / "b.pt"})
    assert heads == [False, False]
