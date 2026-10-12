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
    assert "prelabels ready (40 frames, 40 newly computed)" in capsys.readouterr().out


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

    (tmp_path / "sample_1.json").write_text(json.dumps({"prelabels": {"boxes": []}}))
    (tmp_path / "sample_2.json").write_text(json.dumps({"human_label": "dog"}))
    (tmp_path / "sample_3.json").write_text("")    # power cut mid-write
    _write_index(FakeVolume(), tmp_path)
    index = json.loads((tmp_path / ".labeled-index.json").read_text())
    assert index == {"sample_1": [None, None, 0, 0, 1],
                     "sample_2": ["dog", None, 0, 0, 0]}


def test_uploads_are_sized_by_bytes():
    from mirror_sync import UPLOAD_CHUNK_MAX_FILES, upload_chunks
    jpgs = {f"sample_{i}.jpg": 100_000 for i in range(200)}      # ~19 MB
    assert [len(c) for c in upload_chunks(list(jpgs), jpgs)] == [83, 83, 34]
    sidecars = {f"sample_{i}.json": 2_000 for i in range(5000)}
    assert [len(c) for c in upload_chunks(list(sidecars), sidecars)] == \
        [UPLOAD_CHUNK_MAX_FILES, UPLOAD_CHUNK_MAX_FILES, 1000]
    huge = {"sample_big.jpg": 20 << 20, "sample_a.json": 10}
    assert list(upload_chunks(list(huge), huge)) == [["sample_big.jpg"], ["sample_a.json"]]


def test_nano_boxes_rescue_a_dog_frame_nothing_else_boxed():
    fusion = importlib.import_module("kitchen_training.fusion")
    prelabels = {"shape": [480, 640], "dogs": [], "people": []}  # big model: nothing
    meta = {"detections": {"targets": []}}                       # live Pi: nothing
    assert fusion.fuse("s", "dog", meta, prelabels) == (None, "drop")
    lines, source = fusion.fuse("s", "dog", meta, prelabels, [[64.0, 48.0, 128.0, 96.0]])
    assert source == "fallback" and len(lines) == 1 and lines[0].startswith("1 ")  # dog
