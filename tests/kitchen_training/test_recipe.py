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
