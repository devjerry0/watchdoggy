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
