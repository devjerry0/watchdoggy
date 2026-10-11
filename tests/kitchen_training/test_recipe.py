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
    assert build.augment_mode("auto", big) == "one"
    # Legacy settings files: True meant "augment" -> auto, False -> off.
    assert build.augment_mode(True, big) == "one"
    assert build.augment_mode(False, small) == "off"
    # Explicit choices are honored at any size.
    assert build.augment_mode("all", big) == "all"
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


def _point_champion(tmp_path, monkeypatch):
    champion = tmp_path / "models" / "champion.pt"
    for module in (pipeline, training):
        monkeypatch.setattr(module, "CHAMPION_MODEL", champion)
    monkeypatch.setattr(pipeline, "CHAMPION_META", champion.with_suffix(".json"))
    return champion


def test_promote_champion_writes_weights_and_cutoff(tmp_path, monkeypatch):
    import json
    champion = _point_champion(tmp_path, monkeypatch)
    best = tmp_path / "best.pt"
    best.write_bytes(b"new weights")
    pipeline.promote_champion(best, 1234.5, "20261011-140000")
    assert champion.read_bytes() == b"new weights"
    assert json.loads(champion.with_suffix(".json").read_text()) == {
        "data_cutoff": 1234.5, "run": "20261011-140000"}
    assert not list(champion.parent.glob("*.tmp"))
    # ...and the next run trains only on what came after that cutoff.
    assert pipeline.champion_cutoff() == 1234.5


def test_no_champion_record_means_a_full_run(tmp_path, monkeypatch):
    champion = _point_champion(tmp_path, monkeypatch)
    assert pipeline.champion_cutoff() is None       # no weights at all
    champion.parent.mkdir(parents=True)
    champion.write_bytes(b"weights")
    assert pipeline.champion_cutoff() is None       # weights but no record
    champion.with_suffix(".json").write_text("{not json")
    assert pipeline.champion_cutoff() is None       # torn record


def test_warm_runs_train_new_frames_plus_replay():
    since = 1000.0
    new_auto = {"auto_label": {"verdict": "dog", "labeled_at": 1500.0}}
    old_auto = {"auto_label": {"verdict": "dog", "labeled_at": 500.0}}
    old_hand = {"human_label": "dog", "labeled_at": 500.0}
    relabeled = {"human_label": "dog", "labeled_at": 1500.0,
                 "auto_label": {"verdict": "empty", "labeled_at": 500.0}}
    role = build.training_role
    assert role("a", new_auto, True, since, 0.0) == "new"
    assert role("b", relabeled, False, since, 0.0) == "new"   # a correction is new
    assert role("c", old_hand, False, since, 0.0) == "replay"  # hand labels always
    assert role("d", old_auto, True, since, 0.0) is None       # already learned
    assert role("d", old_auto, True, since, 1.0) == "replay"
    assert role("d", old_auto, True, None, 0.0) == "new"       # fresh run: all
    # The replay draw hits about the requested fraction of old frames.
    drawn = sum(role(f"s{i}", old_auto, True, since, 0.25) == "replay"
                for i in range(4000))
    assert 850 < drawn < 1150
    assert build.replay_fraction(new=500, old_auto=2000) == 0.25
    assert build.replay_fraction(new=5000, old_auto=2000) == 1.0
    assert build.replay_fraction(new=10, old_auto=0) == 0.0
