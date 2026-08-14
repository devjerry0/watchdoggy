"""Build the YOLO dataset: fused labels, stable split, optional augmentation."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from kitchen_training.config import (CLASS_IDS, DATASET_MIRROR,
                                     VAL_FRACTION_MOD, log)
from kitchen_training.dataset import labeled_sidecars, prelabel
from kitchen_training.fusion import fuse
from kitchen_training.perturb import train_variants


def split_for(stem: str) -> str:
    """Stable split: the stem's hash decides, forever."""
    digest = int(hashlib.sha1(stem.encode()).hexdigest(), 16)
    if digest % VAL_FRACTION_MOD == 0:
        return "val"
    return "train"


def _write_augmented(dataset_dir: Path, stem: str, label_text: str,
                     stats: dict) -> None:
    """Train-split only -- the held-out exam stays pristine originals."""
    import cv2
    image = cv2.imread(str(DATASET_MIRROR / f"{stem}.jpg"))
    for tag, variant in train_variants(image).items():
        cv2.imwrite(str(dataset_dir / f"images/train/{stem}_{tag}.jpg"), variant)
        (dataset_dir / f"labels/train/{stem}_{tag}.txt").write_text(label_text)
        stats["augmented"] += 1


def _count_boxes(lines: list[str], into: dict) -> None:
    person_prefix = f"{CLASS_IDS['person']} "
    for line in lines:
        key = "person" if line.startswith(person_prefix) else "dog"
        into[key] += 1


def _write_frame(dataset_dir: Path, stem: str, lines: list[str], augment: bool,
                 stats: dict, force_train: bool = False) -> None:
    # Auto-labeled frames never enter the val split: the held-out exam is
    # human-verified only, so machine error can't grade itself.
    split = "train" if force_train else split_for(stem)
    label_text = "\n".join(lines) + ("\n" if lines else "")
    shutil.copyfile(DATASET_MIRROR / f"{stem}.jpg",
                    dataset_dir / f"images/{split}/{stem}.jpg")
    (dataset_dir / f"labels/{split}/{stem}.txt").write_text(label_text)
    stats[split] += 1
    _count_boxes(lines, stats["boxes"][split])
    if augment and split == "train":
        _write_augmented(dataset_dir, stem, label_text, stats)


def _new_stats() -> dict:
    return {"train": 0, "val": 0, "dropped": [], "fallback": 0, "hand_boxed": 0,
            "augmented": 0, "auto_labeled": 0,
            "boxes": {"train": {"person": 0, "dog": 0},
                      "val": {"person": 0, "dog": 0}},
            "verdicts": {}}


def _log_build(stats: dict) -> None:
    log(f"train {stats['train']} imgs (+{stats['augmented']} augmented, "
        f"{stats['auto_labeled']} auto-labeled) {stats['boxes']['train']} | "
        f"val {stats['val']} imgs {stats['boxes']['val']} (human-verified) | "
        f"hand-boxed {stats['hand_boxed']} | dropped {len(stats['dropped'])} | "
        f"nano-fallback {stats['fallback']}")
    for stem, why in stats["dropped"]:
        log(f"  NEEDS BOXES: {stem} ({why})")


def build(run_dir: Path, augment: bool = False) -> dict:
    samples = labeled_sidecars(include_auto=True)
    log(f"building dataset from {len(samples)} verdicted frames")
    prelabels_by_stem = prelabel([stem for stem, _, _ in samples])

    dataset_dir = run_dir / "dataset"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset_dir / sub).mkdir(parents=True)

    stats = _new_stats()
    source_counters = {"hand": "hand_boxed", "fallback": "fallback"}
    for stem, verdict, meta in samples:
        stats["verdicts"][verdict] = stats["verdicts"].get(verdict, 0) + 1
        lines, source = fuse(stem, verdict, meta, prelabels_by_stem[stem])
        if lines is None:
            stats["dropped"].append(
                (stem, "dog verdict but no model drew a box -- "
                       "draw one by hand on /review"))
            continue
        if source in source_counters:
            stats[source_counters[source]] += 1
        is_auto = not meta.get("human_label")
        stats["auto_labeled"] += is_auto
        _write_frame(dataset_dir, stem, lines, augment, stats,
                     force_train=is_auto)

    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\n  1: dog\n")
    _log_build(stats)
    return stats
