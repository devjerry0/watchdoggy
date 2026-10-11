"""Build the YOLO dataset: fused labels, stable split, near-duplicate
control, human-label weighting, and optional augmentation."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from kitchen_training.config import CLASS_IDS, VAL_FRACTION_MOD, log
from kitchen_training.dataset import labeled_sidecars, prelabel, read_frames
from kitchen_training.fusion import fuse
from kitchen_training.perturb import TRAIN_VARIANTS, train_variants

# Near-duplicate control (train split only; the exam is never thinned):
# a cooking session yields dozens of frames of the same scene. NON-DOG
# frames only: on a fixed camera the static background dominates the hash,
# and measured on this corpus every Hamming floor from 2 to 6 collides
# half or more of the DISTINCT dog frames (a small dog is a handful of
# bits). Dogs are the scarce class and are never dropped; negatives are
# abundant and safely thinned. Human-labeled frames are written first so
# a human frame is never dropped in favor of an auto.
DEDUP_HAMMING_FLOOR = 6
DEDUP_VERDICTS = {"person", "empty", "no_dog"}
_DHASH_SIDE = 8

# Human labels outvote the jury's in the loss -- standard pseudo-label
# down-weighting (0.3-0.5x is the field range), implemented as up-weighting
# human frames by duplication since YOLO has no per-sample loss weights.
HUMAN_WEIGHT_COPIES = 1  # one extra copy = human frames weigh 2x
# Augmentation modes (training page setting). "all" writes every variant per
# train frame -- the recipe that won on the 168-frame set (Aug 2026) and the
# right call for a new install with little data. At 14k frames it made 75%
# of every epoch copies (the run's biggest cost), and real footage from the
# camera itself covers night and motion blur by then (user-decided), so
# "auto" turns copies OFF past AUTO_ALL_VARIANTS_BELOW. "one" stays a manual
# option if the report's stress row (blur/dark robustness) ever slips.
AUGMENT_MODES = ("auto", "all", "one", "off")
AUTO_ALL_VARIANTS_BELOW = 3000  # verdicted frames


def split_for(stem: str) -> str:
    """Stable split: the stem's hash decides, forever."""
    digest = int(hashlib.sha1(stem.encode()).hexdigest(), 16)
    if digest % VAL_FRACTION_MOD == 0:
        return "val"
    return "train"


def _split(stem: str, meta: dict, is_auto: bool) -> str:
    # Auto-labeled frames never enter the val split: the held-out exam is
    # human-verified only, so machine error can't grade itself.
    if is_auto:
        return "train"
    # Escaped failures are PERMANENT exam members (a unit test per failure,
    # the data-engine rule): a false fire the user flagged keeps being
    # tested forever instead of being trained away and forgotten.
    if "user_marked_fp" in meta.get("reasons", []):
        return "val"
    return split_for(stem)


def _dhash(image) -> int:
    import cv2
    import numpy as np
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                       (_DHASH_SIDE + 1, _DHASH_SIDE),
                       interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int(np.packbits(bits).tobytes().hex(), 16)


def _near_duplicate(image, seen: list[int]) -> bool:
    h = _dhash(image)
    if any((h ^ s).bit_count() < DEDUP_HAMMING_FLOOR for s in seen):
        return True
    seen.append(h)
    return False


def augment_mode(augment: bool | str, verdicted: int) -> str:
    """Resolve the setting to "all", "one" or "off". Legacy booleans: True
    meant "augment" and now means auto, False means off."""
    if augment is True or augment == "auto":
        return "all" if verdicted < AUTO_ALL_VARIANTS_BELOW else "off"
    if augment is False:
        return "off"
    return augment


def variant_tags(stem: str, mode: str) -> tuple[str, ...]:
    """Which variants a train frame gets. "one" picks by a stable hash of the
    stem, so a rebuild writes the same copies and the three kinds stay evenly
    spread over the dataset."""
    tags = tuple(TRAIN_VARIANTS)
    if mode == "all":
        return tags
    if mode == "one":
        return (tags[int(hashlib.md5(stem.encode()).hexdigest(), 16) % len(tags)],)
    return ()


def _write_augmented(dataset_dir: Path, stem: str, label_text: str,
                     image, stats: dict, mode: str) -> None:
    """Train-split only -- the held-out exam stays pristine originals."""
    import cv2
    for tag, variant in train_variants(image, variant_tags(stem, mode)).items():
        cv2.imwrite(str(dataset_dir / f"images/train/{stem}_{tag}.jpg"), variant)
        (dataset_dir / f"labels/train/{stem}_{tag}.txt").write_text(label_text)
        stats["augmented"] += 1


def _count_boxes(lines: list[str], into: dict) -> None:
    person_prefix = f"{CLASS_IDS['person']} "
    for line in lines:
        key = "person" if line.startswith(person_prefix) else "dog"
        into[key] += 1


def _write_frame(dataset_dir: Path, stem: str, lines: list[str], augment: str,
                 stats: dict, split: str, human: bool, image, data: bytes) -> None:
    """data = the original JPEG bytes (already read): written as-is, so the
    mirror is never read twice."""
    label_text = "\n".join(lines) + ("\n" if lines else "")
    (dataset_dir / f"images/{split}/{stem}.jpg").write_bytes(data)
    (dataset_dir / f"labels/{split}/{stem}.txt").write_text(label_text)
    stats[split] += 1
    _count_boxes(lines, stats["boxes"][split])
    if split != "train":
        return
    for n in range(HUMAN_WEIGHT_COPIES if human else 0):
        (dataset_dir / f"images/train/{stem}_w{n + 2}.jpg").write_bytes(data)
        (dataset_dir / f"labels/train/{stem}_w{n + 2}.txt").write_text(label_text)
        stats["human_weighted"] += 1
    _write_augmented(dataset_dir, stem, label_text, image, stats, augment)


def _new_stats() -> dict:
    return {"train": 0, "val": 0, "dropped": [], "fallback": 0, "hand_boxed": 0,
            "augmented": 0, "auto_labeled": 0, "human_weighted": 0,
            "dedup_dropped": 0,
            "boxes": {"train": {"person": 0, "dog": 0},
                      "val": {"person": 0, "dog": 0}},
            "verdicts": {}}


def _log_build(stats: dict) -> None:
    log(f"train {stats['train']} imgs (+{stats['augmented']} augmented, "
        f"+{stats['human_weighted']} human-weight copies, "
        f"{stats['auto_labeled']} auto-labeled, "
        f"{stats['dedup_dropped']} near-dupes dropped) "
        f"{stats['boxes']['train']} | "
        f"val {stats['val']} imgs {stats['boxes']['val']} (human-verified) | "
        f"hand-boxed {stats['hand_boxed']} | dropped {len(stats['dropped'])} | "
        f"nano-fallback {stats['fallback']}")
    for stem, why in stats["dropped"]:
        log(f"  NEEDS BOXES: {stem} ({why})")


def build(run_dir: Path, augment: bool | str = False) -> dict:
    samples = labeled_sidecars(include_auto=True)
    mode = augment_mode(augment, len(samples))
    log(f"building dataset from {len(samples)} verdicted frames "
        f"(augmentation: {mode})")
    prelabels_by_stem = prelabel([stem for stem, _, _ in samples])

    dataset_dir = run_dir / "dataset"
    # Idempotent for retried cloud runs: a crashed attempt leaves partial
    # output in the Volume; rebuild from clean rather than trip over it.
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset_dir / sub).mkdir(parents=True)

    stats = _new_stats()
    source_counters = {"hand": "hand_boxed", "fallback": "fallback"}
    seen_hashes: list[int] = []
    # Humans first: with dedup active, a human-labeled frame must never be
    # the one dropped when an auto near-duplicate exists.
    ordered = sorted(samples, key=lambda s: not s[2].get("human_label"))
    frames = read_frames([stem for stem, _, _ in ordered])
    for (stem, verdict, meta), (_, data, image) in zip(ordered, frames):
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
        split = _split(stem, meta, is_auto)
        if (split == "train" and verdict in DEDUP_VERDICTS
                and _near_duplicate(image, seen_hashes)):
            stats["dedup_dropped"] += 1
            continue
        stats["auto_labeled"] += is_auto
        _write_frame(dataset_dir, stem, lines, mode, stats, split,
                     human=not is_auto, image=image, data=data)

    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\n  1: dog\n")
    stats["augment_mode"] = mode
    _log_build(stats)
    return stats
