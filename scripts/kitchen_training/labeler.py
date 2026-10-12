"""The labeler's dataset: hand labels outside the exam plus a stable sample of
agreed auto-labels, with the human exam as val.

Labels carry COCO class ids and data.yaml lists all 80 COCO names, so
yolo26x keeps its 80-class head and every trained weight, including its dog
and person outputs (a fresh 2-class head would relearn dogs from ~600 frames).
"""
from __future__ import annotations

import bisect
import hashlib
import shutil
from pathlib import Path

from kitchen_training.build import DEDUP_VERDICTS, _dhash, _near_duplicate, _split
from kitchen_training.config import log
from kitchen_training.dataset import labeled_sidecars, prelabel, read_frames
from kitchen_training.fusion import fuse

COCO_IDS = {"person": 0, "dog": 16}
# Extra copies of each hand-labeled train frame, so ~1k hand frames make up
# about a third of an epoch next to the auto sample.
LABELER_HUMAN_COPIES = 2
AUTO_SAMPLE_SIZE = 8000  # agreed auto-labels per run; a job can set its own
# Capture saves bursts 2 s apart, so a frame this close to an exam frame is
# nearly the same picture: it stays out of training.
EXAM_GUARD_MS = 30_000


def _stamp(stem: str) -> int:
    return int(stem.removeprefix("sample_"))


def _near_exam(stem: str, exam_stamps: list[int]) -> bool:
    """exam_stamps is sorted: the first exam stamp at or after the window's
    start decides."""
    stamp = _stamp(stem)
    i = bisect.bisect_left(exam_stamps, stamp - EXAM_GUARD_MS)
    return i < len(exam_stamps) and exam_stamps[i] <= stamp + EXAM_GUARD_MS


def _sample_key(sample: tuple) -> str:
    """Stable order for the auto sample: the same inputs pick the same frames,
    and a bigger sample adds frames without changing the smaller one."""
    return hashlib.sha256(sample[0].encode()).hexdigest()


def _coco_names() -> dict[int, str]:
    from ultralytics.utils import ROOT, YAML
    return YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"]


def _write(dataset_dir: Path, split: str, stem: str, lines: list[str],
           data: bytes, copies: int, stats: dict) -> None:
    text = "".join(f"{line}\n" for line in lines)
    for name in [stem] + [f"{stem}_w{n + 2}" for n in range(copies)]:
        (dataset_dir / f"images/{split}/{name}.jpg").write_bytes(data)
        (dataset_dir / f"labels/{split}/{name}.txt").write_text(text)
    stats[split] += 1 + copies


def _log_build(stats: dict) -> None:
    log(f"labeler dataset: epoch {stats['epoch']} imgs = {stats['hand']} hand "
        f"(+{stats['human_copies']} copies) + {stats['auto']} auto sample | "
        f"label sources {stats['sources']} | "
        f"val {stats['val']} of {len(stats['exam'])} exam frames | "
        f"exam guard dropped {stats['exam_guard_dropped']} | "
        f"auto skipped: {stats['auto_no_box']} no yolo26x dog box, "
        f"{stats['dedup_dropped']} near-dupes | "
        f"hand dropped {stats['hand_no_box']} without boxes | "
        f"unreadable {stats['unreadable']}")


def build_dataset(run_dir: Path, sample_size: int = AUTO_SAMPLE_SIZE) -> dict:
    """Write run_dir/dataset and return its counts, plus "data_yaml" and
    "exam": every human exam stem, including frames without usable labels,
    which val leaves out and the gate still scores."""
    samples = labeled_sidecars(include_auto=True)
    hand = [s for s in samples if s[2].get("human_label")]
    autos = [s for s in samples if not s[2].get("human_label")]
    exam = sorted(stem for stem, _, meta in hand
                  if _split(stem, meta, is_auto=False) == "val")
    exam_set, exam_stamps = set(exam), sorted(map(_stamp, exam))
    hand_train = [s for s in hand
                  if s[0] not in exam_set and not _near_exam(s[0], exam_stamps)]
    auto_pool = sorted((s for s in autos if not _near_exam(s[0], exam_stamps)),
                       key=_sample_key)
    guard_dropped = (len(hand) - len(exam) - len(hand_train)
                     + len(autos) - len(auto_pool))
    log(f"exam guard: dropped {guard_dropped} frames within "
        f"{EXAM_GUARD_MS // 1000} s of an exam frame")
    prelabels = prelabel([s[0] for s in hand + auto_pool])

    dataset_dir = run_dir / "dataset"
    if dataset_dir.exists():  # a retried run rebuilds from clean
        shutil.rmtree(dataset_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset_dir / sub).mkdir(parents=True)

    stats = {"train": 0, "val": 0, "hand": 0, "human_copies": 0, "auto": 0,
             "sources": {}, "exam": exam, "exam_guard_dropped": guard_dropped,
             "hand_no_box": 0, "auto_no_box": 0, "dedup_dropped": 0,
             "unreadable": 0}
    seen: list[int] = []
    # Exam frames go to val as they are; hand train frames get the copies
    # and seed the near-duplicate check, so an auto never repeats one.
    jobs = ([(s, "val") for s in hand if s[0] in exam_set]
            + [(s, "train") for s in hand_train])
    frames = read_frames([s[0] for s, _ in jobs])
    for ((stem, verdict, meta), split), (_, data, image) in zip(jobs, frames):
        if image is None:  # empty or corrupt file (see dataset.UNREADABLE)
            stats["unreadable"] += 1
            continue
        lines, source = fuse(stem, verdict, meta, prelabels[stem], COCO_IDS)
        if lines is None:
            stats["hand_no_box"] += 1
            continue
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        if split == "val":
            _write(dataset_dir, "val", stem, lines, data, 0, stats)
            continue
        if verdict in DEDUP_VERDICTS:
            seen.append(_dhash(image))
        _write(dataset_dir, "train", stem, lines, data, LABELER_HUMAN_COPIES, stats)
        stats["hand"] += 1
        stats["human_copies"] += LABELER_HUMAN_COPIES

    frames = read_frames([s[0] for s in auto_pool])
    for (stem, verdict, meta), (_, data, image) in zip(auto_pool, frames):
        if stats["auto"] >= sample_size:
            break
        if image is None:
            stats["unreadable"] += 1
            continue
        # An auto dog trains only on yolo26x's box: both models must agree
        # for a box to train (fuse's live-box fallback is human-only).
        lines, source = fuse(stem, verdict, meta, prelabels[stem], COCO_IDS)
        if lines is None:
            stats["auto_no_box"] += 1
            continue
        if verdict in DEDUP_VERDICTS and _near_duplicate(image, seen):
            stats["dedup_dropped"] += 1
            continue
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        _write(dataset_dir, "train", stem, lines, data, 0, stats)
        stats["auto"] += 1

    names = _coco_names()
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in names.items()))
    stats["epoch"] = stats["train"]
    stats["data_yaml"] = data_yaml
    _log_build(stats)
    return stats
