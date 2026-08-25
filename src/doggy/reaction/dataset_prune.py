"""Capped-storage pruning for the training dataset: labels are sacred.

Oldest UNLABELED samples go first; labeled frames are never deleted. The
old blind oldest-first prune silently ate months of human labels when
capture outgrew the cap (found 2026-08-25 as a collapsing exam count;
restored from backups). If labeled frames alone exceed the cap, log
loudly and stop rather than eat them."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("doggy")

_MB = 1_048_576


def _is_labeled(sidecar: Path) -> bool:
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return False
    return bool(meta.get("human_label") or meta.get("auto_label"))


def delete_sample(image: Path) -> int:
    """Remove one sample (frame + sidecar); returns the bytes freed."""
    freed = 0
    for p in (image, image.with_suffix(".json")):
        if p.is_file():
            freed += p.stat().st_size
            p.unlink()
    return freed


def prune(dataset_dir: Path, cap_bytes: int) -> None:
    samples = sorted(dataset_dir.glob("sample_*.jpg"))
    total = sum(p.stat().st_size for p in dataset_dir.glob("sample_*")
                if p.is_file())
    skipped_labeled = 0
    for image in samples:
        if total <= cap_bytes:
            return
        if _is_labeled(image.with_suffix(".json")):
            skipped_labeled += 1
            continue
        total -= delete_sample(image)
    if total > cap_bytes:
        log.warning(
            "dataset: %d labeled frames exceed the cap by %d MB -- "
            "refusing to delete labels; raise DOGGY_DATASET_CAP_BYTES",
            skipped_labeled, (total - cap_bytes) // _MB)
