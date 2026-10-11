"""Every path, threshold, and tunable of the training pipeline, in one place.

Paths honor KT_* environment overrides so the same package runs on the Mac
(defaults) and inside a Modal container (paths pointed at the Volume).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


REPO = Path(__file__).resolve().parents[2]
DATASET_MIRROR = _path_from_env("KT_DATASET_DIR", REPO / "dataset-pull")
RUNS = _path_from_env("KT_RUNS_DIR", REPO / "training-runs")
PRELABEL_CACHE = RUNS / "prelabel-cache.json"

BASE_MODEL = _path_from_env("KT_BASE_MODEL", REPO / "models/yolo26n.pt")
# Weights of the deployed model: every training run starts here, on the FULL
# dataset, instead of re-learning from BASE_MODEL (user-decided 2026-10-11;
# training only on frames newer than the champion was tried and dropped --
# it saved ~$1-3/run for a forgetting risk and extra bookkeeping), and a run
# the gate deploys replaces it. Missing -> BASE_MODEL (fresh run).
CHAMPION_MODEL = _path_from_env("KT_CHAMPION_MODEL", REPO / "models/champion.pt")
PRELABEL_MODEL = _path_from_env("KT_PRELABEL_MODEL", REPO / "models/yolo26x.pt")

PRELABEL_CONF = 0.45      # big-model boxes below this aren't trusted as labels
FIRE_CONF = 0.7           # the appliance's default alarm threshold
EVAL_CONF_FLOOR = 0.25    # predict floor: low enough to score every threshold
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)  # the eval scores the whole curve
IMAGE_SIZE = 640          # train/prelabel input size
# The NCNN bundle's input (h, w) = the camera frame, so the Pi skips the 25%
# letterbox padding a square export needs (1.30x faster per frame; exam:
# same accuracy, 5-vs-3 discordant dog frames). Score bundles WITHOUT imgsz.
EXPORT_SHAPE = (480, 640)
VAL_FRACTION_MOD = 4      # stem-hash % 4 == 0 -> val (~25%)
MIXED_IOU_DROP = 0.6      # dog box overlapping a person this much is the person

CLASS_IDS = {"person": 0, "dog": 1}


# Hyperparameter sweep, dispatched to Modal in parallel (<=10 concurrent).
# Base recipe (the proven anchor): epochs 80, patience 20, imgsz 640,
# batch 16, freeze 10. Each entry lists only its deltas. Note: lr0 only
# takes effect with an explicit optimizer -- optimizer=auto overrides it.
SWEEP_CONFIGS = {
    "anchor": {},                          # the proven recipe, as-is
    "anchor-s1": {"seed": 1},              # same recipe: run-to-run noise probe
    "full-ft": {"freeze": 0},              # unfreeze the whole backbone
    "head-only": {"freeze": 20},           # touch almost nothing but the head
    "no-mosaic": {"mosaic": 0.0},          # mosaic aug distorts scale cues
    "no-mosaic-ft": {"freeze": 0, "mosaic": 0.0},
    "long": {"epochs": 200, "patience": 50},
    "gentle-ft": {"freeze": 0, "lr0": 0.0005, "cos_lr": True,
                  "optimizer": "AdamW"},
    "cls-heavy": {"cls": 1.0},             # our failure IS classification
    "low-lr": {"lr0": 0.0005, "optimizer": "AdamW"},
}


@dataclass(frozen=True)
class Frame:
    """One labeled frame as the eval sees it."""
    stem: str
    is_dog: bool      # human verdict says a real dog is present
    heldout: bool     # in the val split: never trained on
    jpg: Path


def log(message: str) -> None:
    print(f"[train] {message}", flush=True)


def sh(command: list[str], env: dict | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
