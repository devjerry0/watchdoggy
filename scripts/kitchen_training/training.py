"""The train stage: local GPU, single Modal job, or the parallel Modal sweep."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from kitchen_training.config import (BASE_MODEL, IMAGE_SIZE, MIN_PATIENCE,
                                     PATIENCE_DIVISOR, REPO, SWEEP_CONFIGS,
                                     device, log, sh)

MODAL_SCRIPT = REPO / "scripts/modal_train.py"
MODAL_SETUP_HINT = "Modal job failed. First time? Run: uv run modal setup"


def _train_local(run_dir: Path, epochs: int, batch: int, freeze: int) -> None:
    from ultralytics import YOLO
    log(f"training {BASE_MODEL.name} locally: epochs<={epochs}, "
        f"freeze={freeze}, device={device()}")
    model = YOLO(str(BASE_MODEL))
    model.train(data=str(run_dir / "dataset/data.yaml"),
                epochs=epochs,
                patience=max(MIN_PATIENCE, epochs // PATIENCE_DIVISOR),
                imgsz=IMAGE_SIZE, batch=batch, freeze=freeze, device=device(),
                project=str(run_dir), name="train", exist_ok=True,
                verbose=False, plots=True)
    # Ultralytics decides the save dir (its settings can reroute project
    # paths); ask the trainer where it actually wrote instead of guessing.
    produced = Path(model.trainer.save_dir) / "weights/best.pt"
    if not produced.is_file():
        return
    shutil.copyfile(produced, run_dir / "best.pt")


def _train_modal(run_dir: Path, epochs: int, batch: int, freeze: int) -> None:
    log(f"dispatching training job to Modal: epochs<={epochs}, freeze={freeze}")
    try:
        sh(["uv", "run", "modal", "run", str(MODAL_SCRIPT),
            "--run-dir", str(run_dir), "--weights", str(BASE_MODEL),
            "--epochs", str(epochs), "--batch", str(batch),
            "--freeze", str(freeze)])
    except subprocess.CalledProcessError:
        sys.exit(f"[train] {MODAL_SETUP_HINT}")


TRAIN_BACKENDS = {"local": _train_local, "modal": _train_modal}


def train(run_dir: Path, backend: str, epochs: int, batch: int,
          freeze: int) -> Path:
    # Both backends land the weights at run_dir/best.pt -- the one contract
    # eval/export rely on, wherever the GPU actually was.
    TRAIN_BACKENDS[backend](run_dir, epochs, batch, freeze)
    best = run_dir / "best.pt"
    if not best.is_file():
        sys.exit("[train] FAILED: no best.pt produced")
    log(f"best weights: {best}")
    return best


def sweep(run_dir: Path, gpu: str) -> dict[str, Path]:
    """Fan the sweep configs out to Modal; return name -> local weights."""
    configs_path = run_dir / "sweep-configs.json"
    configs_path.write_text(json.dumps(SWEEP_CONFIGS, indent=1))
    log(f"dispatching {len(SWEEP_CONFIGS)} training jobs to Modal on {gpu}s ...")
    try:
        sh(["uv", "run", "modal", "run", f"{MODAL_SCRIPT}::sweep",
            "--run-dir", str(run_dir), "--weights", str(BASE_MODEL),
            "--configs", str(configs_path)])
    except subprocess.CalledProcessError:
        sys.exit("[train] Modal sweep failed. First time? Run: uv run modal setup")
    weights_by_name = {}
    for name in SWEEP_CONFIGS:
        weights = run_dir / "sweep" / f"{name}.pt"
        if not weights.is_file():
            log(f"WARNING: sweep job {name} returned no weights")
            continue
        weights_by_name[name] = weights
    if not weights_by_name:
        sys.exit("[train] FAILED: sweep produced no weights at all")
    return weights_by_name
