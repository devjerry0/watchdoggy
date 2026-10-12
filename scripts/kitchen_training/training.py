"""The train stage: local GPU, single Modal job, or the parallel Modal sweep."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from kitchen_training.config import (BASE_MODEL, CHAMPION_MODEL, IMAGE_SIZE,
                                     REPO, SWEEP_CONFIGS, device, log, sh)

MODAL_SCRIPT = REPO / "scripts/modal_train.py"
MODAL_SETUP_HINT = "Modal job failed. First time? Run: uv run modal setup"


def start_weights() -> Path:
    """The deployed model's weights when they exist, else the stock base."""
    return CHAMPION_MODEL if CHAMPION_MODEL.is_file() else BASE_MODEL


# Mid-run checkpoints (epoch40.pt, epoch80.pt; ultralytics counts from 0):
# the report scores them beside best.pt and last.pt, so every run shows
# whether the later epochs paid off on the exam.
CHECKPOINT_EVERY = 40


def _train_local(run_dir: Path, epochs: int, batch: int, freeze: int,
                 init: Path | None = None, tag: str = "") -> None:
    from ultralytics import YOLO
    init = init or start_weights()
    log(f"training from {init.name} locally: epochs<={epochs}, "
        f"freeze={freeze}, device={device()}")
    model = YOLO(str(init))
    # Train only the classic (one-to-many) head, the one the Pi runs: NCNN has
    # no top-k op, so the export drops YOLO26's end-to-end head, while YOLO26's
    # loss phases the classic head out (weight 0.8 -> 0.1 over the epochs). The
    # Oct 11 120-epoch run starved it and lost the gate (2026-10-11). With
    # end2end off the trainer rebuilds the model without that branch and uses
    # the classic YOLO loss; best.pt selection then measures the shipped head.
    model.model.yaml["end2end"] = False
    # cache=True decodes each image once into RAM instead of once per epoch
    # -- a 200-epoch run reads the dataset 200x otherwise.
    model.train(data=str(run_dir / "dataset/data.yaml"),
                epochs=epochs,
                # The full schedule, always (user-decided): ultralytics decays
                # the learning rate over the configured epochs and switches
                # mosaic off for the last 10, and that final phase is where
                # late gains come from. Early stopping on a noisy 379-frame val
                # set quit before it. best.pt is still the best epoch.
                patience=epochs,
                imgsz=IMAGE_SIZE, batch=batch, freeze=freeze, device=device(),
                cache=True, project=str(run_dir), name=f"train{tag}", exist_ok=True,
                save_period=CHECKPOINT_EVERY,
                verbose=False, plots=True)
    # Ultralytics decides the save dir (its settings can reroute project
    # paths); ask the trainer where it actually wrote instead of guessing.
    produced = Path(model.trainer.save_dir) / "weights/best.pt"
    if not produced.is_file():
        return
    shutil.copyfile(produced, run_dir / f"best{tag}.pt")


def _train_modal(run_dir: Path, epochs: int, batch: int, freeze: int,
                 init: Path | None = None, tag: str = "") -> None:
    # Mac-side legacy backend: always the stock base, one output.
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
          freeze: int, init: Path | None = None, tag: str = "") -> Path:
    """init = starting weights (default: champion, else base); tag keeps a
    second training's output apart (best{tag}.pt, train{tag}/)."""
    # Both backends land the weights at run_dir/best.pt -- the one contract
    # eval/export rely on, wherever the GPU actually was.
    TRAIN_BACKENDS[backend](run_dir, epochs, batch, freeze, init, tag)
    best = run_dir / f"best{tag}.pt"
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
