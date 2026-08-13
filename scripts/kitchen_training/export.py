"""Export the winner as the NCNN bundle the Pi actually runs."""
from __future__ import annotations

import shutil
from pathlib import Path

from kitchen_training.config import IMAGE_SIZE, log


def export(run_dir: Path, best: Path) -> Path:
    from ultralytics import YOLO
    log("exporting NCNN bundle for the Pi ...")
    YOLO(str(best)).export(format="ncnn", imgsz=IMAGE_SIZE)
    # ultralytics writes <stem>_ncnn_model next to the weights file
    exported = best.with_name(best.stem + "_ncnn_model")
    bundle = run_dir / "kitchen_ncnn_model"
    if bundle.exists():
        shutil.rmtree(bundle)
    shutil.move(str(exported), bundle)
    log(f"deployable model: {bundle}")
    return bundle
