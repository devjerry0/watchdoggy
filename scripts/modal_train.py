"""Modal cloud-GPU backend for the kitchen fine-tune's train stage.

Invoked by scripts/train_kitchen_model.py --backend modal; also runnable
directly against an already-built run dir:

    uv run modal run scripts/modal_train.py --run-dir training-runs/<ts> \
        --weights models/yolo26n.pt

Uploads run_dir/dataset + the base weights to a Modal Volume, fine-tunes on a
T4 (~$0.15 for our 4-minute recipe), and writes best.pt back into run_dir --
the same contract the local backend fulfills. Only this one stage runs in the
cloud: label fusion, eval, and NCNN export stay on the Mac where the frame
mirror and the big prelabel model live.

One-time setup: uv run modal setup   (browser auth, token saved locally)
Patterned after https://modal.com/docs/examples/finetune_yolo
"""
from __future__ import annotations

from pathlib import Path

import modal

MINUTES = 60

app = modal.App("watchdoggy-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # OpenCV inside ultralytics needs the usual headless-Linux graphics libs.
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics~=8.4", "opencv-python~=4.10")
)

volume = modal.Volume.from_name("watchdoggy-train", create_if_missing=True)
VOL = Path("/vol")


@app.function(image=image, gpu="T4", volumes={str(VOL): volume},
              timeout=60 * MINUTES)
def train(run_name: str, epochs: int, batch: int, freeze: int) -> bytes:
    """Fine-tune on the uploaded dataset; return best.pt as bytes (~5MB)."""
    from ultralytics import YOLO

    root = VOL / run_name
    data_yaml = root / "dataset/data.yaml"
    # The yaml was written on the Mac with an absolute Mac path; repoint it.
    lines = [f"path: {root / 'dataset'}" if ln.startswith("path:") else ln
             for ln in data_yaml.read_text().splitlines()]
    data_yaml.write_text("\n".join(lines) + "\n")

    model = YOLO(str(root / "base.pt"))
    # Keep this recipe identical to the local backend in train_kitchen_model.py.
    model.train(data=str(data_yaml), epochs=epochs,
                patience=max(10, epochs // 4), imgsz=640, batch=batch,
                freeze=freeze, device=0, project=str(root), name="train",
                exist_ok=True, verbose=False, plots=True)
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    volume.commit()  # persist the full run (curves, plots) for the dashboard
    return best.read_bytes()


@app.local_entrypoint()
def main(run_dir: str, weights: str = "models/yolo26n.pt",
         epochs: int = 80, batch: int = 16, freeze: int = 10) -> None:
    rd = Path(run_dir).resolve()
    if not (rd / "dataset/data.yaml").is_file():
        raise SystemExit(f"no built dataset in {rd} -- run the build stage first")

    run_name = rd.name
    print(f"[modal] uploading dataset + base weights as {run_name} ...")
    with volume.batch_upload(force=True) as batch_up:
        batch_up.put_directory(str(rd / "dataset"), f"{run_name}/dataset")
        batch_up.put_file(weights, f"{run_name}/base.pt")

    print("[modal] training on a cloud GPU ...")
    best = train.remote(run_name, epochs, batch, freeze)
    (rd / "best.pt").write_bytes(best)
    print(f"[modal] done -> {rd / 'best.pt'} ({len(best) / 1e6:.1f} MB)")
