"""Modal cloud-GPU backend for the kitchen fine-tune's train stage.

Invoked by scripts/train_kitchen_model.py (--backend modal, or --sweep);
also runnable directly against an already-built run dir:

    uv run modal run scripts/modal_train.py --run-dir training-runs/<ts>
    uv run modal run scripts/modal_train.py::sweep --run-dir ... --configs ...

The dataset + base weights upload once to a Modal Volume; each job trains on
its own GPU (type from DOGGY_TRAIN_GPU, default L4; sweep jobs run in
parallel, capped at 10 containers) and returns best.pt as bytes, which land
back in the run dir -- the same contract the local backend fulfills. Only
this stage runs in the cloud: label fusion, eval, and NCNN export stay on
the Mac where the frame mirror and the big prelabel model live.

ultralytics is PINNED to the Mac's version so cloud and local runs stay
comparable (8.4.118 trains YOLO26 with a different loss recipe than 8.4.88).

One-time setup: uv run modal setup   (browser auth, token saved locally)
Patterned after https://modal.com/docs/examples/finetune_yolo
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

MINUTES = 60
GPU = os.environ.get("DOGGY_TRAIN_GPU", "L4")
MAX_PARALLEL = 10

# The proven recipe; sweep configs override individual keys.
BASE_ARGS = dict(epochs=80, patience=20, imgsz=640, batch=16, freeze=10)

app = modal.App("watchdoggy-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # OpenCV inside ultralytics needs the usual headless-Linux graphics libs.
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.88", "opencv-python~=4.10")
)

volume = modal.Volume.from_name("watchdoggy-train", create_if_missing=True)
VOL = Path("/vol")


@app.function(image=image, gpu=GPU, volumes={str(VOL): volume},
              timeout=60 * MINUTES, max_containers=MAX_PARALLEL)
def train(run_name: str, job: str, overrides: dict) -> bytes:
    """Fine-tune one config on the uploaded dataset; return best.pt bytes."""
    from ultralytics import YOLO

    root = VOL / run_name
    args = {**BASE_ARGS, **overrides}
    model = YOLO(str(root / "base.pt"))
    # modal-data.yaml was rewritten at upload time to point inside /vol.
    # Results go to container-local /tmp: the weights travel home as the
    # return value, so nothing needs to persist in the Volume.
    model.train(data=str(root / "modal-data.yaml"), device=0,
                project=f"/tmp/{job}", name="train", exist_ok=True,
                verbose=False, plots=False, **args)
    return (Path(model.trainer.save_dir) / "weights/best.pt").read_bytes()


def _upload(rd: Path, weights: str) -> str:
    """Push run_dir/dataset + base weights to the Volume; return run name."""
    if not (rd / "dataset/data.yaml").is_file():
        raise SystemExit(f"no built dataset in {rd} -- run the build stage first")
    run_name = rd.name
    remote_root = f"/vol/{run_name}"
    lines = [f"path: {remote_root}/dataset" if ln.startswith("path:") else ln
             for ln in (rd / "dataset/data.yaml").read_text().splitlines()]
    (rd / "modal-data.yaml").write_text("\n".join(lines) + "\n")
    print(f"[modal] uploading dataset + base weights as {run_name} ...")
    with volume.batch_upload(force=True) as up:
        up.put_directory(str(rd / "dataset"), f"{run_name}/dataset")
        up.put_file(str(rd / "modal-data.yaml"), f"{run_name}/modal-data.yaml")
        up.put_file(weights, f"{run_name}/base.pt")
    return run_name


@app.local_entrypoint()
def main(run_dir: str, weights: str = "models/yolo26n.pt",
         epochs: int = 80, batch: int = 16, freeze: int = 10) -> None:
    """Single training job (the --backend modal path)."""
    rd = Path(run_dir).resolve()
    run_name = _upload(rd, weights)
    print(f"[modal] training on a cloud {GPU} ...")
    best = train.remote(run_name, "single",
                        {"epochs": epochs, "batch": batch, "freeze": freeze,
                         "patience": max(10, epochs // 4)})
    (rd / "best.pt").write_bytes(best)
    print(f"[modal] done -> {rd / 'best.pt'} ({len(best) / 1e6:.1f} MB)")


@app.local_entrypoint()
def sweep(run_dir: str, configs: str,
          weights: str = "models/yolo26n.pt") -> None:
    """Train every config in the JSON file in parallel; weights land in
    run_dir/sweep/<name>.pt. A failed job warns instead of sinking the fleet."""
    rd = Path(run_dir).resolve()
    cfgs = json.loads(Path(configs).read_text())
    run_name = _upload(rd, weights)
    out_dir = rd / "sweep"
    out_dir.mkdir(exist_ok=True)

    print(f"[modal] {len(cfgs)} jobs in parallel on {GPU}s "
          f"(max {MAX_PARALLEL} concurrent) ...")
    jobs = [(run_name, name, ov) for name, ov in cfgs.items()]
    results = train.starmap(jobs, return_exceptions=True)
    for (_, name, _), blob in zip(jobs, results):
        if isinstance(blob, Exception):
            print(f"[modal] {name}: FAILED -- {blob}")
            continue
        (out_dir / f"{name}.pt").write_bytes(blob)
        print(f"[modal] {name}: done ({len(blob) / 1e6:.1f} MB)")
