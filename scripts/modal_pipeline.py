"""The whole training pipeline as one Modal job, triggerable from the Pi.

    modal run scripts/modal_pipeline.py::kickoff --dataset-dir <frames> \
        --deployed-dir <current_ncnn_bundle> --out-dir <results>

The caller (the Pi's trainer daemon, or a Mac) uploads the raw labeled
frames and the currently-deployed bundle; the cloud does EVERYTHING else:
big-model prelabels (GPU, cached in the Volume), dataset build with blur
augmentation, the proven "long" fine-tune recipe, the full evaluation
(leaderboard scoring, NCNN deploy-truth, robustness stress), and the
best-model-wins deploy gate. The actual pipeline logic lives in
kitchen_training.pipeline (plus gate/consensus); this file only wraps it
in Modal functions and moves bytes. Results land in --out-dir:
summary.json, report.md, prelabels.json (fresh big-model boxes for the
review page), and kitchen_ncnn_model/ when the gate passes.

The Volume ("watchdoggy-train") holds the model weights (seed once with
kickoff --seed-models, from a machine that has models/yolo26?.pt), the
prelabel cache, and the last few run directories.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))  # kitchen_training

MINUTES = 60
GPU = os.environ.get("DOGGY_TRAIN_GPU", "L4")
VOL = Path("/vol")
PIPELINE_ROOT = VOL / "pipeline"

app = modal.App("watchdoggy-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.88", "opencv-python~=4.10",
                    "ncnn", "pnnx")
    .add_local_python_source("kitchen_training")
)

volume = modal.Volume.from_name("watchdoggy-train", create_if_missing=True)


def _tar_bytes(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(directory.rglob("*")):
            tar.add(path, arcname=path.relative_to(directory))
    return buffer.getvalue()


def _point_kitchen_training_at_volume(run_root: Path) -> None:
    """Must run BEFORE importing kitchen_training modules: config reads the
    KT_* environment at import time."""
    os.environ["KT_DATASET_DIR"] = str(run_root / "dataset-mirror")
    os.environ["KT_RUNS_DIR"] = str(PIPELINE_ROOT / "training-runs")
    os.environ["KT_BASE_MODEL"] = str(PIPELINE_ROOT / "models/yolo26n.pt")
    os.environ["KT_PRELABEL_MODEL"] = str(PIPELINE_ROOT / "models/yolo26x.pt")


# cpu/memory matter as much as the GPU here: the dataloader (decode +
# augmentation) is pure CPU, and Modal's default allocation starves it.
@app.function(image=image, gpu=GPU, cpu=8, memory=16384,
              volumes={str(VOL): volume}, timeout=180 * MINUTES)
def run_pipeline(run_name: str, recipe: dict,
                 fire_conf: float = 0.7) -> tuple[dict, bytes | None]:
    """The whole training day in one container. Returns (results, bundle_tar)."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.config import RUNS
    from kitchen_training.pipeline import full_run, prune_old_runs

    results, bundle = full_run(run_name, recipe, fire_conf, run_root)
    # Tar BEFORE pruning: if the Pi's clock regressed, the current run can
    # sort oldest and pruning first would delete the winning bundle.
    bundle_tar = _tar_bytes(bundle) if bundle else None
    prune_old_runs(RUNS)
    volume.commit()
    return results, bundle_tar


@app.function(image=image, gpu=GPU, cpu=4, memory=8192,
              volumes={str(VOL): volume}, timeout=30 * MINUTES)
def run_prelabels(run_name: str) -> dict:
    """GPU phase of the nightly pass: big-model boxes for new frames only."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.pipeline import prelabel_phase

    results = prelabel_phase(run_name)
    volume.commit()
    return results


@app.function(image=image, cpu=4, memory=8192,
              volumes={str(VOL): volume}, timeout=30 * MINUTES)
def run_consensus(run_name: str) -> dict:
    """CPU phase: the jury judges what needs judging (no GPU billed while
    a CPU model loops over frames)."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.pipeline import consensus_phase

    results = consensus_phase(run_name, run_root)
    volume.commit()
    return results


def _upload(dataset_dir: Path, deployed_dir: Path | None,
            seed_models: Path | None) -> str:
    run_name = time.strftime("%Y%m%d-%H%M%S")
    print(f"[pipeline] uploading {dataset_dir} as run {run_name} ...")
    with volume.batch_upload(force=True) as up:
        up.put_directory(str(dataset_dir), f"pipeline/runs/{run_name}/dataset-mirror")
        if deployed_dir and deployed_dir.is_dir():
            up.put_directory(str(deployed_dir),
                             f"pipeline/runs/{run_name}/deployed_ncnn_model")
        if seed_models:
            for name in ("yolo26n.pt", "yolo26x.pt"):
                up.put_file(str(seed_models / name), f"pipeline/models/{name}")
    return run_name


def _write_results(out_dir: Path, results: dict, bundle_tar: bytes | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {key: results[key] for key in
         ("run_name", "gate", "dataset", "recipe", "ncnn_truth",
          "robustness", "robustness_deployed", "slices", "calibration",
          "ncnn_heldout", "exam_suspects")}, indent=1))
    (out_dir / "report.md").write_text(results["report_md"])
    (out_dir / "prelabels.json").write_text(json.dumps(results["prelabels"]))
    if bundle_tar:
        with tarfile.open(fileobj=io.BytesIO(bundle_tar), mode="r:gz") as tar:
            tar.extractall(out_dir / "kitchen_ncnn_model")
    print(f"[pipeline] results -> {out_dir} "
          f"(deploy={'YES' if bundle_tar else 'no'})")


@app.local_entrypoint()
def kickoff(dataset_dir: str, out_dir: str, deployed_dir: str = "",
            seed_models: str = "", epochs: int = 200, batch: int = 16,
            freeze: int = 10, augment: bool = True,
            fire_conf: float = 0.7) -> None:
    """Full pipeline run. --seed-models <dir> uploads yolo26n/x.pt first
    (one-time, from a machine that has them). --fire-conf is the appliance's
    runtime alarm threshold: the gate judges at that operating point."""
    run_name = _upload(Path(dataset_dir),
                       Path(deployed_dir) if deployed_dir else None,
                       Path(seed_models) if seed_models else None)
    recipe = {"epochs": epochs, "batch": batch, "freeze": freeze,
              "augment": augment}
    print(f"[pipeline] running the full pipeline in the cloud "
          f"(gate @ {fire_conf}) ...")
    results, bundle_tar = run_pipeline.remote(run_name, recipe, fire_conf)
    _write_results(Path(out_dir), results, bundle_tar)


@app.local_entrypoint()
def kickoff_prelabels(dataset_dir: str, out_dir: str,
                      deployed_dir: str = "") -> None:
    """Prelabels-only run (fast): fresh ·x boxes for the review page, plus
    consensus auto-verdicts when --deployed-dir provides the second juror."""
    run_name = _upload(Path(dataset_dir),
                       Path(deployed_dir) if deployed_dir else None, None)
    print("[pipeline] prelabeling in the cloud ...")
    results = run_prelabels.remote(run_name)
    results.update(run_consensus.remote(run_name))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prelabels.json").write_text(json.dumps(results["prelabels"]))
    (out / "auto_verdicts.json").write_text(
        json.dumps(results.get("auto_verdicts", {})))
    (out / "disputes.json").write_text(json.dumps(results.get("disputes", {})))
    print(f"[pipeline] {len(results['prelabels'])} new frames prelabeled, "
          f"{len(results.get('auto_verdicts', {}))} auto-labeled, "
          f"{len(results.get('disputes', {}))} labels disputed -> {out}")
