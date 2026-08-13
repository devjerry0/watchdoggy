"""The whole training pipeline as one Modal job, triggerable from the Pi.

    modal run scripts/modal_pipeline.py::kickoff --dataset-dir <frames> \
        --deployed-dir <current_ncnn_bundle> --out-dir <results>

The caller (the Pi's trainer daemon, or a Mac) uploads the raw labeled
frames and the currently-deployed bundle; the cloud does EVERYTHING else:
big-model prelabels (GPU, cached in the Volume), dataset build with blur
augmentation, the proven "long" fine-tune recipe, the full evaluation
(leaderboard scoring, NCNN deploy-truth, robustness stress), and the deploy
gate -- ship only if the new bundle beats-or-ties the deployed one on the
current held-out exam with zero false fires. Results land in --out-dir:
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
KEEP_RUNS = 5

# The proven recipe from the sweep: "long" (200 epochs) won on the full
# dataset with augmentation. The training page can override per run; the
# 10-config sweep stays available manually via train_kitchen_model.py --sweep.
DEFAULT_RECIPE = {"epochs": 200, "batch": 16, "freeze": 10, "augment": True}

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


def _fresh_prelabels(cache_before: set[str]) -> dict:
    from kitchen_training.config import PRELABEL_CACHE
    cache = json.loads(PRELABEL_CACHE.read_text())
    fresh = {}
    for stem in set(cache) - cache_before:
        entry = cache[stem]
        fresh[stem] = ([{"label": "dog", "box": box} for box in entry["dogs"]]
                       + [{"label": "person", "box": box}
                          for box in entry["people"]])
    return fresh


def _deploy_gate(run_dir: Path, new_bundle: Path, deployed_dir: Path) -> dict:
    """Ship only a strict non-regression: zero held-out false fires and at
    least as many held-out catches as the currently-deployed bundle, scored
    on the SAME current exam."""
    from ultralytics import YOLO
    from kitchen_training.config import FIRE_CONF
    from kitchen_training.evaluation import dog_confs, eval_frames, score

    frames = eval_frames(run_dir)
    new_score = score(frames, dog_confs(
        YOLO(str(new_bundle), task="detect"), frames, "cpu"), FIRE_CONF, True)
    if not deployed_dir.is_dir():
        return {"deploy": new_score["false_fires"] == 0, "new": new_score,
                "deployed": None, "reason": "no deployed bundle uploaded"}
    old_score = score(frames, dog_confs(
        YOLO(str(deployed_dir), task="detect"), frames, "cpu"), FIRE_CONF, True)
    passes = (new_score["false_fires"] == 0
              and new_score["caught"] >= old_score["caught"])
    reason = ("beats-or-ties deployed with zero false fires" if passes
              else "does not beat deployed bundle")
    return {"deploy": passes, "new": new_score, "deployed": old_score,
            "reason": reason}


def _prune_old_runs(runs_dir: Path) -> None:
    import shutil
    run_dirs = sorted(p for p in runs_dir.glob("*") if p.is_dir())
    for stale in run_dirs[:-KEEP_RUNS]:
        shutil.rmtree(stale, ignore_errors=True)


@app.function(image=image, gpu=GPU, volumes={str(VOL): volume},
              timeout=120 * MINUTES)
def run_pipeline(run_name: str, recipe: dict) -> tuple[dict, bytes | None]:
    """The whole training day in one container. Returns (results, bundle_tar)."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.build import build
    from kitchen_training.config import BASE_MODEL, PRELABEL_CACHE, RUNS
    from kitchen_training.evaluation import evaluate, ncnn_truth, robustness
    from kitchen_training.export import export
    from kitchen_training.report import report
    from kitchen_training.training import train

    recipe = {**DEFAULT_RECIPE, **recipe}
    print(f"[pipeline] recipe: {recipe}", flush=True)
    cache_before = set(json.loads(PRELABEL_CACHE.read_text())
                       if PRELABEL_CACHE.is_file() else {})
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = build(run_dir, augment=recipe["augment"])
    best = train(run_dir, "local", epochs=recipe["epochs"],
                 batch=recipe["batch"], freeze=recipe["freeze"])
    weights_by_name = {"fine-tune": best, "baseline": BASE_MODEL}
    metrics = evaluate(run_dir, weights_by_name)
    bundle = export(run_dir, best)
    metrics["ncnn_truth"] = ncnn_truth(run_dir, bundle)
    metrics["robustness"] = robustness(run_dir, bundle)
    gate = _deploy_gate(run_dir, bundle, run_root / "deployed_ncnn_model")
    report(run_dir, dataset_stats, metrics, "fine-tune", bundle)

    results = {
        "run_name": run_name,
        "gate": gate,
        "prelabels": _fresh_prelabels(cache_before),
        "report_md": (run_dir / "report.md").read_text(),
        "dataset": {"train": dataset_stats["train"],
                    "val": dataset_stats["val"],
                    "dropped": [stem for stem, _ in dataset_stats["dropped"]]},
        "ncnn_heldout": metrics["ncnn_truth"]["0.7"]["heldout"],
    }
    bundle_tar = _tar_bytes(bundle) if gate["deploy"] else None
    _prune_old_runs(RUNS)
    volume.commit()
    return results, bundle_tar


@app.function(image=image, gpu=GPU, volumes={str(VOL): volume},
              timeout=30 * MINUTES)
def run_prelabels(run_name: str) -> dict:
    """Prelabels only: big-model boxes for every uploaded frame that the
    cache hasn't seen. Cheap; keeps the review page stocked with ·x boxes."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.config import DATASET_MIRROR, PRELABEL_CACHE
    from kitchen_training.dataset import prelabel

    cache_before = set(json.loads(PRELABEL_CACHE.read_text())
                       if PRELABEL_CACHE.is_file() else {})
    stems = [jpg.stem for jpg in sorted(DATASET_MIRROR.glob("sample_*.jpg"))]
    prelabel(stems)
    volume.commit()
    return {"run_name": run_name, "prelabels": _fresh_prelabels(cache_before)}


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
         ("run_name", "gate", "dataset", "ncnn_heldout")}, indent=1))
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
            freeze: int = 10, augment: bool = True) -> None:
    """Full pipeline run. --seed-models <dir> uploads yolo26n/x.pt first
    (one-time, from a machine that has them)."""
    run_name = _upload(Path(dataset_dir),
                       Path(deployed_dir) if deployed_dir else None,
                       Path(seed_models) if seed_models else None)
    recipe = {"epochs": epochs, "batch": batch, "freeze": freeze,
              "augment": augment}
    print("[pipeline] running the full pipeline in the cloud ...")
    results, bundle_tar = run_pipeline.remote(run_name, recipe)
    _write_results(Path(out_dir), results, bundle_tar)


@app.local_entrypoint()
def kickoff_prelabels(dataset_dir: str, out_dir: str) -> None:
    """Prelabels-only run (fast): fresh ·x boxes for the review page."""
    run_name = _upload(Path(dataset_dir), None, None)
    print("[pipeline] prelabeling in the cloud ...")
    results = run_prelabels.remote(run_name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prelabels.json").write_text(json.dumps(results["prelabels"]))
    print(f"[pipeline] {len(results['prelabels'])} new frames prelabeled "
          f"-> {out / 'prelabels.json'}")
