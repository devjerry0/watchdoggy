"""The two cloud runs, as plain functions: the full training day and the
nightly prelabel/consensus pass. `modal_pipeline.py` wraps these in Modal
functions; everything here is infrastructure-free and runs wherever the
KT_* environment points it.

IMPORTANT: this module (and everything it imports) captures the KT_* paths
at import time via kitchen_training.config -- import it only AFTER the
environment is pointed at the right roots (see modal_pipeline's
_point_kitchen_training_at_volume)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from kitchen_training.build import build
from kitchen_training.config import BASE_MODEL, DATASET_MIRROR, PRELABEL_CACHE, RUNS
from kitchen_training.consensus import judge_frames
from kitchen_training.dataset import prelabel
from kitchen_training.evaluation import eval_frames, evaluate, ncnn_truth, robustness
from kitchen_training.export import export
from kitchen_training.gate import deploy_gate, exam_suspects
from kitchen_training.report import report
from kitchen_training.slices import calibration, slice_report
from kitchen_training.training import train

# The proven recipe from the sweep: "long" (200 epochs) won on the full
# dataset with augmentation. The training page can override per run; the
# 10-config sweep stays available manually via train_kitchen_model.py --sweep.
DEFAULT_RECIPE = {"epochs": 200, "batch": 16, "freeze": 10, "augment": True}
KEEP_RUNS = 5


def cached_stems() -> set[str]:
    return set(json.loads(PRELABEL_CACHE.read_text())
               if PRELABEL_CACHE.is_file() else {})


def fresh_prelabels(cache_before: set[str]) -> dict:
    cache = json.loads(PRELABEL_CACHE.read_text())
    fresh = {}
    for stem in set(cache) - cache_before:
        entry = cache[stem]
        fresh[stem] = ([{"label": "dog", "box": box} for box in entry["dogs"]]
                       + [{"label": "person", "box": box}
                          for box in entry["people"]])
    return fresh


def prune_old_runs(runs_dir: Path) -> None:
    run_dirs = sorted(p for p in runs_dir.glob("*") if p.is_dir())
    for stale in run_dirs[:-KEEP_RUNS]:
        shutil.rmtree(stale, ignore_errors=True)


def full_run(run_name: str, recipe: dict, fire_conf: float,
             run_root: Path) -> tuple[dict, Path | None]:
    """Build -> train -> evaluate -> export -> gate -> report. Returns the
    results payload and the NCNN bundle path when the gate says deploy."""
    recipe = {**DEFAULT_RECIPE, **recipe}
    print(f"[pipeline] recipe: {recipe}", flush=True)
    cache_before = cached_stems()
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = build(run_dir, augment=recipe["augment"])
    best = train(run_dir, "local", epochs=recipe["epochs"],
                 batch=recipe["batch"], freeze=recipe["freeze"])
    metrics = evaluate(run_dir, {"fine-tune": best, "baseline": BASE_MODEL})
    bundle = export(run_dir, best)
    metrics["ncnn_truth"] = ncnn_truth(run_dir, bundle)
    metrics["robustness"] = robustness(run_dir, bundle)
    deployed_dir = run_root / "deployed_ncnn_model"
    gate = deploy_gate(run_dir, bundle, deployed_dir, fire_conf)
    new_confs = gate.pop("_new_confs")
    deployed_confs = gate.pop("_deployed_confs", None)
    exam = eval_frames(run_dir)
    metrics["slices"] = slice_report(exam, new_confs, deployed_confs, fire_conf)
    metrics["calibration"] = calibration(exam, new_confs)
    if deployed_dir.is_dir():
        # The incumbent runs the same stress suite: comparisons stay
        # apples-to-apples as the exam grows.
        metrics["robustness_deployed"] = robustness(run_dir, deployed_dir)
    suspects = exam_suspects(run_dir, bundle)
    report(run_dir, dataset_stats, metrics, "fine-tune", bundle)

    results = {
        "run_name": run_name,
        "gate": gate,
        "prelabels": fresh_prelabels(cache_before),
        "report_md": (run_dir / "report.md").read_text(),
        "dataset": {"train": dataset_stats["train"],
                    "val": dataset_stats["val"],
                    "augmented": dataset_stats["augmented"],
                    "hand_boxed": dataset_stats["hand_boxed"],
                    "dropped": [stem for stem, _ in dataset_stats["dropped"]]},
        "recipe": recipe,
        "fire_conf": fire_conf,
        "ncnn_truth": metrics["ncnn_truth"],
        "robustness": metrics["robustness"],
        "robustness_deployed": metrics.get("robustness_deployed"),
        "slices": metrics["slices"],
        "calibration": metrics["calibration"],
        # The headline score IS the gate's new-model score: measured at the
        # appliance's runtime threshold, not a fixed default.
        "ncnn_heldout": gate["new"],
        "exam_suspects": suspects,
    }
    # NOTE: no pruning here -- the caller tars the bundle first, THEN calls
    # prune_old_runs. Pruning first could delete the current run dir when the
    # Pi's clock regressed (run names sort by timestamp), losing the bundle
    # of a gate-passing run before it was captured.
    return results, (bundle if gate["deploy"] else None)


def _mirror_stems() -> list[str]:
    return [jpg.stem for jpg in sorted(DATASET_MIRROR.glob("sample_*.jpg"))]


def prelabel_phase(run_name: str) -> dict:
    """GPU phase: big-model boxes for frames the cache has never seen."""
    cache_before = cached_stems()
    prelabel(_mirror_stems())
    return {"run_name": run_name, "prelabels": fresh_prelabels(cache_before)}


def consensus_phase(run_name: str, run_root: Path) -> dict:
    """CPU phase: consensus auto-verdicts and label-audit disputes when a
    deployed nano provides the second juror. Runs without a GPU -- holding
    one through the judging loop was most of a pass's bill."""
    stems = _mirror_stems()
    cache = prelabel(stems)  # fully cached by the prelabel phase
    auto_verdicts, disputes = judge_frames(stems, cache,
                                           run_root / "deployed_ncnn_model")
    return {"run_name": run_name, "auto_verdicts": auto_verdicts,
            "disputes": disputes}
