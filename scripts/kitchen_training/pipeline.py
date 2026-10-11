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
import os
import shutil
from pathlib import Path

from kitchen_training.build import build
from kitchen_training.config import (BASE_MODEL, CHAMPION_MODEL, PRELABEL_CACHE,
                                     RUNS, mirror_glob)
from kitchen_training.consensus import finish_jury, judge_stems, jury_plan
from kitchen_training.dataset import prelabel
from kitchen_training.evaluation import eval_frames, evaluate, ncnn_truth, robustness
from kitchen_training.export import export
from kitchen_training.gate import deploy_gate, exam_suspects
from kitchen_training.hygiene import (fit_epochs, janitor, mark_completed,
                                      strip_dataset)
from kitchen_training.report import report
from kitchen_training.slices import calibration, slice_report
from kitchen_training.training import start_weights, train

# The proven recipe on the grown corpus: 80 epochs (the old 300-frame-era
# "long" 200-epoch sweep winner predates the v1.0.17 retune; hygiene's
# EPOCH_IMAGE_BUDGET is anchored to 80 x ~2.7k and caps the effective epochs
# anyway). The training page can override per run; the 10-config sweep stays
# available manually via train_kitchen_model.py --sweep.
DEFAULT_RECIPE = {"epochs": 120, "batch": 16, "freeze": 10, "augment": "auto"}
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


def promote_champion(best: Path) -> None:
    """The deployed bundle's weights become the next run's starting point.
    Copy-then-rename: a crash mid-copy never leaves a torn champion."""
    CHAMPION_MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHAMPION_MODEL.with_suffix(".pt.tmp")
    shutil.copyfile(best, tmp)
    os.replace(tmp, CHAMPION_MODEL)
    print(f"[pipeline] champion updated from {best}", flush=True)


def pick_start(run_dir: Path, warm_best: Path, recipe: dict,
               deployed_dir: Path, fire_conf: float) -> tuple[Path, dict]:
    """Warm start vs fresh-from-base on the SAME dataset (Ash & Adams 2020:
    warm-started nets can generalize worse). Both sit the exam; the one with
    fewer errors (then fewer false fires; ties keep warm) goes on to the
    normal gate. Both scores ride along in the recipe."""
    fresh_best = train(run_dir, "local", epochs=recipe["epochs"],
                       batch=recipe["batch"], freeze=recipe["freeze"],
                       init=BASE_MODEL, tag="_fresh")
    scores = {}
    for name, weights in (("warm", warm_best), ("fresh", fresh_best)):
        gate = deploy_gate(run_dir, export(run_dir, weights, f"{name}_ncnn_model"),
                           deployed_dir, fire_conf)
        scores[name] = {"errors": gate["new_errors"], "caught": gate["new"]["caught"],
                        "false_fires": gate["new"]["false_fires"]}
    picked = min(scores, key=lambda n: (scores[n]["errors"], scores[n]["false_fires"]))
    print(f"[pipeline] start compare: warm {scores['warm']} vs fresh "
          f"{scores['fresh']} -> {picked}", flush=True)
    weights = fresh_best if picked == "fresh" else warm_best
    return weights, {**recipe, "init_compare": {**scores, "picked": picked},
                     "init": (BASE_MODEL if picked == "fresh" else CHAMPION_MODEL).name}


def full_run(run_name: str, recipe: dict, fire_conf: float,
             run_root: Path) -> tuple[dict, Path | None]:
    """Build -> train -> evaluate -> export -> gate -> report. Returns the
    results payload and the NCNN bundle path when the gate says deploy."""
    recipe = {**DEFAULT_RECIPE, **recipe}
    print(f"[pipeline] recipe: {recipe}", flush=True)
    cache_before = cached_stems()
    run_dir = RUNS / run_name
    # Janitor BEFORE the first write: a full Volume must heal itself here,
    # not depend on a successful run that a full Volume prevents.
    for stale in janitor(RUNS, KEEP_RUNS, run_name, run_root.parent):
        print(f"[pipeline] janitor: removed stale {stale}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = build(run_dir, augment=recipe["augment"])
    # Every image the trainer will actually iterate per epoch: base frames,
    # blur-augmented copies, and the extra human-weight copies.
    corpus = (dataset_stats["train"] + dataset_stats["augmented"]
              + dataset_stats.get("human_weighted", 0))
    ceiling = float(os.environ.get("KT_JOB_CEILING_SECONDS", 10 * 3600))
    # One-off experiment: a fresh model trains beside the warm start.
    compare = bool(recipe.get("compare_init")) and CHAMPION_MODEL.is_file()
    epochs = fit_epochs(recipe["epochs"], corpus * (2 if compare else 1),
                        ceiling, recipe.get("gpu", ""), recipe["batch"])
    if epochs != recipe["epochs"]:
        print(f"[pipeline] {corpus} train images cannot finish "
              f"{recipe['epochs']} epochs inside the {ceiling / 3600:.0f}h "
              f"ceiling: capping to {epochs}", flush=True)
        recipe = {**recipe, "epochs_requested": recipe["epochs"],
                  "epochs": epochs}
    recipe = {**recipe, "augment_mode": dataset_stats["augment_mode"],
              "init": start_weights().name}
    best = train(run_dir, "local", epochs=recipe["epochs"],
                 batch=recipe["batch"], freeze=recipe["freeze"])
    if compare:
        best, recipe = pick_start(run_dir, best, recipe,
                                  run_root / "deployed_ncnn_model", fire_conf)
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
    if gate["deploy"]:
        promote_champion(best)
    # The built dataset (2+ files per frame) is what hit the Volume's
    # 500k-inode limit; the bundle, weights and report stay for forensics.
    strip_dataset(run_dir)
    mark_completed(run_dir)
    # NOTE: no pruning here -- the caller tars the bundle first, THEN calls
    # prune_old_runs. Pruning first could delete the current run dir when the
    # Pi's clock regressed (run names sort by timestamp), losing the bundle
    # of a gate-passing run before it was captured.
    return results, (bundle if gate["deploy"] else None)


def _mirror_stems() -> list[str]:
    return [jpg.stem for jpg in sorted(mirror_glob("sample_*.jpg"))]


def prelabel_phase(run_name: str, run_root: Path | None = None) -> dict:
    """GPU phase: big-model boxes for frames the cache has never seen."""
    if run_root is not None:
        # Janitor here, not only in consensus: the consensus phase runs AFTER
        # this one, so on bad weeks (prelabels failing) it would never fire.
        for stale in janitor(RUNS, KEEP_RUNS, run_name, run_root.parent):
            print(f"[pipeline] janitor: removed stale {stale}", flush=True)
    cache_before = cached_stems()
    prelabel(_mirror_stems())
    return {"run_name": run_name, "prelabels": fresh_prelabels(cache_before)}


def jury_plan_phase(run_name: str, run_root: Path) -> dict:
    """CPU: which frames the jury still has to judge (after labeling)."""
    # Weekly janitor pass: labeling jobs leave kickoff upload dirs too, and
    # on bad weeks (every run failing) nothing else would ever clean up.
    for stale in janitor(RUNS, KEEP_RUNS, run_name, run_root.parent):
        print(f"[pipeline] janitor: removed stale {stale}", flush=True)
    stems = _mirror_stems()
    cache = prelabel(stems)  # fully cached by the labeling phase
    return jury_plan(stems, cache, run_root / "deployed_ncnn_model")


def judge_chunk_phase(run_root: Path, stems: list[str]) -> dict:
    """CPU: judge one chunk of the plan (many containers run in parallel).
    The plan only lists frames the labeling phase already cached, so read
    the cache directly (prelabel() would log a line per container)."""
    cache = json.loads(PRELABEL_CACHE.read_text())
    return judge_stems(stems, cache, run_root / "deployed_ncnn_model")


def finish_jury_phase(run_name: str, plan: dict, parts: list[dict]) -> dict:
    auto_verdicts, disputes = finish_jury(plan, parts)
    return {"run_name": run_name, "auto_verdicts": auto_verdicts,
            "disputes": disputes}
