"""The two job runners: a prelabel pass and a full gated training run."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from trainer_daemon.apply import (
    apply_auto_verdicts,
    apply_disputes,
    apply_exam_suspects,
    install_bundle,
    merge_prelabels,
)
from trainer_daemon.cloud import batch, billing_summary, modal_run, run_cost
from trainer_daemon.env import DEPLOYED_BUNDLE, DOGGY_ROOT, JOBS_DIR, log, settings

FALLBACK_FIRE_CONF = 0.7


def run_prelabel_job(job: dict) -> str:
    before = billing_summary()
    with tempfile.TemporaryDirectory() as tmp:
        arguments = ["--out-dir", tmp]
        if DEPLOYED_BUNDLE.is_dir():
            arguments += ["--deployed-dir", str(DEPLOYED_BUNDLE)]
        modal_run("kickoff_prelabels", arguments, job["id"])
        merged = merge_prelabels(Path(tmp) / "prelabels.json")
        autos = apply_auto_verdicts(Path(tmp) / "auto_verdicts.json")
        disputes = apply_disputes(Path(tmp) / "disputes.json")
    cost = run_cost(before)
    return (f"{merged} frames prelabeled, {autos} auto-labeled, "
            f"{disputes} labels disputed"
            + (f" (${cost:.2f})" if cost is not None else ""))


def runtime_confidence() -> float:
    """The appliance's live alarm threshold: the gate must judge there."""
    env_file = DOGGY_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("DOGGY_CONFIDENCE="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    break
    return FALLBACK_FIRE_CONF


def recipe_arguments(job: dict) -> list[str]:
    recipe = {**settings(), **(job.get("params") or {})}
    return ["--epochs", str(recipe["epochs"]), "--batch", str(batch(job)),
            "--freeze", str(recipe["freeze"]),
            "--fire-conf", str(runtime_confidence()),
            "--augment" if recipe["augment"] else "--no-augment"]


def _scorecard(summary: dict, started: float, before: dict | None) -> dict:
    # The scorecard the training page renders: gate compare, threshold
    # curve, robustness, dataset shape, recipe, duration.
    card = {key: summary.get(key) for key in
            ("gate", "dataset", "recipe", "fire_conf",
             "ncnn_truth", "robustness", "robustness_deployed")}
    card["duration_s"] = round(time.time() - started)
    card["cost_usd"] = run_cost(before)
    return card


def run_train_job(job: dict) -> str:
    started = time.time()
    before = billing_summary()
    with tempfile.TemporaryDirectory() as tmp:
        arguments = ["--out-dir", tmp] + recipe_arguments(job)
        if DEPLOYED_BUNDLE.is_dir():
            arguments += ["--deployed-dir", str(DEPLOYED_BUNDLE)]
        modal_run("kickoff", arguments, job["id"])
        out = Path(tmp)
        merged = merge_prelabels(out / "prelabels.json")
        summary = json.loads((out / "summary.json").read_text())
        apply_exam_suspects(summary)
        (JOBS_DIR / f"{job['id']}.report.md").write_text(
            (out / "report.md").read_text())
        job["_summary"] = _scorecard(summary, started, before)
        gate = summary["gate"]
        heldout = summary["ncnn_heldout"]
        verdict = (f"held-out {heldout['caught']}/"
                   f"{heldout['caught'] + heldout['missed']} catches, "
                   f"{heldout['false_fires']} FP")
        if not gate["deploy"]:
            return f"kept current model ({gate['reason']}; new: {verdict})"
        install_bundle(out / "kitchen_ncnn_model")
    if merged:
        log(f"merged prelabels for {merged} frames")
    return f"DEPLOYED new model ({verdict})"


RUNNERS = {"prelabel": run_prelabel_job, "train": run_train_job}
