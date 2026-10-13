"""The job runners: a prelabel pass, a full gated training run, and a
labeler run (self-updates live in trainer_daemon.update)."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from trainer_daemon.apply import (
    apply_cloud_results,
    apply_exam_suspects,
    install_bundle,
    labeler_tag,
)
from trainer_daemon.billing import billing_summary, run_cost
from trainer_daemon.cloud import batch, modal_run
from trainer_daemon.env import DEPLOYED_BUNDLE, DOGGY_ROOT, JOBS_DIR, log, settings
from trainer_daemon.queue import jobs, queue_auto
from trainer_daemon.update import run_update_job

FALLBACK_FIRE_CONF = 0.7
# kitchen_training.labeler's defaults (the trainer venv cannot import it).
# The cloud trains the labeler at its own LABELER_BATCH.
LABELER_EPOCHS = 100
LABELER_AUTO_SAMPLE = 8000


def run_prelabel_job(job: dict) -> str:
    before = billing_summary()
    with tempfile.TemporaryDirectory() as tmp:
        arguments = ["--out-dir", tmp]
        if DEPLOYED_BUNDLE.is_dir():
            arguments += ["--deployed-dir", str(DEPLOYED_BUNDLE)]
        modal_run("kickoff_prelabels", arguments, job["id"])
        merged, autos, disputes, cleared = apply_cloud_results(
            Path(tmp) / "prelabels.json",
            Path(tmp) / "auto_verdicts.json",
            Path(tmp) / "disputes.json", Path(tmp) / "clear_auto.json",
            model=labeler_tag(Path(tmp)))
    cost = run_cost(before)
    return (f"{merged} frames prelabeled, {autos} auto-labeled, "
            f"{disputes} labels disputed, {cleared} auto-labels cleared"
            + (f" (${cost:.2f})" if cost is not None else ""))


def runtime_confidence() -> float:
    """The appliance's live alarm threshold: the gate must judge there.
    settings.json is the tunables' home (doggy.core.settings_store); the
    .env line is the pre-migration fallback. Read directly, not via the
    doggy package: the trainer venv has no doggy install."""
    settings_file = DOGGY_ROOT / "settings.json"
    try:
        return float(json.loads(settings_file.read_text())["tunables"]["confidence"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
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
            "--augment", augment_mode(recipe["augment"])] + (
                ["--compare-init"] if recipe.get("compare_init") is True else [])


def augment_mode(value: bool | str) -> str:
    """Settings files written before the modes existed hold a boolean."""
    if value is True:
        return "auto"
    if value is False:
        return "off"
    return value


def _scorecard(summary: dict, started: float, before: dict | None) -> dict:
    # The scorecard the training page renders: gate compare, threshold
    # curve, robustness, dataset shape, recipe, duration.
    card = {key: summary.get(key) for key in
            ("gate", "dataset", "recipe", "fire_conf",
             "ncnn_truth", "robustness", "robustness_deployed",
             "slices", "calibration")}
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
        merged = apply_cloud_results(out / "prelabels.json",
                                     model=labeler_tag(out))[0]
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


def run_labeler_job(job: dict) -> str:
    """Fine-tune the labeler in the cloud. Its recipe is its own: the small
    model's saved epochs and batch never apply. On a swap the cloud has
    already written labeler.pt, so this queues the labeling pass that
    relabels every frame and re-judges the auto-labels (the weekly
    schedule would not queue it on its own)."""
    before = billing_summary()
    params = job.get("params") or {}
    with tempfile.TemporaryDirectory() as tmp:
        arguments = ["--out-dir", tmp,
                     "--epochs", str(params.get("epochs", LABELER_EPOCHS)),
                     "--auto-sample",
                     str(params.get("auto_sample", LABELER_AUTO_SAMPLE))]
        if DEPLOYED_BUNDLE.is_dir():
            arguments += ["--deployed-dir", str(DEPLOYED_BUNDLE)]
        modal_run("kickoff_labeler", arguments, job["id"])
        out = Path(tmp)
        results = json.loads((out / "labeler-run.json").read_text())
        apply_exam_suspects(results)
        # Queue the relabel first: everything after it is bookkeeping.
        if results["promoted"]:
            relabel = _queue_relabel()
        (JOBS_DIR / f"{job['id']}.report.md").write_text(
            (out / "report.md").read_text())
    if results["decision"] == "failed":
        raise RuntimeError(f"labeler run failed: {results['reason']}")
    cost = run_cost(before)
    detail = (f"SWAPPED in the fine-tuned labeler ({results['reason']}); {relabel}"
              if results["promoted"]
              else f"kept the active labeler ({results['reason']})")
    return detail + (f" (${cost:.2f})" if cost is not None else "")


def _queue_relabel() -> str:
    """One labeling pass after a swap, reusing one already waiting."""
    if any(j.get("kind") == "prelabel" and j.get("status") in ("queued", "running")
           for j in jobs()):
        return "a labeling pass was already queued"
    queue_auto("prelabel", "labeler swapped: relabel every frame")
    return "labeling pass queued"


RUNNERS = {"prelabel": run_prelabel_job, "train": run_train_job,
           "labeler": run_labeler_job, "update": run_update_job}
