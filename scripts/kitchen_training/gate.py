"""The deploy gate and the exam's self-audit -- who ships, and which
held-out labels the candidate wants a human to re-check."""
from __future__ import annotations

import json
from pathlib import Path

from kitchen_training.config import DATASET_MIRROR
from kitchen_training.evaluation import curve, dog_confs, eval_frames, score

# The exam's own audit: yesterday's jurors can't catch label errors that only
# the newest model can see, so every candidate disputes the held-out frames it
# confidently disagrees with -- proven necessity: a candidate's "false fires"
# turned out to be sleeping/edge dogs the human and both old jurors missed.
SUSPECT_FIRE_CONF = 0.7   # candidate very sure a "non-dog" frame has a dog


def deploy_gate(run_dir: Path, new_bundle: Path, deployed_dir: Path,
                fire_conf: float) -> dict:
    """The best model wins, period. Both bundles sit the IDENTICAL current
    exam at the appliance's runtime threshold; fewest total held-out errors
    (missed dogs + false fires) deploys. False fires break ties and are
    always displayed, but they are an indicator, not a veto -- the old
    zero-FP hard rule kept vetoing models that caught 20+ more dogs over
    noise-level FP differences on a growing exam. Freshness wins exact
    ties: the challenger trained on newer data."""
    from ultralytics import YOLO  # deferred, like every kitchen_training module
    frames = eval_frames(run_dir)
    new_confs = dog_confs(YOLO(str(new_bundle), task="detect"), frames, "cpu")
    new_score = score(frames, new_confs, fire_conf, True)
    gate = {"fire_conf": fire_conf, "new": new_score,
            "new_errors": new_score["missed"] + new_score["false_fires"]}
    if not deployed_dir.is_dir():
        return {**gate, "deploy": True, "deployed": None, "deployed_errors": None,
                "reason": "no deployed bundle uploaded",
                "deployed_curve": None}
    old_confs = dog_confs(YOLO(str(deployed_dir), task="detect"), frames, "cpu")
    old_score = score(frames, old_confs, fire_conf, True)
    old_errors = old_score["missed"] + old_score["false_fires"]
    passes = (gate["new_errors"] < old_errors
              or (gate["new_errors"] == old_errors
                  and new_score["false_fires"] <= old_score["false_fires"]))
    reason = (f"{gate['new_errors']} errors vs deployed's {old_errors} "
              f"at {fire_conf}"
              + (" -- best model wins" if passes else " -- incumbent stands"))
    return {**gate, "deploy": passes, "deployed": old_score,
            "deployed_errors": old_errors, "reason": reason,
            # The incumbent's full curve rides along so every report compares
            # both models on the same grown exam, not just at one threshold.
            "deployed_curve": curve(frames, old_confs)}


def exam_suspects(run_dir: Path, bundle: Path) -> dict:
    from ultralytics import YOLO  # deferred, like every kitchen_training module
    model = YOLO(str(bundle), task="detect")
    frames = [f for f in eval_frames(run_dir) if f.heldout]
    confs = dog_confs(model, frames, "cpu")
    suspects = {}
    for frame in frames:
        meta = json.loads((DATASET_MIRROR / f"{frame.stem}.json").read_text())
        if meta.get("dispute_settled_at"):
            continue  # a human already arbitrated this one
        conf = confs[frame.stem]
        # One direction only: confident dog sightings on non-dog labels.
        # (Models doubting hard dog frames is expected, not suspicious.)
        if not frame.is_dog and conf >= SUSPECT_FIRE_CONF:
            suspects[frame.stem] = {"model_says": "dog", "nano_conf": conf}
    print(f"[pipeline] candidate disputes {len(suspects)} held-out labels",
          flush=True)
    return suspects
