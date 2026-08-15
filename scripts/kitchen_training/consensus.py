"""Consensus auto-labeling and the one-direction label audit.

The nano is the DEPLOYED kitchen specialist; X is the generic giant. The
jurors must agree before a frame skips the human queue -- except a
near-certain nano dog stands alone (the specialist out-sees X on dark and
half-hidden dogs it was specifically trained on). Genuine disagreement =
the frame stays for the human."""
from __future__ import annotations

import json
from pathlib import Path

from kitchen_training.config import DATASET_MIRROR

# Bars backtested 2026-08-14 against 917 human-labeled frames with the
# deployed champion as juror (scratch: rule_sim.py): this rule-set wrongly
# clears 1/917 dog frames (0.11% -- and that one is a suspected mislabel),
# fabricates 0 dogs, and covers ~36% of a fresh queue vs ~0% before. The
# old capture-reason gate ("innocent captures only") was retired the same
# day: it was designed for a weak jury that missed 26% of hard dogs and
# had become the sole bottleneck on autonomy. Safety nets that remain:
# auto-labels train-split only, disputable by every future candidate.
AUTO_DOG_NANO_CONF = 0.45    # dog when X agrees
AUTO_DOG_SOLO_CONF = 0.85    # dog on nano alone (X misses dark/hidden dogs)
AUTO_CLEAR_NANO_CONF = 0.15  # this quiet + no X dog = confident "no dog"
AUTO_PERSON_NANO_CONF = 0.5
# The label AUDIT keeps the old alarm-grade bar: it second-guesses HUMAN
# labels, and lowering it re-litigates frames the human had right.
AUDIT_DOG_NANO_CONF = 0.6

# The nano jury scan: sweep low so quietness is measurable, at the
# appliance's inference size.
NANO_SCAN_CONF = 0.1
NANO_IMAGE_SIZE = 640


def consensus_verdict(x_entry: dict, nano_dog: float,
                      nano_person: float) -> str | None:
    x_dog = bool(x_entry["dogs"])
    x_person = bool(x_entry["people"])
    if x_dog and nano_dog >= AUTO_DOG_NANO_CONF:
        return "dog"
    if nano_dog >= AUTO_DOG_SOLO_CONF:
        return "dog"
    if not x_dog and nano_dog <= AUTO_CLEAR_NANO_CONF:
        if x_person or nano_person >= AUTO_PERSON_NANO_CONF:
            return "person"
        return "empty"
    return None  # the jurors disagree: a human decides


def _sidecar_meta(stem: str) -> dict:
    sidecar = DATASET_MIRROR / f"{stem}.json"
    return json.loads(sidecar.read_text()) if sidecar.is_file() else {}


def _nano_scores(nano, stem: str) -> tuple[float, float]:
    """(max dog conf, max person conf) from the deployed nano on one frame."""
    prediction = nano.predict(str(DATASET_MIRROR / f"{stem}.jpg"),
                              conf=NANO_SCAN_CONF, imgsz=NANO_IMAGE_SIZE,
                              device="cpu", verbose=False)[0]
    nano_dog, nano_person = 0.0, 0.0
    for box in prediction.boxes:
        label = prediction.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if label == "dog":
            nano_dog = max(nano_dog, conf)
        if label == "person":
            nano_person = max(nano_person, conf)
    return nano_dog, nano_person


def _audit_dispute(meta: dict, x_entry: dict, nano_dog: float) -> dict | None:
    """Label audit, ONE direction only: a confident dog sighting on a
    non-dog label usually means a real dog the human missed. The reverse
    direction ("dog-labeled but we see nothing") was removed: the jury
    misses ~26% of hard borderline dogs, so it mostly second-guessed labels
    the human had right. A human who already arbitrated a dispute is not
    asked twice (dispute_settled_at)."""
    if meta.get("dispute_settled_at"):
        return None
    if meta.get("human_label") not in ("person", "empty", "no_dog"):
        return None
    if nano_dog >= AUDIT_DOG_NANO_CONF and x_entry["dogs"]:
        return {"model_says": "dog", "nano_conf": round(nano_dog, 3)}
    return None


def judge_frames(stems: list[str], cache: dict,
                 deployed_dir: Path) -> tuple[dict, dict]:
    """(auto_verdicts, disputes) for every frame, judged by the deployed
    nano + the big-model cache. Without a deployed nano there is no second
    juror, so nothing is auto-labeled or disputed."""
    if not deployed_dir.is_dir():
        return {}, {}
    from ultralytics import YOLO  # deferred, like every kitchen_training module
    nano = YOLO(str(deployed_dir), task="detect")
    auto_verdicts: dict = {}
    disputes: dict = {}
    for stem in stems:
        meta = _sidecar_meta(stem)
        nano_dog, nano_person = _nano_scores(nano, stem)
        if not meta.get("human_label") and not meta.get("auto_label"):
            verdict = consensus_verdict(cache[stem], nano_dog, nano_person)
            if verdict:
                auto_verdicts[stem] = verdict
            continue
        dispute = _audit_dispute(meta, cache[stem], nano_dog)
        if dispute:
            disputes[stem] = dispute
    print(f"[pipeline] consensus auto-labeled {len(auto_verdicts)} "
          f"unlabeled frames; disputed {len(disputes)} existing labels",
          flush=True)
    return auto_verdicts, disputes
