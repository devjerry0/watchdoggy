"""Consensus auto-labeling and the one-direction label audit.

The nano is the DEPLOYED kitchen specialist; X is the generic giant. The
jurors must agree before a frame skips the human queue -- except a
near-certain nano dog stands alone (the specialist out-sees X on dark and
half-hidden dogs it was specifically trained on). Genuine disagreement =
the frame stays for the human."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kitchen_training.config import DATASET_MIRROR, RUNS

# A jury's verdict on an unchanged frame cannot change until the jury does.
# Frames this exact jury (deployed weights + rule constants) has already
# judged with no action are remembered here and skipped -- re-judging the
# whole refused backlog every pass was most of a pass's compute. A new
# champion or a rule change voids the memory automatically.
JURY_MEMORY = RUNS / "jury-no-action.json"

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


def _jury_id(deployed_dir: Path) -> str:
    """The jury's identity: deployed weights + rule constants."""
    digest = hashlib.sha256(repr((
        AUTO_DOG_NANO_CONF, AUTO_DOG_SOLO_CONF, AUTO_CLEAR_NANO_CONF,
        AUTO_PERSON_NANO_CONF, AUDIT_DOG_NANO_CONF)).encode())
    for weights in sorted(deployed_dir.glob("**/*.bin")):
        digest.update(weights.read_bytes())
    return digest.hexdigest()[:16]


def _load_no_action(jury: str) -> set[str]:
    try:
        stored = json.loads(JURY_MEMORY.read_text())
    except (OSError, ValueError):
        return set()
    if stored.get("jury") != jury:
        return set()  # new champion or new rules: the memory is void
    return set(stored.get("stems", []))


def _needs_judging(meta: dict, x_entry: dict, no_action: set[str],
                   stem: str) -> bool:
    if meta.get("human_label"):
        # Only the one-direction audit applies: non-dog label, not yet
        # settled or flagged, and X must see a dog for a dispute to be
        # possible at all.
        return (meta["human_label"] in ("person", "empty", "no_dog")
                and not meta.get("dispute_settled_at")
                and not meta.get("disputed")
                and bool(x_entry["dogs"])
                and stem not in no_action)
    if meta.get("auto_label"):
        return False
    return stem not in no_action


def judge_frames(stems: list[str], cache: dict,
                 deployed_dir: Path) -> tuple[dict, dict]:
    """(auto_verdicts, disputes) judged by the deployed nano + the big-model
    cache. Without a deployed nano there is no second juror, so nothing is
    auto-labeled or disputed. Frames this jury already declined stay
    declined without re-scoring (see JURY_MEMORY)."""
    if not deployed_dir.is_dir():
        return {}, {}
    jury = _jury_id(deployed_dir)
    no_action = _load_no_action(jury)
    metas = {stem: _sidecar_meta(stem) for stem in stems}
    todo = [stem for stem in stems
            if _needs_judging(metas[stem], cache[stem], no_action, stem)]
    print(f"[pipeline] jury {jury}: judging {len(todo)} frames, "
          f"{len(stems) - len(todo)} skipped (labeled, settled, or already "
          f"declined by this jury)", flush=True)
    auto_verdicts: dict = {}
    disputes: dict = {}
    if todo:
        from ultralytics import YOLO  # deferred, like the whole package
        nano = YOLO(str(deployed_dir), task="detect")
        for stem in todo:
            nano_dog, nano_person = _nano_scores(nano, stem)
            if not metas[stem].get("human_label"):
                verdict = consensus_verdict(cache[stem], nano_dog, nano_person)
                if verdict:
                    auto_verdicts[stem] = verdict
                    continue
                no_action.add(stem)
                continue
            dispute = _audit_dispute(metas[stem], cache[stem], nano_dog)
            if dispute:
                disputes[stem] = dispute
                continue
            no_action.add(stem)
    JURY_MEMORY.write_text(json.dumps({"jury": jury,
                                       "stems": sorted(no_action)}))
    print(f"[pipeline] consensus auto-labeled {len(auto_verdicts)} "
          f"unlabeled frames; disputed {len(disputes)} existing labels",
          flush=True)
    return auto_verdicts, disputes
