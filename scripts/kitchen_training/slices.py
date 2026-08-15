"""Failure-mode slices and confidence calibration for the exam.

One aggregate catch-rate hides regressions in rare-but-important corners
(per-slice gates are the industry lesson). Slices are derived automatically
from what the sidecars and pixels already know; calibration reports how
honestly the model's confidences map to reality (a fixed 0.6 threshold
means different things across generations if calibration drifts). Both are
INDICATORS on the scorecard -- the deploy gate stays best-model-wins."""
from __future__ import annotations

import json
import math

from kitchen_training.config import DATASET_MIRROR, Frame

DARK_MEAN = 60.0        # dim evening scenes (pitch-black never gets captured)
EDGE_MARGIN_FRAC = 0.05  # a dog box touching the frame edge = half-out dog
SMALL_AREA_FRAC = 0.015  # or barely-there small
ECE_BINS = 7             # ~n^(1/3) for a ~300-frame exam
_T_GRID = [x / 20 for x in range(5, 81)]  # 0.25 .. 4.0


def _image_mean(frame: Frame) -> float:
    import cv2
    image = cv2.imread(str(frame.jpg))
    if image is None:
        return 255.0
    return float(image[::8, ::8].mean())


def _dog_boxes(meta: dict) -> list[list[float]]:
    hand = meta.get("human_boxes")
    if isinstance(hand, list):
        return [b["box"] for b in hand if b.get("label") == "dog"]
    pre = (meta.get("prelabels") or {}).get("boxes") or []
    return [b["box"] for b in pre if b.get("label") == "dog"]


def _small_or_edge(meta: dict, shape: tuple[int, int]) -> bool:
    height, width = shape
    for x1, y1, x2, y2 in _dog_boxes(meta):
        area = (x2 - x1) * (y2 - y1) / max(1.0, width * height)
        margin = min(x1, y1, width - x2, height - y2) / max(width, height)
        if area < SMALL_AREA_FRAC or margin < EDGE_MARGIN_FRAC:
            return True
    return False


def _tags(frame: Frame, meta: dict) -> set[str]:
    tags = set()
    if _image_mean(frame) < DARK_MEAN:
        tags.add("dark")
    people = (meta.get("detections", {}).get("people")
              or [b for b in (meta.get("prelabels") or {}).get("boxes", [])
                  if b.get("label") == "person"])
    if people:
        tags.add("person_present")
    if frame.is_dog and _small_or_edge(meta, (480, 640)):
        tags.add("small_or_edge_dog")
    if set(meta.get("reasons", [])) & {"fire", "user_marked_fp"}:
        tags.add("fire_origin")
    if not tags:
        tags.add("plain")
    return tags


def _meta(frame: Frame) -> dict:
    side = DATASET_MIRROR / f"{frame.stem}.json"
    try:
        return json.loads(side.read_text())
    except (OSError, ValueError):
        return {}


def slice_report(frames: list[Frame], confs: dict[str, float],
                 deployed_confs: dict[str, float] | None,
                 fire_conf: float) -> dict:
    """Per-slice held-out tallies for the candidate (and the incumbent,
    when its confs are available, so regressions are visible side by side)."""
    slices: dict[str, dict] = {}
    for frame in frames:
        if not frame.heldout:
            continue
        for tag in _tags(frame, _meta(frame)):
            tally = slices.setdefault(tag, {"dogs": 0, "caught": 0,
                                            "nondogs": 0, "false_fires": 0,
                                            "deployed_caught": 0,
                                            "deployed_false_fires": 0})
            fired = confs.get(frame.stem, 0.0) >= fire_conf
            old_fired = (deployed_confs or {}).get(frame.stem, 0.0) >= fire_conf
            if frame.is_dog:
                tally["dogs"] += 1
                tally["caught"] += fired
                tally["deployed_caught"] += old_fired
                continue
            tally["nondogs"] += 1
            tally["false_fires"] += fired
            tally["deployed_false_fires"] += old_fired
    return slices


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _nll(pairs: list[tuple[float, bool]], t: float) -> float:
    total = 0.0
    for conf, is_dog in pairs:
        p = 1 / (1 + math.exp(-_logit(conf) / t))
        p = min(max(p, 1e-6), 1 - 1e-6)
        total -= math.log(p) if is_dog else math.log(1 - p)
    return total


def calibration(frames: list[Frame], confs: dict[str, float]) -> dict:
    """Fitted temperature + ECE on the held-out exam (diagnostics only:
    the gate and the appliance keep judging raw confidences)."""
    pairs = [(confs.get(f.stem, 0.0), f.is_dog) for f in frames if f.heldout]
    temperature = min(_T_GRID, key=lambda t: _nll(pairs, t))
    bins = [[0, 0.0, 0.0] for _ in range(ECE_BINS)]  # count, conf sum, hit sum
    for conf, is_dog in pairs:
        b = bins[min(ECE_BINS - 1, int(conf * ECE_BINS))]
        b[0] += 1
        b[1] += conf
        b[2] += is_dog
    ece = sum(count * abs(conf_sum / count - hits / count)
              for count, conf_sum, hits in bins if count) / max(1, len(pairs))
    return {"temperature": round(temperature, 2), "ece": round(ece, 4),
            "exam_frames": len(pairs)}
