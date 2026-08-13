"""Label fusion: pure functions from frame facts to YOLO label lines.

The precedence is the whole design: hand boxes > big-model prelabels
(mixed-frame filtered) > nano fallback > drop.
"""
from __future__ import annotations

from kitchen_training.config import CLASS_IDS, MIXED_IOU_DROP, PRELABEL_CONF


def iou(box_a, box_b) -> float:
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection == 0:
        return 0.0
    area = lambda box: max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])  # noqa: E731
    union = area(box_a) + area(box_b) - intersection
    if not union:
        return 0.0
    return intersection / union


def yolo_line(class_id: int, box, width: int, height: int) -> str:
    left, top, right, bottom = box
    return (f"{class_id} {(left+right)/2/width:.6f} {(top+bottom)/2/height:.6f} "
            f"{(right-left)/width:.6f} {(bottom-top)/height:.6f}")


def _person_lines(people, width: int, height: int) -> list[str]:
    return [yolo_line(CLASS_IDS["person"], box, width, height) for box in people]


def _dog_lines(dogs, width: int, height: int) -> list[str]:
    return [yolo_line(CLASS_IDS["dog"], box, width, height) for box in dogs]


def _real_dogs(dogs, people, verdict) -> list:
    """dog_mixed: the human says one of the dog boxes is a person in
    disguise -- drop dog boxes that sit on a person box."""
    if verdict != "dog_mixed":
        return dogs
    return [dog for dog in dogs
            if not any(iou(dog, person) >= MIXED_IOU_DROP for person in people)]


def _fallback_dogs(meta: dict) -> list:
    """The big model missed the dog: fall back to the nano's sidecar box."""
    boxes = [d["box"] for d in meta.get("detections", {}).get("targets", [])
             if d.get("label") == "dog"
             and d.get("confidence", 0) >= PRELABEL_CONF]
    return [list(map(float, box)) for box in boxes]


def fuse(stem: str, verdict: str, meta: dict,
         prelabels: dict) -> tuple[list[str] | None, str]:
    """Label lines for one frame plus their provenance:
    'hand' | 'fused' | 'fallback' | 'drop' (lines=None means drop)."""
    height, width = prelabels["shape"]
    hand_boxes = meta.get("human_boxes")
    if isinstance(hand_boxes, list):
        # Hand-drawn boxes are the complete truth for this frame; no model,
        # fallback, or overlap heuristic gets a say.
        return [yolo_line(CLASS_IDS[b["label"]], b["box"], width, height)
                for b in hand_boxes], "hand"
    if verdict in ("person", "no_dog"):
        return _person_lines(prelabels["people"], width, height), "fused"
    if verdict == "empty":
        return [], "fused"
    # dog / dog_mixed
    dogs = _real_dogs(list(prelabels["dogs"]), prelabels["people"], verdict)
    source = "fused"
    if not dogs:
        dogs = _fallback_dogs(meta)
        source = "fallback"
    if not dogs:
        return None, "drop"
    return (_dog_lines(dogs, width, height)
            + _person_lines(prelabels["people"], width, height)), source
