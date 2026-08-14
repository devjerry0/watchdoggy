"""The dataset's shared vocabulary and sidecar-JSON access.

Every captured frame ships with a `sample_*.json` sidecar holding the
machine's detections and the human's judgment. This module is the single
place that vocabulary lives: verdicts, filmstrip filters, the exam split
hash, and the traversal-guarded path lookups every route uses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import HTTPException
from fastapi import status as http_status

# Human verdicts the review page can attach to a sample. "dog" = a real dog is
# present (a person may be too); "person" = person but no dog (the false-alarm
# class); "empty" = nothing of interest (pure background negative); "skip"
# parks a frame (unclear/blurry) without pretending it was judged. "dog_mixed"
# = a real dog IS present but at least one drawn dog box is actually a person
# (the compound case a plain "dog" can't express; flags the frame for box
# surgery at training prep). "no_dog" is
# the legacy coarse verdict from the catch log's one-tap button, kept valid.
VERDICTS = {"dog", "dog_mixed", "person", "empty", "no_dog", "skip"}
# Review order: highest training signal first, newest first within a class.
REASON_PRIORITY = {"fire": 0, "suppressed": 1, "borderline": 2,
                   "fire_context": 3, "person_activity": 4, "periodic": 5}
UNKNOWN_REASON_PRIORITY = 9
# Filmstrip filters. "unlabeled" is the work queue; "needs_boxes" finds
# dog-verdict frames where NO model produced a dog box (training would drop
# them without hand boxes); "no_prelabels" finds frames the big model hasn't
# scored yet.
FRAME_FILTERS = {"unlabeled", "auto", "disputed", "dog", "dog_mixed",
                 "person", "empty", "skip", "needs_boxes", "no_prelabels",
                 "exam", "all"}
# Mirrors kitchen_training.config.VAL_FRACTION_MOD: a frame's stem hash
# assigns it to the held-out exam forever. Keep the two in sync.
VAL_FRACTION_MOD = 4
NANO_FALLBACK_CONF = 0.45  # mirrors training's PRELABEL_CONF fallback rule


def not_found() -> HTTPException:
    return HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                         detail="not found")


def sidecar_paths(dataset_dir: str) -> list[Path]:
    d = Path(dataset_dir)
    return sorted(d.glob("sample_*.json")) if d.is_dir() else []


def read_meta(side: Path) -> dict | None:
    """A sidecar that fails to parse is skipped, not fatal -- capture may be
    mid-write."""
    try:
        return json.loads(side.read_text())
    except (OSError, ValueError):
        return None


def sidecar_or_404(dataset_dir: str, name: str) -> tuple[str, Path]:
    """Traversal-guarded lookup: request name -> (safe stem, sidecar path)."""
    safe = Path(name).name  # Path().name strips directories -> no traversal
    side = Path(dataset_dir) / f"{safe}.json"
    if not safe.startswith("sample_") or not side.is_file():
        raise not_found()
    return safe, side


def reason_priority(meta: dict) -> int:
    return min((REASON_PRIORITY.get(r, UNKNOWN_REASON_PRIORITY)
                for r in meta.get("reasons", [])),
               default=UNKNOWN_REASON_PRIORITY)


def has_dog_box(meta: dict) -> bool:
    """Would training find ANY dog box for this frame? Mirrors the fuse
    precedence: hand boxes, then big-model prelabels, then nano fallback."""
    hand = meta.get("human_boxes")
    if isinstance(hand, list):
        return any(b.get("label") == "dog" for b in hand)
    pre = (meta.get("prelabels") or {}).get("boxes") or []
    if any(b.get("label") == "dog" for b in pre):
        return True
    targets = meta.get("detections", {}).get("targets", [])
    return any(d.get("label") == "dog"
               and d.get("confidence", 0) >= NANO_FALLBACK_CONF
               for d in targets)


def is_exam(stem: str, verdict: str | None) -> bool:
    """Human-labeled frames whose stable hash lands in the val split --
    exactly what every model is judged on. Auto labels never sit here."""
    if not verdict or verdict == "skip":
        return False
    return int(hashlib.sha1(stem.encode()).hexdigest(),
               16) % VAL_FRACTION_MOD == 0


def matches(name: str, verdict: str | None, meta: dict, stem: str = "") -> bool:
    if name == "all":
        return True
    if name == "exam":
        return is_exam(stem, verdict)
    if name == "unlabeled":
        # Auto-labeled frames are off the human's queue.
        return not verdict and not meta.get("auto_label")
    if name == "auto":
        return not verdict and bool(meta.get("auto_label"))
    if name == "disputed":
        return bool(meta.get("disputed"))
    if name == "no_prelabels":
        return "prelabels" not in meta
    if name == "needs_boxes":
        return verdict in ("dog", "dog_mixed") and not has_dog_box(meta)
    return verdict == name


def _parse_box(raw: dict) -> dict:
    label, box = raw.get("label"), raw.get("box")
    well_formed = (label in ("dog", "person") and isinstance(box, list)
                   and len(box) == 4
                   and all(isinstance(v, (int, float)) for v in box)
                   and box[2] > box[0] and box[3] > box[1])
    if not well_formed:
        raise HTTPException(status_code=422,
                            detail="each box needs label dog|person "
                                   "and box [x1,y1,x2,y2]")
    return {"label": label, "box": [round(float(v), 1) for v in box]}


def parse_boxes(raw: object) -> list[dict]:
    """Validate a client-sent box list (label + geometry; 422 on any flaw)."""
    if not isinstance(raw, list):
        raise HTTPException(status_code=422, detail="boxes must be a list")
    return [_parse_box(b) for b in raw]
