"""Dataset write paths: human verdicts and hand boxes, machine prelabels,
consensus auto-labels, jury disputes, and the catch log's "Not a dog" tap."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status

from doggy.core.config import Settings
from doggy.events.store import EventStore
from doggy.web.routers.dataset.sidecars import (
    VERDICTS,
    apply_autolabel,
    apply_dispute,
    apply_prelabels,
    parse_boxes,
    sidecar_or_404,
)


def _clear_verdict(meta: dict) -> None:
    """Undo: the frame returns to the review queue as if never judged.
    Hand-drawn boxes survive an undo -- that work stays done."""
    meta.pop("human_label", None)
    meta.pop("labeled_at", None)


def _settle_dispute(meta: dict) -> None:
    # Re-judging a frame settles its dispute PERMANENTLY -- the nightly
    # audit must never ask the same question twice.
    if meta.pop("disputed", None) is not None:
        meta["dispute_settled_at"] = time.time()


def _drop_contradicted_boxes(meta: dict, verdict: str, body: dict) -> None:
    """A bare verdict that CONTRADICTS saved hand boxes wins: the boxes are
    cleared rather than silently outranking the human's newest judgment
    (boxes imply a verdict; a conflicting tap means the boxes were wrong)."""
    if "boxes" in body:
        return
    hand = meta.get("human_boxes")
    if not isinstance(hand, list):
        return
    has_dog_box = any(b.get("label") == "dog" for b in hand)
    wants_dog = verdict in ("dog", "dog_mixed")
    if has_dog_box != wants_dog:
        meta.pop("human_boxes", None)


def _attach_hand_boxes(meta: dict, body: dict) -> None:
    if "boxes" not in body:
        return
    # Hand-drawn boxes: the COMPLETE annotation for this frame (every dog
    # and person). Training trusts them over any model.
    meta["human_boxes"] = parse_boxes(body["boxes"])


def _apply_label(meta: dict, verdict: str, body: dict) -> None:
    if verdict == "clear":
        _clear_verdict(meta)
        return
    meta["human_label"] = verdict
    meta["labeled_at"] = time.time()
    # A human verdict supersedes any machine auto-label.
    meta.pop("auto_label", None)
    _settle_dispute(meta)
    _drop_contradicted_boxes(meta, verdict, body)
    _attach_hand_boxes(meta, body)


def build_router(settings: Settings, event_store: EventStore) -> APIRouter:
    router = APIRouter()

    @router.post("/api/dataset/label")
    def api_label(body: dict) -> dict:
        verdict = body.get("verdict")
        if verdict != "clear" and verdict not in VERDICTS:
            raise HTTPException(status_code=422,
                                detail="verdict must be dog, dog_mixed, person, empty, "
                                       "skip, or clear")
        _, side = sidecar_or_404(settings.dataset_dir, str(body.get("name", "")))
        meta = json.loads(side.read_text())
        _apply_label(meta, verdict, body)
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.post("/api/dataset/prelabels")
    def api_prelabels(body: dict) -> dict:
        """Merge big-model boxes (computed off-box, on the Mac) into a
        sidecar. The review page shows these as the machine's best guess and
        seeds the box editor from them -- they are what training will use
        unless a human overrides with hand boxes."""
        _, side = sidecar_or_404(settings.dataset_dir, str(body.get("name", "")))
        clean = parse_boxes(body.get("boxes"))
        meta = json.loads(side.read_text())
        apply_prelabels(meta, str(body.get("model", "?")), clean)
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.post("/api/dataset/autolabel")
    def api_autolabel(body: dict) -> dict:
        """Machine consensus verdict (deployed nano + big model agreeing),
        written by the trainer's nightly pass. Never touches a frame a human
        has judged, and trains only in the train split -- the held-out exam
        stays human-verified."""
        verdict = body.get("verdict")
        if verdict not in ("dog", "person", "empty"):
            raise HTTPException(status_code=422,
                                detail="verdict must be dog, person, or empty")
        _, side = sidecar_or_404(settings.dataset_dir, str(body.get("name", "")))
        meta = json.loads(side.read_text())
        if not apply_autolabel(meta, verdict, time.time()):
            return {"ok": True, "skipped": "human label wins"}
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.post("/api/dataset/dispute")
    def api_dispute(body: dict) -> dict:
        """The nightly jury contradicts an existing label: flag the frame
        for human re-review. Any fresh human verdict clears the flag."""
        _, side = sidecar_or_404(settings.dataset_dir, str(body.get("name", "")))
        model_says = str(body.get("model_says", ""))
        if not model_says:
            raise HTTPException(status_code=422, detail="model_says required")
        meta = json.loads(side.read_text())
        if not apply_dispute(meta, model_says, float(body.get("nano_conf", 0)),
                             time.time()):
            return {"ok": True, "skipped": "human already arbitrated"}
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.post("/api/dataset/mark/{event_id}")
    def api_mark_false_positive(event_id: str) -> dict:
        """The catch log's "Not a dog" button: copy that event's raw frame into
        the dataset labeled as a user-confirmed false positive. Works even with
        capture off -- an explicit user label is always worth keeping."""
        record = next((r for r in event_store.list()
                       if r.id == Path(event_id).name), None)
        if record is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        src = Path(settings.event_log_dir) / record.thumb
        if not src.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="snapshot missing")
        out = Path(settings.dataset_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"sample_{int(time.time() * 1000)}"
        shutil.copyfile(src, out / f"{stem}.jpg")
        (out / f"{stem}.json").write_text(json.dumps({
            "wall_time": time.time(),
            "reasons": ["user_marked_fp"],
            "event_id": record.id,
            "event_confidence": record.confidence,
            # The tap IS the verdict: this frame needs no second review.
            "human_label": "no_dog",
            "labeled_at": time.time(),
        }))
        return {"ok": True}

    @router.post("/api/dataset/clear")
    def api_clear_dataset() -> dict:
        d = Path(settings.dataset_dir)
        if not d.is_dir():
            return {"ok": True}
        for p in d.glob("sample_*"):
            if p.is_file():
                p.unlink()
        return {"ok": True}

    return router
