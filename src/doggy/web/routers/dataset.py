from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status
from fastapi.responses import FileResponse

from doggy.core.config import Settings
from doggy.events.store import EventStore
from doggy.reaction.dataset import DatasetCapture

# Human verdicts the review page can attach to a sample. "dog" = a real dog is
# present (a person may be too); "person" = person but no dog (the false-alarm
# class); "empty" = nothing of interest (pure background negative); "skip"
# parks a frame (unclear/blurry) without pretending it was judged. "dog_mixed"
# = a real dog IS present but at least one drawn dog box is actually a person
# (the compound case a plain "dog" can't express; flags the frame for box
# surgery at training prep). "no_dog" is
# the legacy coarse verdict from the catch log's one-tap button, kept valid.
_VERDICTS = {"dog", "dog_mixed", "person", "empty", "no_dog", "skip"}
# Review order: highest training signal first, newest first within a class.
_REASON_PRIORITY = {"fire": 0, "suppressed": 1, "borderline": 2,
                    "fire_context": 3, "person_activity": 4, "periodic": 5}
# Filmstrip filters. "unlabeled" is the work queue; "needs_boxes" finds
# dog-verdict frames where NO model produced a dog box (training would drop
# them without hand boxes); "no_prelabels" finds frames the big model hasn't
# scored yet.
_FRAME_FILTERS = {"unlabeled", "auto", "disputed", "dog", "dog_mixed",
                  "person", "empty", "skip", "needs_boxes", "no_prelabels",
                  "all"}
_NANO_FALLBACK_CONF = 0.45  # mirrors training's PRELABEL_CONF fallback rule


def build_router(settings: Settings, capture: DatasetCapture,
                 event_store: EventStore) -> APIRouter:
    router = APIRouter()

    def _sidecars() -> list[Path]:
        d = Path(settings.dataset_dir)
        return sorted(d.glob("sample_*.json")) if d.is_dir() else []

    @router.get("/api/dataset")
    def api_dataset() -> dict:
        return capture.stats()

    def _has_dog_box(meta: dict) -> bool:
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
                   and d.get("confidence", 0) >= _NANO_FALLBACK_CONF
                   for d in targets)

    def _matches(name: str, verdict: str | None, meta: dict) -> bool:
        if name == "all":
            return True
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
            return verdict in ("dog", "dog_mixed") and not _has_dog_box(meta)
        return verdict == name

    @router.get("/api/dataset/frames")
    def api_frames(filter: str = "unlabeled") -> dict:
        """The filmstrip: light rows for one filter + counts for every chip.
        Unlabeled sorts highest-signal first (like the queue); everything
        else newest first."""
        if filter not in _FRAME_FILTERS:
            raise HTTPException(status_code=422,
                                detail=f"filter must be one of {sorted(_FRAME_FILTERS)}")
        rows = []
        counts = dict.fromkeys(_FRAME_FILTERS, 0)
        for side in _sidecars():
            try:
                meta = json.loads(side.read_text())
            except (OSError, ValueError):
                continue
            verdict = meta.get("human_label")
            for chip in _FRAME_FILTERS:
                counts[chip] += _matches(chip, verdict, meta)
            if not _matches(filter, verdict, meta):
                continue
            prio = min((_REASON_PRIORITY.get(r, 9)
                        for r in meta.get("reasons", [])), default=9)
            auto = meta.get("auto_label") or {}
            rows.append({"name": side.stem,
                         "image": f"/dataset/{side.stem}.jpg",
                         "verdict": verdict or auto.get("verdict"),
                         "auto": not verdict and bool(auto),
                         "disputed": bool(meta.get("disputed")),
                         "hand_boxes": isinstance(meta.get("human_boxes"), list),
                         "_sort": ((prio, -meta.get("wall_time", 0))
                                   if filter == "unlabeled"
                                   else (-meta.get("labeled_at",
                                                   meta.get("wall_time", 0)),))})
        rows.sort(key=lambda r: r["_sort"])
        for row in rows:
            row.pop("_sort")
        return {"frames": rows[:400], "counts": counts}

    @router.get("/api/dataset/sample/{name}")
    def api_sample(name: str) -> dict:
        """Everything the stage needs to show one frame."""
        safe = Path(name).name  # traversal guard
        side = Path(settings.dataset_dir) / f"{safe}.json"
        if not safe.startswith("sample_") or not side.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        meta = json.loads(side.read_text())
        return {"name": safe, "image": f"/dataset/{safe}.jpg",
                "auto_label": meta.get("auto_label"),
                "disputed": meta.get("disputed"),
                "verdict": meta.get("human_label"),
                "reasons": meta.get("reasons", []),
                "suggested": meta.get("suggested_label"),
                "detections": meta.get("detections", {}),
                "human_boxes": meta.get("human_boxes"),
                "prelabels": meta.get("prelabels")}

    @router.get("/api/dataset/next")
    def api_next_unlabeled() -> dict:
        """The next frame to review: unlabeled, highest-signal reason first."""
        pending = []
        labeled = 0
        for side in _sidecars():
            try:
                meta = json.loads(side.read_text())
            except (OSError, ValueError):
                continue
            if meta.get("human_label"):
                labeled += 1
                continue
            prio = min((_REASON_PRIORITY.get(r, 9) for r in meta.get("reasons", [])),
                       default=9)
            pending.append((prio, -meta.get("wall_time", 0), side.stem, meta))
        pending.sort()
        if not pending:
            return {"remaining": 0, "labeled": labeled, "sample": None}
        _, _, stem, meta = pending[0]
        return {"remaining": len(pending), "labeled": labeled,
                "sample": {"name": stem, "image": f"/dataset/{stem}.jpg",
                           "reasons": meta.get("reasons", []),
                           "suggested": meta.get("suggested_label"),
                           "detections": meta.get("detections", {}),
                           "human_boxes": meta.get("human_boxes"),
                           "prelabels": meta.get("prelabels")}}

    @router.get("/api/dataset/labeled")
    def api_labeled() -> dict:
        """Already-judged frames, most recently labeled first, so verdicts can
        be reviewed and corrected (mislabels happen -- ours included)."""
        rows = []
        for side in _sidecars():
            try:
                meta = json.loads(side.read_text())
            except (OSError, ValueError):
                continue
            if not meta.get("human_label"):
                continue
            rows.append({"name": side.stem, "image": f"/dataset/{side.stem}.jpg",
                         "verdict": meta["human_label"],
                         "labeled_at": meta.get("labeled_at", 0),
                         "reasons": meta.get("reasons", []),
                         "detections": meta.get("detections", {}),
                         "human_boxes": meta.get("human_boxes"),
                         "prelabels": meta.get("prelabels")})
        rows.sort(key=lambda r: -r["labeled_at"])
        return {"labeled": rows[:200]}

    @router.post("/api/dataset/label")
    def api_label(body: dict) -> dict:
        name = Path(str(body.get("name", ""))).name  # traversal guard
        verdict = body.get("verdict")
        if verdict != "clear" and verdict not in _VERDICTS:
            raise HTTPException(status_code=422,
                                detail="verdict must be dog, dog_mixed, person, empty, "
                                       "skip, or clear")
        side = Path(settings.dataset_dir) / f"{name}.json"
        if not name.startswith("sample_") or not side.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        meta = json.loads(side.read_text())
        if verdict == "clear":
            # Undo: the frame returns to the review queue as if never judged.
            # Hand-drawn boxes survive an undo -- that work stays done.
            meta.pop("human_label", None)
            meta.pop("labeled_at", None)
        else:
            meta["human_label"] = verdict
            meta["labeled_at"] = time.time()
            # A human verdict supersedes any machine auto-label, and
            # re-judging a frame settles its dispute.
            meta.pop("auto_label", None)
            meta.pop("disputed", None)
            if "boxes" in body:
                # Hand-drawn boxes: the COMPLETE annotation for this frame
                # (every dog and person). Training trusts them over any model.
                boxes = body["boxes"]
                if not isinstance(boxes, list):
                    raise HTTPException(status_code=422, detail="boxes must be a list")
                clean = []
                for b in boxes:
                    label, box = b.get("label"), b.get("box")
                    if (label not in ("dog", "person") or not isinstance(box, list)
                            or len(box) != 4
                            or not all(isinstance(v, (int, float)) for v in box)
                            or box[2] <= box[0] or box[3] <= box[1]):
                        raise HTTPException(status_code=422,
                                            detail="each box needs label dog|person "
                                                   "and box [x1,y1,x2,y2]")
                    clean.append({"label": label,
                                  "box": [round(float(v), 1) for v in box]})
                meta["human_boxes"] = clean
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.post("/api/dataset/prelabels")
    def api_prelabels(body: dict) -> dict:
        """Merge big-model boxes (computed off-box, on the Mac) into a
        sidecar. The review page shows these as the machine's best guess and
        seeds the box editor from them -- they are what training will use
        unless a human overrides with hand boxes."""
        name = Path(str(body.get("name", ""))).name  # traversal guard
        side = Path(settings.dataset_dir) / f"{name}.json"
        if not name.startswith("sample_") or not side.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        boxes = body.get("boxes")
        if not isinstance(boxes, list):
            raise HTTPException(status_code=422, detail="boxes must be a list")
        clean = []
        for b in boxes:
            label, box = b.get("label"), b.get("box")
            if (label not in ("dog", "person") or not isinstance(box, list)
                    or len(box) != 4
                    or not all(isinstance(v, (int, float)) for v in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                raise HTTPException(status_code=422,
                                    detail="each box needs label dog|person "
                                           "and box [x1,y1,x2,y2]")
            clean.append({"label": label, "box": [round(float(v), 1) for v in box]})
        meta = json.loads(side.read_text())
        meta["prelabels"] = {"model": str(body.get("model", "?"))[:40],
                             "boxes": clean}
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.post("/api/dataset/autolabel")
    def api_autolabel(body: dict) -> dict:
        """Machine consensus verdict (deployed nano + big model agreeing),
        written by the trainer's nightly pass. Never touches a frame a human
        has judged, and trains only in the train split -- the held-out exam
        stays human-verified."""
        name = Path(str(body.get("name", ""))).name  # traversal guard
        verdict = body.get("verdict")
        if verdict not in ("dog", "person", "empty"):
            raise HTTPException(status_code=422,
                                detail="verdict must be dog, person, or empty")
        side = Path(settings.dataset_dir) / f"{name}.json"
        if not name.startswith("sample_") or not side.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        meta = json.loads(side.read_text())
        if meta.get("human_label"):
            return {"ok": True, "skipped": "human label wins"}
        meta["auto_label"] = {"verdict": verdict, "labeled_at": time.time(),
                              "source": "nano+x consensus"}
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.post("/api/dataset/dispute")
    def api_dispute(body: dict) -> dict:
        """The nightly jury contradicts an existing label: flag the frame
        for human re-review. Any fresh human verdict clears the flag."""
        name = Path(str(body.get("name", ""))).name  # traversal guard
        side = Path(settings.dataset_dir) / f"{name}.json"
        if not name.startswith("sample_") or not side.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        model_says = str(body.get("model_says", ""))[:40]
        if not model_says:
            raise HTTPException(status_code=422, detail="model_says required")
        meta = json.loads(side.read_text())
        meta["disputed"] = {"model_says": model_says,
                            "nano_conf": float(body.get("nano_conf", 0)),
                            "flagged_at": time.time()}
        side.write_text(json.dumps(meta))
        return {"ok": True}

    @router.get("/dataset/thumb/{name}")
    def dataset_thumb(name: str) -> FileResponse:
        """Small cached thumbnail for the filmstrip. Full frames are ~100KB
        each; a 400-frame strip of them is megabytes through a Pi -- thumbs
        are ~5KB, generated once and cached beside the dataset."""
        safe = Path(name).name  # traversal guard
        source = Path(settings.dataset_dir) / safe
        if not safe.startswith("sample_") or not safe.endswith(".jpg"):
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        thumb = Path(settings.dataset_dir) / "thumbs" / safe
        if not source.is_file():
            thumb.unlink(missing_ok=True)  # source pruned: drop the stale thumb
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        if not thumb.is_file():
            import cv2
            image = cv2.imread(str(source))
            if image is None:
                raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                    detail="unreadable")
            height = max(1, round(image.shape[0] * 160 / image.shape[1]))
            small = cv2.resize(image, (160, height), interpolation=cv2.INTER_AREA)
            thumb.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(thumb), small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return FileResponse(thumb)

    @router.get("/dataset/{name}")
    def dataset_image(name: str) -> FileResponse:
        # Path(name).name strips any directory components -> no path traversal.
        safe = Path(name).name
        path = Path(settings.dataset_dir) / safe
        if not safe.startswith("sample_") or not safe.endswith(".jpg") or not path.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        return FileResponse(path)

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
        if d.is_dir():
            for p in d.glob("sample_*"):
                if p.is_file():
                    p.unlink()
        return {"ok": True}

    return router
