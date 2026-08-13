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


def build_router(settings: Settings, capture: DatasetCapture,
                 event_store: EventStore) -> APIRouter:
    router = APIRouter()

    def _sidecars() -> list[Path]:
        d = Path(settings.dataset_dir)
        return sorted(d.glob("sample_*.json")) if d.is_dir() else []

    @router.get("/api/dataset")
    def api_dataset() -> dict:
        return capture.stats()

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
