"""Read-only dataset views: stats, the filmstrip, one frame, the queue.
All listings come from the shared SidecarIndex -- per-request re-parsing
of thousands of sidecars is what made these pages crawl on the Pi."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from doggy.core.config import Settings
from doggy.web.routers.dataset.sidecars import (
    FRAME_FILTERS,
    matches,
    reason_priority,
    sidecar_or_404,
)
from doggy.web.sidecar_index import SidecarIndex

# The filmstrip and history rails cap their payloads: listings are unbounded
# as the dataset grows, and the Pi serves these over WiFi.
FRAME_PAGE_LIMIT = 400
LABELED_PAGE_LIMIT = 200


def _add_chip_counts(counts: dict, verdict: str | None, meta: dict,
                     stem: str) -> None:
    for chip in FRAME_FILTERS:
        counts[chip] += matches(chip, verdict, meta, stem)


def _sort_key(filter_name: str, meta: dict) -> tuple:
    """Unlabeled sorts highest-signal first (like the queue); everything
    else newest first."""
    if filter_name == "unlabeled":
        return (reason_priority(meta), -meta.get("wall_time", 0))
    return (-meta.get("labeled_at", meta.get("wall_time", 0)),)


def _frame_row(stem: str, verdict: str | None, meta: dict) -> dict:
    auto = meta.get("auto_label") or {}
    return {"name": stem,
            "image": f"/dataset/{stem}.jpg",
            "verdict": verdict or auto.get("verdict"),
            "auto": not verdict and bool(auto),
            "disputed": bool(meta.get("disputed")),
            "hand_boxes": isinstance(meta.get("human_boxes"), list)}


def build_router(settings: Settings, index: SidecarIndex) -> APIRouter:
    router = APIRouter()

    @router.get("/api/dataset")
    def api_dataset() -> dict:
        """Sample count, byte usage, and per-reason tallies for the dashboard."""
        by_reason: dict[str, int] = {}
        sidecars = index.snapshot()
        for _, meta in sidecars:
            for r in meta.get("reasons", []):
                by_reason[r] = by_reason.get(r, 0) + 1
        return {"samples": len(sidecars), "bytes": index.sample_bytes(),
                "cap_bytes": settings.dataset_cap_bytes,
                "by_reason": by_reason}

    @router.get("/api/dataset/frames")
    def api_frames(filter: str = "unlabeled") -> dict:
        """The filmstrip: light rows for one filter + counts for every chip."""
        if filter not in FRAME_FILTERS:
            raise HTTPException(status_code=422,
                                detail=f"filter must be one of {sorted(FRAME_FILTERS)}")
        keyed = []
        counts = dict.fromkeys(FRAME_FILTERS, 0)
        for stem, meta in index.snapshot():
            verdict = meta.get("human_label")
            _add_chip_counts(counts, verdict, meta, stem)
            if not matches(filter, verdict, meta, stem):
                continue
            keyed.append((_sort_key(filter, meta),
                          _frame_row(stem, verdict, meta)))
        keyed.sort(key=lambda pair: pair[0])
        return {"frames": [row for _, row in keyed[:FRAME_PAGE_LIMIT]],
                "counts": counts}

    @router.get("/api/dataset/sample/{name}")
    def api_sample(name: str) -> dict:
        """Everything the stage needs to show one frame. Read directly (not
        via the index): the editor must always see its own last write."""
        safe, side = sidecar_or_404(settings.dataset_dir, name)
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
        for stem, meta in index.snapshot():
            if meta.get("human_label"):
                labeled += 1
                continue
            pending.append((reason_priority(meta), -meta.get("wall_time", 0),
                            stem, meta))
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
        for stem, meta in index.snapshot():
            if not meta.get("human_label"):
                continue
            rows.append({"name": stem, "image": f"/dataset/{stem}.jpg",
                         "verdict": meta["human_label"],
                         "labeled_at": meta.get("labeled_at", 0),
                         "reasons": meta.get("reasons", []),
                         "detections": meta.get("detections", {}),
                         "human_boxes": meta.get("human_boxes"),
                         "prelabels": meta.get("prelabels")})
        rows.sort(key=lambda r: -r["labeled_at"])
        return {"labeled": rows[:LABELED_PAGE_LIMIT]}

    return router
