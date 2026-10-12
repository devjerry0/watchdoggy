"""One-request application of a whole cloud pass: prelabels, consensus
auto-verdicts, audit disputes, and auto-label clears for hundreds of frames
at once.

The per-frame endpoints stay (the label page uses them); this one exists
for the trainer daemon. Per-frame HTTPS meant a fresh TLS handshake per
frame, which on a Pi busy with inference stretched a big merge to twenty
minutes. All boxes are validated before anything is written, so a
malformed payload applies nothing."""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from doggy.core.config import Settings
from doggy.core.priority import WEB_WORKER_NICE, lower_thread_priority
from doggy.web.sidecar_index import SidecarIndex
from doggy.web.routers.dataset.sidecars import (
    apply_autolabel,
    apply_clear_auto,
    apply_dispute,
    apply_prelabels,
    parse_boxes,
)

_AUTO_VERDICTS = ("dog", "person", "empty")


def _validated_verdicts(raw: dict) -> dict:
    for verdict in raw.values():
        if verdict not in _AUTO_VERDICTS:
            raise HTTPException(status_code=422,
                                detail="auto verdicts must be dog, person, or empty")
    return raw


def build_router(settings: Settings, index: SidecarIndex) -> APIRouter:
    router = APIRouter()

    @router.post("/api/dataset/apply-cloud-results")
    def api_apply_cloud_results(body: dict) -> dict:
        lower_thread_priority(WEB_WORKER_NICE)  # 1k-sidecar writes: never outrun NCNN
        now = time.time()
        model = str(body.get("model", "?"))
        prelabels = {stem: parse_boxes(raw)
                     for stem, raw in (body.get("prelabels") or {}).items()}
        verdicts = _validated_verdicts(body.get("auto_verdicts") or {})
        disputes = body.get("disputes") or {}
        clears = body.get("clear_auto") or []  # stems the jury stopped vouching for
        if not isinstance(clears, list):
            raise HTTPException(status_code=422,
                                detail="clear_auto must be a list of names")
        clears = {str(stem) for stem in clears}
        applied = {"prelabels": 0, "auto_verdicts": 0, "disputes": 0,
                   "clear_auto": 0, "missing": 0}
        for stem in sorted({*prelabels, *verdicts, *disputes, *clears}):
            safe = Path(str(stem)).name  # traversal guard
            side = Path(settings.dataset_dir) / f"{safe}.json"
            if not safe.startswith("sample_") or not side.is_file():
                applied["missing"] += 1
                continue
            try:
                meta = json.loads(side.read_text())
            except ValueError:
                # A power cut mid-write leaves an empty sidecar (Oct 2026);
                # one bad file must not drop the rest of the batch.
                applied["missing"] += 1
                continue
            if safe in prelabels:
                apply_prelabels(meta, model, prelabels[safe])
                applied["prelabels"] += 1
            if safe in verdicts:
                applied["auto_verdicts"] += apply_autolabel(meta, verdicts[safe],
                                                            now)
            if safe in clears:
                applied["clear_auto"] += apply_clear_auto(meta)
            model_says = str((disputes.get(safe) or {}).get("model_says", ""))
            if model_says:
                nano_conf = float(disputes[safe].get("nano_conf", 0))
                applied["disputes"] += apply_dispute(meta, model_says,
                                                     nano_conf, now)
            side.write_text(json.dumps(meta))
            index.upsert(side.stem)  # in-place edit: the dir mtime won't move
        return {"ok": True, **applied}

    return router
