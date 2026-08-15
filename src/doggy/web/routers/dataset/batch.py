"""One-request application of a whole cloud pass: prelabels, consensus
auto-verdicts, and audit disputes for hundreds of frames at once.

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
from doggy.web.routers.dataset.sidecars import (
    apply_autolabel,
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


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.post("/api/dataset/apply-cloud-results")
    def api_apply_cloud_results(body: dict) -> dict:
        now = time.time()
        model = str(body.get("model", "?"))
        prelabels = {stem: parse_boxes(raw)
                     for stem, raw in (body.get("prelabels") or {}).items()}
        verdicts = _validated_verdicts(body.get("auto_verdicts") or {})
        disputes = body.get("disputes") or {}
        applied = {"prelabels": 0, "auto_verdicts": 0, "disputes": 0,
                   "missing": 0}
        for stem in sorted({*prelabels, *verdicts, *disputes}):
            safe = Path(str(stem)).name  # traversal guard
            side = Path(settings.dataset_dir) / f"{safe}.json"
            if not safe.startswith("sample_") or not side.is_file():
                applied["missing"] += 1
                continue
            meta = json.loads(side.read_text())
            if safe in prelabels:
                apply_prelabels(meta, model, prelabels[safe])
                applied["prelabels"] += 1
            if safe in verdicts:
                applied["auto_verdicts"] += apply_autolabel(meta, verdicts[safe],
                                                            now)
            model_says = str((disputes.get(safe) or {}).get("model_says", ""))
            if model_says:
                nano_conf = float(disputes[safe].get("nano_conf", 0))
                applied["disputes"] += apply_dispute(meta, model_says,
                                                     nano_conf, now)
            side.write_text(json.dumps(meta))
        return {"ok": True, **applied}

    return router
