"""Pushing cloud results back into the appliance: sidecar merges through
the local web API, and the root-helper model install."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from trainer_daemon.env import STAGING_BUNDLE, api_post, log


BATCH_APPLY_TIMEOUT_S = 180
# Frames per batch request. Oct 2026: a 39k-frame catch-up sent as ONE
# request outran the timeout, and the per-frame fallback then redid every
# frame over HTTPS (~4/s, hours) while the server was still applying the
# batch. 1,000 frames take well under a minute on a busy Pi.
APPLY_CHUNK = 1000
# The box-model tag of results that name no labeler (written before
# labeler swaps existed): stock yolo26x drew them.
STOCK_LABELER_TAG = "yolo26x"


def _load(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())


def labeler_tag(out_dir: Path) -> str:
    """The labeler that drew a cloud pass's boxes (modal_pipeline writes
    labeler.json beside them)."""
    path = out_dir / "labeler.json"
    tag = json.loads(path.read_text()) if path.is_file() else None
    return tag or STOCK_LABELER_TAG


def apply_cloud_results(prelabels_file: Path,
                        verdicts_file: Path | None = None,
                        disputes_file: Path | None = None,
                        model: str = STOCK_LABELER_TAG) -> tuple[int, int, int]:
    """(prelabeled, auto_labeled, disputed) counts, applied in batch
    requests of APPLY_CHUNK frames. Per-frame HTTPS (a TLS handshake each)
    took seconds per frame on a Pi busy with inference; a batch takes one.
    A failed batch falls back to the per-frame endpoints for its frames."""
    prelabels = _load(prelabels_file)
    verdicts = _load(verdicts_file)
    disputes = _load(disputes_file)
    stems = sorted({*prelabels, *verdicts, *disputes})
    totals, started = (0, 0, 0), time.monotonic()
    for start in range(0, len(stems), APPLY_CHUNK):
        chunk = stems[start:start + APPLY_CHUNK]
        counts = _apply_chunk(
            {s: prelabels[s] for s in chunk if s in prelabels},
            {s: verdicts[s] for s in chunk if s in verdicts},
            {s: disputes[s] for s in chunk if s in disputes}, model)
        totals = tuple(a + b for a, b in zip(totals, counts))
        if len(stems) > APPLY_CHUNK:
            log(f"applied {start + len(chunk)}/{len(stems)} frames")
    if stems:
        took = max(time.monotonic() - started, 1e-6)
        log(f"apply took {took:.0f}s ({len(stems) / took:.0f} frames/s)")
    return totals


def _apply_chunk(prelabels: dict, verdicts: dict, disputes: dict,
                 model: str) -> tuple[int, int, int]:
    payload = {"model": model, "prelabels": prelabels,
               "auto_verdicts": verdicts, "disputes": disputes}
    try:
        counts = api_post("/api/dataset/apply-cloud-results", payload,
                          timeout=BATCH_APPLY_TIMEOUT_S)
        return (counts["prelabels"], counts["auto_verdicts"],
                counts["disputes"])
    except Exception as exc:
        log(f"WARNING: batch apply failed ({exc}); using per-frame endpoints")
        return (merge_prelabels(prelabels, model), apply_auto_verdicts(verdicts),
                apply_disputes(disputes))


def merge_prelabels(boxes_by_stem: dict, model: str) -> int:
    merged = 0
    for stem, boxes in boxes_by_stem.items():
        try:
            api_post("/api/dataset/prelabels",
                     {"name": stem, "model": model, "boxes": boxes})
            merged += 1
        except Exception as exc:
            log(f"WARNING: prelabel merge failed for {stem}: {exc}")
    return merged


def apply_auto_verdicts(verdicts: dict) -> int:
    applied = 0
    for stem, verdict in verdicts.items():
        try:
            api_post("/api/dataset/autolabel", {"name": stem, "verdict": verdict})
            applied += 1
        except Exception as exc:
            log(f"WARNING: auto-label failed for {stem}: {exc}")
    return applied


def apply_disputes(disputes: dict) -> int:
    applied = 0
    for stem, dispute in disputes.items():
        try:
            api_post("/api/dataset/dispute", {"name": stem, **dispute})
            applied += 1
        except Exception as exc:
            log(f"WARNING: dispute flag failed for {stem}: {exc}")
    return applied


def apply_exam_suspects(summary: dict) -> None:
    # The candidate audits its own exam: strong disagreements with
    # held-out labels become Disputed flags for the human.
    for stem, dispute in (summary.get("exam_suspects") or {}).items():
        try:
            api_post("/api/dataset/dispute", {"name": stem, **dispute})
        except Exception as exc:
            log(f"WARNING: exam-suspect flag failed for {stem}: {exc}")


def install_bundle(bundle_dir: Path) -> None:
    if STAGING_BUNDLE.exists():
        shutil.rmtree(STAGING_BUNDLE)
    shutil.copytree(bundle_dir, STAGING_BUNDLE)
    subprocess.run(["sudo", "/usr/local/bin/doggy-install-model"], check=True)
    subprocess.run(["sudo", "/usr/bin/systemctl", "restart", "doggy"], check=True)
