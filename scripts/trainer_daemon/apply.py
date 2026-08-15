"""Pushing cloud results back into the appliance: sidecar merges through
the local web API, and the root-helper model install."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from trainer_daemon.env import STAGING_BUNDLE, api_post, log


BATCH_APPLY_TIMEOUT_S = 180


def _load(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())


def apply_cloud_results(prelabels_file: Path,
                        verdicts_file: Path | None = None,
                        disputes_file: Path | None = None) -> tuple[int, int, int]:
    """(prelabeled, auto_labeled, disputed) counts, applied in ONE request.
    Per-frame HTTPS (a TLS handshake each) took seconds per frame on a Pi
    busy with inference; the batch endpoint takes one. Falls back to the
    per-frame endpoints if the batch call fails."""
    payload = {"model": "yolo26x",
               "prelabels": _load(prelabels_file),
               "auto_verdicts": _load(verdicts_file),
               "disputes": _load(disputes_file)}
    if not any((payload["prelabels"], payload["auto_verdicts"],
                payload["disputes"])):
        return 0, 0, 0
    try:
        counts = api_post("/api/dataset/apply-cloud-results", payload,
                          timeout=BATCH_APPLY_TIMEOUT_S)
        return (counts["prelabels"], counts["auto_verdicts"],
                counts["disputes"])
    except Exception as exc:
        log(f"WARNING: batch apply failed ({exc}); using per-frame endpoints")
        autos = apply_auto_verdicts(verdicts_file) if verdicts_file else 0
        flags = apply_disputes(disputes_file) if disputes_file else 0
        return merge_prelabels(prelabels_file), autos, flags


def merge_prelabels(prelabels_file: Path) -> int:
    if not prelabels_file.is_file():
        return 0
    boxes_by_stem = json.loads(prelabels_file.read_text())
    merged = 0
    for stem, boxes in boxes_by_stem.items():
        try:
            api_post("/api/dataset/prelabels",
                     {"name": stem, "model": "yolo26x", "boxes": boxes})
            merged += 1
        except Exception as exc:
            log(f"WARNING: prelabel merge failed for {stem}: {exc}")
    return merged


def apply_auto_verdicts(verdicts_file: Path) -> int:
    if not verdicts_file.is_file():
        return 0
    applied = 0
    for stem, verdict in json.loads(verdicts_file.read_text()).items():
        try:
            api_post("/api/dataset/autolabel", {"name": stem, "verdict": verdict})
            applied += 1
        except Exception as exc:
            log(f"WARNING: auto-label failed for {stem}: {exc}")
    return applied


def apply_disputes(disputes_file: Path) -> int:
    if not disputes_file.is_file():
        return 0
    applied = 0
    for stem, dispute in json.loads(disputes_file.read_text()).items():
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
