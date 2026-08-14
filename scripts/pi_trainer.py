#!/usr/bin/env python3
"""The Pi's trainer daemon: one pass per invocation (systemd timer, ~30 min).

Runs as the dedicated `trainer` user -- the only UID on the appliance with
cloud egress (nftables exception; the detector stays provably offline). Each
pass: pick up a queued job from the web UI, or synthesize one from the auto
rules, then send it to Modal and apply the results:

  prelabel  fresh big-model boxes for new frames, merged into sidecars via
            the local web API (so the review page shows them).
            Auto rule: >= AUTO_PRELABEL_MIN frames lack prelabels.
  train     the full cloud pipeline; if the deploy gate passes, the returned
            NCNN bundle is installed and the detector restarted.
            Auto rule: >= MIN_NEW_LABELS frames labeled since the last done
            train job AND that job is older than TRAIN_INTERVAL seconds.

Stdlib only. Progress is written to job_<id>.result.json files, which the
web's /api/training/status overlays on the originals (different users can't
edit each other's files).
"""
from __future__ import annotations

import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

DOGGY_ROOT = Path("/home/doggy/doggy")
JOBS_DIR = DOGGY_ROOT / "jobs"
DATASET_DIR = DOGGY_ROOT / "dataset"
DEPLOYED_BUNDLE = DOGGY_ROOT / "models/kitchen_ncnn_model"
STAGING_BUNDLE = Path.home() / "staging_ncnn_model"
MODAL = Path.home() / "modal-env/bin/modal"
PIPELINE = DOGGY_ROOT / "scripts/modal_pipeline.py"
LOCAL_API = "https://localhost:8443"

AUTO_PRELABEL_MIN = 10
PRELABEL_COOLDOWN = 6 * 3600.0
STALE_RUNNING = 3 * 3600.0
# Recipe + schedule defaults; the training page's settings file overrides.
SETTINGS_DEFAULTS = {"epochs": 200, "batch": 16, "freeze": 10, "augment": True,
                     "train_interval_hours": 48, "min_new_labels": 5,
                     "nightly_prelabel_hour": 2, "gpu": "auto"}
# gpu=auto tiers by labeled-frame count. The nano model is dataloader-bound
# on small sets -- a bigger GPU only pays once epochs are long enough.
GPU_TIERS = ((2500, "A10G"), (0, "L4"))


def _settings() -> dict:
    merged = dict(SETTINGS_DEFAULTS)
    path = JOBS_DIR / "trainer-settings.json"
    if path.is_file():
        try:
            merged.update(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return merged


def log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def _api(path: str, payload: dict) -> None:
    context = ssl.create_default_context()  # household CA: skip verification
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(
        f"{LOCAL_API}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, context=context, timeout=15):
        pass


def _jobs() -> list[dict]:
    jobs = []
    for path in sorted(JOBS_DIR.glob("job_*.json")):
        if path.name.endswith(".result.json"):
            continue
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        result_path = path.with_name(f"{path.stem}.result.json")
        if result_path.is_file():
            job.update(json.loads(result_path.read_text()))
        jobs.append(job)
    return jobs


def _write_result(job_id: str, status: str, detail: str,
                  extra: dict | None = None) -> None:
    payload = {"status": status, "detail": detail, "updated_at": time.time()}
    payload.update(extra or {})
    (JOBS_DIR / f"{job_id}.result.json").write_text(json.dumps(payload))


def _sidecar_stats() -> tuple[int, int, float]:
    """(missing_prelabels, labeled_count, newest_labeled_at)."""
    missing, labeled, newest = 0, 0, 0.0
    for sidecar in DATASET_DIR.glob("sample_*.json"):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if "prelabels" not in meta:
            missing += 1
        if not meta.get("human_label"):
            continue
        labeled += 1
        newest = max(newest, meta.get("labeled_at", 0.0))
    return missing, labeled, newest


def _labels_since(when: float) -> int:
    count = 0
    for sidecar in DATASET_DIR.glob("sample_*.json"):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("human_label") and meta.get("labeled_at", 0.0) > when:
            count += 1
    return count


def _last_nightly_slot(hour: int) -> float:
    """The most recent occurrence of the nightly hour, as a timestamp."""
    slot = time.localtime()
    seconds_today = slot.tm_hour * 3600 + slot.tm_min * 60 + slot.tm_sec
    slot_offset = hour * 3600
    if seconds_today >= slot_offset:
        return time.time() - (seconds_today - slot_offset)
    return time.time() - seconds_today - 24 * 3600 + slot_offset


def _synthesize_job(jobs: list[dict]) -> dict | None:
    def newest_done(kind: str) -> float:
        return max((j.get("updated_at", 0.0) for j in jobs
                    if j.get("kind") == kind and j.get("status") == "done"),
                   default=0.0)

    conf = _settings()
    now = time.time()
    last_train = newest_done("train")
    new_labels = _labels_since(last_train)
    if (now - last_train >= conf["train_interval_hours"] * 3600.0
            and new_labels >= conf["min_new_labels"]):
        return _queue_auto("train",
                           f"auto: {new_labels} new labels since last run")
    missing, _, _ = _sidecar_stats()
    last_prelabel = newest_done("prelabel")
    # Nightly: after the configured hour, prelabel EVERY new frame once, so
    # the morning's label queue is already stocked with ·x boxes.
    if missing > 0 and last_prelabel < _last_nightly_slot(conf["nightly_prelabel_hour"]):
        return _queue_auto("prelabel",
                           f"nightly: {missing} new frames to prelabel")
    if missing >= AUTO_PRELABEL_MIN and now - last_prelabel >= PRELABEL_COOLDOWN:
        return _queue_auto("prelabel", f"auto: {missing} frames lack prelabels")
    return None


def _queue_auto(kind: str, why: str) -> dict:
    job = {"id": f"job_{int(time.time() * 1000)}", "kind": kind,
           "status": "queued", "requested_at": time.time(),
           "updated_at": time.time(), "detail": why, "auto": True}
    (JOBS_DIR / f"{job['id']}.json").write_text(json.dumps(job))
    log(f"auto-queued {kind}: {why}")
    return job


def _merge_prelabels(prelabels_file: Path) -> int:
    if not prelabels_file.is_file():
        return 0
    boxes_by_stem = json.loads(prelabels_file.read_text())
    merged = 0
    for stem, boxes in boxes_by_stem.items():
        try:
            _api("/api/dataset/prelabels",
                 {"name": stem, "model": "yolo26x", "boxes": boxes})
            merged += 1
        except Exception as exc:
            log(f"WARNING: prelabel merge failed for {stem}: {exc}")
    return merged


def _billing_summary() -> dict | None:
    """Workspace spend this month, straight from Modal. The workspace runs
    only this appliance, so before/after deltas attribute cost per run."""
    try:
        proc = subprocess.run([str(MODAL), "billing", "summary", "--json"],
                              capture_output=True, timeout=60, check=True)
        summary = json.loads(proc.stdout)
        return {"metered_cost": float(summary.get("metered_cost", 0)),
                "billed_cost": float(summary.get("billed_cost", 0)),
                "credits_used": -float(summary.get("adjustments", {})
                                       .get("credits", 0)),
                "fetched_at": time.time()}
    except Exception as exc:
        log(f"WARNING: billing summary unavailable: {exc}")
        return None


def _write_billing() -> dict | None:
    summary = _billing_summary()
    if summary:
        (JOBS_DIR / "billing.json").write_text(json.dumps(summary))
    return summary


def _gpu() -> str:
    chosen = _settings().get("gpu", "auto")
    if chosen != "auto":
        return chosen
    _, labeled, _ = _sidecar_stats()
    return next(tier for floor, tier in GPU_TIERS if labeled >= floor)


def _modal(entrypoint: str, arguments: list[str], job_id: str) -> None:
    # The full cloud-run output streams into the job's log file, which the
    # training page tails live via /api/training/log/{job_id}.
    gpu = _gpu()
    log(f"cloud GPU: {gpu}")
    with open(JOBS_DIR / f"{job_id}.log", "ab") as log_file:
        subprocess.run([str(MODAL), "run", f"{PIPELINE}::{entrypoint}",
                        "--dataset-dir", str(DATASET_DIR)] + arguments,
                       check=True, cwd=DOGGY_ROOT, timeout=3 * 3600,
                       stdout=log_file, stderr=subprocess.STDOUT,
                       env={**os.environ, "DOGGY_TRAIN_GPU": gpu})


def _install_bundle(bundle_dir: Path) -> None:
    if STAGING_BUNDLE.exists():
        shutil.rmtree(STAGING_BUNDLE)
    shutil.copytree(bundle_dir, STAGING_BUNDLE)
    subprocess.run(["sudo", "/usr/local/bin/doggy-install-model"], check=True)
    subprocess.run(["sudo", "/usr/bin/systemctl", "restart", "doggy"], check=True)


def _run_cost(before: dict | None) -> float | None:
    after = _write_billing()
    if not (before and after):
        return None
    return round(max(0.0, after["metered_cost"] - before["metered_cost"]), 2)


def _run_prelabel_job(job: dict) -> str:
    before = _billing_summary()
    with tempfile.TemporaryDirectory() as tmp:
        _modal("kickoff_prelabels", ["--out-dir", tmp], job["id"])
        merged = _merge_prelabels(Path(tmp) / "prelabels.json")
    cost = _run_cost(before)
    return (f"{merged} frames prelabeled"
            + (f" (${cost:.2f})" if cost is not None else ""))


def _runtime_confidence() -> float:
    """The appliance's live alarm threshold: the gate must judge there."""
    env = DOGGY_ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("DOGGY_CONFIDENCE="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    break
    return 0.7


def _recipe_arguments(job: dict) -> list[str]:
    recipe = {**_settings(), **(job.get("params") or {})}
    return ["--epochs", str(recipe["epochs"]), "--batch", str(recipe["batch"]),
            "--freeze", str(recipe["freeze"]),
            "--fire-conf", str(_runtime_confidence()),
            "--augment" if recipe["augment"] else "--no-augment"]


def _run_train_job(job: dict) -> str:
    started = time.time()
    before = _billing_summary()
    with tempfile.TemporaryDirectory() as tmp:
        arguments = ["--out-dir", tmp] + _recipe_arguments(job)
        if DEPLOYED_BUNDLE.is_dir():
            arguments += ["--deployed-dir", str(DEPLOYED_BUNDLE)]
        _modal("kickoff", arguments, job["id"])
        out = Path(tmp)
        merged = _merge_prelabels(out / "prelabels.json")
        summary = json.loads((out / "summary.json").read_text())
        (JOBS_DIR / f"{job['id']}.report.md").write_text(
            (out / "report.md").read_text())
        # The scorecard the training page renders: gate compare, threshold
        # curve, robustness, dataset shape, recipe, duration.
        job["_summary"] = {key: summary.get(key) for key in
                           ("gate", "dataset", "recipe", "fire_conf",
                            "ncnn_truth", "robustness")}
        job["_summary"]["duration_s"] = round(time.time() - started)
        job["_summary"]["cost_usd"] = _run_cost(before)
        gate = summary["gate"]
        heldout = summary["ncnn_heldout"]
        verdict = (f"held-out {heldout['caught']}/"
                   f"{heldout['caught'] + heldout['missed']} catches, "
                   f"{heldout['false_fires']} FP")
        if not gate["deploy"]:
            return f"kept current model ({gate['reason']}; new: {verdict})"
        _install_bundle(out / "kitchen_ncnn_model")
    if merged:
        log(f"merged prelabels for {merged} frames")
    return f"DEPLOYED new model ({verdict})"


_RUNNERS = {"prelabel": _run_prelabel_job, "train": _run_train_job}


def main() -> int:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _write_billing()  # keep the training page's budget card fresh
    jobs = _jobs()
    for job in jobs:
        if job.get("status") != "running":
            continue
        if time.time() - job.get("updated_at", 0.0) < STALE_RUNNING:
            log(f"{job['id']} still running; nothing to do")
            return 0
        _write_result(job["id"], "failed", "stale: gave up after 3h")

    queued = [j for j in _jobs() if j.get("status") == "queued"]
    job = min(queued, key=lambda j: j.get("requested_at", 0.0), default=None)
    if job is None:
        job = _synthesize_job(_jobs())
    if job is None:
        log("nothing to do")
        return 0

    log(f"running {job['kind']} job {job['id']}")
    _write_result(job["id"], "running", "")
    try:
        detail = _RUNNERS[job["kind"]](job)
    except Exception as exc:
        log(f"job {job['id']} FAILED: {exc}")
        _write_result(job["id"], "failed", str(exc)[:300])
        return 1
    extra = {"summary": job["_summary"]} if "_summary" in job else None
    _write_result(job["id"], "done", detail, extra)
    log(f"job {job['id']} done: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
