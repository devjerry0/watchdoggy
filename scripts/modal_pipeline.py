"""The whole training pipeline as one Modal job, triggerable from the Pi.

    modal run scripts/modal_pipeline.py::kickoff --dataset-dir <frames> \
        --deployed-dir <current_ncnn_bundle> --out-dir <results>

The caller (the Pi's trainer daemon, or a Mac) uploads the raw labeled
frames and the currently-deployed bundle; the cloud does EVERYTHING else:
big-model prelabels (GPU, cached in the Volume), dataset build with blur
augmentation, the proven "long" fine-tune recipe, the full evaluation
(leaderboard scoring, NCNN deploy-truth, robustness stress), and the
best-model-wins deploy gate. The actual pipeline logic lives in
kitchen_training.pipeline (plus gate/consensus); this file only wraps it
in Modal functions and moves bytes. Results land in --out-dir:
summary.json, report.md, prelabels.json (fresh big-model boxes for the
review page), labeler.json (the labeler that drew them), and
kitchen_ncnn_model/ when the gate passes. kickoff_labeler runs the
labeler job (kitchen_training.labeler) and writes labeler-run.json (its
results) and report.md.

The Volume ("watchdoggy-train-3", v2) holds the model weights (seed once with
kickoff --seed-models, from a machine that has models/yolo26?.pt), the
prelabel cache, and the last few run directories.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))  # kitchen_training

MINUTES = 60
# One ceiling for EVERY cloud job (user-decided): high enough that no
# legitimate run is ever killed and its work lost, low enough to bound a
# hung container's bill (~$25 worst case on L40S + cpu/memory). The Pi-side
# wrappers must outlast this number -- trainer_daemon/env.py derives them
# from it.
CLOUD_JOB_CEILING = 10 * 60 * MINUTES
# Read on the launching machine only: inside the container this env var is
# absent, so the GPU travels to the run in the recipe (kickoff).
GPU = os.environ.get("DOGGY_TRAIN_GPU", "L40S")
# Labeling (yolo26x inference) has its own best GPU; see
# trainer_daemon/env.py AUTO_LABEL_GPU for the measurement.
LABEL_GPU = os.environ.get("DOGGY_LABEL_GPU", "L40S")
VOL = Path("/vol")
PIPELINE_ROOT = VOL / "pipeline"

app = modal.App("watchdoggy-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.88", "opencv-python~=4.10",
                    "ncnn", "pnnx")
    .add_local_python_source("kitchen_training", "mirror_sync")
)

# watchdoggy-train-3, a v2 volume (2026-10-11). watchdoggy-train-2 (v1) held
# ~192k mirror files in one folder; Modal's v1 guidance is under 50k files
# and per-file reads ran ~100x slower than a fresh volume (5-14 frames/s).
# v2 read the same files at ~1,100/s. The mirror lives in day folders
# (mirror_sync.shard_of; v2 caps a folder at 262,144 files). Never
# create_if_missing: that would make a v1 volume.
volume = modal.Volume.from_name("watchdoggy-train-3", version=2)


def _tar_bytes(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(directory.rglob("*")):
            tar.add(path, arcname=path.relative_to(directory))
    return buffer.getvalue()


def _point_kitchen_training_at_volume(run_root: Path) -> None:
    """Must run BEFORE importing kitchen_training modules: config reads the
    KT_* environment at import time."""
    _ = run_root  # runs share the delta-synced mirror; only bundles are per-run
    os.environ["KT_DATASET_DIR"] = str(PIPELINE_ROOT / "mirror")
    os.environ["KT_RUNS_DIR"] = str(PIPELINE_ROOT / "training-runs")
    os.environ["KT_BASE_MODEL"] = str(PIPELINE_ROOT / "models/yolo26n.pt")
    # The deployed model's weights; training starts here (fresh from
    # yolo26n when absent) and a deployed run replaces them.
    os.environ["KT_CHAMPION_MODEL"] = str(PIPELINE_ROOT / "models/champion.pt")
    os.environ["KT_MIRROR_SHARDED"] = "1"  # day folders on the volume
    # Stock yolo26x; a fine-tuned labeler.pt beside it takes over when
    # present (config.labeler_identity).
    os.environ["KT_PRELABEL_MODEL"] = str(PIPELINE_ROOT / "models/yolo26x.pt")
    # The ceiling-fit epoch cap (kitchen_training.hygiene.fit_epochs) must
    # know the real function timeout so a run never overruns it.
    os.environ["KT_JOB_CEILING_SECONDS"] = str(CLOUD_JOB_CEILING)
    from kitchen_training.config import cap_ncnn_threads
    cap_ncnn_threads()


# cpu/memory matter as much as the GPU here: the dataloader (decode +
# augmentation) is pure CPU, and Modal's default allocation starves it.
@app.function(image=image, gpu=GPU, cpu=8, memory=16384,
              volumes={str(VOL): volume}, timeout=CLOUD_JOB_CEILING)
def run_pipeline(run_name: str, recipe: dict,
                 fire_conf: float = 0.7) -> tuple[dict, bytes | None]:
    """The whole training day in one container. Returns (results, bundle_tar)."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.config import RUNS
    from kitchen_training.pipeline import full_run, prune_old_runs

    results, bundle = full_run(run_name, recipe, fire_conf, run_root)
    # Tar BEFORE pruning: if the Pi's clock regressed, the current run can
    # sort oldest and pruning first would delete the winning bundle.
    bundle_tar = _tar_bytes(bundle) if bundle else None
    prune_old_runs(RUNS)
    volume.commit()
    return results, bundle_tar


@app.function(image=image, gpu=LABEL_GPU, cpu=4, memory=8192,
              volumes={str(VOL): volume}, timeout=CLOUD_JOB_CEILING)
def run_prelabels(run_name: str) -> dict:
    """GPU phase of the nightly pass: big-model boxes for new frames only."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.pipeline import prelabel_phase

    results = prelabel_phase(run_name, run_root)
    volume.commit()
    return results


# memory is a floor: the RAM cache of ~11.5k 640x480 frames is ~11 GB, and
# the probe run says whether it needs more.
@app.function(image=image, gpu=GPU, cpu=8, memory=32768,
              volumes={str(VOL): volume}, timeout=CLOUD_JOB_CEILING)
def run_labeler(run_name: str, epochs: int, sample_size: int, gpu: str) -> dict:
    """The labeler job in one container: build, fine-tune, gate, and promote
    labeler.pt on a clear win. The replay runs the deployed NCNN nano on CPU
    here, so its threads must be capped (_point_kitchen_training_at_volume)."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training import labeler

    results = labeler.run_labeler(run_name, run_root, epochs, sample_size, gpu)
    volume.commit()  # a promoted labeler.pt must outlive the container
    return results


# The jury runs the deployed NCNN nano one frame per call (ultralytics'
# NCNN backend reads only im[0] of a batch), ~26 frames/s per container, so
# it is batched across containers instead: 65k frames in 4,000-frame chunks
# finish in minutes, not ~40 min in one container. Same CPU-hours.
JURY_CHUNK = 4000


@app.function(image=image, cpu=4, memory=8192,
              volumes={str(VOL): volume}, timeout=CLOUD_JOB_CEILING)
def run_consensus_chunk(run_name: str, stems: list[str]) -> dict:
    """CPU: one chunk of the jury (no GPU billed while the nano judges)."""
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.pipeline import judge_chunk_phase

    return judge_chunk_phase(run_root, stems)


@app.function(image=image, cpu=1, memory=2048,
              volumes={str(VOL): volume}, timeout=CLOUD_JOB_CEILING)
def run_label_pass(run_name: str) -> dict:
    """The whole weekly labeling pass as ONE cloud call: GPU labeling, then
    the jury fanned out over CPU containers, then the merge. A Pi that drops
    off mid-pass can no longer strand the jury step (it used to chain the
    two calls itself); the Pi collects this call's result by id instead."""
    started = time.monotonic()
    results = run_prelabels.remote(run_name)
    labeled_at = time.monotonic()
    volume.reload()  # see the cache the labeling container just committed
    run_root = PIPELINE_ROOT / "runs" / run_name
    _point_kitchen_training_at_volume(run_root)
    from kitchen_training.pipeline import finish_jury_phase, jury_plan_phase

    plan = jury_plan_phase(run_name, run_root)
    todo = plan["todo"]
    chunks = [todo[i:i + JURY_CHUNK] for i in range(0, len(todo), JURY_CHUNK)]
    print(f"[pipeline] jury fanned out over {len(chunks)} containers", flush=True)
    parts, judged, jury_started = [], 0, time.monotonic()
    # Unordered: report each chunk as it lands (the merge doesn't care).
    for part in run_consensus_chunk.map([run_name] * len(chunks), chunks,
                                        order_outputs=False):
        parts.append(part)
        judged += part["judged"]  # a disputed frame is also declined
        elapsed = time.monotonic() - jury_started
        print(f"[pipeline] jury {len(parts)}/{len(chunks)} chunks done "
              f"({judged}/{len(todo)} frames, {elapsed / 60:.1f} min, "
              f"{judged / max(elapsed, 1e-6):.0f} frames/s overall)",
              flush=True)
    results.update(finish_jury_phase(run_name, plan, parts))
    volume.commit()
    done = time.monotonic()
    print(f"[pipeline] label pass took {(done - started) / 60:.1f} min: "
          f"labeling {(labeled_at - started) / 60:.1f}, jury plan "
          f"{(jury_started - labeled_at) / 60:.1f}, jury "
          f"{(done - jury_started) / 60:.1f}", flush=True)
    return results


def _upload(dataset_dir: Path, deployed_dir: Path | None,
            seed_models: Path | None) -> str:
    run_name = time.strftime("%Y%m%d-%H%M%S")
    print(f"[pipeline] syncing {dataset_dir} and starting run {run_name} ...")
    from mirror_sync import sync_mirror  # sibling module: no ML deps
    sync_mirror(volume, dataset_dir)
    with volume.batch_upload(force=True) as up:
        if deployed_dir and deployed_dir.is_dir():
            up.put_directory(str(deployed_dir),
                             f"pipeline/runs/{run_name}/deployed_ncnn_model")
        if seed_models:
            for name in ("yolo26n.pt", "yolo26x.pt"):
                up.put_file(str(seed_models / name), f"pipeline/models/{name}")
    return run_name


def _write_results(out_dir: Path, results: dict, bundle_tar: bytes | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {key: results[key] for key in
         ("run_name", "gate", "dataset", "recipe", "ncnn_truth",
          "robustness", "robustness_deployed", "slices", "calibration",
          "ncnn_heldout", "exam_suspects")}
        | {"timings": results.get("timings")}, indent=1))
    (out_dir / "report.md").write_text(results["report_md"])
    (out_dir / "prelabels.json").write_text(json.dumps(results["prelabels"]))
    (out_dir / "labeler.json").write_text(json.dumps(results.get("labeler")))
    if bundle_tar:
        with tarfile.open(fileobj=io.BytesIO(bundle_tar), mode="r:gz") as tar:
            tar.extractall(out_dir / "kitchen_ncnn_model")
    print(f"[pipeline] results -> {out_dir} "
          f"(deploy={'YES' if bundle_tar else 'no'})")


@app.local_entrypoint()
def kickoff(dataset_dir: str, out_dir: str, deployed_dir: str = "",
            seed_models: str = "", epochs: int = 200, batch: int = 16,
            freeze: int = 10, augment: str = "auto", compare_init: bool = False,
            fire_conf: float = 0.7, call_file: str = "") -> None:
    """Full pipeline run. --seed-models <dir> uploads yolo26n/x.pt first
    (one-time, from a machine that has them). --fire-conf is the appliance's
    runtime alarm threshold: the gate judges at that operating point."""
    run_name = _upload(Path(dataset_dir),
                       Path(deployed_dir) if deployed_dir else None,
                       Path(seed_models) if seed_models else None)
    recipe = {"epochs": epochs, "batch": batch, "freeze": freeze,
              "augment": augment, "gpu": GPU, "compare_init": compare_init}
    print(f"[pipeline] running the full pipeline in the cloud "
          f"(gate @ {fire_conf}) ...")
    call = run_pipeline.spawn(run_name, recipe, fire_conf)
    _remember_call(call_file, call)
    results, bundle_tar = _wait(call)
    _write_results(Path(out_dir), results, bundle_tar)


@app.local_entrypoint()
def kickoff_prelabels(dataset_dir: str, out_dir: str,
                      deployed_dir: str = "", call_file: str = "") -> None:
    """Labeling pass: fresh ·x boxes for the review page, plus consensus
    auto-verdicts when --deployed-dir provides the second juror."""
    run_name = _upload(Path(dataset_dir),
                       Path(deployed_dir) if deployed_dir else None, None)
    print("[pipeline] labeling in the cloud ...")
    call = run_label_pass.spawn(run_name)
    _remember_call(call_file, call)
    _write_label_results(Path(out_dir), _wait(call))


@app.local_entrypoint()
def kickoff_labeler(dataset_dir: str, out_dir: str, deployed_dir: str = "",
                    epochs: int = 100, auto_sample: int = 8000,
                    call_file: str = "") -> None:
    """Labeler run: fine-tune yolo26x on the kitchen and swap it in when it
    beats the active labeler on the exam. --deployed-dir is the small model
    the jury replay needs. The defaults are labeler.LABELER_EPOCHS and
    AUTO_SAMPLE_SIZE; a probe passes a few epochs."""
    run_name = _upload(Path(dataset_dir),
                       Path(deployed_dir) if deployed_dir else None, None)
    print(f"[pipeline] training the labeler in the cloud ({epochs} epochs, "
          f"{auto_sample} auto-labels sampled) ...")
    call = run_labeler.spawn(run_name, epochs, auto_sample, GPU)
    _remember_call(call_file, call)
    _write_labeler_results(Path(out_dir), _wait(call))


# A waiting kickoff checks its cloud call this often, and gives up after
# DEAD_CHECKS checks in a row find it ended (one could race the result).
WAIT_POLL_SECONDS = 60
DEAD_CHECKS = 2
DEAD_STATUSES = {"FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT"}


def _wait(call):
    """call.get() that notices a dead run. On 2026-10-13 a labeler
    container vanished mid-epoch: Modal marked the input FAILURE but get()
    never returned, and the Pi waited 12 h with every other job queued."""
    dead = 0
    while True:
        try:
            return call.get(timeout=WAIT_POLL_SECONDS)
        except TimeoutError:
            statuses = _statuses(call.get_call_graph())
            dead = dead + 1 if statuses & DEAD_STATUSES else 0
            if dead >= DEAD_CHECKS:
                raise RuntimeError(
                    f"cloud call {call.object_id} ended without a result "
                    f"({', '.join(sorted(statuses))})") from None


def _statuses(nodes) -> set[str]:
    """Every status in a call graph, children included: run_label_pass maps
    run_consensus_chunk, and a dead chunk leaves the parent PENDING."""
    return {status for node in nodes
            for status in (node.status.name, *_statuses(node.children))}


def _remember_call(call_file: str, call) -> None:
    """Save the spawned call's id before waiting on it: if the connection
    drops (Pi Wi-Fi, a heartbeat stall), the run keeps going in the cloud
    (modal run --detach) and the trainer collects the result by this id.
    Modal keeps call results for 7 days."""
    print(f"[pipeline] cloud call {call.object_id}", flush=True)
    if call_file:
        Path(call_file).write_text(call.object_id)


@app.local_entrypoint()
def collect(call_id: str, out_dir: str, kind: str) -> None:
    """Fetch a spawned run's result by call id (waits if it is still
    running, fails if it ended without a result) and write the same out_dir
    files its kickoff would have."""
    print(f"[pipeline] collecting cloud call {call_id} ...", flush=True)
    result = _wait(modal.FunctionCall.from_id(call_id))
    if kind == "train":
        _write_results(Path(out_dir), *result)
    elif kind == "labeler":
        _write_labeler_results(Path(out_dir), result)
    else:
        _write_label_results(Path(out_dir), result)


def _write_labeler_results(out: Path, results: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "labeler-run.json").write_text(json.dumps(
        {key: value for key, value in results.items() if key != "report_md"},
        indent=1))
    (out / "report.md").write_text(results["report_md"])
    print(f"[pipeline] labeler {results['decision']}: {results['reason']} "
          f"-> {out}")


def _write_label_results(out: Path, results: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "prelabels.json").write_text(json.dumps(results["prelabels"]))
    # The labeler that drew these boxes: the Pi tags each sidecar with it.
    (out / "labeler.json").write_text(json.dumps(results.get("labeler")))
    (out / "auto_verdicts.json").write_text(
        json.dumps(results.get("auto_verdicts", {})))
    (out / "disputes.json").write_text(json.dumps(results.get("disputes", {})))
    (out / "clear_auto.json").write_text(
        json.dumps(results.get("clear_auto", [])))
    print(f"[pipeline] {len(results['prelabels'])} new frames prelabeled, "
          f"{len(results.get('auto_verdicts', {}))} auto-labeled, "
          f"{len(results.get('disputes', {}))} labels disputed, "
          f"{len(results.get('clear_auto', []))} auto-labels cleared -> {out}")
