"""Score models the way the appliance fires, plus ranking and deploy-truth.

Rock-solid rules: each model gets ONE prediction pass recording its max dog
confidence per frame, so the whole threshold curve comes from the same
evidence; the held-out split is the headline (training frames can only
flatter a model); box-level mAP50 on the held-out split cross-checks the
frame-level metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

from kitchen_training.config import (DATASET_MIRROR, EVAL_CONF_FLOOR, FIRE_CONF,
                                     IMAGE_SIZE, THRESHOLDS, Frame, device, log)
from kitchen_training.dataset import labeled_sidecars

CONF_DECIMALS = 3


def eval_frames(run_dir: Path) -> list[Frame]:
    val_stems = {p.stem for p in (run_dir / "dataset/images/val").glob("*.jpg")}
    frames = [Frame(stem, verdict in ("dog", "dog_mixed"), stem in val_stems,
                    DATASET_MIRROR / f"{stem}.jpg")
              for stem, verdict, _ in labeled_sidecars()]
    if not frames:
        sys.exit("[train] FAILED: no labeled frames to evaluate")
    return frames


def _max_dog_conf(model, source, run_device: str) -> float:
    """Max dog confidence in one image (a path or a decoded array)."""
    prediction = model.predict(source, conf=EVAL_CONF_FLOOR, imgsz=IMAGE_SIZE,
                               device=run_device, verbose=False)[0]
    return round(max(
        (float(box.conf[0]) for box in prediction.boxes
         if prediction.names[int(box.cls[0])] == "dog"), default=0.0),
        CONF_DECIMALS)


def dog_confs(model, frames: list[Frame], run_device: str) -> dict[str, float]:
    """Max dog confidence per frame: ONE prediction pass yields the whole
    threshold curve."""
    return {frame.stem: _max_dog_conf(model, str(frame.jpg), run_device)
            for frame in frames}


def score(frames: list[Frame], confs: dict, threshold: float,
          heldout_only: bool) -> dict:
    results = [(frame.is_dog, confs[frame.stem] >= threshold) for frame in frames
               if frame.heldout or not heldout_only]
    return {"caught": sum(1 for is_dog, fired in results if is_dog and fired),
            "missed": sum(1 for is_dog, fired in results if is_dog and not fired),
            "false_fires": sum(1 for is_dog, fired in results
                               if not is_dog and fired),
            "quiet": sum(1 for is_dog, fired in results
                         if not is_dog and not fired)}


def curve(frames: list[Frame], confs: dict) -> dict:
    return {f"{t:.1f}": {"all": score(frames, confs, t, False),
                         "heldout": score(frames, confs, t, True)}
            for t in THRESHOLDS}


def _val_map50(model, run_dir: Path) -> float | None:
    try:
        return round(float(model.val(
            data=str(run_dir / "dataset/data.yaml"), device=device(),
            verbose=False, plots=False).box.map50), CONF_DECIMALS)
    except Exception:
        return None  # the 80-class baseline can't val on 2-class data


def evaluate(run_dir: Path, weights_by_name: dict[str, Path]) -> dict:
    """Score every model on every labeled frame, the way the appliance fires."""
    from ultralytics import YOLO

    frames = eval_frames(run_dir)
    result = {"n_frames": len(frames),
              "heldout_dogs": sum(1 for f in frames if f.heldout and f.is_dog),
              "heldout_nondogs": sum(1 for f in frames
                                     if f.heldout and not f.is_dog),
              "models": {}}
    log(f"evaluating {len(weights_by_name)} models on {len(frames)} frames "
        f"({result['heldout_dogs']}+{result['heldout_nondogs']} held out)")

    for name, weights in weights_by_name.items():
        model = YOLO(str(weights))
        confs = dog_confs(model, frames, device())
        result["models"][name] = {"map50_val": _val_map50(model, run_dir),
                                  "confs": confs,
                                  "curve": curve(frames, confs)}
        heldout = result["models"][name]["curve"][f"{FIRE_CONF:.1f}"]["heldout"]
        log(f"  {name}: held-out @{FIRE_CONF} catches "
            f"{heldout['caught']}/{heldout['caught'] + heldout['missed']}, "
            f"false-fires {heldout['false_fires']} | val mAP50 "
            f"{result['models'][name]['map50_val']}")
    return result


def rank_key(model_metrics: dict):
    """Best model first: fewest held-out false fires, then most held-out
    catches, then the all-frames tiebreaks, then mAP50 -- all at FIRE_CONF."""
    heldout = model_metrics["curve"][f"{FIRE_CONF:.1f}"]["heldout"]
    all_frames = model_metrics["curve"][f"{FIRE_CONF:.1f}"]["all"]
    return (heldout["false_fires"], -heldout["caught"],
            all_frames["false_fires"], -all_frames["caught"],
            -(model_metrics["map50_val"] or 0.0))


def pick_winner(metrics: dict) -> str:
    """Best non-baseline model by rank_key."""
    candidates = {name: m for name, m in metrics["models"].items()
                  if name != "baseline"}
    return min(candidates, key=lambda name: rank_key(candidates[name]))


def _tally_variants(model, frame: Frame, variants: dict, scores: dict) -> None:
    """Score one frame's stress variants into the running per-variant tallies."""
    for name, variant in variants.items():
        fired = _max_dog_conf(model, variant, "cpu") >= FIRE_CONF
        tally = scores.setdefault(name, {"caught": 0, "dogs": 0,
                                         "false_fires": 0, "nondogs": 0})
        if frame.is_dog:
            tally["dogs"] += 1
            tally["caught"] += fired
            continue
        tally["nondogs"] += 1
        tally["false_fires"] += fired


def robustness(run_dir: Path, ncnn_dir: Path) -> dict:
    """Stress the deployable bundle on held-out frames: reliability is what
    survives blur, bad light, compression, and a bumped camera mount."""
    import cv2
    from ultralytics import YOLO
    from kitchen_training.perturb import stress_variants
    log("stress-testing the NCNN bundle on held-out frames ...")
    model = YOLO(str(ncnn_dir), task="detect")
    heldout = [f for f in eval_frames(run_dir) if f.heldout]
    scores: dict[str, dict] = {}
    for frame in heldout:
        image = cv2.imread(str(frame.jpg))
        variants = {"original": image, **stress_variants(image)}
        _tally_variants(model, frame, variants, scores)
    for name, tally in scores.items():
        log(f"  {name:>8}: catches {tally['caught']}/{tally['dogs']}, "
            f"false fires {tally['false_fires']}/{tally['nondogs']}")
    return scores


def ncnn_truth(run_dir: Path, ncnn_dir: Path) -> dict:
    """Deploy-truth: score the exported NCNN bundle itself. The NCNN export
    silently swaps YOLO26's end2end head for the classic NMS head, so its
    confidence distribution differs from the .pt the leaderboard scored --
    and the Pi runs THIS, not the .pt."""
    from ultralytics import YOLO
    log("scoring the NCNN bundle (deploy-truth) ...")
    frames = eval_frames(run_dir)
    confs = dog_confs(YOLO(str(ncnn_dir), task="detect"), frames, "cpu")
    out = curve(frames, confs)
    out["max_nondog_conf"] = round(max(
        (confs[f.stem] for f in frames if not f.is_dog), default=0.0),
        CONF_DECIMALS)
    heldout = out[f"{FIRE_CONF:.1f}"]["heldout"]
    log(f"  NCNN held-out @{FIRE_CONF}: catches "
        f"{heldout['caught']}/{heldout['caught'] + heldout['missed']}, "
        f"false-fires {heldout['false_fires']} | max non-dog conf "
        f"{out['max_nondog_conf']}")
    return out
