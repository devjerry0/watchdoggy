#!/usr/bin/env python3
"""Fine-tune the kitchen dog detector on the labeled dataset, end to end.

One command runs the whole training day:

    uv run python scripts/train_kitchen_model.py --host doggy@doggypi.local

Stages (each skippable):
  pull       rsync the labeled dataset from the Pi
  build      fuse human verdicts + big-model prelabels into a YOLO dataset
  train      fine-tune the nano model (frozen backbone, early stopping);
             --backend modal sends this one stage to a Modal cloud GPU
  evaluate   baseline vs fine-tune, scored the way the appliance fires
  export     NCNN bundle ready to rsync to the Pi
  report     dataset stats, metrics, and changed frames, in the run dir

Everything lands in training-runs/<timestamp>/ so runs are comparable.
This script IS the future dashboard button's backend: keep it argument-driven
and side-effect free outside its run directory (and the dataset mirror).

Label fusion rules (the heart of it):
  hand boxes drawn on /review are the complete truth for a frame and
             override everything below.
  dog        keep big-model dog+person boxes; if the big model missed the dog,
             fall back to the nano's sidecar box; drop the frame if neither.
  dog_mixed  big-model dog boxes that heavily overlap a person box are the
             misclassified person -- dropped; the rest are real.
  person     every dog box is a lie (that is the point); keep person boxes.
  no_dog     legacy coarse verdict: same as person.
  empty      background: empty label file (a hard negative).
  skip       excluded.

The split is STABLE: a frame's stem hash decides train/val forever, so growing
the dataset never moves old frames across the split (no quiet contamination).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET_MIRROR = REPO / "dataset-pull"
RUNS = REPO / "training-runs"
PRELABEL_CACHE = RUNS / "prelabel-cache.json"

BASE_MODEL = REPO / "models/yolo26n.pt"
PRELABEL_MODEL = REPO / "models/yolo26x.pt"
PRELABEL_CONF = 0.45
FIRE_CONF = 0.7          # the appliance's default alarm threshold
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)  # the eval scores the whole curve
VAL_FRACTION_MOD = 4     # stem-hash % 4 == 0 -> val (~25%)
MIXED_IOU_DROP = 0.6     # dog box overlapping a person this much is the person

CLS = {"person": 0, "dog": 1}


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def sh(cmd: list[str], env: dict | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


# Hyperparameter sweep, dispatched to Modal in parallel (<=10 concurrent).
# Base recipe (the proven anchor): epochs 80, patience 20, imgsz 640,
# batch 16, freeze 10. Each entry lists only its deltas. Note: lr0 only
# takes effect with an explicit optimizer -- optimizer=auto overrides it.
SWEEP_CONFIGS = {
    "anchor": {},                          # the proven recipe, as-is
    "anchor-s1": {"seed": 1},              # same recipe: run-to-run noise probe
    "full-ft": {"freeze": 0},              # unfreeze the whole backbone
    "head-only": {"freeze": 20},           # touch almost nothing but the head
    "no-mosaic": {"mosaic": 0.0},          # mosaic aug distorts scale cues
    "no-mosaic-ft": {"freeze": 0, "mosaic": 0.0},
    "long": {"epochs": 200, "patience": 50},
    "gentle-ft": {"freeze": 0, "lr0": 0.0005, "cos_lr": True,
                  "optimizer": "AdamW"},
    "cls-heavy": {"cls": 1.0},             # our failure IS classification
    "low-lr": {"lr0": 0.0005, "optimizer": "AdamW"},
}


# -- stage: pull ------------------------------------------------------------

def pull(host: str) -> None:
    log(f"pulling dataset from {host} ...")
    DATASET_MIRROR.mkdir(exist_ok=True)
    sh(["rsync", "-az", f"{host}:doggy/dataset/", str(DATASET_MIRROR) + "/"])
    n = len(list(DATASET_MIRROR.glob("sample_*.json")))
    log(f"dataset mirror: {n} samples")


# -- prelabels (cached: the big model runs once per frame, ever) ------------

def _load_cache() -> dict:
    if PRELABEL_CACHE.is_file():
        return json.loads(PRELABEL_CACHE.read_text())
    return {}


def prelabel(stems: list[str]) -> dict:
    cache = _load_cache()
    todo = [s for s in stems if s not in cache]
    if todo:
        from ultralytics import YOLO
        log(f"prelabeling {len(todo)} new frames with {PRELABEL_MODEL.name} ...")
        model = YOLO(str(PRELABEL_MODEL))
        for stem in todo:
            jpg = DATASET_MIRROR / f"{stem}.jpg"
            res = model.predict(str(jpg), conf=PRELABEL_CONF, imgsz=640,
                                device=_device(), verbose=False)[0]
            dogs, people = [], []
            for b in res.boxes:
                label = res.names[int(b.cls[0])]
                box = [round(float(v), 1) for v in b.xyxy[0].tolist()]
                if label == "dog":
                    dogs.append(box)
                elif label == "person":
                    people.append(box)
            cache[stem] = {"dogs": dogs, "people": people,
                           "shape": list(res.orig_shape)}
        RUNS.mkdir(exist_ok=True)
        PRELABEL_CACHE.write_text(json.dumps(cache))
    log(f"prelabels ready ({len(stems)} frames, {len(todo)} newly computed)")
    return cache


def prelabel_push(pi_url: str) -> None:
    """Run the big model over EVERY mirrored frame (cache-aware) and push the
    boxes into the Pi's sidecars, so the review page shows the exact boxes
    training will use and the box editor seeds from them."""
    import ssl
    import urllib.request

    stems = [p.stem for p in sorted(DATASET_MIRROR.glob("sample_*.jpg"))
             if p.with_suffix(".json").is_file()]
    cache = prelabel(stems)
    ctx = ssl.create_default_context()  # household CA isn't in the system store
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    pushed, failed = 0, 0
    for stem in stems:
        pre = cache[stem]
        boxes = ([{"label": "dog", "box": b} for b in pre["dogs"]]
                 + [{"label": "person", "box": b} for b in pre["people"]])
        req = urllib.request.Request(
            f"{pi_url}/api/dataset/prelabels",
            data=json.dumps({"name": stem, "model": PRELABEL_MODEL.stem,
                             "boxes": boxes}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=10):
                pushed += 1
        except Exception as exc:
            failed += 1
            if failed <= 3:
                log(f"WARNING: push failed for {stem}: {exc}")
    log(f"prelabels pushed for {pushed}/{len(stems)} frames"
        + (f" ({failed} failed)" if failed else ""))


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


# -- stage: build -----------------------------------------------------------

def build(run_dir: Path) -> dict:
    samples = []
    for side in sorted(DATASET_MIRROR.glob("sample_*.json")):
        meta = json.loads(side.read_text())
        verdict = meta.get("human_label")
        if not verdict or verdict == "skip":
            continue
        if not side.with_suffix(".jpg").is_file():
            continue
        samples.append((side.stem, verdict, meta))
    log(f"building dataset from {len(samples)} verdicted frames")
    cache = prelabel([s for s, _, _ in samples])

    ds = run_dir / "dataset"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ds / sub).mkdir(parents=True)

    stats = {"train": 0, "val": 0, "dropped": [], "fallback": 0, "hand_boxed": 0,
             "boxes": {"train": {"person": 0, "dog": 0},
                       "val": {"person": 0, "dog": 0}},
             "verdicts": {}}

    def yolo_line(cls: int, box, w, h) -> str:
        x1, y1, x2, y2 = box
        return (f"{cls} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")

    for stem, verdict, meta in samples:
        pre = cache[stem]
        h, w = pre["shape"]
        dogs, people = list(pre["dogs"]), list(pre["people"])
        stats["verdicts"][verdict] = stats["verdicts"].get(verdict, 0) + 1

        lines: list[str] = []
        hand = meta.get("human_boxes")
        if isinstance(hand, list):
            # Hand-drawn boxes are the complete truth for this frame; no
            # model, fallback, or overlap heuristic gets a say.
            lines = [yolo_line(CLS[b["label"]], b["box"], w, h) for b in hand]
            stats["hand_boxed"] += 1
        elif verdict in ("dog", "dog_mixed"):
            if verdict == "dog_mixed":
                # The human says one of the dog boxes is a person in disguise.
                dogs = [d for d in dogs
                        if not any(_iou(d, p) >= MIXED_IOU_DROP for p in people)]
            if not dogs:
                fallback = [d["box"] for d in
                            meta.get("detections", {}).get("targets", [])
                            if d.get("label") == "dog"
                            and d.get("confidence", 0) >= PRELABEL_CONF]
                dogs = [list(map(float, b)) for b in fallback]
                if dogs:
                    stats["fallback"] += 1
            if not dogs:
                stats["dropped"].append(
                    (stem, "dog verdict but no model drew a box -- "
                           "draw one by hand on /review"))
                continue
            lines += [yolo_line(CLS["dog"], b, w, h) for b in dogs]
            lines += [yolo_line(CLS["person"], b, w, h) for b in people]
        elif verdict in ("person", "no_dog"):
            lines += [yolo_line(CLS["person"], b, w, h) for b in people]
        elif verdict == "empty":
            pass

        # Stable split: the stem's hash decides, forever.
        digest = int(hashlib.sha1(stem.encode()).hexdigest(), 16)
        split = "val" if digest % VAL_FRACTION_MOD == 0 else "train"
        shutil.copyfile(DATASET_MIRROR / f"{stem}.jpg", ds / f"images/{split}/{stem}.jpg")
        (ds / f"labels/{split}/{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        stats[split] += 1
        for ln in lines:
            key = "person" if ln.startswith(f"{CLS['person']} ") else "dog"
            stats["boxes"][split][key] += 1

    (ds / "data.yaml").write_text(
        f"path: {ds.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\n  1: dog\n")
    log(f"train {stats['train']} imgs {stats['boxes']['train']} | "
        f"val {stats['val']} imgs {stats['boxes']['val']} | "
        f"hand-boxed {stats['hand_boxed']} | dropped {len(stats['dropped'])} | "
        f"nano-fallback {stats['fallback']}")
    for stem, why in stats["dropped"]:
        log(f"  NEEDS BOXES: {stem} ({why})")
    return stats


# -- stage: train -----------------------------------------------------------

def train(run_dir: Path, backend: str, epochs: int, batch: int, freeze: int) -> Path:
    # Both backends land the weights at run_dir/best.pt -- the one contract
    # eval/export rely on, wherever the GPU actually was.
    best = run_dir / "best.pt"
    if backend == "modal":
        log(f"dispatching training job to Modal: epochs<={epochs}, freeze={freeze}")
        try:
            sh(["uv", "run", "modal", "run", str(REPO / "scripts/modal_train.py"),
                "--run-dir", str(run_dir), "--weights", str(BASE_MODEL),
                "--epochs", str(epochs), "--batch", str(batch),
                "--freeze", str(freeze)])
        except subprocess.CalledProcessError:
            sys.exit("[train] Modal job failed. First time? Run: uv run modal setup")
    else:
        from ultralytics import YOLO
        log(f"training {BASE_MODEL.name} locally: epochs<={epochs}, "
            f"freeze={freeze}, device={_device()}")
        model = YOLO(str(BASE_MODEL))
        model.train(data=str(run_dir / "dataset/data.yaml"),
                    epochs=epochs, patience=max(10, epochs // 4), imgsz=640,
                    batch=batch, freeze=freeze, device=_device(),
                    project=str(run_dir), name="train", exist_ok=True,
                    verbose=False, plots=True)
        # Ultralytics decides the save dir (its settings can reroute project
        # paths); ask the trainer where it actually wrote instead of guessing.
        produced = Path(model.trainer.save_dir) / "weights/best.pt"
        if produced.is_file():
            shutil.copyfile(produced, best)
    if not best.is_file():
        sys.exit("[train] FAILED: no best.pt produced")
    log(f"best weights: {best}")
    return best


def sweep(run_dir: Path, gpu: str) -> dict[str, Path]:
    """Fan the sweep configs out to Modal; return name -> local weights."""
    cfg_path = run_dir / "sweep-configs.json"
    cfg_path.write_text(json.dumps(SWEEP_CONFIGS, indent=1))
    log(f"dispatching {len(SWEEP_CONFIGS)} training jobs to Modal on {gpu}s ...")
    try:
        sh(["uv", "run", "modal", "run", f"{REPO / 'scripts/modal_train.py'}::sweep",
            "--run-dir", str(run_dir), "--weights", str(BASE_MODEL),
            "--configs", str(cfg_path)])
    except subprocess.CalledProcessError:
        sys.exit("[train] Modal sweep failed. First time? Run: uv run modal setup")
    out = {}
    for name in SWEEP_CONFIGS:
        p = run_dir / "sweep" / f"{name}.pt"
        if p.is_file():
            out[name] = p
        else:
            log(f"WARNING: sweep job {name} returned no weights")
    if not out:
        sys.exit("[train] FAILED: sweep produced no weights at all")
    return out


# -- stage: evaluate --------------------------------------------------------

def evaluate(run_dir: Path, weights_by_name: dict[str, Path]) -> dict:
    """Score every model on every labeled frame, the way the appliance fires.

    Rock-solid rules: each model gets ONE prediction pass recording its max
    dog confidence per frame, so the whole threshold curve comes from the
    same evidence; the held-out split is the headline (training frames can
    only flatter a model); box-level mAP50 on the held-out split
    cross-checks the frame-level metric.
    """
    from ultralytics import YOLO

    val_stems = {p.stem for p in (run_dir / "dataset/images/val").glob("*.jpg")}
    frames = []
    for side in sorted(DATASET_MIRROR.glob("sample_*.json")):
        meta = json.loads(side.read_text())
        verdict = meta.get("human_label")
        if verdict in (None, "skip"):
            continue
        jpg = side.with_suffix(".jpg")
        if not jpg.is_file():
            continue
        frames.append((side.stem, verdict in ("dog", "dog_mixed"),
                       side.stem in val_stems, jpg))
    if not frames:
        sys.exit("[train] FAILED: no labeled frames to evaluate")

    result = {"n_frames": len(frames),
              "heldout_dogs": sum(1 for _, t, v, _ in frames if v and t),
              "heldout_nondogs": sum(1 for _, t, v, _ in frames if v and not t),
              "models": {}}
    log(f"evaluating {len(weights_by_name)} models on {len(frames)} frames "
        f"({result['heldout_dogs']}+{result['heldout_nondogs']} held out)")

    for name, w in weights_by_name.items():
        model = YOLO(str(w))
        confs = {}
        for stem, _, _, jpg in frames:
            res = model.predict(str(jpg), conf=0.25, imgsz=640,
                                device=_device(), verbose=False)[0]
            confs[stem] = round(max(
                (float(b.conf[0]) for b in res.boxes
                 if res.names[int(b.cls[0])] == "dog"), default=0.0), 3)
        try:
            map50 = round(float(model.val(
                data=str(run_dir / "dataset/data.yaml"), device=_device(),
                verbose=False, plots=False).box.map50), 3)
        except Exception:
            map50 = None  # the 80-class baseline can't val on 2-class data

        def score(thr: float, heldout_only: bool) -> dict:
            rs = [(truth, confs[stem] >= thr) for stem, truth, isval, _ in frames
                  if isval or not heldout_only]
            return {"caught": sum(1 for t, f in rs if t and f),
                    "missed": sum(1 for t, f in rs if t and not f),
                    "false_fires": sum(1 for t, f in rs if not t and f),
                    "quiet": sum(1 for t, f in rs if not t and not f)}

        result["models"][name] = {
            "map50_val": map50, "confs": confs,
            "curve": {f"{t:.1f}": {"all": score(t, False),
                                   "heldout": score(t, True)}
                      for t in THRESHOLDS}}
        h = result["models"][name]["curve"][f"{FIRE_CONF:.1f}"]["heldout"]
        log(f"  {name}: held-out @{FIRE_CONF} catches "
            f"{h['caught']}/{h['caught'] + h['missed']}, "
            f"false-fires {h['false_fires']} | val mAP50 {map50}")
    return result


def rank_key(m: dict):
    """Best model first: fewest held-out false fires, then most held-out
    catches, then the all-frames tiebreaks, then mAP50 -- all at FIRE_CONF."""
    h = m["curve"][f"{FIRE_CONF:.1f}"]["heldout"]
    a = m["curve"][f"{FIRE_CONF:.1f}"]["all"]
    return (h["false_fires"], -h["caught"], a["false_fires"], -a["caught"],
            -(m["map50_val"] or 0.0))


# -- stage: export ----------------------------------------------------------

def export(run_dir: Path, best: Path) -> Path:
    from ultralytics import YOLO
    log("exporting NCNN bundle for the Pi ...")
    YOLO(str(best)).export(format="ncnn", imgsz=640)
    # ultralytics writes <stem>_ncnn_model next to the weights file
    src = best.with_name(best.stem + "_ncnn_model")
    dst = run_dir / "kitchen_ncnn_model"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), dst)
    log(f"deployable model: {dst}")
    return dst


# -- stage: report ----------------------------------------------------------

def pick_winner(metrics: dict) -> str:
    """Best non-baseline model by rank_key."""
    candidates = {n: m for n, m in metrics["models"].items() if n != "baseline"}
    return min(candidates, key=lambda n: rank_key(candidates[n]))


def report(run_dir: Path, dataset_stats: dict, metrics: dict, winner: str,
           ncnn: Path | None) -> None:
    (run_dir / "report.json").write_text(json.dumps(
        {"dataset": dataset_stats, "metrics": metrics, "winner": winner,
         "run_dir": str(run_dir)}, indent=1, default=str))

    n_dog = metrics["heldout_dogs"]
    n_non = metrics["heldout_nondogs"]

    def cell(m, thr, scope):
        s = m["curve"][thr][scope]
        return f"{s['caught']}/{s['caught'] + s['missed']} · {s['false_fires']} FP"

    ranked = sorted(metrics["models"], key=lambda n: rank_key(metrics["models"][n]))
    md = [f"# Training run {run_dir.name}", "",
          f"Dataset: {dataset_stats['train']} train / {dataset_stats['val']} val frames "
          f"({dataset_stats['verdicts']})", "",
          f"Held-out = {n_dog} dog + {n_non} non-dog frames the models never "
          f"trained on. One frame there is worth ~{100 // max(n_dog, 1)} points -- "
          "treat single-frame gaps as noise, not signal.", "",
          "| model | held-out @0.6 | held-out @0.7 | held-out @0.8 | "
          "all frames @0.7 (most seen in training) | val mAP50 |",
          "|---|---|---|---|---|---|"]
    for n in ranked:
        m = metrics["models"][n]
        tag = " **(winner)**" if n == winner else (" (baseline)" if n == "baseline" else "")
        md.append(f"| {n}{tag} | {cell(m, '0.6', 'heldout')} | "
                  f"{cell(m, '0.7', 'heldout')} | {cell(m, '0.8', 'heldout')} | "
                  f"{cell(m, '0.7', 'all')} | {m['map50_val'] if m['map50_val'] is not None else '--'} |")

    base_confs = metrics["models"]["baseline"]["confs"]
    win = metrics["models"][winner]["confs"]
    truth_by_stem = {}  # rebuild truth from confs' keys via sidecars
    for side in sorted(DATASET_MIRROR.glob("sample_*.json")):
        if side.stem in win:
            v = json.loads(side.read_text()).get("human_label")
            truth_by_stem[side.stem] = v in ("dog", "dog_mixed")
    md += ["", f"Frames where {winner} changed the verdict vs baseline (@{FIRE_CONF}):"]
    changed = []
    for stem, truth in truth_by_stem.items():
        b, t = base_confs[stem] >= FIRE_CONF, win[stem] >= FIRE_CONF
        if b != t:
            changed.append(f"- {stem}: truth={'dog' if truth else 'no-dog'}, now "
                           f"{'fires' if t else 'quiet'} "
                           f"({'improvement' if t == truth else 'REGRESSION'})")
    md += changed or ["- none"]

    if dataset_stats.get("dropped"):
        md += ["", "Frames excluded from training -- open /review, tap them in "
                   "the history rail, and draw their boxes:"]
        md += [f"- {stem}" for stem, _ in dataset_stats["dropped"]]

    if ncnn:
        md += ["", "## Deploy", "```",
               f"rsync -az {ncnn}/ doggy@doggypi.local:doggy/models/kitchen_ncnn_model/",
               "ssh doggy@doggypi.local \"sed -i "
               "'s|DOGGY_MODEL_PATH=.*|DOGGY_MODEL_PATH=models/kitchen_ncnn_model|' "
               "~/doggy/.env && sudo systemctl restart doggy\"", "```"]
    (run_dir / "report.md").write_text("\n".join(md) + "\n")
    log(f"report: {run_dir / 'report.md'}")


# -- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="doggy@doggypi.local")
    ap.add_argument("--backend", choices=("local", "modal"), default="local",
                    help="where the train stage runs: this machine's GPU, or "
                         "a Modal cloud GPU (scripts/modal_train.py)")
    ap.add_argument("--sweep", action="store_true",
                    help="train every SWEEP_CONFIGS variant on Modal in "
                         "parallel and leaderboard them (implies --backend modal)")
    ap.add_argument("--gpu", default="L4",
                    help="Modal GPU type (T4, L4, A10G, A100, H100)")
    ap.add_argument("--push-prelabels", action="store_true",
                    help="no training: big-model boxes for every frame, "
                         "pushed into the Pi's sidecars for the review page")
    ap.add_argument("--pi-url", default="https://doggypi.local:8443")
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=10,
                    help="backbone layers to freeze (small datasets need this)")
    args = ap.parse_args()
    os.environ["DOGGY_TRAIN_GPU"] = args.gpu  # read by scripts/modal_train.py

    if args.push_prelabels:
        if not args.skip_pull:
            pull(args.host)
        prelabel_push(args.pi_url)
        return

    run_dir = RUNS / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    log(f"run dir: {run_dir}")

    if not args.skip_pull:
        pull(args.host)
    dataset_stats = build(run_dir)
    if args.sweep:
        weights_by_name = sweep(run_dir, args.gpu)
    else:
        best = train(run_dir, args.backend, args.epochs, args.batch, args.freeze)
        weights_by_name = {"fine-tune": best}
    weights_by_name["baseline"] = BASE_MODEL
    metrics = evaluate(run_dir, weights_by_name)
    winner = pick_winner(metrics)
    log(f"winner: {winner}")
    ncnn = None
    if not args.skip_export:
        # A broken exporter must never cost us the training + eval results.
        try:
            ncnn = export(run_dir, weights_by_name[winner])
        except Exception as exc:
            log(f"WARNING: NCNN export failed ({exc}); report continues without it")
    report(run_dir, dataset_stats, metrics, winner, ncnn)
    log("done.")


if __name__ == "__main__":
    main()
