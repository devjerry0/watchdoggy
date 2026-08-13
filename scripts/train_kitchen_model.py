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
FIRE_CONF = 0.7          # the appliance's alarm threshold, for would-fire eval
VAL_FRACTION_MOD = 4     # stem-hash % 4 == 0 -> val (~25%)
MIXED_IOU_DROP = 0.6     # dog box overlapping a person this much is the person

CLS = {"person": 0, "dog": 1}


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


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

    stats = {"train": 0, "val": 0, "dropped": [], "fallback": 0,
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
        if verdict in ("dog", "dog_mixed"):
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
                stats["dropped"].append((stem, "dog verdict but no box found"))
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
        f"dropped {len(stats['dropped'])} | nano-fallback {stats['fallback']}")
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


# -- stage: evaluate --------------------------------------------------------

def evaluate(run_dir: Path, best: Path) -> dict:
    from ultralytics import YOLO
    log("evaluating: baseline vs fine-tune, appliance-style (would it fire?)")
    base, tuned = YOLO(str(BASE_MODEL)), YOLO(str(best))
    val_stems = {p.stem for p in (run_dir / "dataset/images/val").glob("*.jpg")}

    def would_fire(model, jpg: Path) -> bool:
        res = model.predict(str(jpg), conf=0.25, imgsz=640,
                            device=_device(), verbose=False)[0]
        return any(res.names[int(b.cls[0])] == "dog"
                   and float(b.conf[0]) >= FIRE_CONF for b in res.boxes)

    rows = []
    for side in sorted(DATASET_MIRROR.glob("sample_*.json")):
        meta = json.loads(side.read_text())
        verdict = meta.get("human_label")
        if verdict in (None, "skip"):
            continue
        jpg = side.with_suffix(".jpg")
        if not jpg.is_file():
            continue
        rows.append({"stem": side.stem,
                     "truth": verdict in ("dog", "dog_mixed"),
                     "val": side.stem in val_stems,
                     "base": would_fire(base, jpg),
                     "tuned": would_fire(tuned, jpg)})

    def score(rs, key):
        tp = sum(1 for r in rs if r["truth"] and r[key])
        fn = sum(1 for r in rs if r["truth"] and not r[key])
        fp = sum(1 for r in rs if not r["truth"] and r[key])
        tn = sum(1 for r in rs if not r["truth"] and not r[key])
        return {"caught": tp, "missed": fn, "false_fires": fp, "quiet": tn}

    result = {"all": {k: score(rows, k) for k in ("base", "tuned")},
              "heldout": {k: score([r for r in rows if r["val"]], k)
                          for k in ("base", "tuned")},
              "changed": [
                  {"stem": r["stem"], "truth": "dog" if r["truth"] else "no-dog",
                   "now": "fires" if r["tuned"] else "quiet",
                   "improved": r["tuned"] == r["truth"]}
                  for r in rows if r["base"] != r["tuned"]]}
    return result


# -- stage: export ----------------------------------------------------------

def export(run_dir: Path, best: Path) -> Path:
    from ultralytics import YOLO
    log("exporting NCNN bundle for the Pi ...")
    YOLO(str(best)).export(format="ncnn", imgsz=640)
    src = best.parent / "best_ncnn_model"
    dst = run_dir / "kitchen_ncnn_model"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), dst)
    log(f"deployable model: {dst}")
    return dst


# -- stage: report ----------------------------------------------------------

def report(run_dir: Path, dataset_stats: dict, metrics: dict,
           ncnn: Path | None) -> None:
    (run_dir / "report.json").write_text(json.dumps(
        {"dataset": dataset_stats, "metrics": metrics,
         "run_dir": str(run_dir)}, indent=1, default=str))

    def line(scope, key):
        s = metrics[scope][key]
        return (f"catches {s['caught']}/{s['caught']+s['missed']} dogs, "
                f"false-fires {s['false_fires']}/{s['false_fires']+s['quiet']} non-dog frames")

    md = [f"# Training run {run_dir.name}", "",
          f"Dataset: {dataset_stats['train']} train / {dataset_stats['val']} val frames "
          f"({dataset_stats['verdicts']})", "",
          "| model | all frames | held-out val |", "|---|---|---|",
          f"| baseline | {line('all','base')} | {line('heldout','base')} |",
          f"| fine-tune | {line('all','tuned')} | {line('heldout','tuned')} |", "",
          "Changed frames:"]
    md += [f"- {c['stem']}: truth={c['truth']}, now {c['now']} "
           f"({'improvement' if c['improved'] else 'REGRESSION'})"
           for c in metrics["changed"]] or ["- none"]
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
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=10,
                    help="backbone layers to freeze (small datasets need this)")
    args = ap.parse_args()

    run_dir = RUNS / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    log(f"run dir: {run_dir}")

    if not args.skip_pull:
        pull(args.host)
    dataset_stats = build(run_dir)
    best = train(run_dir, args.backend, args.epochs, args.batch, args.freeze)
    metrics = evaluate(run_dir, best)
    ncnn = None
    if not args.skip_export:
        # A broken exporter must never cost us the training + eval results.
        try:
            ncnn = export(run_dir, best)
        except Exception as exc:
            log(f"WARNING: NCNN export failed ({exc}); report continues without it")
    report(run_dir, dataset_stats, metrics, ncnn)
    log("done.")


if __name__ == "__main__":
    main()
