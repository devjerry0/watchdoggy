"""The dataset mirror: pulling it from the Pi, and the big-model prelabels."""
from __future__ import annotations

import json
from pathlib import Path

from kitchen_training.config import (DATASET_MIRROR, IMAGE_SIZE,
                                     PRELABEL_CACHE, PRELABEL_CONF,
                                     PRELABEL_MODEL, RUNS, device, log, sh)

PUSH_TIMEOUT_SECONDS = 10
PUSH_WARNINGS_SHOWN = 3   # after this many, failures are only counted


def pull(host: str) -> None:
    log(f"pulling dataset from {host} ...")
    DATASET_MIRROR.mkdir(exist_ok=True)
    sh(["rsync", "-az", f"{host}:doggy/dataset/", str(DATASET_MIRROR) + "/"])
    count = len(list(DATASET_MIRROR.glob("sample_*.json")))
    log(f"dataset mirror: {count} samples")


def labeled_sidecars(include_auto: bool = False) -> list[tuple[str, str, dict]]:
    """(stem, verdict, meta) for every judged frame whose image exists.

    By default HUMAN verdicts only -- the evaluation exam must never be
    machine-graded. include_auto adds machine-consensus auto labels (build
    uses them for extra training data; they are confined to the train split).
    """
    out = []
    for sidecar in sorted(DATASET_MIRROR.glob("sample_*.json")):
        meta = json.loads(sidecar.read_text())
        verdict = meta.get("human_label")
        if not verdict and include_auto:
            verdict = (meta.get("auto_label") or {}).get("verdict")
        if not verdict:
            continue
        if verdict == "skip":
            continue
        if not sidecar.with_suffix(".jpg").is_file():
            continue
        out.append((sidecar.stem, verdict, meta))
    return out


# -- prelabels (cached: the big model runs once per frame, ever) ------------

def _load_cache() -> dict:
    if not PRELABEL_CACHE.is_file():
        return {}
    return json.loads(PRELABEL_CACHE.read_text())


def _prelabel_one(model, jpg: Path) -> dict:
    prediction = model.predict(str(jpg), conf=PRELABEL_CONF, imgsz=IMAGE_SIZE,
                               device=device(), verbose=False)[0]
    dogs, people = [], []
    buckets = {"dog": dogs, "person": people}
    for box in prediction.boxes:
        label = prediction.names[int(box.cls[0])]
        if label not in buckets:
            continue
        buckets[label].append([round(float(v), 1) for v in box.xyxy[0].tolist()])
    return {"dogs": dogs, "people": people, "shape": list(prediction.orig_shape)}


def prelabel(stems: list[str]) -> dict:
    cache = _load_cache()
    todo = [stem for stem in stems if stem not in cache]
    if todo:
        from ultralytics import YOLO
        log(f"prelabeling {len(todo)} new frames with {PRELABEL_MODEL.name} ...")
        model = YOLO(str(PRELABEL_MODEL))
        for stem in todo:
            cache[stem] = _prelabel_one(model, DATASET_MIRROR / f"{stem}.jpg")
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

    stems = [jpg.stem for jpg in sorted(DATASET_MIRROR.glob("sample_*.jpg"))
             if jpg.with_suffix(".json").is_file()]
    cache = prelabel(stems)
    context = ssl.create_default_context()  # household CA isn't in the system store
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    pushed, failed = 0, 0
    for stem in stems:
        prelabels = cache[stem]
        boxes = ([{"label": "dog", "box": box} for box in prelabels["dogs"]]
                 + [{"label": "person", "box": box} for box in prelabels["people"]])
        request = urllib.request.Request(
            f"{pi_url}/api/dataset/prelabels",
            data=json.dumps({"name": stem, "model": PRELABEL_MODEL.stem,
                             "boxes": boxes}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, context=context,
                                        timeout=PUSH_TIMEOUT_SECONDS):
                pushed += 1
        except Exception as exc:
            failed += 1
            if failed <= PUSH_WARNINGS_SHOWN:
                log(f"WARNING: push failed for {stem}: {exc}")
    log(f"prelabels pushed for {pushed}/{len(stems)} frames"
        + (f" ({failed} failed)" if failed else ""))
