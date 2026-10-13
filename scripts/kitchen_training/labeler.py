"""The labeler run: fine-tune yolo26x on the kitchen, gate it against the
active labeler on the human exam, and swap it in on a clear win.

The dataset is hand labels outside the exam plus a stable sample of agreed
auto-labels, with the human exam as val. Labels carry COCO class ids and
data.yaml lists all 80 COCO names, so yolo26x keeps its 80-class head and
every trained weight, including its dog and person outputs (a fresh 2-class
head would relearn dogs from ~600 frames).
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path

from kitchen_training.build import DEDUP_VERDICTS, _dhash, _near_duplicate, _split
from kitchen_training.config import (EVAL_CONF_FLOOR, IMAGE_SIZE, LABELER_TAG,
                                     PRELABEL_CONF, PRELABEL_MODEL, RUNS,
                                     STOCK_PRELABEL_MODEL, device, duration,
                                     log, mirror_path, phase)
from kitchen_training.consensus import _nano_scores, consensus_verdict
from kitchen_training.dataset import (PRELABEL_BATCH, labeled_sidecars,
                                      prelabel, read_frames)
from kitchen_training.fusion import fuse
from kitchen_training.hygiene import (fit_epochs, janitor, mark_completed,
                                      strip_dataset)
from kitchen_training.pipeline import KEEP_RUNS
from kitchen_training.training import CHECKPOINT_EVERY

COCO_IDS = {"person": 0, "dog": 16}
# Extra copies of each hand-labeled train frame, so ~1k hand frames make up
# about a third of an epoch next to the auto sample.
LABELER_HUMAN_COPIES = 2
AUTO_SAMPLE_SIZE = 8000  # agreed auto-labels per run; a job can set its own
# Capture saves bursts 2 s apart, so a frame this close to an exam frame is
# nearly the same picture: it stays out of training.
EXAM_GUARD_MS = 30_000

LABELER_EPOCHS = 100  # the full schedule; a probe job asks for a few
LABELER_BATCH = 16    # the probe run shows whether 32 fits and pays
# Ultralytics' own fine-tune optimizer for 80 classes: optimizer=auto picks
# AdamW at lr0 0.002 * 5 / (4 + 80), beta1 0.9, no bias warmup, but only
# under 10k iterations. The full run is ~17.7k, where auto switches to MuSGD
# at 0.01, so pinning keeps the probe and the full run on one recipe.
LABELER_OPTIMIZER = {"optimizer": "AdamW", "lr0": 0.000119, "momentum": 0.9,
                     "warmup_bias_lr": 0.0}
SWEEP = (0.3, PRELABEL_CONF, 0.6)  # the report's confidence sweep
SIGN_TEST_ALPHA = 0.05


def _stamp(stem: str) -> int:
    return int(stem.removeprefix("sample_"))


def _near_exam(stem: str, exam_stamps: list[int]) -> bool:
    """exam_stamps is sorted: the first exam stamp at or after the window's
    start decides."""
    stamp = _stamp(stem)
    i = bisect.bisect_left(exam_stamps, stamp - EXAM_GUARD_MS)
    return i < len(exam_stamps) and exam_stamps[i] <= stamp + EXAM_GUARD_MS


def _sample_key(sample: tuple) -> str:
    """Stable order for the auto sample: the same inputs pick the same frames,
    and a bigger sample adds frames without changing the smaller one."""
    return hashlib.sha256(sample[0].encode()).hexdigest()


def _coco_names() -> dict[int, str]:
    from ultralytics.utils import ROOT, YAML
    return YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"]


def _write(dataset_dir: Path, split: str, stem: str, lines: list[str],
           data: bytes, copies: int, stats: dict) -> None:
    text = "".join(f"{line}\n" for line in lines)
    for name in [stem] + [f"{stem}_w{n + 2}" for n in range(copies)]:
        (dataset_dir / f"images/{split}/{name}.jpg").write_bytes(data)
        (dataset_dir / f"labels/{split}/{name}.txt").write_text(text)
    stats[split] += 1 + copies


def _log_build(stats: dict) -> None:
    log(f"labeler dataset: epoch {stats['epoch']} imgs = {stats['hand']} hand "
        f"(+{stats['human_copies']} copies) + {stats['auto']} auto sample | "
        f"label sources {stats['sources']} | "
        f"val {stats['val']} of {len(stats['exam'])} exam frames | "
        f"exam guard dropped {stats['exam_guard_dropped']} | "
        f"auto skipped: {stats['auto_no_box']} no yolo26x dog box, "
        f"{stats['dedup_dropped']} near-dupes | "
        f"hand dropped {stats['hand_no_box']} without boxes | "
        f"unreadable {stats['unreadable']}")


def build_dataset(run_dir: Path, sample_size: int = AUTO_SAMPLE_SIZE) -> dict:
    """Write run_dir/dataset and return its counts, plus "data_yaml" and
    "exam": every human exam stem, including frames without usable labels,
    which val leaves out and the gate still scores."""
    samples = labeled_sidecars(include_auto=True)
    hand = [s for s in samples if s[2].get("human_label")]
    autos = [s for s in samples if not s[2].get("human_label")]
    exam = sorted(stem for stem, _, meta in hand
                  if _split(stem, meta, is_auto=False) == "val")
    exam_set, exam_stamps = set(exam), sorted(map(_stamp, exam))
    hand_train = [s for s in hand
                  if s[0] not in exam_set and not _near_exam(s[0], exam_stamps)]
    auto_pool = sorted((s for s in autos if not _near_exam(s[0], exam_stamps)),
                       key=_sample_key)
    guard_dropped = (len(hand) - len(exam) - len(hand_train)
                     + len(autos) - len(auto_pool))
    log(f"exam guard: dropped {guard_dropped} frames within "
        f"{EXAM_GUARD_MS // 1000} s of an exam frame")
    prelabels = prelabel([s[0] for s in hand + auto_pool])

    dataset_dir = run_dir / "dataset"
    if dataset_dir.exists():  # a retried run rebuilds from clean
        shutil.rmtree(dataset_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset_dir / sub).mkdir(parents=True)

    stats = {"train": 0, "val": 0, "hand": 0, "human_copies": 0, "auto": 0,
             "sources": {}, "exam": exam, "exam_guard_dropped": guard_dropped,
             "hand_no_box": 0, "auto_no_box": 0, "dedup_dropped": 0,
             "unreadable": 0}
    seen: list[int] = []
    # Exam frames go to val as they are; hand train frames get the copies
    # and seed the near-duplicate check, so an auto never repeats one.
    jobs = ([(s, "val") for s in hand if s[0] in exam_set]
            + [(s, "train") for s in hand_train])
    frames = read_frames([s[0] for s, _ in jobs])
    for ((stem, verdict, meta), split), (_, data, image) in zip(jobs, frames):
        if image is None:  # empty or corrupt file (see dataset.UNREADABLE)
            stats["unreadable"] += 1
            continue
        lines, source = fuse(stem, verdict, meta, prelabels[stem], COCO_IDS)
        if lines is None:
            stats["hand_no_box"] += 1
            continue
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        if split == "val":
            _write(dataset_dir, "val", stem, lines, data, 0, stats)
            continue
        if verdict in DEDUP_VERDICTS:
            seen.append(_dhash(image))
        _write(dataset_dir, "train", stem, lines, data, LABELER_HUMAN_COPIES, stats)
        stats["hand"] += 1
        stats["human_copies"] += LABELER_HUMAN_COPIES

    frames = read_frames([s[0] for s in auto_pool])
    for (stem, verdict, meta), (_, data, image) in zip(auto_pool, frames):
        if stats["auto"] >= sample_size:
            break
        if image is None:
            stats["unreadable"] += 1
            continue
        # An auto dog trains only on yolo26x's box: both models must agree
        # for a box to train (fuse's live-box fallback is human-only).
        lines, source = fuse(stem, verdict, meta, prelabels[stem], COCO_IDS)
        if lines is None:
            stats["auto_no_box"] += 1
            continue
        if verdict in DEDUP_VERDICTS and _near_duplicate(image, seen):
            stats["dedup_dropped"] += 1
            continue
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        _write(dataset_dir, "train", stem, lines, data, 0, stats)
        stats["auto"] += 1

    names = _coco_names()
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in names.items()))
    stats["epoch"] = stats["train"]
    stats["data_yaml"] = data_yaml
    _log_build(stats)
    return stats


# -- training ----------------------------------------------------------------

def _train(run_dir: Path, data_yaml: Path, epochs: int) -> Path | None:
    """Fine-tune stock yolo26x. Returns the weights folder, or None when
    training crashed (a crash can leave an earlier epoch's last.pt behind,
    which must never be promoted)."""
    from ultralytics import YOLO
    log(f"training the labeler from {STOCK_PRELABEL_MODEL.name}: {epochs} "
        f"epochs, batch {LABELER_BATCH}, device={device()}")
    model = YOLO(str(STOCK_PRELABEL_MODEL))
    # No end2end override: the labeler labels as a .pt with YOLO26's
    # end-to-end head (dataset._label_batch), so that head is the one to
    # train. The small model turns it off because NCNN ships the classic head.
    try:
        model.train(data=str(data_yaml), epochs=epochs,
                    patience=epochs,  # the full schedule (training._train_local)
                    freeze=10, close_mosaic=10, imgsz=IMAGE_SIZE,
                    batch=LABELER_BATCH, cache="ram", device=device(),
                    project=str(run_dir), name="train", exist_ok=True,
                    save_period=CHECKPOINT_EVERY, verbose=False,
                    **LABELER_OPTIMIZER)
    except Exception:
        traceback.print_exc()
        log("labeler training FAILED")
        return None
    return Path(model.trainer.save_dir) / "weights"


def _steady_rate(results_csv: Path, corpus: int) -> int | None:
    """Trained images/s from epoch 2 on, from Ultralytics' results.csv
    (its time column is seconds since training began, at each epoch's end).
    The first epoch carries warmup and setup, which would make a short probe
    under-report the rate a full run sees. None below two epochs."""
    try:
        times = [float(row["time"]) for row in
                 csv.DictReader(results_csv.read_text().splitlines())]
    except (OSError, KeyError, ValueError):
        return None
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return round(corpus * (len(times) - 1) / (times[-1] - times[0]))


def _checkpoints(weights: Path) -> dict[str, Path]:
    """Report rows: the epoch checkpoints (epoch0 is about stock) and best.pt,
    whose selection scores agreement with stock's own boxes on the exam."""
    epochs = sorted((p for p in weights.glob("epoch*.pt") if p.stem != "epoch0"),
                    key=lambda p: int(p.stem[5:]))
    best = weights / "best.pt"
    return {p.stem: p for p in epochs + ([best] if best.is_file() else [])}


# -- the gate ----------------------------------------------------------------

def _person_present(verdict: str, meta: dict) -> bool:
    """The human saw a person, or a hand box or the live detector did."""
    return (verdict in ("person", "dog_mixed")
            or any(b.get("label") == "person"
                   for b in meta.get("human_boxes") or [])
            or bool(meta.get("detections", {}).get("people")))


def _exam_truth(exam: list[str]) -> dict[str, tuple[bool, bool]]:
    """(dog frame, person present) per exam frame from the human label; dog
    and dog_mixed are dog frames, as in evaluation.eval_frames."""
    truth = {}
    for stem in exam:
        meta = json.loads(mirror_path(f"{stem}.json").read_text())
        verdict = meta["human_label"]
        truth[stem] = (verdict in ("dog", "dog_mixed"),
                       _person_present(verdict, meta))
    return truth


def _judged(prediction) -> dict:
    """Max dog confidence, plus the dog and person boxes the labeler would
    cache at the label bar (dataset._boxes, predicted at PRELABEL_CONF)."""
    entry = {"dog": 0.0, "dogs": [], "people": []}
    for box in prediction.boxes:
        label, conf = prediction.names[int(box.cls[0])], float(box.conf[0])
        if label == "dog":
            entry["dog"] = max(entry["dog"], conf)
        if label in ("dog", "person") and conf >= PRELABEL_CONF:
            entry["dogs" if label == "dog" else "people"].append(
                [round(float(v), 1) for v in box.xyxy[0].tolist()])
    return entry


def _scores(weights: Path, images: list) -> dict[str, dict]:
    """One labeler's pass over the exam, predicting as dataset._label_batch
    labels (.pt, IMAGE_SIZE, PRELABEL_BATCH frames per call), with a low
    floor so the sweep sees confidences under the label bar. YOLO26's
    end-to-end head stays on: evaluation.evaluate turns it off to score the
    head NCNN ships, and the labeler never runs that head."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    scores = {}
    for i in range(0, len(images), PRELABEL_BATCH):
        chunk = images[i:i + PRELABEL_BATCH]
        predictions = model.predict([image for _, image in chunk],
                                    conf=EVAL_CONF_FLOOR, imgsz=IMAGE_SIZE,
                                    device=device(), verbose=False)
        for (stem, _), prediction in zip(chunk, predictions):
            scores[stem] = _judged(prediction)
    return scores


def _nano(deployed_dir: Path, images: list) -> dict[str, tuple[float, float]]:
    """The deployed small model's (dog, person) per exam frame, one frame per
    call: ultralytics' NCNN backend reads only im[0] of a batch."""
    from ultralytics import YOLO
    nano = YOLO(str(deployed_dir), task="detect")
    return {stem: _nano_scores(nano, image) for stem, image in images}


def _tally(truth: dict, entries: dict, threshold: float) -> dict:
    out = dict.fromkeys(("caught", "missed", "false_dogs", "errors",
                         "person_missed", "person_false_dogs"), 0)
    for stem, (is_dog, person) in truth.items():
        seen = entries[stem]["dog"] >= threshold
        if is_dog and seen:
            out["caught"] += 1
        elif is_dog:
            out["missed"] += 1
            out["person_missed"] += person
        elif seen:
            out["false_dogs"] += 1
            out["person_false_dogs"] += person
    out["errors"] = out["missed"] + out["false_dogs"]
    return out


def sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact sign test: the chance of at least this many of the
    disagreements going the candidate's way if both labelers were equal."""
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def score_gate(truth: dict, active: dict, candidate: dict) -> dict:
    """Both labelers on the exam at the label bar, frame by frame (R4). A dog
    frame no model boxed counts as a miss."""
    def right(entries: dict, stem: str) -> bool:
        return (entries[stem]["dog"] >= PRELABEL_CONF) == truth[stem][0]

    wins = sum(1 for s in truth if right(candidate, s) and not right(active, s))
    losses = sum(1 for s in truth if right(active, s) and not right(candidate, s))
    return {"conf": PRELABEL_CONF,
            "exam": {"frames": len(truth),
                     "dogs": sum(dog for dog, _ in truth.values()),
                     "person_present": sum(p for _, p in truth.values())},
            "active": _tally(truth, active, PRELABEL_CONF),
            "candidate": _tally(truth, candidate, PRELABEL_CONF),
            "wins": wins, "losses": losses, "p": sign_test_p(wins, losses),
            "sweep": {f"{t:.2f}": {"active": _tally(truth, active, t),
                                   "candidate": _tally(truth, candidate, t)}
                      for t in SWEEP}}


def replay(truth: dict, entries: dict, nano: dict) -> dict:
    """The jury rules on the exam with this labeler's boxes and the deployed
    small model (R5). Wrongly cleared: a dog frame the jury labels person or
    empty. Invented: a frame without a dog the jury labels dog. Frames the
    jurors disagree on go to a human and count as neither."""
    out = dict.fromkeys(("settled", "wrongly_cleared", "invented", "to_human"), 0)
    for stem, (is_dog, _) in truth.items():
        verdict = consensus_verdict(entries[stem], *nano[stem])
        if verdict is None:
            out["to_human"] += 1
        elif (verdict == "dog") == is_dog:
            out["settled"] += 1
        elif is_dog:
            out["wrongly_cleared"] += 1
        else:
            out["invented"] += 1
    return out


def decide(gate: dict, replayed: dict | None) -> tuple[str, str]:
    """R6: swap only on fewer exam errors, a significant share of the paired
    disagreements, and a jury replay no worse than the active labeler's."""
    active, candidate = gate["active"]["errors"], gate["candidate"]["errors"]
    errors = f"{candidate} exam errors vs the active labeler's {active}"
    if replayed is None:
        return "no swap", ("no deployed small-model bundle: the jury replay "
                           f"cannot run ({errors})")
    if candidate >= active:
        return "no swap", errors
    bad = {side: replayed[side]["wrongly_cleared"] + replayed[side]["invented"]
           for side in ("active", "candidate")}
    if bad["candidate"] > bad["active"]:
        return "no swap", (f"{errors}, with a worse jury replay "
                           f"({bad['candidate']} vs {bad['active']} wrongly "
                           "cleared plus invented dogs)")
    split = (f"{gate['wins']} vs {gate['losses']} disagreements, "
             f"p = {gate['p']:.3f}")
    if gate["p"] >= SIGN_TEST_ALPHA:
        return "inconclusive", (f"inconclusive, active labeler stays: {errors}; "
                                f"{split} can be noise")
    return "swap", f"{errors}; {split}; jury replay no worse"


def exam_suspects(truth: dict, candidate: dict) -> dict:
    """The candidate audits its own exam, as gate.exam_suspects does for the
    small model: a dog it would label on a frame the human labeled without
    one becomes a Disputed flag (one direction only). The jury's audit runs
    the ACTIVE labeler, which missed these dogs, so a candidate held back
    by mislabeled exam frames would otherwise stay held back."""
    suspects = {}
    for stem, (is_dog, _) in truth.items():
        conf = candidate[stem]["dog"]
        if is_dog or conf < PRELABEL_CONF:
            continue
        if json.loads(mirror_path(f"{stem}.json").read_text()).get("dispute_settled_at"):
            continue  # a human already arbitrated this one
        suspects[stem] = {"model_says": "dog", "nano_conf": round(conf, 3)}
    log(f"candidate disputes {len(suspects)} exam labels")
    return suspects


def _judge(exam: list[str], weights: Path, deployed_dir: Path,
           timings: dict) -> tuple[dict, dict | None, dict, dict]:
    """(gate, replay or None without a deployed bundle, checkpoint rows,
    exam suspects)."""
    truth = _exam_truth(exam)
    with phase("gate", timings):
        images = [(stem, image) for stem, _, image in read_frames(list(truth))
                  if image is not None]
        unreadable = len(truth) - len(images)
        truth = {stem: truth[stem] for stem, _ in images}
        active = _scores(PRELABEL_MODEL, images)
        candidate = _scores(weights / "last.pt", images)
        gate = {**score_gate(truth, active, candidate), "unreadable": unreadable}
        suspects = exam_suspects(truth, candidate)
        replayed = None
        if deployed_dir.is_dir():
            nano = _nano(deployed_dir, images)
            replayed = {"active": replay(truth, active, nano),
                        "candidate": replay(truth, candidate, nano)}
    with phase("checkpoints", timings):
        rows = {name: _tally(truth, _scores(path, images), PRELABEL_CONF)
                for name, path in _checkpoints(weights).items()}
    return gate, replayed, rows, suspects


def promote(last: Path) -> Path:
    """last.pt becomes labeler.pt beside the stock model, copy-then-rename
    (pipeline.promote_champion), so a crash mid-copy never leaves a torn
    labeler. yolo26x.pt and its box cache stay: removing labeler.pt is the
    rollback."""
    target = STOCK_PRELABEL_MODEL.with_name("labeler.pt")
    tmp = target.with_suffix(".pt.tmp")
    shutil.copyfile(last, tmp)
    os.replace(tmp, target)
    log(f"labeler swapped in: {target} from {last}")
    return target


# -- the report and the run --------------------------------------------------

def _gate_md(gate: dict, replayed: dict | None, rows: dict) -> list[str]:
    exam = gate["exam"]
    lines = ["", f"## Gate: the exam at the label bar ({gate['conf']})", "",
             f"{exam['frames']} exam frames, {exam['dogs']} with a dog and "
             f"{exam['person_present']} with a person present"
             + (f"; {gate['unreadable']} unreadable frames left out"
                if gate["unreadable"] else "") + ".", "",
             "| labeler | caught | missed dogs | false dogs | errors | "
             "person-present missed | person-present false dogs |",
             "|---|---|---|---|---|---|---|"]
    for side in ("active", "candidate"):
        t = gate[side]
        lines.append(f"| {side} | {t['caught']}/{exam['dogs']} | {t['missed']} | "
                     f"{t['false_dogs']} | {t['errors']} | {t['person_missed']} | "
                     f"{t['person_false_dogs']} |")
    lines += ["", f"Paired disagreements: the candidate is right on "
                  f"{gate['wins']} frames the active labeler gets wrong and "
                  f"wrong on {gate['losses']} it gets right. One-sided exact "
                  f"sign test p = {gate['p']:.3f}; a swap needs p < "
                  f"{SIGN_TEST_ALPHA}.",
              "", "## Jury replay with the deployed small model", ""]
    if replayed is None:
        lines.append("Not run: no deployed small-model bundle.")
    else:
        lines += ["| labeler | settled right | wrongly cleared dogs | "
                  "invented dogs | left to a human |", "|---|---|---|---|---|"]
        for side in ("active", "candidate"):
            t = replayed[side]
            lines.append(f"| {side} | {t['settled']} | {t['wrongly_cleared']} | "
                         f"{t['invented']} | {t['to_human']} |")
    lines += ["", "## Confidence sweep", "",
              "| threshold | active missed | active false dogs | "
              "candidate missed | candidate false dogs |", "|---|---|---|---|---|"]
    for threshold, sides in gate["sweep"].items():
        a, c = sides["active"], sides["candidate"]
        lines.append(f"| {threshold} | {a['missed']} | {a['false_dogs']} | "
                     f"{c['missed']} | {c['false_dogs']} |")
    lines += ["", "## Checkpoints (report rows; the decision uses last.pt)", "",
              "| weights | missed dogs | false dogs | errors |", "|---|---|---|---|"]
    lines += [f"| {name} | {t['missed']} | {t['false_dogs']} | {t['errors']} |"
              for name, t in rows.items()] or ["| none | | | |"]
    return lines


def _report_md(result: dict) -> str:
    data = result["dataset"]
    rate = (f"{result['images_per_second']} img/s trained"
            if result["images_per_second"] is not None else "no rate measured")
    lines = [f"# Labeler run {result['run_name']}", "",
             f"Decision: **{result['decision']}** ({result['reason']}). "
             f"Active labeler: {result['labeler']}.", "",
             f"Dataset: {data['epoch']} train images per epoch ({data['hand']} "
             f"hand frames + {data['human_copies']} copies + {data['auto']} auto "
             f"sample); val {data['val']} of {data['exam']} exam frames; exam "
             f"guard dropped {data['exam_guard_dropped']}; label sources "
             f"{data['sources']}.", "",
             f"Training: {result['epochs']} epochs (requested "
             f"{result['epochs_requested']}), batch {result['batch']}, GPU "
             f"{result['gpu'] or 'unnamed'}, {rate}."]
    if result["gate"]:
        lines += _gate_md(result["gate"], result["replay"], result["checkpoints"])
    if result["exam_suspects"]:
        lines += ["", f"Flagged {len(result['exam_suspects'])} exam labels as "
                  "Disputed: the candidate sees a dog the hand label lacks."]
    lines += ["", "Phase timings: " + " | ".join(
        f"{name} {duration(seconds)}" for name, seconds in result["timings"].items())]
    return "\n".join(lines) + "\n"


def run_labeler(run_name: str, run_root: Path, epochs: int = LABELER_EPOCHS,
                sample_size: int = AUTO_SAMPLE_SIZE, gpu: str = "") -> dict:
    """Build, train, gate, decide, report: the whole labeler job in one cloud
    call. run_root holds the job's deployed small-model bundle
    (run_root/deployed_ncnn_model), as for pipeline.full_run. A probe is the
    same call with a few epochs. Returns the results the Pi writes out."""
    started, timings = time.monotonic(), {}
    run_dir = RUNS / run_name
    # Janitor before the first write (pipeline.full_run): a full volume heals
    # at the start of a run, because it would prevent the run that cleans up.
    for stale in janitor(RUNS, KEEP_RUNS, run_name, run_root.parent):
        log(f"janitor: removed stale {stale}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with phase("build", timings):
        stats = build_dataset(run_dir, sample_size)
    corpus = stats["epoch"]
    ceiling = float(os.environ.get("KT_JOB_CEILING_SECONDS", 10 * 3600))
    fitted = fit_epochs(epochs, corpus, ceiling, gpu, LABELER_BATCH, model="labeler")
    if fitted != epochs:
        log(f"{corpus} train images cannot finish {epochs} epochs inside the "
            f"{ceiling / 3600:.0f}h ceiling: capping to {fitted}")
    with phase("train", timings):
        weights = _train(run_dir, stats["data_yaml"], fitted)
    result = {"run_name": run_name, "labeler": LABELER_TAG,
              "decision": "failed", "promoted": False,
              "reason": "training crashed", "gate": None, "replay": None,
              "checkpoints": {}, "exam_suspects": {},
              "epochs": fitted, "epochs_requested": epochs,
              "batch": LABELER_BATCH, "gpu": gpu, "images_per_second": None,
              "dataset": {**{k: v for k, v in stats.items()
                             if k not in ("exam", "data_yaml")},
                          "exam": len(stats["exam"])},
              "timings": timings}
    if weights is not None and not (weights / "last.pt").is_file():
        result["reason"] = "training produced no last.pt"
    elif weights is not None:
        seen = corpus * fitted
        overall = round(seen / max(timings["train"], 1))
        steady = _steady_rate(weights.parent / "results.csv", corpus)
        result["images_per_second"] = steady or overall
        log(f"trained {seen} images ({corpus} x {fitted} epochs): {overall} "
            f"img/s overall, " + (f"{steady} img/s from epoch 2 on (use this "
                                  "one for LABELER_IMAGES_PER_SECOND)"
                                  if steady else "no steady rate (one epoch)"))
        gate, replayed, rows, suspects = _judge(
            stats["exam"], weights, run_root / "deployed_ncnn_model", timings)
        decision, reason = decide(gate, replayed)
        result.update(decision=decision, reason=reason, gate=gate,
                      replay=replayed, checkpoints=rows, exam_suspects=suspects)
        if decision == "swap":
            promote(weights / "last.pt")
            result["promoted"] = True
    log(f"labeler decision: {result['decision']} ({result['reason']})")
    # The built dataset (2+ files per frame) is what hit the volume's
    # 500k-inode cap; weights and the report stay.
    strip_dataset(run_dir)
    timings["total"] = round(time.monotonic() - started, 1)
    log("timings: " + " | ".join(f"{name} {duration(seconds)}"
                                 for name, seconds in timings.items()))
    result["report_md"] = _report_md(result)
    (run_dir / "report.md").write_text(result["report_md"])
    mark_completed(run_dir)
    return result
