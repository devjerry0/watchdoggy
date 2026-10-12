"""The labeler's dataset (U3): hand labels outside the exam plus a stable
sample of agreed auto-labels, written with COCO class ids."""
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

# The package re-exports a build() function that shadows the module name.
build = importlib.import_module("kitchen_training.build")
config = importlib.import_module("kitchen_training.config")
dataset = importlib.import_module("kitchen_training.dataset")

T0 = 1_791_000_000_000  # sample_<ms> stems
MINUTE = 60_000
SHAPE = [48, 64]
LIVE_DOG = {"targets": [{"label": "dog", "confidence": 0.9, "box": [8, 8, 32, 32]}]}


def _stem(split: str, at: int) -> str:
    """The first stem at or after `at` whose hash puts a hand frame in split."""
    while build.split_for(f"sample_{at}") != split:
        at += 1
    return f"sample_{at}"


class Mirror:
    """A fixture mirror of real JPEGs and sidecars, with yolo26x's cached
    boxes faked per stem."""

    def __init__(self, root: Path, run_dir: Path, labeler):
        self.root, self.run_dir, self.labeler = root, run_dir, labeler
        self.boxes: dict[str, dict] = {}

    def add(self, stem: str, seed: int | None = None, x_dogs=(), x_people=(),
            **meta) -> None:
        import cv2
        import numpy as np
        rng = np.random.default_rng(int(stem[7:]) if seed is None else seed)
        image = rng.integers(0, 256, (*SHAPE, 3), dtype=np.uint8)
        (self.root / f"{stem}.jpg").write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
        (self.root / f"{stem}.json").write_text(json.dumps(meta))
        self.boxes[stem] = {"dogs": list(x_dogs), "people": list(x_people)}

    def prelabel(self, stems):
        return {s: {"dogs": [], "people": [], "shape": SHAPE, **self.boxes.get(s, {})}
                for s in stems}

    def build(self, **kwargs) -> dict:
        return self.labeler.build_dataset(self.run_dir, **kwargs)

    def files(self, split: str) -> list[str]:
        images = sorted(p.stem for p in (self.run_dir / f"dataset/images/{split}").glob("*.jpg"))
        labels = sorted(p.stem for p in (self.run_dir / f"dataset/labels/{split}").glob("*.txt"))
        assert images == labels  # every image has its label file
        return images

    def stems(self, split: str) -> set[str]:
        return {name.split("_w")[0] for name in self.files(split)}

    def classes(self, stem: str, split: str = "train") -> list[str]:
        text = (self.run_dir / f"dataset/labels/{split}/{stem}.txt").read_text()
        return sorted(line.split()[0] for line in text.splitlines())


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    labeler = importlib.import_module("kitchen_training.labeler")
    root = tmp_path / "mirror"
    root.mkdir()
    monkeypatch.setattr(config, "DATASET_MIRROR", root)
    monkeypatch.setattr(dataset, "DATASET_MIRROR", root)
    monkeypatch.setattr(config, "MIRROR_SHARDED", False)
    fixture = Mirror(root, tmp_path / "run", labeler)
    monkeypatch.setattr(labeler, "prelabel", fixture.prelabel)
    return fixture


def test_no_exam_stem_enters_training_even_when_auto_labeled(mirror):
    exam = _stem("val", T0)
    mirror.add(exam, human_label="dog", auto_label={"verdict": "dog"},
               x_dogs=[[8, 8, 32, 32]])
    far = _stem("train", T0 + 10 * MINUTE)
    mirror.add(far, human_label="empty")
    stats = mirror.build()
    assert exam in stats["exam"] and exam in mirror.stems("val")
    assert exam not in mirror.stems("train")
    assert mirror.stems("train") == {far}


def test_a_flagged_false_fire_is_exam_even_when_its_hash_says_train(mirror):
    flagged = _stem("train", T0)
    mirror.add(flagged, human_label="person", reasons=["user_marked_fp"],
               x_people=[[0, 0, 16, 16]])
    stats = mirror.build()
    assert stats["exam"] == [flagged]
    assert flagged in mirror.stems("val")
    assert flagged not in mirror.stems("train")


def test_frames_within_30_seconds_of_an_exam_frame_stay_out_of_training(mirror, capsys):
    exam = _stem("val", T0)
    at = int(exam[7:])
    mirror.add(exam, human_label="empty")
    near_hand = _stem("train", at + 2000)
    assert int(near_hand[7:]) - at < 3000
    mirror.add(near_hand, human_label="empty")
    near_auto = f"sample_{at - 2000}"
    mirror.add(near_auto, auto_label={"verdict": "empty"})
    far_hand = _stem("train", at + 2 * MINUTE)
    mirror.add(far_hand, human_label="empty")
    far_auto = f"sample_{at - 2 * MINUTE}"
    mirror.add(far_auto, auto_label={"verdict": "empty"})
    stats = mirror.build()
    assert mirror.stems("train") == {far_hand, far_auto}
    assert stats["exam_guard_dropped"] == 2
    assert "2 frames within 30 s of an exam frame" in capsys.readouterr().out


def test_hand_dog_frames_carry_coco_ids_and_the_live_box_fills_in(mirror):
    missed = _stem("train", T0)
    mirror.add(missed, human_label="dog", detections=LIVE_DOG)
    boxed = _stem("train", T0 + 10 * MINUTE)
    mirror.add(boxed, human_label="dog", x_dogs=[[8, 8, 32, 32]],
               x_people=[[0, 0, 16, 16]])
    drawn = _stem("train", T0 + 20 * MINUTE)
    mirror.add(drawn, human_label="dog",
               human_boxes=[{"label": "dog", "box": [8, 8, 32, 32]}])
    stats = mirror.build()
    assert mirror.classes(missed) == ["16"]
    assert mirror.classes(boxed) == ["0", "16"]
    assert mirror.classes(drawn) == ["16"]
    assert stats["sources"] == {"fallback": 1, "fused": 1, "hand": 1}


def test_an_auto_dog_frame_yolo26x_never_boxed_is_skipped(mirror):
    lonely = f"sample_{T0}"
    mirror.add(lonely, auto_label={"verdict": "dog"}, detections=LIVE_DOG)
    agreed = f"sample_{T0 + 10 * MINUTE}"
    mirror.add(agreed, auto_label={"verdict": "dog"}, x_dogs=[[8, 8, 32, 32]])
    stats = mirror.build()
    assert mirror.stems("train") == {agreed}
    assert mirror.classes(agreed) == ["16"]
    assert stats["auto_no_box"] == 1


def test_the_auto_sample_is_stable_and_sized_by_the_job(mirror):
    assert mirror.labeler.AUTO_SAMPLE_SIZE == 8000
    for n in range(12):
        mirror.add(f"sample_{T0 + n * MINUTE}", auto_label={"verdict": "empty"})
    mirror.build(sample_size=5)
    first = mirror.stems("train")
    mirror.build(sample_size=5)
    assert mirror.stems("train") == first and len(first) == 5
    mirror.build(sample_size=8)
    bigger = mirror.stems("train")
    assert len(bigger) == 8 and first < bigger


def test_near_duplicate_autos_do_not_use_up_the_sample(mirror):
    same = [f"sample_{T0 + n * MINUTE}" for n in range(3)]
    for stem in same:
        mirror.add(stem, seed=7, auto_label={"verdict": "empty"})
    for n in range(3, 6):
        mirror.add(f"sample_{T0 + n * MINUTE}", auto_label={"verdict": "empty"})
    mirror.build(sample_size=4)
    train = mirror.stems("train")
    assert len(train) == 4 and len(train & set(same)) == 1


def test_data_yaml_keeps_the_80_coco_names(mirror):
    import yaml
    stats = mirror.build()
    data = yaml.safe_load(Path(stats["data_yaml"]).read_text())
    assert len(data["names"]) == 80
    assert (data["names"][0], data["names"][16]) == ("person", "dog")
    assert (data["train"], data["val"]) == ("images/train", "images/val")


def test_hand_frames_get_extra_copies_and_autos_none(mirror):
    copies = mirror.labeler.LABELER_HUMAN_COPIES
    assert copies == 2
    hand = _stem("train", T0)
    mirror.add(hand, human_label="empty")
    auto = f"sample_{T0 + 10 * MINUTE}"
    mirror.add(auto, auto_label={"verdict": "empty"})
    exam = _stem("val", T0 + 20 * MINUTE)
    mirror.add(exam, human_label="empty")
    stats = mirror.build()
    assert mirror.files("train") == sorted(
        [hand, auto] + [f"{hand}_w{n + 2}" for n in range(copies)])
    assert mirror.files("val") == [exam]  # the exam is never copied
    assert stats["epoch"] == 2 + copies


# -- U4: training, gate, jury replay, decision, promotion ------------------

def _entry(dog: float, person: float = 0.0) -> dict:
    """One labeler's view of a frame: max dog confidence plus the boxes it
    would cache at the label bar."""
    bar = config.PRELABEL_CONF
    return {"dog": dog, "dogs": [[8, 8, 32, 32]] if dog >= bar else [],
            "people": [[0, 0, 16, 16]] if person >= bar else []}


def _labeler():
    return importlib.import_module("kitchen_training.labeler")


def test_gate_counts_errors_disagreements_and_a_dog_neither_model_boxed():
    labeler = _labeler()
    truth = {"d1": (True, False), "d2": (True, False), "d3": (True, False),
             "d4": (True, True), "n1": (False, True), "n2": (False, False),
             "n3": (False, False)}
    active = {"d1": _entry(0.8), "d2": _entry(0.0), "d3": _entry(0.0),
              "d4": _entry(0.6), "n1": _entry(0.5), "n2": _entry(0.0),
              "n3": _entry(0.0)}
    candidate = {"d1": _entry(0.8), "d2": _entry(0.7), "d3": _entry(0.0),
                 "d4": _entry(0.3), "n1": _entry(0.0), "n2": _entry(0.0),
                 "n3": _entry(0.9)}
    gate = labeler.score_gate(truth, active, candidate)
    assert gate["exam"] == {"frames": 7, "dogs": 4, "person_present": 2}
    # d3: neither model boxed the dog, and it still counts as a miss for both.
    assert gate["active"] == {"caught": 2, "missed": 2, "false_dogs": 1,
                              "errors": 3, "person_missed": 0,
                              "person_false_dogs": 1}
    assert gate["candidate"] == {"caught": 2, "missed": 2, "false_dogs": 1,
                                 "errors": 3, "person_missed": 1,
                                 "person_false_dogs": 0}
    assert (gate["wins"], gate["losses"]) == (2, 2)  # d2, n1 vs d4, n3
    assert gate["p"] == labeler.sign_test_p(2, 2)
    low = gate["sweep"]["0.30"]
    assert (low["active"]["missed"], low["candidate"]["missed"]) == (2, 1)
    assert set(gate["sweep"]) == {"0.30", "0.45", "0.60"}


def test_sign_test_is_one_sided_and_exact():
    labeler = _labeler()
    assert round(labeler.sign_test_p(7, 4), 3) == 0.274  # inconclusive
    assert labeler.sign_test_p(9, 1) == 11 / 1024        # significant
    assert labeler.sign_test_p(0, 0) == 1.0


def test_jury_replay_counts_wrongly_cleared_and_invented_dogs():
    labeler = _labeler()
    truth = {"cleared": (True, False), "invented": (False, False),
             "human": (True, False), "settled": (True, False)}
    entries = {"cleared": _entry(0.0), "invented": _entry(0.7),
               "human": _entry(0.7), "settled": _entry(0.7)}
    nano = {"cleared": (0.1, 0.0), "invented": (0.5, 0.0),
            "human": (0.3, 0.0), "settled": (0.6, 0.0)}
    assert labeler.replay(truth, entries, nano) == {
        "settled": 1, "wrongly_cleared": 1, "invented": 1, "to_human": 1}


def _decision_inputs(active_errors, candidate_errors, wins, losses,
                     active_bad=1, candidate_bad=1):
    labeler = _labeler()
    gate = {"active": {"errors": active_errors},
            "candidate": {"errors": candidate_errors},
            "wins": wins, "losses": losses,
            "p": labeler.sign_test_p(wins, losses)}
    replay = {"active": {"wrongly_cleared": active_bad, "invented": 0},
              "candidate": {"wrongly_cleared": candidate_bad, "invented": 0}}
    return gate, replay


def test_r6_swaps_only_on_a_significant_win_with_no_worse_replay():
    decide = _labeler().decide
    assert decide(*_decision_inputs(19, 10, 9, 0, 1, 2))[0] == "no swap"
    assert decide(*_decision_inputs(19, 10, 9, 0, 1, 1))[0] == "swap"
    decision, reason = decide(*_decision_inputs(20, 17, 7, 4))
    assert decision == "inconclusive"
    assert "inconclusive, active labeler stays" in reason and "p = 0.274" in reason
    assert decide(*_decision_inputs(10, 10, 3, 3))[0] == "no swap"
    gate, _ = _decision_inputs(19, 10, 9, 0)
    decision, reason = decide(gate, None)
    assert decision == "no swap" and "no deployed small-model bundle" in reason


def test_promotion_writes_labeler_pt_atomically_and_keeps_stock(tmp_path, monkeypatch):
    labeler = _labeler()
    stock = tmp_path / "models" / "yolo26x.pt"
    stock.parent.mkdir()
    stock.write_bytes(b"stock weights")
    monkeypatch.setattr(labeler, "STOCK_PRELABEL_MODEL", stock)
    last = tmp_path / "last.pt"
    last.write_bytes(b"kitchen weights")
    target = labeler.promote(last)
    assert target == stock.with_name("labeler.pt")
    assert target.read_bytes() == b"kitchen weights"
    assert stock.read_bytes() == b"stock weights"
    assert sorted(p.name for p in stock.parent.iterdir()) == ["labeler.pt", "yolo26x.pt"]


class _Box:
    def __init__(self, cls: int, conf: float):
        import types
        self.cls, self.conf = [cls], [conf]
        self.xyxy = [types.SimpleNamespace(tolist=lambda: [8.0, 8.0, 32.0, 32.0])]


class _Result:
    names = {0: "person", 16: "dog"}

    def __init__(self, dog: float, person: float):
        self.boxes = ([_Box(16, dog)] if dog else []) + ([_Box(0, person)] if person else [])


class FakeUltralytics:
    """YOLO stand-in: training writes ultralytics' weights folder; predict
    answers per weights file from a table keyed by frame (the fake
    read_frames hands out stems as images)."""

    def __init__(self):
        self.scores: dict[str, dict] = {}
        self.inits: list[Path] = []
        self.trains: list[dict] = []
        self.predicts: list[tuple[str, dict]] = []
        self.heads: list[bool] = []
        self.crash = self.no_weights = False

    def YOLO(self, path, task=None):
        import types
        world = self

        class Model:
            def __init__(self):
                self.path = Path(path)
                self.model = types.SimpleNamespace(yaml={"end2end": True}, end2end=True)
                world.inits.append(self.path)

            def train(self, **kwargs):
                world.trains.append(kwargs)
                world.heads.append(self.model.yaml["end2end"])
                if world.crash:
                    raise RuntimeError("CUDA out of memory")
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                (save_dir / "weights").mkdir(parents=True)
                self.trainer = types.SimpleNamespace(save_dir=str(save_dir))
                if world.no_weights:
                    return
                for name in ("last", "best", "epoch0", "epoch40"):
                    (save_dir / "weights" / f"{name}.pt").write_bytes(name.encode())

            def predict(self, source, **kwargs):
                world.predicts.append((self.path.name, kwargs))
                world.heads.append(self.model.end2end)
                table = world.scores.get(self.path.name, world.scores["last.pt"])
                frames = source if isinstance(source, list) else [source]
                return [_Result(*table[frame]) for frame in frames]

        return Model()


DOGS = [f"sample_{T0 + n * MINUTE}" for n in range(12)]
QUIET = [f"sample_{T0 + n * MINUTE}" for n in range(12, 16)]


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A labeler run on fakes: a 16-frame exam where the candidate catches
    10 dogs the stock labeler misses, a deployed nano that agrees with
    whoever sees the dogs, and stock yolo26x on the volume."""
    import types
    labeler = _labeler()
    root = tmp_path / "mirror"
    root.mkdir()
    monkeypatch.setattr(config, "DATASET_MIRROR", root)
    monkeypatch.setattr(config, "MIRROR_SHARDED", False)
    for stem in DOGS:
        (root / f"{stem}.json").write_text(json.dumps({"human_label": "dog"}))
    for stem in QUIET:
        (root / f"{stem}.json").write_text(json.dumps({"human_label": "empty"}))
    stock = tmp_path / "models" / "yolo26x.pt"
    stock.parent.mkdir()
    stock.write_bytes(b"stock weights")
    monkeypatch.setattr(labeler, "STOCK_PRELABEL_MODEL", stock)
    monkeypatch.setattr(labeler, "PRELABEL_MODEL", stock)
    monkeypatch.setattr(labeler, "RUNS", tmp_path / "training-runs")
    monkeypatch.setattr(labeler, "device", lambda: "cpu")

    def fake_build(run_dir, sample_size):
        (run_dir / "dataset").mkdir(parents=True)
        (run_dir / "dataset" / "data.yaml").write_text("names: {}\n")
        return {"train": 3000, "val": 16, "hand": 1000, "human_copies": 2000,
                "auto": sample_size, "sources": {"fused": 3000},
                "exam": DOGS + QUIET, "exam_guard_dropped": 4,
                "hand_no_box": 1, "auto_no_box": 2, "dedup_dropped": 3,
                "unreadable": 0, "epoch": 3000,
                "data_yaml": run_dir / "dataset" / "data.yaml"}

    monkeypatch.setattr(labeler, "build_dataset", fake_build)
    monkeypatch.setattr(labeler, "read_frames",
                        lambda stems: iter((s, b"jpg", s) for s in stems))
    fake = FakeUltralytics()
    stock_sees = set(DOGS[:2])
    fake.scores = {
        "yolo26x.pt": {s: (0.8 if s in stock_sees else 0.0, 0.0) for s in DOGS + QUIET},
        "last.pt": {s: (0.8 if s in DOGS else 0.0, 0.0) for s in DOGS + QUIET},
        "best.pt": {s: (0.8 if s in DOGS[:6] else 0.0, 0.0) for s in DOGS + QUIET},
        "deployed_ncnn_model": {s: (0.6 if s in DOGS else 0.05, 0.0) for s in DOGS + QUIET},
    }
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=fake.YOLO))
    run_root = tmp_path / "runs" / "job1"
    (run_root / "deployed_ncnn_model").mkdir(parents=True)
    return types.SimpleNamespace(labeler=labeler, fake=fake, stock=stock,
                                 run_root=run_root,
                                 run_dir=tmp_path / "training-runs" / "job1",
                                 target=stock.with_name("labeler.pt"))


def test_a_winning_candidate_trains_on_the_recipe_and_is_swapped_in(run):
    result = run.labeler.run_labeler("job1", run.run_root, epochs=3,
                                     sample_size=500, gpu="L40S")
    assert result["decision"] == "swap" and result["promoted"] is True
    assert run.target.read_bytes() == b"last"           # last.pt, not best.pt
    assert run.stock.read_bytes() == b"stock weights"
    # KTD2: stock yolo26x always, Ultralytics' own AdamW recipe pinned.
    assert run.fake.inits[0] == run.stock
    train = run.fake.trains[0]
    assert {k: train[k] for k in ("epochs", "patience", "freeze", "optimizer",
                                  "lr0", "momentum", "warmup_bias_lr",
                                  "close_mosaic", "cache", "imgsz", "batch",
                                  "save_period")} == {
        "epochs": 3, "patience": 3, "freeze": 10, "optimizer": "AdamW",
        "lr0": 0.000119, "momentum": 0.9, "warmup_bias_lr": 0.0,
        "close_mosaic": 10, "cache": "ram", "imgsz": 640,
        "batch": run.labeler.LABELER_BATCH, "save_period": 40}
    assert run.labeler.LABELER_BATCH == 16 and run.labeler.LABELER_EPOCHS == 100
    gate = result["gate"]
    assert (gate["active"]["missed"], gate["candidate"]["missed"]) == (10, 0)
    assert (gate["wins"], gate["losses"]) == (10, 0) and gate["p"] < 0.05
    assert result["replay"]["candidate"]["wrongly_cleared"] == 0
    assert result["replay"]["active"]["to_human"] == 10
    assert set(result["checkpoints"]) == {"epoch40", "best"}  # report rows only
    assert result["checkpoints"]["best"]["missed"] == 6
    assert result["dataset"]["exam"] == 16 and result["dataset"]["auto"] == 500
    assert result["images_per_second"] > 0 and result["epochs"] == 3
    assert {"build", "train", "gate", "total"} <= set(result["timings"])
    report = result["report_md"]
    for text in ("swap", "p = 0.001", "wrongly cleared", "0.30", "epoch40",
                 "img/s", "3000 train images per epoch", "timings"):
        assert text in report
    assert (run.run_dir / "report.md").read_text() == report
    assert not (run.run_dir / "dataset").exists()        # inode cap
    assert (run.run_dir / ".completed").is_file()


def test_the_labeler_trains_and_is_scored_with_its_end_to_end_head(run):
    """The labeler labels as a .pt with YOLO26's end-to-end head
    (dataset._label_batch), so it must train and sit the gate with that head.
    The small model's end2end = False (training._train_local,
    evaluation.evaluate) is for the head NCNN ships and must not leak here."""
    run.labeler.run_labeler("job1", run.run_root, epochs=3)
    assert run.fake.heads and all(run.fake.heads)
    labelers = [kw for name, kw in run.fake.predicts if name.endswith(".pt")]
    assert labelers and all(kw["imgsz"] == 640 for kw in labelers)
    assert all(kw["conf"] == config.EVAL_CONF_FLOOR for kw in labelers)
    nano = [kw for name, kw in run.fake.predicts if name == "deployed_ncnn_model"]
    assert nano and all("imgsz" not in kw for kw in nano)


def test_no_deployed_bundle_means_no_swap_with_the_reason_in_the_report(run):
    import shutil
    shutil.rmtree(run.run_root / "deployed_ncnn_model")
    result = run.labeler.run_labeler("job1", run.run_root, epochs=3)
    assert result["decision"] == "no swap" and result["promoted"] is False
    assert result["replay"] is None
    assert "no deployed small-model bundle" in result["report_md"]
    assert result["gate"]["candidate"]["errors"] == 0  # still scored
    assert not run.target.exists()


@pytest.mark.parametrize("failure", ["crash", "no_weights"])
def test_failed_training_never_promotes_and_reports_failure(run, failure):
    run.target.write_bytes(b"previous kitchen weights")
    setattr(run.fake, failure, True)
    result = run.labeler.run_labeler("job1", run.run_root, epochs=3)
    assert result["decision"] == "failed" and result["promoted"] is False
    assert run.target.read_bytes() == b"previous kitchen weights"
    assert result["gate"] is None and result["images_per_second"] is None
    assert "failed" in result["report_md"]
    assert not (run.run_dir / "dataset").exists()
