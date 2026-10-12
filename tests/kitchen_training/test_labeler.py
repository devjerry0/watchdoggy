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
