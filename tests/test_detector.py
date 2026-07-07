import numpy as np
import pytest

from doggy.detection import Detection
from doggy.detector import StubDetector, select_device


def test_stub_detector_scripts_returns():
    stub = StubDetector([[Detection("dog", 0.9, (0, 0, 1, 1))], []])
    assert stub.detect(np.zeros((2, 2))) == [Detection("dog", 0.9, (0, 0, 1, 1))]
    assert stub.detect(np.zeros((2, 2))) == []
    assert stub.detect(np.zeros((2, 2))) == []  # exhausted -> empty


def test_select_device_returns_known_value():
    assert select_device() in {"mps", "cpu"}


@pytest.mark.slow
def test_yolo_detects_dog_and_ignores_empty_room():
    from pathlib import Path
    from doggy.detector import YoloDetector
    from doggy.core.config import Settings, TunableSettings
    from doggy.core.runtime import RuntimeSettings
    import cv2

    det = YoloDetector(Path("models/yolo26n.pt"), RuntimeSettings(TunableSettings(confidence=0.4)))
    dog = cv2.imread("tests/fixtures/dog.jpg")
    empty = cv2.imread("tests/fixtures/empty_room.jpg")
    assert any(d.label == "dog" for d in det.detect(dog))
    assert not any(d.label == "dog" for d in det.detect(empty))
