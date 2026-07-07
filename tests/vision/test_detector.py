import numpy as np
import pytest

from doggy.core.config import TunableSettings
from doggy.vision.detection import Detection
from doggy.vision.detector import StubDetector, keep_detection, select_device


def test_stub_detector_scripts_returns():
    stub = StubDetector([[Detection("dog", 0.9, (0, 0, 1, 1))], []])
    assert stub.detect(np.zeros((2, 2))) == [Detection("dog", 0.9, (0, 0, 1, 1))]
    assert stub.detect(np.zeros((2, 2))) == []
    assert stub.detect(np.zeros((2, 2))) == []  # exhausted -> empty


def test_select_device_returns_known_value():
    assert select_device() in {"mps", "cpu"}


def _keep_cfg(**over):
    base = dict(confidence=0.55, inventory_confidence=0.4, inventory_enabled=True)
    base.update(over)
    return TunableSettings(**base)


def test_keep_detection_alarm_threshold_not_loosened_by_inventory():
    # The laxer inventory bar must never admit a watched-class or person box.
    assert keep_detection("dog", 0.45, _keep_cfg()) is False
    assert keep_detection("person", 0.45, _keep_cfg()) is False


def test_keep_detection_inventory_uses_its_own_bar():
    assert keep_detection("cup", 0.45, _keep_cfg()) is True
    assert keep_detection("cup", 0.35, _keep_cfg()) is False


def test_keep_detection_inventory_disabled_drops_inventory():
    assert keep_detection("cup", 0.45, _keep_cfg(inventory_enabled=False)) is False


def test_keep_detection_target_above_threshold_kept():
    assert keep_detection("dog", 0.6, _keep_cfg()) is True


@pytest.mark.slow
def test_yolo_detects_dog_and_ignores_empty_room():
    from pathlib import Path
    from doggy.vision.detector import YoloDetector
    from doggy.core.config import TunableSettings
    from doggy.core.runtime import RuntimeSettings
    import cv2

    det = YoloDetector(Path("models/yolo26n.pt"), RuntimeSettings(TunableSettings(confidence=0.4)))
    dog = cv2.imread("tests/fixtures/dog.jpg")
    empty = cv2.imread("tests/fixtures/empty_room.jpg")
    assert any(d.label == "dog" for d in det.detect(dog))
    assert not any(d.label == "dog" for d in det.detect(empty))
