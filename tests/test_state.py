import dataclasses

import numpy as np
import pytest

from doggy.config import TunableSettings
from doggy.detection import Detection
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore


def test_detection_is_frozen():
    d = Detection(label="dog", confidence=0.9, box=(1, 2, 3, 4))
    assert d.box == (1, 2, 3, 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.confidence = 0.1  # type: ignore[misc]


def test_runtime_settings_atomic_swap():
    rs = RuntimeSettings(TunableSettings(confidence=0.5))
    assert rs.get().confidence == 0.5
    rs.update(TunableSettings(confidence=0.9))
    assert rs.get().confidence == 0.9


def test_frame_buffer_keeps_latest():
    fb = FrameBuffer()
    assert fb.get() is None
    fb.set(np.zeros((2, 2), dtype=np.uint8))
    fb.set(np.ones((2, 2), dtype=np.uint8))
    assert fb.get().sum() == 4  # the newest frame won


def test_status_store_update_and_events():
    ss = StatusStore()
    ss.update(state="CONFIRMING", fps=5.0)
    snap = ss.snapshot()
    assert snap.state == "CONFIRMING"
    assert snap.fps == 5.0
    ss.add_event({"ts": 1.0, "confidence": 0.9, "thumb": "fire_1.0.jpg"})
    assert ss.events()[-1]["confidence"] == 0.9
