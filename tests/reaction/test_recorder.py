import numpy as np

from doggy.events.store import EventStore
from doggy.reaction.recorder import Recorder

FRAME = np.zeros((16, 16, 3), dtype=np.uint8)


def test_recorder_writes_thumbnail_and_log(tmp_path):
    recorder = Recorder(EventStore(tmp_path, 10, 0))
    event = recorder.record(FRAME, 0.87, latency_s=1.0, wall_time=1000.0, mono_ts=123.0)
    thumb = tmp_path / event.thumb
    assert thumb.exists()
    assert (tmp_path / "events.jsonl").exists()
    assert event.confidence == 0.87


def test_recorder_delegates_to_store(tmp_path):
    store = EventStore(tmp_path, 10, 0)
    recorder = Recorder(store)
    ev = recorder.record(np.zeros((8, 8, 3), np.uint8), confidence=0.7, latency_s=1.2,
                         wall_time=1000.0, mono_ts=5.0)
    assert ev.latency_s == 1.2 and len(store.list()) == 1
