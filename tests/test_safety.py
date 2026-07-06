import numpy as np

from doggy.config import TunableSettings
from doggy.events import EventStore
from doggy.safety import SafetyGovernor
from doggy.state import RuntimeSettings

FRAME = np.zeros((16, 16, 3), dtype=np.uint8)


def gov(tmp_path, **over):
    base = dict(safety_enabled=True, max_fires_per_hour=2)
    base.update(over)
    rs = RuntimeSettings(TunableSettings(**base))
    return SafetyGovernor(rs, EventStore(tmp_path, 10, 0))


def test_allows_when_enabled_and_under_limit(tmp_path):
    g = gov(tmp_path)
    assert g.allow_fire(now=0.0) is True


def test_master_off_switch_blocks(tmp_path):
    g = gov(tmp_path, safety_enabled=False)
    assert g.allow_fire(now=0.0) is False


def test_rate_limit_blocks_after_max(tmp_path):
    g = gov(tmp_path, max_fires_per_hour=2)
    g.record_fire(FRAME, 0.9, latency_s=1.0, wall_time=1000.0, now=0.0)
    g.record_fire(FRAME, 0.9, latency_s=1.0, wall_time=1010.0, now=10.0)
    assert g.allow_fire(now=20.0) is False


def test_rate_limit_window_rolls_off(tmp_path):
    g = gov(tmp_path, max_fires_per_hour=2)
    g.record_fire(FRAME, 0.9, latency_s=1.0, wall_time=1000.0, now=0.0)
    g.record_fire(FRAME, 0.9, latency_s=1.0, wall_time=1010.0, now=10.0)
    assert g.allow_fire(now=3601.0) is True  # first fire aged out of the hour


def test_record_fire_writes_thumbnail_and_log(tmp_path):
    g = gov(tmp_path)
    event = g.record_fire(FRAME, 0.87, latency_s=1.0, wall_time=1000.0, now=123.0)
    thumb = tmp_path / event.thumb
    assert thumb.exists()
    assert (tmp_path / "events.jsonl").exists()
    assert event.confidence == 0.87


def test_record_fire_delegates_to_store(tmp_path):
    from doggy.events import EventStore
    from doggy.config import Settings
    from doggy.state import RuntimeSettings
    import numpy as np
    store = EventStore(tmp_path, 10, 0)
    gov = SafetyGovernor(RuntimeSettings(Settings().tunable()), store)
    ev = gov.record_fire(np.zeros((8, 8, 3), np.uint8), confidence=0.7, latency_s=1.2,
                         wall_time=1000.0, now=5.0)
    assert ev.latency_s == 1.2 and len(store.list()) == 1
