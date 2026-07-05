import numpy as np

from doggy.config import TunableSettings
from doggy.safety import SafetyGovernor
from doggy.state import RuntimeSettings

FRAME = np.zeros((16, 16, 3), dtype=np.uint8)


def gov(tmp_path, **over):
    base = dict(safety_enabled=True, max_fires_per_hour=2)
    base.update(over)
    rs = RuntimeSettings(TunableSettings(**base))
    return SafetyGovernor(rs, event_log_dir=tmp_path)


def test_allows_when_enabled_and_under_limit(tmp_path):
    g = gov(tmp_path)
    assert g.allow_fire(now=0.0) is True


def test_master_off_switch_blocks(tmp_path):
    g = gov(tmp_path, safety_enabled=False)
    assert g.allow_fire(now=0.0) is False


def test_rate_limit_blocks_after_max(tmp_path):
    g = gov(tmp_path, max_fires_per_hour=2)
    g.record_fire(FRAME, 0.9, now=0.0)
    g.record_fire(FRAME, 0.9, now=10.0)
    assert g.allow_fire(now=20.0) is False


def test_rate_limit_window_rolls_off(tmp_path):
    g = gov(tmp_path, max_fires_per_hour=2)
    g.record_fire(FRAME, 0.9, now=0.0)
    g.record_fire(FRAME, 0.9, now=10.0)
    assert g.allow_fire(now=3601.0) is True  # first fire aged out of the hour


def test_record_fire_writes_thumbnail_and_log(tmp_path):
    g = gov(tmp_path)
    event = g.record_fire(FRAME, 0.87, now=123.0)
    thumb = tmp_path / event["thumb"]
    assert thumb.exists()
    assert (tmp_path / "events.jsonl").exists()
    assert event["confidence"] == 0.87
