import numpy as np
import pytest

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.decision.gate import FireGate
from doggy.events.store import EventStore
from doggy.reaction.hub import DogCaught
from doggy.reaction.outcome import OutcomeWatcher
from doggy.reaction.sound import FakeAlerter
from doggy.vision.analysis import FrameAnalysis
from doggy.vision.detection import Detection


def _img():
    return np.zeros((16, 16, 3), np.uint8)


def _analysis(candidates=(), inventory=()):
    return FrameAnalysis(shape=(100, 100, 3), people=[],
                         targets=list(candidates), candidates=list(candidates),
                         inventory=list(inventory))


def test_escalates_while_occupied_then_stops_at_max(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    fake = FakeAlerter()
    cfg = TunableSettings(escalation_enabled=True, escalation_seconds=8,
                          escalation_max_strikes=3, escalation_volume_step=0.2,
                          max_volume=0.5)
    runtime = RuntimeSettings(cfg)
    w = OutcomeWatcher(store, FireGate(runtime), fake, runtime)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 17.9, cfg)
    assert fake.calls == 0                      # not yet 8s since strike 1
    w.on_frame(_analysis([dog]), 18.1, cfg)     # strike 2
    w.on_frame(_analysis([dog]), 26.2, cfg)     # strike 3
    w.on_frame(_analysis([dog]), 40.0, cfg)     # max reached: no strike 4
    assert fake.calls == 2
    assert fake.volumes == [pytest.approx(0.7), pytest.approx(0.9)]
    assert store.list()[0].strikes == 3


class NoneAlerter:
    """Alerter whose playback always fails: records the attempt, returns None
    (like BaseAlerter with an empty clips dir)."""

    def __init__(self) -> None:
        self.calls = 0

    def alert(self, volume: float | None = None) -> str | None:
        self.calls += 1
        return None


def _escalation_cfg(**overrides):
    kwargs = dict(escalation_enabled=True, escalation_seconds=8,
                  escalation_max_strikes=3, escalation_volume_step=0.2,
                  max_volume=0.5)
    kwargs.update(overrides)
    return TunableSettings(**kwargs)


def test_gate_denial_blocks_escalation_strikes(tmp_path):
    # A snoozed gate must veto follow-up strikes entirely: no alert attempt,
    # no strike bump, and no last_strike_ts consumption.
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    fake = FakeAlerter()
    cfg = _escalation_cfg()
    runtime = RuntimeSettings(cfg)
    gate = FireGate(runtime)
    gate.snooze(3600, now=10.0)
    w = OutcomeWatcher(store, gate, fake, runtime)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 18.1, cfg)          # 8.1s occupied: timing eligible
    assert fake.calls == 0
    assert store.list()[0].strikes == 1
    w.on_frame(_analysis([dog]), 26.2, cfg)          # another 8.1s: still denied
    assert fake.calls == 0
    assert store.list()[0].strikes == 1


def test_escalation_disabled_never_strikes(tmp_path):
    # With the flag off (the default) an occupied zone must ride out the full
    # watch silently: no alerts, strikes stay at 1, timeout path unchanged.
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    fake = FakeAlerter()
    cfg = _escalation_cfg(escalation_enabled=False)
    runtime = RuntimeSettings(cfg)
    w = OutcomeWatcher(store, FireGate(runtime), fake, runtime, clock=lambda: 2000.0)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 18.1, cfg)          # past escalation_seconds
    w.on_frame(_analysis([dog]), 26.2, cfg)
    w.on_frame(_analysis([dog]), 71.0, cfg)          # occupied 61s: give up
    assert fake.calls == 0
    rec = store.list()[0]
    assert rec.strikes == 1
    assert rec.clear_seconds is None
    assert rec.outcome_at == 2000.0


def test_failed_alert_consumes_no_bookkeeping(tmp_path):
    # alert() returning None (no clip to play) must not count as a strike:
    # nothing recorded against the gate or store, and last_strike_ts stays
    # put so the very next eligible frame retries the alert.
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    alerter = NoneAlerter()
    cfg = _escalation_cfg()
    runtime = RuntimeSettings(cfg)
    gate = FireGate(runtime)
    w = OutcomeWatcher(store, gate, alerter, runtime)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 18.1, cfg)          # eligible: attempt fails
    w.on_frame(_analysis([dog]), 18.6, cfg)          # timer not reset: retries
    assert alerter.calls == 2
    assert store.list()[0].strikes == 1
    assert gate.fires_last_hour(18.6) == 0
