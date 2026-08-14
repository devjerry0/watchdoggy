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


def _cfg():
    return TunableSettings()


def _runtime():
    return RuntimeSettings(_cfg())


def _gate():
    return FireGate(_runtime())


def _analysis(candidates=(), inventory=()):
    return FrameAnalysis(shape=(100, 100, 3), people=[],
                         targets=list(candidates), candidates=list(candidates),
                         inventory=list(inventory))


def _escalation_cfg(**overrides):
    kwargs = dict(escalation_enabled=True, escalation_seconds=8,
                  escalation_max_strikes=3, escalation_volume_step=0.2,
                  max_volume=0.5)
    kwargs.update(overrides)
    return TunableSettings(**kwargs)


class FlakyOutcomeStore(EventStore):
    """First attach_outcome raises OSError (an SD-card hiccup mid-write);
    later calls hit the real store."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.failures = 0

    def attach_outcome(self, *args, **kwargs) -> None:
        if self.failures == 0:
            self.failures += 1
            raise OSError("simulated I/O error")
        super().attach_outcome(*args, **kwargs)


class BrokenStrikeStore(EventStore):
    """bump_strikes always fails; everything else is the real store."""

    def bump_strikes(self, id: str) -> None:
        raise OSError("simulated I/O error")


def test_store_failure_on_finalize_is_logged_and_watcher_recovers(tmp_path, caplog):
    # A failing outcome write on the detect thread must not escape on_frame
    # (it would kill the pipeline); the incident must still clear so the next
    # fire is watched and recorded normally.
    store = FlakyOutcomeStore(tmp_path, 100, 0)
    first = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    second = store.add(_img(), 0.9, 1.0, 1030.0, 40.0)
    w = OutcomeWatcher(store, _gate(), FakeAlerter(), _runtime(), clock=lambda: 2000.0)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(first, _img(), 10.0))
    w.on_frame(_analysis([dog]), 11.0, _cfg())
    w.on_frame(_analysis([]), 12.0, _cfg())
    w.on_frame(_analysis([]), 14.1, _cfg())          # finalize: write blows up
    assert any("failed to record outcome" in r.message for r in caplog.records)
    w.on_dog_caught(DogCaught(second, _img(), 40.0))
    w.on_frame(_analysis([]), 41.0, _cfg())
    w.on_frame(_analysis([]), 43.5, _cfg())          # 2.5s clear: finalize
    recs = {rec.id: rec for rec in store.list()}
    assert recs[first.id].outcome_at is None         # the write really failed
    assert recs[second.id].clear_seconds == pytest.approx(1.0)
    assert recs[second.id].outcome_at == 2000.0


def test_strike_bookkeeping_survives_bump_strikes_failure(tmp_path, caplog):
    # The record's strike counter is best-effort: a failing bump_strikes must
    # not crash the detect thread, and the in-memory ladder must still advance
    # so the next escalation waits the full spacing.
    store = BrokenStrikeStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    fake = FakeAlerter()
    cfg = _escalation_cfg()
    runtime = RuntimeSettings(cfg)
    w = OutcomeWatcher(store, FireGate(runtime), fake, runtime)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 18.1, cfg)          # strike 2: store write fails
    assert fake.calls == 1
    assert any("failed to record strike" in r.message for r in caplog.records)
    w.on_frame(_analysis([dog]), 18.6, cfg)          # 0.5s since strike 2: too soon
    assert fake.calls == 1
    w.on_frame(_analysis([dog]), 26.2, cfg)          # full 8s spacing: strike 3
    assert fake.calls == 2
    assert store.list()[0].strikes == 1              # bump never landed
