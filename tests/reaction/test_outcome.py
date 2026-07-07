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


def test_clear_time_measured_after_debounce(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    w = OutcomeWatcher(store, _gate(), FakeAlerter(), _runtime(), clock=lambda: 2000.0)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 11.0, _cfg())      # still there
    w.on_frame(_analysis([]), 14.0, _cfg())          # gone at 14.0
    w.on_frame(_analysis([]), 15.5, _cfg())          # 1.5s clear: not yet
    assert store.list()[0].outcome_at is None
    w.on_frame(_analysis([]), 16.1, _cfg())          # 2.1s clear: finalize
    rec = store.list()[0]
    assert rec.clear_seconds == pytest.approx(4.0)   # 14.0 - 10.0
    assert rec.outcome_at == 2000.0


def test_reoccupancy_flicker_restarts_clear_debounce(tmp_path):
    # A brief re-appearance during the clear debounce must restart the count:
    # clear time is measured from the LAST departure, not the first.
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    w = OutcomeWatcher(store, _gate(), FakeAlerter(), _runtime(), clock=lambda: 2000.0)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 11.0, _cfg())       # still there
    w.on_frame(_analysis([]), 14.0, _cfg())          # gone at 14.0
    w.on_frame(_analysis([dog]), 15.0, _cfg())       # back: debounce must reset
    w.on_frame(_analysis([]), 15.5, _cfg())          # gone again at 15.5
    w.on_frame(_analysis([]), 17.0, _cfg())          # 1.5s since 15.5: not yet
    rec = store.list()[0]
    assert rec.clear_seconds is None                 # no early finalize off 14.0
    assert rec.outcome_at is None
    w.on_frame(_analysis([]), 17.6, _cfg())          # 2.1s since 15.5: finalize
    rec = store.list()[0]
    assert rec.clear_seconds == pytest.approx(5.5)   # 15.5 - 10.0, not 14.0 - 10.0
    assert rec.outcome_at == 2000.0


def test_timeout_records_not_deterred(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    w = OutcomeWatcher(store, _gate(), FakeAlerter(), _runtime(), clock=lambda: 2000.0)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 11.0, _cfg())       # still there
    assert store.list()[0].outcome_at is None
    w.on_frame(_analysis([dog]), 71.0, _cfg())       # occupied 61s: give up
    rec = store.list()[0]
    assert rec.clear_seconds is None                 # never cleared: not deterred
    assert rec.outcome_at == 2000.0


def test_taken_is_inventory_diff(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    w = OutcomeWatcher(store, _gate(), FakeAlerter(), _runtime(), clock=lambda: 2000.0)
    sandwich = Detection("sandwich", 0.8, (5, 5, 15, 15))
    # Two sightings before the fire so the 2-of-5 debounce marks it present.
    w.on_frame(_analysis([], [sandwich]), 8.0, _cfg())
    w.on_frame(_analysis([], [sandwich]), 9.0, _cfg())
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    # Dog gone and sandwich gone; four empty frames push the sandwich out of
    # the tracker's 5-frame window before the clear debounce finalizes.
    w.on_frame(_analysis([]), 11.0, _cfg())
    w.on_frame(_analysis([]), 12.0, _cfg())
    w.on_frame(_analysis([]), 12.5, _cfg())
    w.on_frame(_analysis([]), 13.1, _cfg())          # 2.1s clear: finalize
    rec = store.list()[0]
    assert rec.taken == ["sandwich"]
    assert rec.clear_seconds == pytest.approx(1.0)   # 11.0 - 10.0
    assert rec.outcome_at == 2000.0


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


def test_new_fire_finalizes_previous_incident_as_not_cleared(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    first = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    second = store.add(_img(), 0.9, 1.0, 1030.0, 40.0)
    w = OutcomeWatcher(store, _gate(), FakeAlerter(), _runtime(), clock=lambda: 2000.0)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(first, _img(), 10.0))
    w.on_frame(_analysis([dog]), 11.0, _cfg())
    w.on_dog_caught(DogCaught(second, _img(), 40.0))
    recs = {rec.id: rec for rec in store.list()}
    assert recs[first.id].clear_seconds is None      # never cleared before the new fire
    assert recs[first.id].outcome_at == 2000.0
    assert recs[second.id].outcome_at is None        # new incident still open
    w.on_frame(_analysis([]), 41.0, _cfg())
    w.on_frame(_analysis([]), 43.5, _cfg())          # 2.5s clear: finalize
    recs = {rec.id: rec for rec in store.list()}
    assert recs[second.id].clear_seconds == pytest.approx(1.0)
