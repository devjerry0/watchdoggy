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
