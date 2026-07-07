import random

from doggy.core.config import TunableSettings
from doggy.vision.detection import Detection
from doggy.core.runtime import RuntimeSettings
from doggy.decision.trigger import TriggerLogic, TriggerState

DOG = [Detection(label="dog", confidence=0.9, box=(0, 0, 10, 10))]
NONE: list[Detection] = []


def make(**over):
    base = dict(confirm_seconds=1.0, window_m=2, window_n=3,
               cooldown_min_seconds=10, cooldown_max_seconds=10, confidence=0.5)
    base.update(over)
    return TriggerLogic(RuntimeSettings(TunableSettings(**base)),
                        rng=random.Random(0))


def test_single_frame_does_not_fire():
    t = make()
    assert t.update(DOG, now=0.0) is False
    assert t.state is TriggerState.CONFIRMING


def test_fires_after_confirm_seconds():
    t = make()
    assert t.update(DOG, now=0.0) is False
    assert t.update(DOG, now=0.5) is False
    fired = t.update(DOG, now=1.0)  # 1.0s >= confirm_seconds
    assert fired is True
    assert t.state is TriggerState.COOLDOWN


def test_low_confidence_ignored():
    t = make(confidence=0.8)
    low = [Detection(label="dog", confidence=0.6, box=(0, 0, 1, 1))]
    assert t.update(low, now=0.0) is False
    assert t.state is TriggerState.IDLE


def test_lost_dog_resets_to_idle():
    t = make()
    t.update(DOG, now=0.0)
    t.update(NONE, now=0.1)
    t.update(NONE, now=0.2)  # window no longer M-of-N
    assert t.state is TriggerState.IDLE


def test_flicker_tolerated_by_m_of_n():
    t = make()  # window_m=2, window_n=3
    assert t.update(DOG, now=0.0) is False
    assert t.update(NONE, now=0.5) is False   # one dropped frame
    fired = t.update(DOG, now=1.0)            # 2 of last 3 had a dog, 1.0s elapsed
    assert fired is True


def test_cooldown_blocks_refire():
    t = make()
    t.update(DOG, now=0.0)
    assert t.update(DOG, now=1.0) is True     # fires, cooldown=10s
    assert t.update(DOG, now=2.0) is False    # still cooling down
    assert t.state is TriggerState.COOLDOWN


def test_refires_after_cooldown_with_fresh_confirm():
    t = make()
    t.update(DOG, now=0.0)
    assert t.update(DOG, now=1.0) is True
    assert t.update(DOG, now=12.0) is False   # cooldown expired -> fresh CONFIRMING
    assert t.update(DOG, now=13.0) is True     # confirmed again


def test_no_fire_without_m_of_n_at_low_frame_rate():
    # At low FPS the window never fills before confirm_seconds; a lone dog frame
    # among misses must NOT fire (frame-rate independence + M-of-N).
    t = make(window_m=4, window_n=6, confirm_seconds=1.0)
    assert t.update(DOG, now=0.0) is False
    assert t.update(NONE, now=0.33) is False
    assert t.update(NONE, now=0.67) is False
    assert t.update(NONE, now=1.0) is False   # 1.0>=confirm but sum=1<4 -> no fire
    assert t.update(NONE, now=1.33) is False


def test_sustained_dog_fires_once_m_of_n_met():
    t = make(window_m=4, window_n=6, confirm_seconds=1.0)
    fired = False
    for ts in [0.0, 0.33, 0.67, 1.0]:
        fired = t.update(DOG, now=ts)
    assert fired is True  # 4 dog frames -> m_of_n (4>=4) and 1.0>=1.0


def test_fire_confidence_is_window_peak_not_fire_edge_frame():
    # Repro of the "conf 0" bug: with M-of-N flicker tolerance the trigger can
    # fire on a frame that currently has NO dog. fire_confidence must still report
    # the peak confidence that confirmed the fire, not the empty fire-edge frame.
    t = make()  # window_m=2, window_n=3, confirm_seconds=1.0
    high = [Detection(label="dog", confidence=0.9, box=(0, 0, 10, 10))]
    assert t.update(high, now=0.0) is False
    assert t.update(high, now=0.5) is False
    fired = t.update(NONE, now=1.0)   # fires on an empty (flicker) frame
    assert fired is True
    assert t.fire_confidence == 0.9


def test_fire_confidence_tracks_max_across_confirm_window():
    # peak of the confirming frames (0.72, 0.95, 0.8) is what gets reported
    t = make()
    def dog(c):
        return [Detection(label="dog", confidence=c, box=(0, 0, 10, 10))]
    assert t.update(dog(0.72), now=0.0) is False
    assert t.update(dog(0.95), now=0.5) is False
    assert t.update(dog(0.80), now=1.0) is True
    assert t.fire_confidence == 0.95


def test_fire_latency_is_time_since_first_sighting():
    t = make()  # window_m=2, window_n=3, confirm_seconds=1.0
    d = [Detection("dog", 0.9, (0, 0, 10, 10))]
    t.update(d, now=0.0)
    t.update(d, now=0.5)
    assert t.update(d, now=1.0) is True
    assert t.fire_latency == 1.0


def test_detect_only_label_never_fires():
    # A persistent "bird" with alert_labels=("dog",) is detect-only: it must sail
    # through the M-of-N + confirm window (which fires a dog by now=1.0) without
    # ever firing or even entering CONFIRMING.
    t = make(target_labels=("dog", "bird"), alert_labels=("dog",))
    bird = [Detection(label="bird", confidence=0.9, box=(0, 0, 10, 10))]
    for ts in [0.0, 0.5, 1.0, 1.5, 2.0]:
        assert t.update(bird, now=ts) is False
    assert t.state is TriggerState.IDLE


def test_monitor_mode_never_fires():
    # Empty alert set = monitor mode: even a sustained alert-grade dog never fires.
    t = make(alert_labels=())
    for ts in [0.0, 0.5, 1.0, 1.5, 2.0]:
        assert t.update(DOG, now=ts) is False
    assert t.state is TriggerState.IDLE


def test_state_objects_round_trip():
    t = make()  # window_m=2, n=3, confirm=1.0, cooldown 10
    assert t.state is TriggerState.IDLE
    t.update(DOG, now=0.0)
    assert t.state is TriggerState.CONFIRMING
    t.update(DOG, now=1.0)
    assert t.state is TriggerState.COOLDOWN
    t.update(NONE, now=20.0)
    assert t.state is TriggerState.IDLE  # expired, no dog
