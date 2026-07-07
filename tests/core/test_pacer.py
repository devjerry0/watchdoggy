from doggy.core.pacer import Pacer

def make(clock_values):
    it = iter(clock_values)
    slept = []
    p = Pacer(clock=lambda: next(it), sleep=slept.append)
    return p, slept

def test_first_call_never_sleeps():
    p, slept = make([0.0])
    p.wait(1.0)
    assert slept == []

def test_sleeps_remainder_when_called_too_soon():
    p, slept = make([0.0, 0.3, 1.0])   # last=0.0; now=0.3 -> sleep 0.7; last=1.0
    p.wait(1.0)
    p.wait(1.0)
    assert slept == [0.7]

def test_no_sleep_when_interval_already_elapsed():
    p, slept = make([0.0, 2.0, 2.0])
    p.wait(1.0)
    p.wait(1.0)
    assert slept == []
