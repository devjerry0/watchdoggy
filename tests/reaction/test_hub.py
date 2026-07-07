import numpy as np

from doggy.events.store import EventRecord
from doggy.reaction.hub import DogCaught, ReactionHub, SafeReaction


def _event():
    record = EventRecord(id="fire_1", ts=0.0, wall_time=None, confidence=0.9,
                         latency_s=1.0, thumb="fire_1.jpg")
    return DogCaught(record, np.zeros((8, 8, 3), np.uint8), 0.0)


def test_hub_publishes_in_registration_order():
    calls = []

    class R:
        def __init__(self, tag):
            self.tag = tag

        def on_dog_caught(self, event):
            calls.append(self.tag)

    hub = ReactionHub([R("sound"), R("clip")])
    hub.publish(_event())
    assert calls == ["sound", "clip"]


def test_safe_reaction_swallows_and_logs(caplog):
    class Boom:
        def on_dog_caught(self, event):
            raise RuntimeError("kaput")

    ok = []

    class Fine:
        def on_dog_caught(self, event):
            ok.append(1)

    hub = ReactionHub([SafeReaction(Boom()), SafeReaction(Fine())])
    hub.publish(_event())  # must not raise
    assert ok == [1]
    assert any("kaput" in r.message or "dog_caught" in r.message for r in caplog.records)
