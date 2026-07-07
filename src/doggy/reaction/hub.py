from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from doggy.events.store import EventRecord

log = logging.getLogger("doggy")


@dataclass(frozen=True)
class DogCaught:
    """A confirmed catch, broadcast to every reaction after the fire is recorded."""

    record: EventRecord      # the persisted catch (id, confidence, latency, thumb)
    frame: np.ndarray        # raw frame at the fire edge
    mono_ts: float           # pipeline monotonic time of the fire


class Reaction(Protocol):
    def on_dog_caught(self, event: DogCaught) -> None: ...


class ReactionHub:
    """Observer subject: fans a DogCaught out to each reaction in registration order.

    Deliberately dumb -- no try/except. Crash isolation is a separate concern
    owned by SafeReaction (Decorator), so an un-wrapped reaction can still abort
    the fan-out if a caller ever wants that.
    """

    def __init__(self, reactions: Sequence[Reaction]) -> None:
        self._reactions = list(reactions)

    def publish(self, event: DogCaught) -> None:
        for reaction in self._reactions:
            reaction.on_dog_caught(event)


class SafeReaction:
    """Decorator: logs (log.exception) and swallows any error from the inner
    reaction so one failing reaction never stops the others or kills the detect
    loop. The composition root wraps every registered reaction."""

    def __init__(self, inner: Reaction) -> None:
        self._inner = inner

    def on_dog_caught(self, event: DogCaught) -> None:
        try:
            self._inner.on_dog_caught(event)
        except Exception:
            log.exception("reaction %s failed on dog_caught", type(self._inner).__name__)
