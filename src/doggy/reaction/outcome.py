from __future__ import annotations

import logging
import time
from typing import Callable

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.decision.gate import FireGate
from doggy.events.store import EventStore
from doggy.reaction.hub import DogCaught
from doggy.reaction.sound import Alerter
from doggy.vision.analysis import FrameAnalysis, InventoryTracker

log = logging.getLogger("doggy")

# An incident is "cleared" after the zone stays target-free this long; the
# debounce absorbs detection flicker and is subtracted from the measurement.
CLEAR_DEBOUNCE_SECONDS = 2.0
# Give up measuring after this long: the target never left (not deterred).
MAX_WATCH_SECONDS = 60.0


class OutcomeWatcher:
    """Measures what happened after each fire: how long until the target left
    the zone, and which counter items disappeared during the incident.

    Per-frame stage + hub Reaction, like ClipService. Single-threaded: both
    entry points run on the pipeline thread. `clock` is wall time and is only
    stamped onto the stored outcome; measurements use pipeline monotonic time.
    """

    def __init__(self, store: EventStore, gate: FireGate, alerter: Alerter,
                 runtime: RuntimeSettings,
                 clock: Callable[[], float] = time.time) -> None:
        self._store = store
        self._gate = gate
        self._alerter = alerter
        self._runtime = runtime
        self._clock = clock
        self._tracker = InventoryTracker()
        self._incident: dict | None = None

    def on_dog_caught(self, event: DogCaught) -> None:
        if self._incident is not None:
            # A new confirmed fire while still watching: the old incident
            # never cleared. Close it honestly before tracking the new one.
            self._finalize(clear_seconds=None)
        self._incident = {
            "id": event.record.id,
            "fire_ts": event.mono_ts,
            "last_strike_ts": event.mono_ts,
            "strikes": 1,
            "clear_since": None,
            "before": self._tracker.labels(),
        }

    def on_frame(self, analysis: FrameAnalysis, now: float,
                 cfg: TunableSettings) -> None:
        self._tracker.update([d.label for d in analysis.inventory])
        if self._incident is None:
            return
        inc = self._incident
        if analysis.candidates:
            inc["clear_since"] = None
            if (cfg.escalation_enabled
                    and inc["strikes"] < cfg.escalation_max_strikes
                    and now - inc["last_strike_ts"] >= cfg.escalation_seconds
                    and self._gate.allow_escalation(now)):
                level = min(1.0, cfg.max_volume
                            + inc["strikes"] * cfg.escalation_volume_step)
                if self._alerter.alert(volume=level):
                    self._gate.note_fire(now)
                    try:
                        # Best-effort: the in-memory ladder below must advance
                        # even if the record's counter can't be persisted.
                        self._store.bump_strikes(inc["id"])
                    except Exception:
                        log.exception("failed to record strike for %s", inc["id"])
                    inc["strikes"] += 1
                    inc["last_strike_ts"] = now
            if now - inc["fire_ts"] >= MAX_WATCH_SECONDS:
                self._finalize(clear_seconds=None)
            return
        if inc["clear_since"] is None:
            inc["clear_since"] = now
        if now - inc["clear_since"] >= CLEAR_DEBOUNCE_SECONDS:
            self._finalize(clear_seconds=inc["clear_since"] - inc["fire_ts"])

    def _finalize(self, clear_seconds: float | None) -> None:
        # The incident is cleared before the store write so a failing disk
        # (this runs on the detect thread) can't wedge the watcher.
        inc, self._incident = self._incident, None
        taken = sorted(inc["before"] - self._tracker.labels())
        try:
            self._store.attach_outcome(
                inc["id"], clear_seconds=clear_seconds, taken=taken,
                wall_time=self._clock())
        except Exception:
            log.exception("failed to record outcome for %s", inc["id"])
