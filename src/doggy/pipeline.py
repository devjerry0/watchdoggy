from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable

import numpy as np

from doggy.vision.camera import Camera
from doggy.reaction.clips import ClipService
from doggy.core.config import Settings
from doggy.vision.analysis import DetectionAnalyzer, InventoryTracker
from doggy.core.pacer import Pacer
from doggy.hardware.power import PowerMonitor
from doggy.decision.gate import FireGate
from doggy.reaction.hub import DogCaught, ReactionHub
from doggy.reaction.outcome import OutcomeWatcher
from doggy.reaction.recorder import Recorder
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import CONFIDENCE_DECIMALS, FrameBuffer, StatusStore
from doggy.hardware.thermal import ThermalGovernor
from doggy.decision.trigger import TriggerLogic
from doggy.vision.annotate import annotate

log = logging.getLogger("doggy")

# Idle poll while the detect loop waits for the capture thread's first frame.
_IDLE_POLL_SECONDS = 0.01
# Decimal places for the FPS readout (confidence uses CONFIDENCE_DECIMALS).
_FPS_DECIMALS = 1


class Pipeline:
    def __init__(self, *, settings: Settings, analyzer: DetectionAnalyzer, camera: Camera,
                 runtime: RuntimeSettings, status: StatusStore,
                 raw_buffer: FrameBuffer, annotated_buffer: FrameBuffer,
                 gate: FireGate, recorder: Recorder, hub: ReactionHub,
                 clip_service: ClipService, outcome: OutcomeWatcher,
                 clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None) -> None:
        self.settings = settings
        self.analyzer = analyzer
        self.camera = camera
        self.runtime = runtime
        self.status = status
        self.raw_buffer = raw_buffer
        self.annotated_buffer = annotated_buffer
        self.gate = gate
        self.recorder = recorder
        self.hub = hub
        # Per-frame stage here and a hub Reaction (registers its pending clip on fire).
        self.clip_service = clip_service
        # Per-frame stage here and a hub Reaction (opens its incident on fire).
        self.outcome = outcome
        self.clock = clock
        self.trigger = TriggerLogic(runtime, rng=rng or random.Random())
        self.inventory_tracker = InventoryTracker()
        self.pacer = Pacer(clock=clock)
        self.governor = ThermalGovernor()
        self.power = PowerMonitor(clock=clock)

    def run_once(self, frame: np.ndarray) -> bool:
        """Process a single frame: detect, annotate, trigger, maybe fire."""
        now = self.clock()
        cfg = self.runtime.get()
        analysis = self.analyzer.analyze(frame, cfg)
        on_counter = self.inventory_tracker.update([d.label for d in analysis.inventory])
        # `targets` are drawn; `candidates` are the in-zone subset that may trigger.
        show_people = analysis.people if cfg.person_suppression_enabled else None
        points = cfg.zone_points if cfg.zone_enabled else []
        annotated = annotate(frame, analysis.targets, analysis.candidates, points,
                             people=show_people,
                             inventory=analysis.inventory if cfg.show_inventory_boxes else None)
        self.annotated_buffer.set(annotated)
        self.clip_service.on_frame(annotated, now, cfg)
        top = max((d.confidence for d in analysis.candidates), default=0.0)
        fired = self.trigger.update(analysis.candidates, now)
        muted = not self.gate.allow(now)
        if fired and not muted:
            # Log the peak confidence over the confirm window (fire_confidence), not
            # this frame's `top`: the fire edge can land on a flicker frame that logs 0.
            # `now` = injected monotonic clock; time.time() = wall-clock for the record.
            record = self.recorder.record(
                frame, self.trigger.fire_confidence, self.trigger.fire_latency,
                time.time(), now,
            )
            self.gate.note_fire(now)
            self.hub.publish(DogCaught(record, frame, now))
            self.status.update(last_fire_ts=record.ts, last_fire_thumb=record.thumb)
        # After the fire block so the fire frame that opens an incident is
        # also its first observation.
        self.outcome.on_frame(analysis, now, cfg)
        self.clip_service.finalize_due(now, cfg)
        self.status.update(state=self.trigger.state.value, confidence=round(top, CONFIDENCE_DECIMALS),
                           # targets counts what is drawn/"in view"; certainty still comes from candidates.
                           targets=len(analysis.targets),
                           people=len(analysis.people) if cfg.person_suppression_enabled else 0,
                           on_counter=on_counter,
                           fires_this_hour=self.gate.fires_last_hour(now), muted=muted,
                           snoozed_until_seconds=self.gate.snooze_remaining(now))
        return fired and not muted

    def _capture_loop(self, stop: threading.Event) -> None:
        try:
            for frame in self.camera.frames():
                if stop.is_set():
                    return
                self.raw_buffer.set(frame)
        except Exception:
            log.exception("capture thread failed; signaling shutdown")
            stop.set()

    def run(self, stop: threading.Event) -> None:
        cap = threading.Thread(target=self._capture_loop, args=(stop,), daemon=True)
        cap.start()
        last = self.clock()
        while not stop.is_set():
            frame = self.raw_buffer.get()
            if frame is None:
                time.sleep(_IDLE_POLL_SECONDS)
                continue
            cfg = self.runtime.get()
            temp = self.governor.read_temp_c()
            interval = self.governor.effective_interval(temp, cfg)
            power = self.power.read()
            self.status.update(
                temp_c=temp, detect_interval_effective=interval,
                undervolt_now=power.undervolt_now if power else None,
                undervolt_since_boot=power.undervolt_since_boot if power else None,
            )
            self.pacer.wait(interval)
            self.run_once(frame)
            now = self.clock()
            dt = now - last
            if dt > 0:
                self.status.update(fps=round(1.0 / dt, _FPS_DECIMALS))
            last = now
        self.camera.close()
