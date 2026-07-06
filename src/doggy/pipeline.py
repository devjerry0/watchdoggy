from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable

import cv2
import numpy as np

from doggy.alerter import Alerter
from doggy.camera import Camera
from doggy.config import Settings
from doggy.detection import PERSON_LABEL, TARGET_LABEL
from doggy.detector import Detector
from doggy.pacer import Pacer
from doggy.people import suppress_dogs_overlapping_people
from doggy.power import PowerMonitor
from doggy.safety import SafetyGovernor
from doggy.state import CONFIDENCE_DECIMALS, FrameBuffer, RuntimeSettings, StatusStore
from doggy.thermal import ThermalGovernor
from doggy.trigger import TriggerLogic
from doggy.zone import ZoneFilter

log = logging.getLogger("doggy")

# Idle poll interval while the detect loop waits for the capture thread's
# first frame.
_IDLE_POLL_SECONDS = 0.01
# Detection-overlay styling (OpenCV uses BGR).
_BOX_THICKNESS = 2
_LABEL_FONT_SCALE = 0.5
_LABEL_THICKNESS = 1
_LABEL_Y_OFFSET = 6  # pixels above the box to place the label
# Decimal places for the FPS readout (confidence precision is shared: CONFIDENCE_DECIMALS).
_FPS_DECIMALS = 1
_DOG_ACTIVE_COLOR = (0, 0, 255)     # red BGR — in-zone / will trigger
_DOG_IGNORED_COLOR = (150, 150, 150)  # grey — outside zone, ignored
_PERSON_COLOR = (255, 0, 0)         # blue BGR — shown, never alerted on
_ZONE_COLOR = (0, 165, 255)         # orange BGR
_ZONE_ALPHA = 0.25


def _draw_box(out, box, label, confidence, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, _BOX_THICKNESS)
    cv2.putText(out, f"{label} {confidence:.2f}", (x1, max(0, y1 - _LABEL_Y_OFFSET)),
                cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, color, _LABEL_THICKNESS)


def annotate(frame, detections, in_zone=None, zone_points=None, people=None):
    out = frame.copy()
    h, w = frame.shape[0], frame.shape[1]
    if zone_points and len(zone_points) >= 3:
        pts = np.array([[int(x * w), int(y * h)] for x, y in zone_points], np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], _ZONE_COLOR)
        cv2.addWeighted(overlay, _ZONE_ALPHA, out, 1 - _ZONE_ALPHA, 0, out)
        cv2.polylines(out, [pts], True, _ZONE_COLOR, _BOX_THICKNESS)
    active = detections if in_zone is None else in_zone
    for p in people or []:
        _draw_box(out, p.box, p.label, p.confidence, _PERSON_COLOR)
    for d in detections:
        color = _DOG_ACTIVE_COLOR if d in active else _DOG_IGNORED_COLOR
        _draw_box(out, d.box, d.label, d.confidence, color)
    return out


class Pipeline:
    def __init__(self, *, settings: Settings, detector: Detector, camera: Camera,
                 alerter: Alerter, runtime: RuntimeSettings, status: StatusStore,
                 raw_buffer: FrameBuffer, annotated_buffer: FrameBuffer,
                 safety: SafetyGovernor, clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None) -> None:
        self.settings = settings
        self.detector = detector
        self.camera = camera
        self.alerter = alerter
        self.runtime = runtime
        self.status = status
        self.raw_buffer = raw_buffer
        self.annotated_buffer = annotated_buffer
        self.safety = safety
        self.clock = clock
        self.trigger = TriggerLogic(runtime, rng=rng or random.Random())
        self.zone = ZoneFilter()
        self.pacer = Pacer(clock=clock)
        self.governor = ThermalGovernor()
        self.power = PowerMonitor(clock=clock)

    def run_once(self, frame: np.ndarray) -> bool:
        """Process a single frame: detect, annotate, trigger, maybe fire."""
        now = self.clock()
        cfg = self.runtime.get()
        detections = self.detector.detect(frame)
        dogs = [d for d in detections if d.label == TARGET_LABEL]
        people = [d for d in detections if d.label == PERSON_LABEL]
        if cfg.person_suppression_enabled and people:
            # A "dog" whose box is near-coincident with a person is a misclassified
            # human -> drop it before it can count or fire. A real dog near a person
            # keeps its own distinct box (low IoU) and survives.
            dogs = suppress_dogs_overlapping_people(dogs, people, cfg.person_iou_threshold)
        show_people = people if cfg.person_suppression_enabled else None
        points = cfg.zone_points if cfg.zone_enabled else []
        in_zone = self.zone.filter(dogs, points, frame.shape)
        self.annotated_buffer.set(annotate(frame, dogs, in_zone, points, people=show_people))
        top = max((d.confidence for d in in_zone), default=0.0)
        fired = self.trigger.update(in_zone, now)
        muted = not self.safety.allow_fire(now)
        if fired and not muted:
            self.alerter.alert()
            # Log the confidence that actually triggered the fire (peak over the
            # confirm window), not this frame's `top` -- the fire edge can land on
            # a flicker frame with no current detection, which logged "conf 0".
            # `now` is the injected monotonic clock (event ts / rate limiting);
            # time.time() supplies wall-clock time for the persisted record.
            record = self.safety.record_fire(
                frame, self.trigger.fire_confidence, self.trigger.fire_latency,
                time.time(), now,
            )
            self.status.update(last_fire_ts=record.ts, last_fire_thumb=record.thumb)
        self.status.update(state=self.trigger.state.value, confidence=round(top, CONFIDENCE_DECIMALS),
                           dogs=len(in_zone), people=len(people) if cfg.person_suppression_enabled else 0,
                           fires_this_hour=self.safety.fires_last_hour(now), muted=muted)
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
