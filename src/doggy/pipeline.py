from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable

import cv2
import numpy as np

from doggy.alerter import Alerter
from doggy.vision.camera import Camera
from doggy.clips import ClipBuffer, encode_clip
from doggy.core.config import Settings
from doggy.vision.detection import PERSON_LABEL, TARGET_LABEL
from doggy.vision.detector import Detector
from doggy.events.store import EventStore
from doggy.core.pacer import Pacer
from doggy.people import suppress_dogs_overlapping_people
from doggy.hardware.power import PowerMonitor
from doggy.safety import SafetyGovernor
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import CONFIDENCE_DECIMALS, FrameBuffer, StatusStore
from doggy.hardware.thermal import ThermalGovernor
from doggy.trigger import TriggerLogic
from doggy.vision.annotate import annotate
from doggy.zone import ZoneFilter

log = logging.getLogger("doggy")

# Idle poll interval while the detect loop waits for the capture thread's
# first frame.
_IDLE_POLL_SECONDS = 0.01
# Decimal places for the FPS readout (confidence precision is shared: CONFIDENCE_DECIMALS).
_FPS_DECIMALS = 1


class Pipeline:
    def __init__(self, *, settings: Settings, detector: Detector, camera: Camera,
                 alerter: Alerter, runtime: RuntimeSettings, status: StatusStore,
                 raw_buffer: FrameBuffer, annotated_buffer: FrameBuffer,
                 safety: SafetyGovernor, event_store: EventStore,
                 clock: Callable[[], float] = time.monotonic,
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
        self.event_store = event_store
        self.clock = clock
        self.trigger = TriggerLogic(runtime, rng=rng or random.Random())
        self.zone = ZoneFilter()
        self.pacer = Pacer(clock=clock)
        self.governor = ThermalGovernor()
        self.power = PowerMonitor(clock=clock)
        # Rolling in-memory JPEG buffer for opt-in per-catch clips (no SD writes
        # until a fire asks for a slice). Pending clips finalize after post-roll.
        self._clip_buffer = ClipBuffer(settings.clip_window_seconds)
        self._pending_clips: list[dict] = []

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
        annotated = annotate(frame, dogs, in_zone, points, people=show_people)
        self.annotated_buffer.set(annotated)
        if cfg.clips_enabled:
            # Buffer the ANNOTATED frame so any resulting clip shows the boxes.
            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                self._clip_buffer.push(now, buf.tobytes())
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
            if cfg.clips_enabled:
                # Defer encoding until post-roll has elapsed so the clip captures
                # a few seconds after the catch as well as the pre-roll.
                self._pending_clips.append(
                    {"id": record.id, "fire_ts": now, "end": now + cfg.clip_postroll_seconds})
        self._finalize_clips(now, cfg)
        self.status.update(state=self.trigger.state.value, confidence=round(top, CONFIDENCE_DECIMALS),
                           dogs=len(in_zone), people=len(people) if cfg.person_suppression_enabled else 0,
                           fires_this_hour=self.safety.fires_last_hour(now), muted=muted,
                           snoozed_until_seconds=self.safety.snooze_remaining(now))
        return fired and not muted

    def _finalize_clips(self, now: float, cfg) -> None:
        """Encode any pending clips whose post-roll window has elapsed."""
        if not self._pending_clips:
            return
        still_pending = []
        for p in self._pending_clips:
            if p["end"] > now:
                still_pending.append(p)
                continue
            frames = self._clip_buffer.slice(p["fire_ts"] - cfg.clip_preroll_seconds, p["end"])
            if frames:
                try:
                    path = encode_clip(frames, cfg.clip_fps, self.event_store.dir / f"{p['id']}.mp4")
                    self.event_store.attach_clip(p["id"], path.name)
                except Exception:
                    log.exception("failed to encode clip for %s", p["id"])
        self._pending_clips = still_pending

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
