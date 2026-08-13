from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.reaction.clips import ClipBuffer
from doggy.reaction.hub import DogCaught
from doggy.vision.analysis import FrameAnalysis
from doggy.vision.detection import Detection

log = logging.getLogger("doggy")

# The confidence band where the model is confused -- the hard examples a
# fine-tune learns the most from.
BORDERLINE_LOW = 0.25
BORDERLINE_HIGH = 0.75
# Rate limits: at most one sample per reason per this many seconds, so a person
# standing in frame for a minute yields a couple of samples, not hundreds.
SAMPLE_COOLDOWN_SECONDS = 10.0
# Background negatives (empty counter, cooking, lighting changes) once an hour.
PERIODIC_SECONDS = 3600.0
# On a fire, also save the RAW approach sequence from a rolling pre-roll ring:
# frames spaced CONTEXT_SPACING seconds over the last CONTEXT_WINDOW seconds.
# The clip buffer can't serve this -- it deliberately stores ANNOTATED frames
# (boxes burned in), which would poison training.
CONTEXT_WINDOW = 12.0
CONTEXT_SPACING = 2.0
CONTEXT_MAX_FRAMES = 6
# While a person is (or was recently) in frame, sample every couple of minutes:
# cooking / dishwasher sessions are where the head-only misclassifications live,
# including the frames where the model sees nothing (prime hard negatives).
PERSON_ACTIVITY_SECONDS = 120.0
PERSON_RECENT_SECONDS = 60.0


def _det(d: Detection) -> dict:
    return {"label": d.label, "confidence": round(d.confidence, 3), "box": list(d.box)}


class DatasetCapture:
    """Collects a fine-tuning dataset from the moments that matter.

    Per-frame stage + hub Reaction (the ClipService shape). Saves the RAW frame
    (never the annotated one -- boxes burned into pixels would poison training)
    plus a JSON sidecar of everything the detector saw, when:

    - a fire happens (hub event; true and false positives alike),
    - a filter suppressed a target as a misclassified person,
    - any detection lands in the borderline-confidence band,
    - nothing happened for an hour (periodic negatives).

    Storage is capped; oldest samples are pruned. Frames never leave the Pi.
    Single-threaded: both entry points run on the detect thread.
    """

    def __init__(self, dataset_dir: Path, cap_bytes: int, runtime: RuntimeSettings,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time) -> None:
        self._dir = Path(dataset_dir)
        self._cap = cap_bytes
        # For the hub path: fires arrive without cfg, so the enabled decision
        # is read at the fire moment (the ClipService pattern).
        self._runtime = runtime
        self._clock = clock
        self._wall = wall_clock
        self._last_by_reason: dict[str, float] = {}
        self._last_person_seen: float | None = None
        # Rolling raw-frame ring (mono ts -> jpeg) feeding fire context saves.
        self._ring = ClipBuffer(CONTEXT_WINDOW)

    # -- per-frame stage ----------------------------------------------------

    def on_frame(self, frame: np.ndarray, analysis: FrameAnalysis, now: float,
                 cfg: TunableSettings) -> None:
        if not cfg.dataset_enabled:
            return
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            self._ring.push(now, buf.tobytes())
        reasons = []
        # Low-conf people count as "person seen": the bent-over head-only phase
        # often scores 0.3-0.6 as person -- exactly the sessions to sample.
        if analysis.people or any(d.label == "person" for d in analysis.lowconf):
            self._last_person_seen = now
        if analysis.suppressed and self._due("suppressed", now):
            reasons.append("suppressed")
        if self._due("borderline", now) and any(
            BORDERLINE_LOW <= d.confidence <= BORDERLINE_HIGH
            for d in (*analysis.targets, *analysis.people, *analysis.lowconf)
        ):
            reasons.append("borderline")
        if self._due("periodic", now, PERIODIC_SECONDS):
            reasons.append("periodic")
        if (self._last_person_seen is not None
                and now - self._last_person_seen <= PERSON_RECENT_SECONDS
                and self._due("person_activity", now, PERSON_ACTIVITY_SECONDS)):
            # The person may be bent out of the model's sight right now (the
            # head-only phase) -- that is exactly the frame we want.
            reasons.append("person_activity")
        if reasons:
            for r in reasons:
                self._last_by_reason[r] = now
            self._save(frame, analysis, reasons)

    # -- hub Reaction -------------------------------------------------------

    def on_dog_caught(self, event: DogCaught) -> None:
        # Fires are always kept (no cooldown): each is a labeled-by-outcome
        # example, and the hourly fire cap already bounds their rate.
        if not self._runtime.get().dataset_enabled:
            return
        self._save(event.frame, None, ["fire"], event_id=event.record.id)
        # The approach sequence: spaced raw frames from before the fire (the
        # dog walking in / the person bending down), each its own sample.
        picked: list[bytes] = []
        last_ts = None
        for ts, jpeg in self._ring.slice_timed(event.mono_ts - CONTEXT_WINDOW, event.mono_ts):
            if last_ts is None or ts - last_ts >= CONTEXT_SPACING:
                picked.append(jpeg)
                last_ts = ts
        for jpeg in picked[-CONTEXT_MAX_FRAMES:]:
            self._save_bytes(jpeg, ["fire_context"], event_id=event.record.id)

    # -- internals ----------------------------------------------------------

    def _due(self, reason: str, now: float,
             interval: float = SAMPLE_COOLDOWN_SECONDS) -> bool:
        last = self._last_by_reason.get(reason)
        return last is None or now - last >= interval

    def _stem(self) -> str:
        # Millisecond stems; bump on collision (context bursts save several
        # frames inside one tick).
        ms = int(self._wall() * 1000)
        while (self._dir / f"sample_{ms}.json").exists():
            ms += 1
        return f"sample_{ms}"

    def _save(self, frame: np.ndarray, analysis: FrameAnalysis | None,
              reasons: list[str], event_id: str | None = None) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            stem = self._stem()
            sidecar: dict = {"wall_time": self._wall(), "reasons": reasons}
            if event_id:
                sidecar["event_id"] = event_id
            if analysis is not None:
                sidecar["detections"] = {
                    "targets": [_det(d) for d in analysis.targets],
                    "people": [_det(d) for d in analysis.people],
                    "suppressed": [_det(d) for d in analysis.suppressed],
                    "candidates": [_det(d) for d in analysis.candidates],
                    "lowconf": [_det(d) for d in analysis.lowconf],
                }
            cv2.imwrite(str(self._dir / f"{stem}.jpg"), frame)
            (self._dir / f"{stem}.json").write_text(json.dumps(sidecar))
            self._prune()
        except OSError:
            # A full/failing SD card must never crash the detect loop.
            log.exception("dataset: failed to save sample")

    def _save_bytes(self, jpeg: bytes, reasons: list[str],
                    event_id: str | None = None) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            stem = self._stem()
            sidecar: dict = {"wall_time": self._wall(), "reasons": reasons}
            if event_id:
                sidecar["event_id"] = event_id
            (self._dir / f"{stem}.jpg").write_bytes(jpeg)
            (self._dir / f"{stem}.json").write_text(json.dumps(sidecar))
            self._prune()
        except OSError:
            log.exception("dataset: failed to save context sample")

    def _prune(self) -> None:
        samples = sorted(self._dir.glob("sample_*.jpg"))
        total = sum(p.stat().st_size for p in self._dir.glob("sample_*") if p.is_file())
        while samples and total > self._cap:
            oldest = samples.pop(0)
            side = oldest.with_suffix(".json")
            for p in (oldest, side):
                if p.is_file():
                    total -= p.stat().st_size
                    p.unlink()

    def stats(self) -> dict:
        """Sample count, byte usage, and per-reason tallies for the dashboard."""
        count = 0
        by_reason: dict[str, int] = {}
        total = 0
        if self._dir.is_dir():
            for side in self._dir.glob("sample_*.json"):
                count += 1
                try:
                    for r in json.loads(side.read_text()).get("reasons", []):
                        by_reason[r] = by_reason.get(r, 0) + 1
                except (OSError, ValueError):
                    continue
            total = sum(p.stat().st_size for p in self._dir.glob("sample_*")
                        if p.is_file())
        return {"samples": count, "bytes": total, "cap_bytes": self._cap,
                "by_reason": by_reason}
