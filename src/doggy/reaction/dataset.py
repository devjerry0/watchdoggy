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
from doggy.reaction.capture_policy import CapturePolicy, too_dark
from doggy.reaction.clips import ClipBuffer
from doggy.reaction.hub import DogCaught
from doggy.vision.analysis import FrameAnalysis
from doggy.vision.detection import Detection

log = logging.getLogger("doggy")

# On a fire, also save the RAW approach sequence from a rolling pre-roll ring:
# frames spaced CONTEXT_SPACING seconds over the last CONTEXT_WINDOW seconds.
# The clip buffer can't serve this -- it deliberately stores ANNOTATED frames
# (boxes burned in), which would poison training.
CONTEXT_WINDOW = 12.0
CONTEXT_SPACING = 2.0
CONTEXT_MAX_FRAMES = 6


def _det(d: Detection) -> dict:
    return {"label": d.label, "confidence": round(d.confidence, 3), "box": list(d.box)}


def _detections(analysis: FrameAnalysis) -> dict:
    return {
        "targets": [_det(d) for d in analysis.targets],
        "people": [_det(d) for d in analysis.people],
        "suppressed": [_det(d) for d in analysis.suppressed],
        "candidates": [_det(d) for d in analysis.candidates],
        "lowconf": [_det(d) for d in analysis.lowconf],
    }


class DatasetCapture:
    """Collects a fine-tuning dataset from the moments that matter.

    Per-frame stage + hub Reaction (the ClipService shape). Saves the RAW frame
    (never the annotated one -- boxes burned into pixels would poison training)
    plus a JSON sidecar of everything the detector saw, when:

    - a fire happens (hub event; true and false positives alike),
    - a filter suppressed a target as a misclassified person,
    - any detection lands in the borderline-confidence band,
    - nothing happened for an hour (periodic negatives).

    CapturePolicy decides WHEN to sample; this class owns the storage.
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
        self._policy = CapturePolicy()
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
        reasons = self._policy.reasons_due(analysis, now)
        if not reasons:
            return
        # Darkness and duplicate checks only run when a trigger wants to save.
        if too_dark(frame):
            return
        if self._policy.duplicate(frame, reasons):
            return
        self._policy.mark_saved(reasons, now, frame)
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
        window = self._ring.slice_timed(event.mono_ts - CONTEXT_WINDOW,
                                        event.mono_ts)
        for jpeg in _spaced(window)[-CONTEXT_MAX_FRAMES:]:
            self._save_bytes(jpeg, ["fire_context"], event_id=event.record.id)

    # -- internals ----------------------------------------------------------

    def _stem(self) -> str:
        # Millisecond stems; bump on collision (context bursts save several
        # frames inside one tick).
        ms = int(self._wall() * 1000)
        while (self._dir / f"sample_{ms}.json").exists():
            ms += 1
        return f"sample_{ms}"

    def _save(self, frame: np.ndarray, analysis: FrameAnalysis | None,
              reasons: list[str], event_id: str | None = None) -> None:
        detections = _detections(analysis) if analysis is not None else None
        self._persist(reasons, event_id, detections, "sample",
                      lambda stem: cv2.imwrite(str(self._dir / f"{stem}.jpg"),
                                               frame))

    def _save_bytes(self, jpeg: bytes, reasons: list[str],
                    event_id: str | None = None) -> None:
        self._persist(reasons, event_id, None, "context sample",
                      lambda stem: (self._dir / f"{stem}.jpg").write_bytes(jpeg))

    def _persist(self, reasons: list[str], event_id: str | None,
                 detections: dict | None, what: str,
                 write_image: Callable[[str], object]) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            stem = self._stem()
            sidecar: dict = {"wall_time": self._wall(), "reasons": reasons}
            if event_id:
                sidecar["event_id"] = event_id
            if detections is not None:
                sidecar["detections"] = detections
            write_image(stem)
            (self._dir / f"{stem}.json").write_text(json.dumps(sidecar))
            self._prune()
        except OSError:
            # A full/failing SD card must never crash the detect loop.
            log.exception("dataset: failed to save %s", what)

    def _prune(self) -> None:
        samples = sorted(self._dir.glob("sample_*.jpg"))
        total = sum(p.stat().st_size for p in self._dir.glob("sample_*") if p.is_file())
        while samples and total > self._cap:
            total -= self._delete_sample(samples.pop(0))

    def _delete_sample(self, image: Path) -> int:
        """Remove one sample (frame + sidecar); returns the bytes freed."""
        freed = 0
        for p in (image, image.with_suffix(".json")):
            if p.is_file():
                freed += p.stat().st_size
                p.unlink()
        return freed

    def stats(self) -> dict:
        """Sample count, byte usage, and per-reason tallies for the dashboard."""
        count = 0
        by_reason: dict[str, int] = {}
        total = 0
        if self._dir.is_dir():
            for side in self._dir.glob("sample_*.json"):
                count += 1
                _tally_reasons(side, by_reason)
            total = sum(p.stat().st_size for p in self._dir.glob("sample_*")
                        if p.is_file())
        return {"samples": count, "bytes": total, "cap_bytes": self._cap,
                "by_reason": by_reason}


def _spaced(frames: list[tuple[float, bytes]]) -> list[bytes]:
    """Thin a timestamped burst to frames >= CONTEXT_SPACING apart."""
    picked: list[bytes] = []
    last_ts = None
    for ts, jpeg in frames:
        if last_ts is None or ts - last_ts >= CONTEXT_SPACING:
            picked.append(jpeg)
            last_ts = ts
    return picked


def _tally_reasons(side: Path, by_reason: dict[str, int]) -> None:
    try:
        for r in json.loads(side.read_text()).get("reasons", []):
            by_reason[r] = by_reason.get(r, 0) + 1
    except (OSError, ValueError):
        return
