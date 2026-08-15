"""When to harvest a training frame: the sampling policy for DatasetCapture.

Tracks per-reason cooldowns and recent-person memory, and turns one frame's
analysis into the list of capture reasons that are due right now. Storage
(saving, pruning, the context ring) stays in `dataset`."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from doggy.vision.analysis import FrameAnalysis

# The confidence band where the model is confused -- the hard examples a
# fine-tune learns the most from.
BORDERLINE_LOW = 0.25
BORDERLINE_HIGH = 0.75
# Rate limits: at most one sample per reason per this many seconds, so a person
# standing in frame for a minute yields a couple of samples, not hundreds.
SAMPLE_COOLDOWN_SECONDS = 10.0
# Background negatives (empty counter, cooking, lighting changes) once an hour.
PERIODIC_SECONDS = 3600.0
# While a person is (or was recently) in frame, sample every couple of minutes:
# cooking / dishwasher sessions are where the head-only misclassifications live,
# including the frames where the model sees nothing (prime hard negatives).
PERSON_ACTIVITY_SECONDS = 120.0
PERSON_RECENT_SECONDS = 60.0
# Lights-off failsafe: a pitch-dark frame teaches nothing and the model's
# night-time person hallucinations once flooded the queue with hundreds of
# black frames. Harvest triggers skip frames this dark; fires still archive.
DARK_MEAN_FLOOR = 15.0
# Subsample stride for the darkness check: a sparse mean is plenty to
# recognize a lights-off frame.
_DARK_CHECK_STRIDE = 8

# Near-duplicate suppression: a person cooking for twenty minutes yields
# dozens of near-identical harvests that bloat the label queue and teach
# nothing new (industry practice drops ~half of near-dupes at zero accuracy
# cost). A 64-bit difference hash over a small grayscale thumbnail, compared
# by Hamming distance against the most recent saves, catches them on-Pi for
# microseconds per frame. Fires and their context frames are EXEMPT: every
# fire is labeled-by-outcome and always kept.
DEDUP_HAMMING_FLOOR = 6
_DEDUP_RING_SIZE = 16
_DHASH_SIDE = 8
DEDUP_EXEMPT_REASONS = {"fire", "fire_context"}

# Flicker trigger: the model changing its mind about the same scene across
# nearby frames is a stronger uncertainty signal than one mid-band score
# (track-based active learning). Two or more crossings of the borderline
# band within the window flags the frame.
FLICKER_WINDOW_SECONDS = 30.0
FLICKER_MIN_CROSSINGS = 2


def too_dark(frame: np.ndarray) -> bool:
    return float(frame[::_DARK_CHECK_STRIDE, ::_DARK_CHECK_STRIDE].mean()) < DARK_MEAN_FLOOR


def dhash(frame: np.ndarray) -> int:
    """64-bit difference hash of a frame (grayscale 9x8 gradient signs)."""
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       (_DHASH_SIDE + 1, _DHASH_SIDE), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int(np.packbits(bits).tobytes().hex(), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class CapturePolicy:
    """Single-threaded (detect thread), like its owner."""

    def __init__(self) -> None:
        self._last_by_reason: dict[str, float] = {}
        self._last_person_seen: float | None = None
        self._recent_hashes: deque[int] = deque(maxlen=_DEDUP_RING_SIZE)
        # Timestamps of borderline-band entries/exits for the flicker signal.
        self._band_crossings: deque[float] = deque(maxlen=8)
        self._in_band = False

    def reasons_due(self, analysis: FrameAnalysis, now: float) -> list[str]:
        # Low-conf people count as "person seen": the bent-over head-only phase
        # often scores 0.3-0.6 as person -- exactly the sessions to sample.
        if analysis.people or any(d.label == "person" for d in analysis.lowconf):
            self._last_person_seen = now
        self._track_band(analysis, now)
        reasons = []
        if analysis.suppressed and self._due("suppressed", now):
            reasons.append("suppressed")
        if self._due("borderline", now) and self._any_borderline(analysis):
            reasons.append("borderline")
        if self._due("flicker", now) and self._flickering(now):
            # The model changing its mind about a steady scene: prime
            # uncertainty, stronger than any single mid-band score.
            reasons.append("flicker")
        if self._due("periodic", now, PERIODIC_SECONDS):
            reasons.append("periodic")
        if self._person_activity_due(now):
            # The person may be bent out of the model's sight right now (the
            # head-only phase) -- that is exactly the frame we want.
            reasons.append("person_activity")
        return reasons

    def duplicate(self, frame: np.ndarray, reasons: list[str],
                  analysis: FrameAnalysis) -> bool:
        """Near-duplicate of a recent save. Never for fires, and never for
        any frame the model sees a dog in: on a fixed camera the background
        dominates the hash, so distinct dog moments collide -- only dogless
        scenes (the abundant class) are safe to thin. A True verdict does
        NOT consume cooldowns -- the next distinct frame still saves."""
        if any(r in DEDUP_EXEMPT_REASONS for r in reasons):
            return False
        if any(d.label == "dog" for d in (*analysis.targets, *analysis.lowconf,
                                          *analysis.suppressed)):
            return False
        h = dhash(frame)
        if any(hamming(h, seen) < DEDUP_HAMMING_FLOOR
               for seen in self._recent_hashes):
            return True
        return False

    def mark_saved(self, reasons: list[str], now: float,
                   frame: np.ndarray | None = None) -> None:
        for r in reasons:
            self._last_by_reason[r] = now
        if frame is not None:
            self._recent_hashes.append(dhash(frame))

    def _track_band(self, analysis: FrameAnalysis, now: float) -> None:
        in_band = self._any_borderline(analysis)
        if in_band != self._in_band:
            self._band_crossings.append(now)
            self._in_band = in_band

    def _flickering(self, now: float) -> bool:
        recent = sum(1 for t in self._band_crossings
                     if now - t <= FLICKER_WINDOW_SECONDS)
        return recent >= FLICKER_MIN_CROSSINGS

    def _due(self, reason: str, now: float,
             interval: float = SAMPLE_COOLDOWN_SECONDS) -> bool:
        last = self._last_by_reason.get(reason)
        return last is None or now - last >= interval

    @staticmethod
    def _any_borderline(analysis: FrameAnalysis) -> bool:
        return any(
            BORDERLINE_LOW <= d.confidence <= BORDERLINE_HIGH
            for d in (*analysis.targets, *analysis.people, *analysis.lowconf)
        )

    def _person_activity_due(self, now: float) -> bool:
        return (self._last_person_seen is not None
                and now - self._last_person_seen <= PERSON_RECENT_SECONDS
                and self._due("person_activity", now, PERSON_ACTIVITY_SECONDS))
