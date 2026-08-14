"""When to harvest a training frame: the sampling policy for DatasetCapture.

Tracks per-reason cooldowns and recent-person memory, and turns one frame's
analysis into the list of capture reasons that are due right now. Storage
(saving, pruning, the context ring) stays in `dataset`."""
from __future__ import annotations

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


def too_dark(frame: np.ndarray) -> bool:
    return float(frame[::_DARK_CHECK_STRIDE, ::_DARK_CHECK_STRIDE].mean()) < DARK_MEAN_FLOOR


class CapturePolicy:
    """Single-threaded (detect thread), like its owner."""

    def __init__(self) -> None:
        self._last_by_reason: dict[str, float] = {}
        self._last_person_seen: float | None = None

    def reasons_due(self, analysis: FrameAnalysis, now: float) -> list[str]:
        # Low-conf people count as "person seen": the bent-over head-only phase
        # often scores 0.3-0.6 as person -- exactly the sessions to sample.
        if analysis.people or any(d.label == "person" for d in analysis.lowconf):
            self._last_person_seen = now
        reasons = []
        if analysis.suppressed and self._due("suppressed", now):
            reasons.append("suppressed")
        if self._due("borderline", now) and self._any_borderline(analysis):
            reasons.append("borderline")
        if self._due("periodic", now, PERIODIC_SECONDS):
            reasons.append("periodic")
        if self._person_activity_due(now):
            # The person may be bent out of the model's sight right now (the
            # head-only phase) -- that is exactly the frame we want.
            reasons.append("person_activity")
        return reasons

    def mark_saved(self, reasons: list[str], now: float) -> None:
        for r in reasons:
            self._last_by_reason[r] = now

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
