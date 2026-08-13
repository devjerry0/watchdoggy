from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Callable

from doggy.vision.detection import Detection

if TYPE_CHECKING:
    from doggy.core.config import TunableSettings
    from doggy.vision.analysis import FrameAnalysis

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes (0.0 when disjoint, 1.0 when equal)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _suppress_against_boxes(
    targets: list[Detection], person_boxes: list[Box], iou_threshold: float
) -> list[Detection]:
    """Drop targets whose box is near-coincident (IoU >= threshold) with any of
    ``person_boxes``. The threshold is deliberately high so mere overlap never
    eats a real animal -- only same-pixels double-labels are removed."""
    return [
        d for d in targets
        if not any(iou(d.box, pb) >= iou_threshold for pb in person_boxes)
    ]


def suppress_targets_overlapping_people(
    targets: list[Detection], people: list[Detection], iou_threshold: float
) -> list[Detection]:
    """Same-frame suppression: drop targets coincident with a person box in this
    frame. A person misclassified as a target produces a near-identical box (high
    IoU) -- one human double-labeled. A real animal near/behind a person has its
    own distinct box (low IoU) and is kept."""
    return _suppress_against_boxes(targets, [p.box for p in people], iou_threshold)


class PersonSuppressionFilter:
    """Filter link: drop targets that are actually misclassified people.

    The model flickers person<->dog on the same body from frame to frame, so
    same-frame coincidence alone misses the "dog" frames (which carry no person
    box). This filter also REMEMBERS where people were for the last
    ``person_memory_seconds`` and suppresses a target that lands on a recently
    seen person location. Stateful, so instantiated once and reused; the clock
    is injectable for tests.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._recent: deque[tuple[float, Box]] = deque()  # (seen_at, person box)

    def apply(self, analysis: "FrameAnalysis", cfg: "TunableSettings") -> None:
        if not cfg.person_suppression_enabled:
            return
        now = self._clock()
        for p in analysis.people:
            self._recent.append((now, p.box))
        # Forget people older than the memory window.
        horizon = now - cfg.person_memory_seconds
        while self._recent and self._recent[0][0] < horizon:
            self._recent.popleft()
        if not self._recent:
            return
        person_boxes = [box for _, box in self._recent]
        survivors = _suppress_against_boxes(
            analysis.targets, person_boxes, cfg.person_iou_threshold)
        # Record what was dropped -- these misclassification specimens are gold
        # for training-data capture. Observational only.
        analysis.suppressed.extend(d for d in analysis.targets if d not in survivors)
        analysis.targets = survivors
        # Reseed from the alert set, not from `targets`, so detect-only animals
        # never re-enter the candidate list.
        alertable = set(cfg.alert_labels)
        analysis.candidates = [d for d in analysis.targets if d.label in alertable]
