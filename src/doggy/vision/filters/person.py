from __future__ import annotations

from typing import TYPE_CHECKING

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


def suppress_targets_overlapping_people(
    targets: list[Detection], people: list[Detection], iou_threshold: float
) -> list[Detection]:
    """Drop target detections whose box is near-coincident with a person's box.

    A person misclassified as a target produces a box almost identical to the
    "person" box (high IoU) -- that pair is one human double-labeled, so suppress
    the target. A REAL animal merely near or behind a person has its own distinct
    box that only clips the person's at the edges (low IoU); it is kept and still
    fires. The threshold is deliberately high so overlap alone never eats a real
    animal -- only same-pixels double-labels are removed.
    """
    return [
        d for d in targets
        if not any(iou(d.box, p.box) >= iou_threshold for p in people)
    ]


class PersonSuppressionFilter:
    """Filter link: drop targets that are actually misclassified people.

    Applies only when suppression is enabled and people are present: narrows
    `analysis.targets` to the survivors, then reseeds `analysis.candidates` from
    them so downstream links (zone) act on the suppressed set.
    """

    def apply(self, analysis: "FrameAnalysis", cfg: "TunableSettings") -> None:
        if not (cfg.person_suppression_enabled and analysis.people):
            return
        analysis.targets = suppress_targets_overlapping_people(
            analysis.targets, analysis.people, cfg.person_iou_threshold)
        # Reseed from the alert set, not from `targets`, so detect-only animals
        # never re-enter the candidate list.
        alertable = set(cfg.alert_labels)
        analysis.candidates = [d for d in analysis.targets if d.label in alertable]
