from __future__ import annotations

from typing import TYPE_CHECKING

from doggy.vision.detection import Detection

if TYPE_CHECKING:
    from doggy.core.config import TunableSettings
    from doggy.vision.analysis import FrameAnalysis

# Joint rule, calibrated from a month of real triggering frames (2026-08-13):
# true dogs never exceeded 11.1% of the frame and their boxes stay squat
# (height <= 33%, except a hind-legs stand at 54% height but only 7.7% area);
# bent-over people misread as dogs are BOTH big (>= 11.5%) and tall (>= 46%).
# A box over JOINT_AREA that is also taller than JOINT_HEIGHT is a person.
JOINT_AREA = 0.10
JOINT_HEIGHT = 0.40


def is_oversize(box: tuple[int, int, int, int], shape: tuple,
                area_ceiling: float) -> bool:
    """True when a target box is too big to be a dog on this camera. The flat
    ceiling catches huge boxes outright; the joint big-AND-tall rule catches
    the foreshortened bent-over-person silhouette at lower areas."""
    h, w = shape[0], shape[1]
    if h <= 0 or w <= 0:
        return False
    x1, y1, x2, y2 = box
    area_frac = max(0, x2 - x1) * max(0, y2 - y1) / (w * h)
    height_frac = max(0, y2 - y1) / h
    if area_frac > area_ceiling:
        return True
    return area_frac > JOINT_AREA and height_frac > JOINT_HEIGHT


class OversizeFilter:
    """Filter link: a "dog" box too big to be a dog is a misclassified person.

    The counter-mounted camera sees people foreshortened and hunched -- a
    silhouette outside COCO's mostly eye-level training distribution -- and the
    model confidently labels them "dog" WITHOUT a person box, so IoU-based
    person suppression never sees them. Geometry still separates them: no real
    dog fills a sixth of this frame. Narrows `candidates` only (the box still
    draws, greyed like out-of-zone) and records drops in `analysis.suppressed`
    so the training-data capture keeps the specimens.
    """

    def apply(self, analysis: "FrameAnalysis", cfg: "TunableSettings") -> None:
        if not cfg.oversize_suppression_enabled:
            return
        judged: list[tuple[Detection, bool]] = [
            (d, is_oversize(d.box, analysis.shape, cfg.oversize_area_ceiling))
            for d in analysis.candidates]
        analysis.suppressed.extend(d for d, oversize in judged if oversize)
        analysis.candidates = [d for d, oversize in judged if not oversize]
