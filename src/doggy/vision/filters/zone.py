from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from doggy.vision.detection import Detection

if TYPE_CHECKING:
    from doggy.core.config import TunableSettings
    from doggy.vision.analysis import FrameAnalysis

_MIN_POLYGON_POINTS = 3


class ZoneInclusionFilter:
    """Filter link: keep only candidates whose box overlaps a normalized zone.

    The polygon (points in [0,1]) is rasterized to a frame-sized 0/1 mask and
    cached; the mask is rebuilt only when the points or the frame shape change.
    Fewer than 3 points means "no zone" -> every detection passes through.
    Narrows `analysis.candidates` and `analysis.inventory` (the zone is what
    defines "the counter"); drawn targets are unaffected.
    """

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None
        self._key: tuple | None = None

    def _ensure_mask(self, points: list[tuple[float, float]], shape: tuple) -> None:
        h, w = shape[0], shape[1]
        key = (tuple(points), (h, w))
        if key == self._key:
            return
        mask = np.zeros((h, w), np.uint8)
        pts = np.array([[int(x * w), int(y * h)] for x, y in points], np.int32)
        cv2.fillPoly(mask, [pts], 1)
        self._mask, self._key = mask, key

    def overlap_fraction(self, box: tuple[int, int, int, int],
                         points: list[tuple[float, float]], shape: tuple) -> float:
        """Fraction of the box's area that lies inside the zone (0.0..1.0).

        Uses the cached mask: in-zone pixels under the clipped box over the
        clipped box's area. A degenerate (zero-area) box overlaps nothing -> 0.0.
        """
        self._ensure_mask(points, shape)
        assert self._mask is not None
        h, w = shape[0], shape[1]
        x1, y1, x2, y2 = box
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 0.0
        return float(self._mask[y1:y2, x1:x2].sum()) / area

    def in_zone(self, box: tuple[int, int, int, int],
                points: list[tuple[float, float]], shape: tuple,
                min_overlap: float = 0.0) -> bool:
        if len(points) < _MIN_POLYGON_POINTS:
            return True
        frac = self.overlap_fraction(box, points, shape)
        # min_overlap == 0 keeps the original any-pixel behaviour; a positive
        # threshold requires that much of the box to be genuinely inside.
        return frac > 0 if min_overlap == 0 else frac >= min_overlap

    def filter(self, detections: list[Detection],
               points: list[tuple[float, float]], shape: tuple,
               min_overlap: float = 0.0) -> list[Detection]:
        if len(points) < _MIN_POLYGON_POINTS:
            return list(detections)
        return [d for d in detections
                if self.in_zone(d.box, points, shape, min_overlap)]

    def apply(self, analysis: "FrameAnalysis", cfg: "TunableSettings") -> None:
        if not (cfg.zone_enabled and len(cfg.zone_points) >= _MIN_POLYGON_POINTS):
            return
        # Candidates can fire the deterrent, so require a real amount of the box
        # inside the watch area -- a box that only scrapes the edge is dropped.
        analysis.candidates = self.filter(
            analysis.candidates, cfg.zone_points, analysis.shape, cfg.zone_overlap)
        # Inventory never alerts; an item half-off the counter should still be
        # counted, so keep any-overlap here (min_overlap left at its 0.0 default).
        analysis.inventory = self.filter(
            analysis.inventory, cfg.zone_points, analysis.shape)
