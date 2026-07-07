from __future__ import annotations

import cv2
import numpy as np

from doggy.vision.detection import Detection

_MIN_POLYGON_POINTS = 3


class ZoneFilter:
    """Keep only detections whose box overlaps a normalized polygon zone.

    The polygon (points in [0,1]) is rasterized to a frame-sized 0/1 mask and
    cached; the mask is rebuilt only when the points or the frame shape change.
    Fewer than 3 points means "no zone" -> every detection passes through.
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

    def in_zone(self, box: tuple[int, int, int, int],
                points: list[tuple[float, float]], shape: tuple) -> bool:
        if len(points) < _MIN_POLYGON_POINTS:
            return True
        self._ensure_mask(points, shape)
        assert self._mask is not None
        h, w = shape[0], shape[1]
        x1, y1, x2, y2 = box
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return False
        return bool(self._mask[y1:y2, x1:x2].any())

    def filter(self, detections: list[Detection],
               points: list[tuple[float, float]], shape: tuple) -> list[Detection]:
        if len(points) < _MIN_POLYGON_POINTS:
            return list(detections)
        return [d for d in detections if self.in_zone(d.box, points, shape)]
