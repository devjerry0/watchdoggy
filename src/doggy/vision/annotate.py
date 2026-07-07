from __future__ import annotations

import cv2
import numpy as np

# Detection-overlay styling (OpenCV uses BGR).
_BOX_THICKNESS = 2
_LABEL_FONT_SCALE = 0.5
_LABEL_THICKNESS = 1
_LABEL_Y_OFFSET = 6  # pixels above the box to place the label
_DOG_ACTIVE_COLOR = (0, 0, 255)     # red BGR — in-zone / will trigger
_DOG_IGNORED_COLOR = (150, 150, 150)  # grey — outside zone, ignored
_PERSON_COLOR = (255, 0, 0)         # blue BGR — shown, never alerted on
_INVENTORY_COLOR = (140, 160, 180)  # muted sand BGR
_ZONE_COLOR = (0, 165, 255)         # orange BGR
_ZONE_ALPHA = 0.25


def _draw_box(out, box, label, confidence, color, thickness=_BOX_THICKNESS):
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(out, f"{label} {confidence:.2f}", (x1, max(0, y1 - _LABEL_Y_OFFSET)),
                cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, color, _LABEL_THICKNESS)


def annotate(frame, detections, in_zone=None, zone_points=None, people=None,
             inventory=None):
    out = frame.copy()
    h, w = frame.shape[0], frame.shape[1]
    if zone_points and len(zone_points) >= 3:
        pts = np.array([[int(x * w), int(y * h)] for x, y in zone_points], np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], _ZONE_COLOR)
        cv2.addWeighted(overlay, _ZONE_ALPHA, out, 1 - _ZONE_ALPHA, 0, out)
        cv2.polylines(out, [pts], True, _ZONE_COLOR, _BOX_THICKNESS)
    active = detections if in_zone is None else in_zone
    for i in inventory or []:
        _draw_box(out, i.box, i.label, i.confidence, _INVENTORY_COLOR, thickness=1)
    for p in people or []:
        _draw_box(out, p.box, p.label, p.confidence, _PERSON_COLOR)
    for d in detections:
        color = _DOG_ACTIVE_COLOR if d in active else _DOG_IGNORED_COLOR
        _draw_box(out, d.box, d.label, d.confidence, color)
    return out
