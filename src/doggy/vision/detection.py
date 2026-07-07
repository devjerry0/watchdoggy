from __future__ import annotations

from dataclasses import dataclass

# The COCO class this system acts on. Shared by the detector (filter) and the
# trigger (match) so the target label lives in exactly one place.
TARGET_LABEL = "dog"
# Also detected (never alerted on) to suppress people misclassified as dogs.
PERSON_LABEL = "person"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
