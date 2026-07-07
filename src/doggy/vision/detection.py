from __future__ import annotations

from dataclasses import dataclass

# The classes the watcher may act on, offered as the dashboard menu. person is
# also always detected (never alerted on) to suppress misclassified humans.
# Mirrored by TunableSettings._ALLOWED_TARGETS (importing it here would create
# a core -> vision cycle).
ANIMAL_TARGETS = ("dog", "cat", "bird")
PERSON_LABEL = "person"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
