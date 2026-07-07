from __future__ import annotations

from dataclasses import dataclass

# The classes the watcher may act on, offered as the dashboard menu. person is
# also always detected (never alerted on) to suppress misclassified humans.
# Mirrored by TunableSettings._ALLOWED_TARGETS (importing it here would create
# a core -> vision cycle).
ANIMAL_TARGETS = ("dog", "cat", "bird")
PERSON_LABEL = "person"

# Things a counter raider steals or knocks over. Observed, never targeted:
# these classes can appear in the inventory readout and theft forensics but
# cannot fire the deterrent. Fixtures (oven, sink...) are excluded: they
# never move, so they would only be noise.
INVENTORY_LABELS = frozenset({
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
})


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
