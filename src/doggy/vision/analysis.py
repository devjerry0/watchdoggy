from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from doggy.core.config import TunableSettings
from doggy.vision.detection import Detection, INVENTORY_LABELS, PERSON_LABEL
from doggy.vision.detector import Detector
from doggy.vision.filters.base import FilterChain


@dataclass
class FrameAnalysis:
    """The detection state for one frame, narrowed in place by the filter chain.

    - `targets`: watched-class detections that survived suppression (all drawn).
    - `candidates`: the subset of `targets` still eligible to trigger (post-zone).
    - `people`: person-labeled detections (used for suppression / overlay).
    - `inventory`: food/tableware on the counter; observed only, never a
      target or candidate, so it can never fire the deterrent.
    """

    shape: tuple[int, ...]
    people: list[Detection]
    targets: list[Detection]
    candidates: list[Detection]
    inventory: list[Detection] = field(default_factory=list)


class DetectionAnalyzer:
    """Runs the detector then the filter chain, producing a `FrameAnalysis`."""

    def __init__(self, detector: Detector, chain: FilterChain) -> None:
        self._detector = detector
        self._chain = chain

    def analyze(self, frame: np.ndarray, cfg: TunableSettings) -> FrameAnalysis:
        detections = self._detector.detect(frame)
        detected = set(cfg.target_labels)
        alertable = set(cfg.alert_labels)
        targets = [d for d in detections if d.label in detected]
        people = [d for d in detections if d.label == PERSON_LABEL]
        analysis = FrameAnalysis(
            shape=frame.shape, people=people, targets=targets,
            # Only alert-class animals may trigger; detect-only ones are
            # drawn (in the ignored grey) but never enter the candidate set.
            candidates=[d for d in targets if d.label in alertable],
            inventory=[d for d in detections if d.label in INVENTORY_LABELS]
            if cfg.inventory_enabled else [])
        self._chain.run(analysis, cfg)
        return analysis


class InventoryTracker:
    """Debounced presence for inventory labels: an item counts as on the
    counter when seen in at least 2 of the last 5 analyzed frames. Counts are
    the max simultaneous instances seen in those frames (flicker-proof)."""

    WINDOW = 5
    NEEDED = 2

    def __init__(self) -> None:
        self._recent: deque[Counter] = deque(maxlen=self.WINDOW)

    def update(self, labels: list[str]) -> list[dict]:
        self._recent.append(Counter(labels))
        return self._present()

    def labels(self) -> set[str]:
        return {item["label"] for item in self._present()}

    def _present(self) -> list[dict]:
        seen: Counter = Counter()
        appearances: Counter = Counter()
        for frame_counts in self._recent:
            for label, count in frame_counts.items():
                appearances[label] += 1
                seen[label] = max(seen[label], count)
        return [{"label": label, "count": seen[label]}
                for label in sorted(seen)
                if appearances[label] >= self.NEEDED]
