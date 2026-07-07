from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doggy.core.config import TunableSettings
from doggy.vision.detection import Detection, PERSON_LABEL
from doggy.vision.detector import Detector
from doggy.vision.filters.base import FilterChain


@dataclass
class FrameAnalysis:
    """The detection state for one frame, narrowed in place by the filter chain.

    - `targets`: watched-class detections that survived suppression (all drawn).
    - `candidates`: the subset of `targets` still eligible to trigger (post-zone).
    - `people`: person-labeled detections (used for suppression / overlay).
    """

    shape: tuple[int, ...]
    people: list[Detection]
    targets: list[Detection]
    candidates: list[Detection]


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
            candidates=[d for d in targets if d.label in alertable])
        self._chain.run(analysis, cfg)
        return analysis
