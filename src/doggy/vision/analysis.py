from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doggy.core.config import TunableSettings
from doggy.vision.detection import Detection, PERSON_LABEL, TARGET_LABEL
from doggy.vision.detector import Detector
from doggy.vision.filters.base import FilterChain


@dataclass
class FrameAnalysis:
    """The detection state for one frame, narrowed in place by the filter chain.

    - `dogs`: dog-labeled detections that survived suppression (all drawn).
    - `candidates`: the subset of `dogs` still eligible to trigger (post-zone).
    - `people`: person-labeled detections (used for suppression / overlay).
    """

    shape: tuple[int, ...]
    people: list[Detection]
    dogs: list[Detection]
    candidates: list[Detection]


class DetectionAnalyzer:
    """Runs the detector then the filter chain, producing a `FrameAnalysis`."""

    def __init__(self, detector: Detector, chain: FilterChain) -> None:
        self._detector = detector
        self._chain = chain

    def analyze(self, frame: np.ndarray, cfg: TunableSettings) -> FrameAnalysis:
        detections = self._detector.detect(frame)
        dogs = [d for d in detections if d.label == TARGET_LABEL]
        people = [d for d in detections if d.label == PERSON_LABEL]
        analysis = FrameAnalysis(
            shape=frame.shape, people=people, dogs=dogs, candidates=list(dogs))
        self._chain.run(analysis, cfg)
        return analysis
