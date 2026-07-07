from __future__ import annotations

import platform
from pathlib import Path
from typing import Protocol

import numpy as np

from doggy.core.config import Settings
from doggy.detection import Detection, PERSON_LABEL, TARGET_LABEL
from doggy.core.runtime import RuntimeSettings

# Classes the detector surfaces: the target (dog) plus people, which the pipeline
# uses to suppress people misclassified as dogs. Both come free from one inference.
_RETURNED_LABELS = frozenset({TARGET_LABEL, PERSON_LABEL})


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class StubDetector:
    """Returns scripted detections; used by pipeline/web tests (no model)."""

    def __init__(self, scripted: list[list[Detection]]) -> None:
        self._scripted = list(scripted)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._scripted:
            return self._scripted.pop(0)
        return []


def select_device() -> str:
    try:
        import torch

        if platform.machine() == "arm64" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YoloDetector:
    """Ultralytics YOLO wrapper: runs the model and returns dog + person detections."""

    def __init__(self, model_path: Path, runtime: RuntimeSettings, device: str | None = None) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._runtime = runtime
        self._device = device or select_device()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self._runtime.get().confidence, device=self._device, verbose=False
        )
        out: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                label = names[int(box.cls[0])]
                if label not in _RETURNED_LABELS:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(label, float(box.conf[0]), (x1, y1, x2, y2)))
        return out


def build_detector(settings: Settings, runtime: RuntimeSettings) -> Detector:
    return YoloDetector(settings.model_path, runtime)
