from __future__ import annotations

import platform
from pathlib import Path
from typing import Protocol

import numpy as np

from doggy.core.config import Settings
from doggy.vision.detection import Detection, PERSON_LABEL
from doggy.core.runtime import RuntimeSettings


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
    """Ultralytics YOLO wrapper: returns watched-class + person detections.

    People come free from the same inference and feed the misclassification
    suppression filter; they are never alerted on.
    """

    def __init__(self, model_path: Path, runtime: RuntimeSettings, device: str | None = None) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._runtime = runtime
        self._device = device or select_device()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        cfg = self._runtime.get()
        wanted = set(cfg.target_labels) | {PERSON_LABEL}
        results = self._model.predict(
            frame, conf=cfg.confidence, device=self._device, verbose=False
        )
        out: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                label = names[int(box.cls[0])]
                if label not in wanted:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(label, float(box.conf[0]), (x1, y1, x2, y2)))
        return out


def build_detector(settings: Settings, runtime: RuntimeSettings) -> Detector:
    return YoloDetector(settings.model_path, runtime)
