from __future__ import annotations

import platform
from pathlib import Path
from typing import Protocol

import numpy as np

from doggy.config import Settings
from doggy.detection import Detection


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

        if torch.cuda.is_available():
            return "cuda"
        if platform.machine() == "arm64" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YoloDetector:
    def __init__(self, model_path: Path, confidence: float, device: str | None = None) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._device = device or select_device()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self._confidence, device=self._device, verbose=False
        )
        out: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                label = names[int(box.cls[0])]
                if label != "dog":
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(label, float(box.conf[0]), (x1, y1, x2, y2)))
        return out


def build_detector(settings: Settings) -> Detector:
    return YoloDetector(settings.model_path, settings.confidence)
