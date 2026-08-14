from __future__ import annotations

import platform
from pathlib import Path
from typing import Protocol

import numpy as np

from doggy.core.config import Settings, TunableSettings
from doggy.vision.detection import Detection, INVENTORY_LABELS, PERSON_LABEL
from doggy.core.runtime import RuntimeSettings


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


# With training-data capture on, watched-class and person detections are ALSO
# surfaced below the alarm threshold, down to this floor: the model's
# half-guesses (bent people at 0.4, uncertain "dogs") are the confusion-zone
# frames the fine-tune learns the most from. The analyzer routes sub-threshold
# ones into `FrameAnalysis.lowconf` -- observational only, never alarm inputs.
DATASET_FLOOR = 0.25


def keep_detection(label: str, score: float, cfg: TunableSettings) -> bool:
    """Per-class threshold re-check after the model's low-water predict pass."""
    wanted = set(cfg.target_labels) | {PERSON_LABEL}
    if label in wanted:
        floor = DATASET_FLOOR if cfg.dataset_enabled else cfg.confidence
        return score >= floor
    return (cfg.inventory_enabled and label in INVENTORY_LABELS
            and score >= cfg.inventory_confidence)


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
        # Inventory rides the same inference pass at its own (laxer) threshold;
        # predict at the lower bar, then re-apply each class's own bar below.
        # Dataset capture lowers the bar further so confusion-zone detections
        # exist to be captured (the analyzer keeps them out of alarm inputs).
        conf = (min(cfg.confidence, cfg.inventory_confidence)
                if cfg.inventory_enabled else cfg.confidence)
        if cfg.dataset_enabled:
            conf = min(conf, DATASET_FLOOR)
        results = self._model.predict(
            frame, conf=conf, device=self._device, verbose=False
        )
        out: list[Detection] = []
        for r in results:
            out.extend(_result_detections(r, cfg))
        return out


def _result_detections(result, cfg) -> list[Detection]:
    """One YOLO result's boxes, filtered to the classes/thresholds we keep."""
    out: list[Detection] = []
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        score = float(box.conf[0])
        if not keep_detection(label, score, cfg):
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        out.append(Detection(label, score, (x1, y1, x2, y2)))
    return out


def build_detector(settings: Settings, runtime: RuntimeSettings) -> Detector:
    return YoloDetector(settings.model_path, runtime)
