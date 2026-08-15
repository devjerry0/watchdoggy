from __future__ import annotations

import logging
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np

from doggy.core.config import Settings, TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.vision import inference_worker
from doggy.vision.detection import Detection, INVENTORY_LABELS, PERSON_LABEL

log = logging.getLogger("doggy")

# One inference normally takes well under a second; a worker silent for this
# long is stuck (or its child died mid-call) and gets replaced.
INFERENCE_TIMEOUT_S = 30.0


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
    """YOLO detections via a dedicated inference child process.

    People come free from the same inference and feed the misclassification
    suppression filter; they are never alerted on.

    The child process exists because the ncnn binding holds the GIL for the
    whole forward pass (see vision/inference_worker.py) -- in-process
    inference froze every other thread, including the dashboard. The model
    lives only in the child; a dead or stuck child is replaced and the
    frame is reported empty (the M-of-N trigger absorbs a missed frame).
    """

    def __init__(self, model_path: Path, runtime: RuntimeSettings,
                 device: str | None = None) -> None:
        self._model_path = str(model_path)
        self._runtime = runtime
        self._device = device or select_device()
        self._pool = self._start_pool()

    def _start_pool(self) -> ProcessPoolExecutor:
        # spawn, never fork: this process carries a dozen live threads
        # (uvicorn, capture, soothing) and forking them is undefined joy.
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=inference_worker.init,
            initargs=(self._model_path, self._device))

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
        try:
            future = self._pool.submit(inference_worker.predict, frame, conf)
            raw = future.result(timeout=INFERENCE_TIMEOUT_S)
        except Exception:
            # A crashed/stuck child must not take down the detect loop.
            log.exception("inference worker failed; replacing it")
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = self._start_pool()
            return []
        return [Detection(label, score, box) for label, score, box in raw
                if keep_detection(label, score, cfg)]


def build_detector(settings: Settings, runtime: RuntimeSettings) -> Detector:
    return YoloDetector(settings.model_path, runtime)
