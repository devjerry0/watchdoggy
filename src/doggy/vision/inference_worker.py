"""The YOLO/NCNN forward pass runs here, in a dedicated child process.

Proven on-device (py-spy + ticker-gap test, 2026-08-14): the ncnn python
binding holds the GIL for the ENTIRE extract -- hundreds of ms per frame,
every cycle -- freezing every other thread in the process. Dashboard
requests took 1-2s despite ~70ms of actual work, and no Python-side tuning
can help: a C call that holds the GIL is not preemptible. Isolating
inference in a spawn child gives the appliance back its threads; the frame
crosses the process boundary (~1ms) instead of the GIL blocking everything.

Module-level state is per-child: the pool's initializer loads the model
once, then every predict reuses it."""
from __future__ import annotations

import numpy as np

_model = None
_device = "cpu"

RawDetection = tuple[str, float, tuple[int, int, int, int]]

# Tiny warmup frame: forces the backend to load/compile in init() so the
# first real frame doesn't pay it.
_WARMUP_SIDE = 32


def init(model_path: str, device: str) -> None:
    global _model, _device
    from ultralytics import YOLO  # heavy: child-process only
    _model = YOLO(model_path)
    _device = device
    warmup = np.zeros((_WARMUP_SIDE, _WARMUP_SIDE, 3), dtype=np.uint8)
    _model.predict(warmup, device=device, verbose=False)


def _raw_boxes(result) -> list[RawDetection]:
    out: list[RawDetection] = []
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        score = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        out.append((label, score, (x1, y1, x2, y2)))
    return out


def predict(frame: np.ndarray, conf: float) -> list[RawDetection]:
    results = _model.predict(frame, conf=conf, device=_device, verbose=False)
    out: list[RawDetection] = []
    for r in results:
        out.extend(_raw_boxes(r))
    return out
