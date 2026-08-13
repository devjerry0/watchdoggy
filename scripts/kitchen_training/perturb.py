"""Image corruptions, shared by training augmentation and the stress eval.

Two consumers, one source of truth so they can never drift apart:
  train_variants   what --augment adds to the TRAIN split. Photometric only:
                   nothing here moves a pixel's position, so every label box
                   stays exactly valid.
  stress_variants  what the robustness eval throws at the deployable bundle.
                   The train variants plus corruptions training never saw --
                   including shift, which crops and rescales (it moves boxes,
                   which is fine for frame-level scoring but is why it must
                   never become a train variant).
"""
from __future__ import annotations

import cv2
import numpy as np

MOTION_BLUR_KERNEL = 9    # horizontal streak: a trotting dog at 2fps exposure
DEFOCUS_KERNEL = 5
DARKEN_FACTOR = 0.5       # dusk-dark kitchen
BRIGHTEN_FACTOR = 1.6     # morning sun straight into the lens
JPEG_CRUSH_QUALITY = 30   # far below the camera's actual write quality
SHIFT_FRACTION = 0.05     # small reframing: bumped mount, thermal drift


def motion_blur(image: np.ndarray) -> np.ndarray:
    streak = np.zeros((MOTION_BLUR_KERNEL, MOTION_BLUR_KERNEL), np.float32)
    streak[MOTION_BLUR_KERNEL // 2, :] = 1.0 / MOTION_BLUR_KERNEL
    return cv2.filter2D(image, -1, streak)


def defocus(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (DEFOCUS_KERNEL, DEFOCUS_KERNEL), 0)


def darken(image: np.ndarray) -> np.ndarray:
    return (image * DARKEN_FACTOR).astype(np.uint8)


def brighten(image: np.ndarray) -> np.ndarray:
    return np.clip(image * BRIGHTEN_FACTOR, 0, 255).astype(np.uint8)


def jpeg_crush(image: np.ndarray) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_CRUSH_QUALITY])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def shift(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    margin = int(SHIFT_FRACTION * width)
    cropped = image[margin:height - margin, margin:width - margin]
    return cv2.resize(cropped, (width, height))


def train_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    """Photometric variants that keep every box valid: motion blur (a dog
    trotting through a 2fps exposure), defocus, and a dusk-dark kitchen."""
    return {"mblur": motion_blur(image),
            "gblur": defocus(image),
            "dark": darken(image)}


def stress_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    return {**train_variants(image),
            "bright": brighten(image),
            "jpeg": jpeg_crush(image),
            "shift": shift(image)}
