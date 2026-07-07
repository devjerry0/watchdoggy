from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator, Protocol

import cv2
import numpy as np

from doggy.core.config import Settings

# Consecutive failed reads tolerated before OpenCVCamera gives up, and the
# backoff between reconnect attempts.
_DEFAULT_MAX_RECONNECTS = 5
_RECONNECT_BACKOFF_SECONDS = 0.5


class Camera(Protocol):
    def frames(self) -> Iterator[np.ndarray]: ...
    def close(self) -> None: ...


class FakeCamera:
    """Yields a fixed list of frames (in-memory) or, via from_video, a file."""

    def __init__(self, frames: list[np.ndarray], loop: bool = False) -> None:
        self._frames = frames
        self._loop = loop

    @classmethod
    def from_video(cls, path: Path, loop: bool = False) -> "FakeCamera":
        cap = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return cls(frames, loop=loop)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            for f in self._frames:
                yield f
            if not self._loop:
                return

    def close(self) -> None:
        self._frames = []


class OpenCVCamera:
    """USB webcam via cv2.VideoCapture; reconnects on transient read failures."""

    def __init__(self, index: int, max_reconnects: int = _DEFAULT_MAX_RECONNECTS) -> None:
        self._index = index
        self._max_reconnects = max_reconnects
        self._cap = cv2.VideoCapture(index)

    def frames(self) -> Iterator[np.ndarray]:
        failures = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                failures += 1
                if failures > self._max_reconnects:
                    raise RuntimeError(f"camera {self._index} lost after {failures} failures")
                self._cap.release()
                time.sleep(_RECONNECT_BACKOFF_SECONDS)
                self._cap = cv2.VideoCapture(self._index)
                continue
            failures = 0
            yield frame

    def close(self) -> None:
        self._cap.release()


def _build_opencv_camera(settings: Settings) -> Camera:
    return OpenCVCamera(settings.camera_index)


def _build_file_camera(settings: Settings) -> Camera:
    if settings.camera_path:
        return FakeCamera.from_video(settings.camera_path, loop=True)
    return FakeCamera([], loop=False)


_BACKENDS: dict[str, Callable[[Settings], Camera]] = {
    "opencv": _build_opencv_camera,
    "file": _build_file_camera,
}


def build_camera(settings: Settings) -> Camera:
    try:
        builder = _BACKENDS[settings.camera_backend]
    except KeyError:
        raise ValueError(f"unknown camera backend: {settings.camera_backend!r}") from None
    return builder(settings)
