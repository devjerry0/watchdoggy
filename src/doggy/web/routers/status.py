from __future__ import annotations

import threading
import time
from dataclasses import asdict

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.events.store import EventStore
from doggy.web.routers.events import _event_dict

# Poll interval for the MJPEG stream. A frame is encoded only when the detect
# loop has produced a NEW one (FrameBuffer.version), and only ONCE for all
# viewers: re-encoding the same 640x480 frame ~10x/s per viewer cost ~0.3
# of a core, which NCNN's barrier-synced threads felt as lost FPS.
_MJPEG_FRAME_INTERVAL_SECONDS = 0.1
_MJPEG_JPEG_QUALITY = 75
# Safety net against a browser with a dozen dashboard tabs: each extra viewer
# still costs a 10 Hz generator plus ~100 KB/s of WiFi. Beyond this the
# OLDEST stream is evicted (it gets a final "moved to a newer tab" frame):
# the tab the user just opened is the one they are looking at.
_MJPEG_MAX_VIEWERS = 4
_EVICTED_TEXT = ("Live view moved to a newer tab", "click to resume here")


class MjpegFanout:
    """Shares one JPEG encode per new frame across every live viewer, and
    keeps at most ``max_viewers`` streams open by evicting the oldest."""

    class Slot:
        def __init__(self, fanout: "MjpegFanout", order: int) -> None:
            self._fanout = fanout
            self.order = order
            self.evicted = False
            self._open = True

        def release(self) -> None:
            # Idempotent: the generator's finally, the response's background
            # task, and an eviction may each call this.
            with self._fanout._lock:
                self._release_locked()

        def _release_locked(self) -> None:
            if self._open:
                self._open = False
                self._fanout._viewers -= 1
                self._fanout._live.pop(self.order, None)

    def __init__(self, buffer: FrameBuffer, max_viewers: int,
                 quality: int) -> None:
        self._buffer = buffer
        self._max = max_viewers
        self._quality = quality
        self._lock = threading.Lock()
        self._viewers = 0
        self._next_order = 0
        self._live: dict[int, MjpegFanout.Slot] = {}  # order -> open slot
        self._version = -1
        self._jpeg: bytes | None = None
        self._evicted_jpeg: bytes | None = None

    @property
    def viewers(self) -> int:
        return self._viewers

    def acquire(self) -> "MjpegFanout.Slot":
        """A slot for a new viewer; at the cap, the oldest open viewer is
        evicted so the newest always wins."""
        with self._lock:
            while self._viewers >= self._max and self._live:
                oldest = self._live[min(self._live)]
                oldest.evicted = True
                oldest._release_locked()
            self._viewers += 1
            slot = MjpegFanout.Slot(self, self._next_order)
            self._next_order += 1
            self._live[slot.order] = slot
            return slot

    def evicted_frame(self) -> bytes:
        """The one frame an evicted viewer receives before its stream ends."""
        with self._lock:
            if self._evicted_jpeg is None:
                canvas = np.zeros((480, 640, 3), np.uint8)
                canvas[:] = (28, 24, 20)
                for i, line in enumerate(_EVICTED_TEXT):
                    cv2.putText(canvas, line, (40, 220 + 44 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9 - 0.2 * i,
                                (200, 200, 200), 2, cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", canvas)
                self._evicted_jpeg = buf.tobytes() if ok else b""
            return self._evicted_jpeg

    def frame(self) -> tuple[bytes | None, int]:
        """(jpeg, version) of the newest annotated frame; encodes at most once
        per version no matter how many viewers ask."""
        frame, version = self._buffer.get_versioned()
        if frame is None:
            return None, version
        with self._lock:
            if version != self._version:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
                if not ok:
                    return None, version
                self._jpeg, self._version = buf.tobytes(), version
            return self._jpeg, self._version


def build_router(runtime: RuntimeSettings, annotated_buffer: FrameBuffer,
                 status: StatusStore, event_store: EventStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/status")
    def api_status() -> dict:
        return {
            **asdict(status.snapshot()),
            "settings": runtime.get().model_dump(mode="json"),
            "events": [_event_dict(r) for r in event_store.list(limit=10)],
        }

    fanout = MjpegFanout(annotated_buffer, _MJPEG_MAX_VIEWERS, _MJPEG_JPEG_QUALITY)

    @router.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        slot = fanout.acquire()

        def part(jpeg: bytes) -> bytes:
            return (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg + b"\r\n")

        def gen():
            sent = -1
            try:
                while not slot.evicted:
                    jpeg, version = fanout.frame()
                    if jpeg is not None and version != sent:
                        sent = version
                        yield part(jpeg)
                    time.sleep(_MJPEG_FRAME_INTERVAL_SECONDS)
                # A newer tab took the slot: leave this one a frame that says
                # so, then end the stream (the img keeps showing it).
                yield part(fanout.evicted_frame())
            finally:
                slot.release()

        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame",
                                 background=BackgroundTask(slot.release))

    return router
