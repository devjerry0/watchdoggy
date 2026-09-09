from __future__ import annotations

import threading
import time
from dataclasses import asdict

import cv2
from fastapi import APIRouter, HTTPException
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
# stream answers 503 and the dashboard says so instead of a broken image.
_MJPEG_MAX_VIEWERS = 4


class MjpegFanout:
    """Shares one JPEG encode per new frame across every live viewer, and
    caps how many viewers may hold a stream at once."""

    class Slot:
        def __init__(self, fanout: "MjpegFanout") -> None:
            self._fanout = fanout
            self._open = True

        def release(self) -> None:
            # Idempotent: both the generator's finally and the response's
            # background task call this, whichever the client's disconnect
            # reaches first.
            with self._fanout._lock:
                if self._open:
                    self._open = False
                    self._fanout._viewers -= 1

    def __init__(self, buffer: FrameBuffer, max_viewers: int,
                 quality: int) -> None:
        self._buffer = buffer
        self._max = max_viewers
        self._quality = quality
        self._lock = threading.Lock()
        self._viewers = 0
        self._version = -1
        self._jpeg: bytes | None = None

    @property
    def viewers(self) -> int:
        return self._viewers

    def acquire(self) -> "MjpegFanout.Slot | None":
        with self._lock:
            if self._viewers >= self._max:
                return None
            self._viewers += 1
            return MjpegFanout.Slot(self)

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
        if slot is None:
            raise HTTPException(
                status_code=503,
                detail=f"too many live viewers ({_MJPEG_MAX_VIEWERS}); "
                       "close another dashboard tab")

        def gen():
            sent = -1
            try:
                while True:
                    jpeg, version = fanout.frame()
                    if jpeg is not None and version != sent:
                        sent = version
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + jpeg + b"\r\n")
                    time.sleep(_MJPEG_FRAME_INTERVAL_SECONDS)
            finally:
                slot.release()

        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame",
                                 background=BackgroundTask(slot.release))

    return router
