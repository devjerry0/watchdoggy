from __future__ import annotations

import time
from dataclasses import asdict

import cv2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.events.store import EventStore
from doggy.web.routers.events import _event_dict

# Poll interval for the MJPEG stream. A frame is encoded only when the detect
# loop has produced a NEW one (FrameBuffer.version): re-encoding the same
# 640x480 frame ~10x/s per viewer cost ~0.3 of a core, which NCNN's
# barrier-synced threads felt as lost FPS.
_MJPEG_FRAME_INTERVAL_SECONDS = 0.1
_MJPEG_JPEG_QUALITY = 75


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

    @router.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        def gen():
            sent = -1
            while True:
                frame, version = annotated_buffer.get_versioned()
                if frame is not None and version != sent:
                    ok, buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _MJPEG_JPEG_QUALITY])
                    if ok:
                        sent = version
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + buf.tobytes() + b"\r\n")
                time.sleep(_MJPEG_FRAME_INTERVAL_SECONDS)

        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    return router
