"""Live-view fan-out: N dashboard tabs must cost one JPEG encode per new
frame (not N), and a runaway number of tabs must be refused instead of
starving the detector."""
import numpy as np

from doggy.core.status import FrameBuffer
from doggy.web.routers import status as status_router
from doggy.web.routers.status import MjpegFanout


def _frame(v):
    return np.full((8, 8, 3), v, np.uint8)


def test_one_encode_per_frame_version_shared_by_viewers(monkeypatch):
    buf = FrameBuffer()
    fanout = MjpegFanout(buf, max_viewers=4, quality=75)
    encodes = []
    real = status_router.cv2.imencode
    monkeypatch.setattr(status_router.cv2, "imencode",
                        lambda *a, **k: encodes.append(1) or real(*a, **k))
    assert fanout.frame() == (None, 0)          # nothing captured yet
    buf.set(_frame(10))
    j1, v1 = fanout.frame()
    j2, v2 = fanout.frame()                     # a second viewer, same frame
    assert j1 is j2 and v1 == v2 and len(encodes) == 1
    buf.set(_frame(200))
    j3, v3 = fanout.frame()
    assert v3 == v1 + 1 and j3 != j1 and len(encodes) == 2


def test_viewer_cap_and_idempotent_release():
    fanout = MjpegFanout(FrameBuffer(), max_viewers=2, quality=75)
    a, b = fanout.acquire(), fanout.acquire()
    assert a is not None and b is not None
    assert fanout.acquire() is None             # third tab refused
    a.release()
    a.release()                                 # generator finally + background task
    assert fanout.viewers == 1
    assert fanout.acquire() is not None         # slot freed exactly once
