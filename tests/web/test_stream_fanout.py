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


def test_newest_viewer_evicts_the_oldest_and_release_is_idempotent():
    fanout = MjpegFanout(FrameBuffer(), max_viewers=2, quality=75)
    a, b = fanout.acquire(), fanout.acquire()
    c = fanout.acquire()                        # third tab: oldest (a) is evicted
    assert a.evicted and not b.evicted and not c.evicted
    assert fanout.viewers == 2
    a.release()                                 # evicted generator's finally
    a.release()                                 # ...and the response background task
    assert fanout.viewers == 2                  # eviction already freed a's slot
    b.release()
    assert fanout.viewers == 1
    d = fanout.acquire()                        # room again: nobody evicted
    assert not c.evicted and not d.evicted and fanout.viewers == 2


def test_evicted_frame_is_a_jpeg_encoded_once(monkeypatch):
    fanout = MjpegFanout(FrameBuffer(), max_viewers=1, quality=75)
    first = fanout.evicted_frame()
    assert first[:2] == b"\xff\xd8"           # JPEG SOI marker
    assert fanout.evicted_frame() is first
