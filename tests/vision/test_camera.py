import numpy as np

from doggy.vision.camera import FakeCamera, build_camera
from doggy.core.config import Settings


def test_fake_camera_yields_given_frames():
    frames = [np.full((4, 4), i, dtype=np.uint8) for i in range(3)]
    cam = FakeCamera(frames)
    out = list(cam.frames())
    assert len(out) == 3
    assert out[1][0, 0] == 1
    cam.close()


def test_build_camera_file_backend_uses_fake(tmp_path):
    # camera_backend=file with no path still returns a Camera object
    s = Settings(camera_backend="file", camera_path=None)
    cam = build_camera(s)
    assert hasattr(cam, "frames")
    cam.close()


def test_opencv_camera_caps_fps_and_buffer(monkeypatch):
    # Decoding the webcam's default 30 fps cost ~0.2 core the detector never
    # used; the capture must ask for a modest rate and a 1-frame buffer.
    import cv2
    from doggy.vision import camera as cam_mod

    class FakeCap:
        def __init__(self, index):
            self.props = {}
        def set(self, prop, value):
            self.props[prop] = value
            return True
        def read(self):
            return False, None
        def release(self):
            pass

    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", FakeCap)
    cam = cam_mod.OpenCVCamera(0)
    assert cam._cap.props[cv2.CAP_PROP_FPS] == cam_mod._CAMERA_FPS
    assert cam._cap.props[cv2.CAP_PROP_BUFFERSIZE] == cam_mod._CAMERA_BUFFERSIZE
