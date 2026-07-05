import numpy as np

from doggy.camera import FakeCamera, build_camera
from doggy.config import Settings


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
