import numpy as np
from splatproj.camera.camera import Intrinsics, Extrinsics, Camera


def test_intrinsics_matrix_shape():
    intr = Intrinsics(fx=1400.0, fy=1400.0, cx=800.0, cy=600.0, width=1600, height=1200)
    K = intr.as_matrix()
    assert K.shape == (3, 3)
    assert K[0, 0] == 1400.0
    assert K[2, 2] == 1.0


def test_extrinsics_valid_shapes():
    ext = Extrinsics(R=np.eye(3), t=np.zeros(3))
    assert ext.R.shape == (3, 3)


def test_extrinsics_rejects_bad_shape():
    try:
        Extrinsics(R=np.eye(2), t=np.zeros(3))
        assert False, "should have raised"
    except AssertionError:
        pass


def test_camera_construction():
    intr = Intrinsics(1400.0, 1400.0, 800.0, 600.0, 1600, 1200)
    ext = Extrinsics(R=np.eye(3), t=np.zeros(3))
    cam = Camera(intrinsics=intr, extrinsics=ext, image_name="photo_001.jpg")
    assert cam.image_name == "photo_001.jpg"
