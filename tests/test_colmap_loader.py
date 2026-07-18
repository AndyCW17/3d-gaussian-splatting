import numpy as np
from splatproj.data.colmap_loader import parse_cameras_txt
from splatproj.data.colmap_loader import parse_images_txt
from splatproj.camera.camera import Intrinsics

def test_parse_single_camera(tmp_path):
    cameras_txt = tmp_path / "camera.txt"
    cameras_txt.write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "1 PINHOLE 1600 1200 1400.0 1400.0 800.0 600.0\n"
    )

    cameras = parse_cameras_txt(str(cameras_txt))


    assert len(cameras)  == 1
    assert cameras[1].fx == 1400.0
    assert cameras[1].width == 1600



def test_parse_multiple_cameras(tmp_path):
    cameras_txt = tmp_path / "cameras.txt"
    cameras_txt.write_text(
        "1 PINHOLE 1600 1200 1400.0 1400.0 800.0 600.0\n"
        "2 PINHOLE 800 600 700.0 700.0 400.0 300.0\n"
    )

    cameras = parse_cameras_txt(str(cameras_txt))

    assert len(cameras) == 2
    assert cameras[2].fx == 700.0


def test_rejects_non_pinhole_model(tmp_path):
    cameras_txt = tmp_path / "cameras.txt"
    cameras_txt.write_text("1 RADIAL 1600 1200 1400.0 800.0 600.0 0.01\n")

    try:
        parse_cameras_txt(str(cameras_txt))
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_parse_single_image(tmp_path):
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "1 1.0 0.0 0.0 0.0 0.5 0.0 2.0 1 photo_001.jpg\n"
        "100.0 200.0 -1\n"  # points line, should be skipped
    )
    cameras = {1: Intrinsics(fx=1400.0, fy=1400.0, cx=800.0, cy=600.0, width=1600, height=1200)}

    images = parse_images_txt(str(images_txt), cameras)

    assert len(images) == 1
    assert images[0].image_name == "photo_001.jpg"
    assert np.allclose(images[0].extrinsics.t, [0.5, 0.0, 2.0])
    assert np.allclose(images[0].extrinsics.R, np.eye(3))  # identity quaternion


def test_parse_multiple_images(tmp_path):
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 photo_001.jpg\n"
        "100.0 200.0 -1\n"
        "2 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1 photo_002.jpg\n"
        "150.0 250.0 -1\n"
    )
    cameras = {1: Intrinsics(fx=1400.0, fy=1400.0, cx=800.0, cy=600.0, width=1600, height=1200)}

    images = parse_images_txt(str(images_txt), cameras)

    assert len(images) == 2
    assert images[1].image_name == "photo_002.jpg"


def test_missing_camera_id_raises(tmp_path):
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 99 photo_001.jpg\n"
        "100.0 200.0 -1\n"
    )
    cameras = {1: Intrinsics(fx=1400.0, fy=1400.0, cx=800.0, cy=600.0, width=1600, height=1200)}

    try:
        parse_images_txt(str(images_txt), cameras)
        assert False, "should have raised ValueError"
    except ValueError:
        pass