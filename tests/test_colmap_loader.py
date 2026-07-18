from splatproj.data.colmap_loader import parse_cameras_txt


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