import numpy as np
from splatproj.data.point_cloud import SparsePointCloud
from splatproj.data.colmap_loader import parse_points3d_txt


def test_sparse_point_cloud_construction():
    cloud = SparsePointCloud(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
    )
    assert cloud.points.shape[0] == 5


def test_sparse_point_cloud_rejects_mismatched_lengths():
    try:
        SparsePointCloud(
            points=np.zeros((5, 3), dtype=np.float32),
            colors=np.zeros((3, 3), dtype=np.uint8),  # wrong length on purpose
        )
        assert False, "should have raised"
    except AssertionError:
        pass


def test_parse_single_point(tmp_path):
    points_txt = tmp_path / "points3D.txt"
    points_txt.write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "1 2.1 0.5 1.2 128 130 127 0.856 3 0 4 1 5 2\n"
    )

    cloud = parse_points3d_txt(str(points_txt))

    assert cloud.points.shape == (1, 3)
    assert np.allclose(cloud.points[0], [2.1, 0.5, 1.2])
    assert list(cloud.colors[0]) == [128, 130, 127]


def test_parse_multiple_points(tmp_path):
    points_txt = tmp_path / "points3D.txt"
    points_txt.write_text(
        "1 0.0 0.0 0.0 255 0 0 0.5 1 0\n"
        "2 1.0 1.0 1.0 0 255 0 0.3 1 1\n"
    )

    cloud = parse_points3d_txt(str(points_txt))

    assert cloud.points.shape == (2, 3)
    assert cloud.colors.shape == (2, 3)