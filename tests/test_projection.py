import numpy as np
from splatproj.camera.camera import Intrinsics, Extrinsics, Camera
from splatproj.camera.projection import project_points


def make_identity_camera(width=1600, height=1200, focal=1400.0):
    """Camera at the world origin, loking straight down +z, no rotation."""
    intr = Intrinsics(fx=focal, fy=focal, cx=width / 2, cy=height / 2,
                      width=width, height=height)
    ext = Extrinsics(R=np.eye(3), t=np.zeros(3))
    return Camera(intrinsics=intr, extrinsics=ext, image_name="test.jpg")


def test_point_on_optical_axis_lands_at_image_center():
    # a point straight ahead of an un-rotated camera should project to
    # exactly the optical center (cx, cy)
    camera = make_identity_camera()
    point = np.array([[0.0, 0.0, 5.0]])

    pixels, depths = project_points(camera, point)

    assert np.allclose(pixels[0], [800, 600])
    assert depths[0] == 5.0


def test_farther_point_has_greater_depth_same_pixel():
    # Two points on the same ray from the camera should land on the exact
    # same pixel, but at different depths --- this checks that depth and
    # pixel position are computed independently, as they should be.
    camera = make_identity_camera()
    near = np.array([[0.0, 0.0, 2.0]])
    far = np.array([[0.0, 0.0, 8.0]])

    pixels_near, depths_near = project_points(camera, near)
    pixels_far, depths_far = project_points(camera, far)

    assert np.allclose(pixels_near, pixels_far)
    assert depths_far[0] > depths_near[0]


def test_off_axis_point_moves_away_from_center():
    # A point shifted sideways (+x) should project to a pixel with a
    # larger u coordinate than the center --- checks the sign/direction
    # of the math is correct, not just that it runs without crashing.
    camera = make_identity_camera()
    point = np.array([[1.0, 0.0, 5.0]])

    pixels, _ = project_points(camera, point)

    assert pixels[0, 0] > 800.0  # u > cx


def test_batch_of_points():
    # Confirms the function handles N > 1 points at once, since training
    # will always call this with many points, never just one.
    camera = make_identity_camera()
    points = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [-1.0, -1.0, 5.0]])

    pixels, depths = project_points(camera, points)

    assert pixels.shape == (3, 2)
    assert depths.shape == (3,)