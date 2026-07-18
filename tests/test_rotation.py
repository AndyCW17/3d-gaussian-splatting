import numpy as np
from splatproj.camera.rotation import quaternion_to_rotation_matrix


def test_identity_quaternion_gives_identity_matrix():
    # qw=1, qx=qy=qz=0 represents "no rotation" --- should give back
    # the identity matrix exactly.
    R = quaternion_to_rotation_matrix(1.0, 0.0, 0.0, 0.0)
    assert np.allclose(R, np.eye(3))


def test_output_is_a_valid_rotation_matrix():
    # Any unit quaternion should produce an orthonormal matrix:
    # R @ R.T == identity. This is the real defining property of a
    # rotation matrix, so it's a much stronger check than comparing
    # to hardcoded numbers.
    R = quaternion_to_rotation_matrix(0.7071, 0.7071, 0.0, 0.0)
    should_be_identity = R @ R.T
    assert np.allclose(should_be_identity, np.eye(3), atol=1e-4)


def test_determinant_is_one():
    # A valid rotation (not a reflection) always has determinant +1.
    R = quaternion_to_rotation_matrix(0.9239, 0.0, 0.3827, 0.0)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-4)