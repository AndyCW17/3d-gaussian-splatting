"""
Quaternion <-> rotation matrix conversion.

A quaternion (qw, qx, qy, qz) is a compact 4-number way to represent a
3D rotation. This is the standard conversion formula --- the same one
used by COLMAP, and later by our Gaussian model, since gradient descent
behaves much better optimizing a quaternion than optimizing a raw 3x3
matrix directly (nudging 9 numbers can easily produce an invalid,
non-rotation matrix; nudging a quaternion mostly stays close to valid).
"""

import numpy as np


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Converts a unit quaternion into a 3x3 rotation matrix."""

    # Normalize first --- COLMAP's quaternions should already be unit
    # length, but guarding against tiny floating-point drift is cheap
    # and avoids a subtly-wrong rotation matrix later.
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),       2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),       1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float32)

    return R