"""
Projects 3D world points into 2D pixel coordinates, given a camera.

This is the standard pinhole camera model, done in three steps:
  1. World -> camera space:  p_cam = R @ p_world + t
  2. Perspective divide:     x' = p_cam.x / p_cam.z,  y' = p_cam.y / p_cam.z
     (this is *the* step that makes far-away things look smaller ---
     dividing by depth is what "perspective" means, geometrically)
  3. Camera space -> pixels: apply the intrinsics (focal length, optical center)

Points behind the camera (z <= 0) don't get filtered out here --- we return
their depth so the caller  can decide what to do with them. Keeping this
function's job narrow (just "do the math") makes it easier to test and
reuse later.
"""

import numpy as np
from splatproj.camera.camera import Camera


def project_points(camera: Camera, points_world:np.array) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        camera: a posed camera(intrisics+ extrinsics)
        points_world: (N, 3) array of 3d points in world space

    Returns:
        pixels: (N, 2) array of (u, v) pixel coordinates
        depths: (N,) arrays of camera-space z (distance in front of the camera)
    """
    assert points_world.ndim == 2 and points_world.shape[1] == 3, \
        f"expected (N, 3), got {points_world.shape}"
    
    R = camera.extrinsics.R
    t = camera.extrinsics.t

    # Step 1: world coordinates to camera space,
    # R @ p for each point, done for all points at once with matrix multiplication:
    # (N,3) @ (3,3).T + t
    points_cam = points_world @ R.T + t #points_cam is coordinates of the object in camera space 

    # Step 2: depth is extracted from the projected coordinates 
    # to calculate how big or small an object is
    depths = points_cam[:, 2]

    x_normalized = points_cam[:, 0] / (depths + 1e-8)
    y_normalized = points_cam[:, 1] / (depths + 1e-8)
    # small epsilon for zero division error,handles edge cases where position is same depth as camera

    # Step 3: apply intrisics
    # maps the coordinates of the object in camera space onto the actual pixel grid for image
    
    # 3x3 intrinsics matrix K, standard pinhole camera form:
    #             [ fx  0  cx ]
    #             [ 0  fy  cy ]
    #             [ 0   0   1 ]
    
    K = camera.intrinsics.as_matrix()
    u = K[0, 0] * x_normalized +K[0, 2] # adding cx and cy because camera centers in the dead center
    v = K[1, 1] * y_normalized +K[1, 2] # , while images start from top-left


    pixels = np.stack([u, v], axis=1)
    return pixels, depths