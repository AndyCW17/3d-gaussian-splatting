"""
Parses COLMAP's plain-text sparse reconstruction output info into our Camera/INtrisics objects.
We only support the PINHOLE model for now --- COLMAP does supports many other models.

Example contents of cameras.txt:
# CAMERA_ID, MODEL, WIDTH, HEIGHT, INTR_PARAMS[]
1 PINHOLE 1600 1200 1400.0 1400.0 800.0 600.0

# Example contents of images.txt:
# (the second line lists 2d keypoint COLMAP matched in that image won't be used here)

# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
# 4 0.699 0.0 0.0 0.715 0.1 0.2 5.0 1 photo_003.jpg
# 2124.5 1051.8 3 2140.1 1049.2 -1 ...

"""

import numpy as np
from splatproj.camera.camera import Intrinsics
from splatproj.camera.camera import Camera, Extrinsics
from splatproj.camera.rotation import quaternion_to_rotation_matrix
from splatproj.data.point_cloud import SparsePointCloud


def parse_cameras_txt(path: str) -> dict[int, Intrinsics]:
    """
    Reads a COLMAP cameras.txt file
    Returns:
        dict mapping camera_id -> Intrinsics
    """
    cameras: dict[int, Intrinsics] = {}

    with open (path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): # skip blank lines and comments
                continue
        
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])

            if model != "PINHOLE":
                raise ValueError(
                    f"Camera {camera_id}: unsupported '{model}', "
                    f"only PINHOLE is supported. Re-run COLMAP with "
                    f"--ImageReader.camera_model PINHOLE."
                )
            
            fx, fy, cx, cy = (float(p) for p in parts[4:8])

            cameras[camera_id] = Intrinsics(
                fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
            )

    return cameras


def parse_images_txt(path: str, cameras: dict[int, Intrinsics] ) -> list[Camera]:
    """
    Reads a COLMAP images.txt file. Each image occupies two lines: a pose
    line (which we parse) followed by a 2D-points line (which we skip).

    Args:
        path: path to images.txt
        cameras: output of parse_cameras_txt, used to look up each
                  image's CAMERA_ID and attach the right Intrinsics

    Returns:
        list of Camera objects, one per posed image
    """
    result: list[Camera] = []

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # Pose lines and points alternate, so step through two at a time.
    for i in range(0, len(lines), 2):
        pose_line = lines[i]

        parts = pose_line.split()
        qw, qx, qy, qz = (float(p) for p in parts[1:5])
        tx, ty, tz = (float(p) for p in parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]

        if camera_id not in cameras:
            raise ValueError(
                f"Image '{image_name} reference camera_id {camera_id}, "
                f"wasn't found in cameras.txt."
            )
        
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float32)

        result.append(Camera(
            intrinsics=cameras[camera_id], 
            extrinsics=Extrinsics(R=R, t=t),
            image_name=image_name,
        ))
    return result


def parse_points3d_txt(path: str) -> SparsePointCloud:
    """
    Reads a COLMAP points3D.txt file. We only keep position and color ---
    per-point error and the image track are COLMAP-internal bookkeeping,
    no need for our training.
    """
    points = []
    colors = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            x, y, z = (float(p) for p in parts[1:4])
            r, g, b = (int(p) for p in parts[4:7])

            points.append([x, y , z])
            colors.append([r, g, b])

    return SparsePointCloud(
        points=np.array(points, dtype=np.float32),
        colors=np.array(colors, dtype=np.uint8),
    )