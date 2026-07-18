"""
Parses COLMAP's plain-text sparse reconstruction output info into our Camera/INtrisics objects.
We only support the PINHOLE model for now --- COLMAP does supports many other models.

Example contents of camera.txt:
# CAMERA_ID, MODEL, WIDTH, HEIGHT, INTR_PARAMS[]
1 PINHOLE 1600 1200 1400.0 1400.0 800.0 600.0

"""


from splatproj.camera.camera import Intrinsics


def parse_cameras_txt(path: str) -> dict[int, Intrinsics]:
    """
    Reads a COLMAP camera.txt file
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