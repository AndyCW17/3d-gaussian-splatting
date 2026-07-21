"""
Loads a real COLMAP text export through our Day 1 parsers and prints a
summary --- first real-data run of parse_cameras_txt, parse_images_txt,
and parse_points3d_txt, which so far have only been tested against tiny
hand-written fixtures.

Run from the project root:
    PYTHONPATH=src python3 scripts/load_colmap_sanity_check.py <path_to_sparse_txt_folder>
"""


import sys
from splatproj.data.colmap_loader import (
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
)


def main():
    if len(sys.argv) !=2:
        print("Usage: python3 load_colmap_sanity_check.py <path_to_sparse_txt_folder>")
        sys.exit(1)
    
    folder = sys.argv[1]
    print(folder)

    cameras = parse_cameras_txt(f"{folder}/cameras.txt")
    print(f"Parsed {len(cameras)} camera(s):")
    for cam_id, intr in cameras.items():
        print(f"  camera {cam_id}: {intr.width}x{intr.height}, fx={intr.fx:.1f}")
    
    images = parse_images_txt(f"{folder}/images.txt", cameras)
    print(f"\nParsed {len(images)} posed image(s):")
    for cam in images[:5]:  # just the first 5, not all 26
        print(f"  {cam.image_name}: t={cam.extrinsics.t}")
    if len(images) > 5:
        print(f"  ... and {len(images) - 5} more")

    cloud = parse_points3d_txt(f"{folder}/points3D.txt")
    print(f"\nParsed sparse point cloud: {cloud.points.shape[0]} points")
    print(f"  position range: min={cloud.points.min(axis=0)}, max={cloud.points.max(axis=0)}")


if __name__ == "__main__":
    main()