"""
Loads real COLMAP output, builds a GaussianModel from the actual sparse
point cloud, and renders from one of the actual recovered camera poses ---
first fully real-data pass through the whole Day + Day2 pipeline.

Run from the project root, inside Docker (needs GPU + gsplat):
    PYTHONPATH=src python3 scripts/render_real_camera.py <sparse_txt_folder> <output.png>
"""


import sys
from dataclasses import replace
from splatproj.data.colmap_loader import parse_cameras_txt, parse_images_txt, parse_points3d_txt
from splatproj.data.image_io import save_render
from splatproj.model.gaussians import GaussianModel
from splatproj.model.renderer import render


def main():
    folder, out_path = sys.argv[1], sys.argv[2]

    cameras = parse_cameras_txt(f"{folder}/cameras.txt")
    images = parse_images_txt(f"{folder}/images.txt", cameras)
    cloud = parse_points3d_txt(f"{folder}/points3d.txt")

    print(f"Loaded {len(images)} cameras, {cloud.points.shape[0]} points")


    # Downscale to a manageable render resolution --- 7008px wide would
    # be slow and isn't needed for a visual sanity check.
    scale_factor = 800/ images[0].intrinsics.width
    small_intrinsics = images[0].intrinsics.scaled(scale_factor)
    camera = replace(images[0], intrinsics=small_intrinsics)

    print("Rendering from {camera.image_name} at {small_intrinsics.width}x{small_intrinsics.height}")

    model = GaussianModel(cloud, initial_scale=0.02)
    image, _ = render(model, camera, device="cuda")

    save_render(image, out_path)
    print(f"Saved to {out_path}.")


if __name__ == "__main__":
    main()