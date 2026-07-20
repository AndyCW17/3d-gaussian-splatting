"""
Renders one synthetic scene and saves it as an image --- a visual gut
check that the pipeline (point cloud -> Gaussians -> gsplat render)
produces sane output, before we test it against real COLMAP data.

Run from the project root:
    PYTHONPATH=src python3 scripts/render_sanity_check.py
"""

import numpy as np
from splatproj.camera.camera import Intrinsics, Extrinsics, Camera
from splatproj.data.point_cloud import SparsePointCloud
from splatproj.data.image_io import save_render
from splatproj.model.gaussians import GaussianModel
from splatproj.model.renderer import render


def main():
    # A camera sitting 3 units back from the origin, looking at it ---
    # same synthetic setup as our render test, just larger for visibility.
    width, height = 400, 400
    intr = Intrinsics(fx=300.0, fy=300.0, cx=width / 2, cy=height / 2,
                       width=width, height=height)
    ext = Extrinsics(R=np.eye(3, dtype=np.float32),
                      t=np.array([0.0, 0.0, 3.0], dtype=np.float32))
    camera = Camera(intrinsics=intr, extrinsics=ext, image_name="sanity_check")

    # A cluster of 200 random colored points near the origin --- standing
    # in for real COLMAP output, which we don't have loaded yet.
    n = 200
    points = (np.random.rand(n, 3).astype(np.float32) - 0.5)
    colors = np.random.randint(0, 255, size=(n, 3)).astype(np.uint8)
    cloud = SparsePointCloud(points=points, colors=colors)

    model = GaussianModel(cloud)
    image = render(model, camera, device="cuda")

    save_render(image, "outputs/sanity_check.png")
    print("Saved to outputs/sanity_check.png")


if __name__ == "__main__":
    main()