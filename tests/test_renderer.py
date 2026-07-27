import numpy as np
import torch
import pytest
from splatproj.camera.camera import Intrinsics, Extrinsics, Camera
from splatproj.data.point_cloud import SparsePointCloud
from splatproj.model.gaussians import GaussianModel
from splatproj.model.renderer import render


def make_test_camera(width=64, height=64):
    intr = Intrinsics(fx=50.0, fy=50.0, cx=width / 2, cy=height / 2, width=width, height=height)
    ext = Extrinsics(R=np.eye(3, dtype=np.float32), t=np.array([0.0, 0.0, -3.0], dtype=np.float32))
    return Camera(intrinsics=intr, extrinsics=ext, image_name="synthetic.jpg")


def make_test_point_cloud(n=20):
    points = (np.random.rand(n, 3).astype(np.float32) - 0.5)  # small cluster near origin
    colors = np.random.randint(0, 255, size=(n, 3)).astype(np.uint8)
    return SparsePointCloud(points=points, colors=colors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat requires a CUDA GPU")
def test_render_output_shape():
    camera = make_test_camera(width=64, height=64)
    cloud = make_test_point_cloud(n=20)
    model = GaussianModel(cloud)

    image, _ = render(model, camera, device="cuda")

    assert image.shape == (64, 64, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat requires a CUDA GPU")
def test_render_output_has_no_nans():
    camera = make_test_camera()
    cloud = make_test_point_cloud(n=20)
    model = GaussianModel(cloud)

    image, _ = render(model, camera, device="cuda")

    assert not torch.isnan(image).any()
    assert image.min() >= 0.0