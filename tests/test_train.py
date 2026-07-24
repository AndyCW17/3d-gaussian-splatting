import numpy as np
import torch
import pytest
from splatproj.camera.camera import Intrinsics, Extrinsics, Camera
from splatproj.data.point_cloud import SparsePointCloud
from splatproj.model.gaussians import GaussianModel
from splatproj.model.renderer import render
from splatproj.training.optimizer import build_optimizer
from splatproj.training.train import train


def make_test_camera(width=64, height=64):
    intr = Intrinsics(fx=50.0, fy=50.0, cx=width / 2, cy=height / 2, width=width, height=height)
    ext = Extrinsics(R=np.eye(3, dtype=np.float32), t=np.array([0.0, 0.0, 3.0], dtype=np.float32))
    return Camera(intrinsics=intr, extrinsics=ext, image_name="synthetic.jpg")


class TinyDataset:
    """Minimal in-memory dataset --- just enough to satisfy what train()
    needs (__len__, __getitem__), without touching disk. Real training
    later uses SceneDataset instead."""

    def __init__(self, camera, target):
        self.camera = camera
        self.target = target

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.camera, self.target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat requires a CUDA GPU")
def test_training_loss_decreases():
    torch.manual_seed(0)
    camera = make_test_camera()

    n = 30
    points = (np.random.rand(n, 3).astype(np.float32) - 0.5)
    true_colors = np.random.randint(0, 255, size=(n, 3)).astype(np.uint8)
    cloud = SparsePointCloud(points=points, colors=true_colors)

    # The "ground truth" --- render once with correct colors, treat that
    # as the real photo we're training against.
    true_model = GaussianModel(cloud)
    with torch.no_grad():
        target_image = render(true_model, camera, device="cuda").clone()

    # The "student" --- same shape/positions, but wrong (random) colors.
    # Only this model gets trained.
    wrong_colors = np.random.randint(0, 255, size=(n, 3)).astype(np.uint8)
    wrong_cloud = SparsePointCloud(points=points, colors=wrong_colors)
    student_model = GaussianModel(wrong_cloud).to("cuda")

    dataset = TinyDataset(camera, target_image)
    optimizer = build_optimizer(student_model)

    loss_history = train(student_model, dataset, optimizer, num_iterations=100,
                          device="cuda", log_every=25)

    # The real assertion: loss should be meaningfully lower at the end
    # than at the start --- this is the actual proof training works.
    assert loss_history[-1] < loss_history[0] * 0.5