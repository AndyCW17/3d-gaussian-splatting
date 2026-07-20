import numpy as np
import torch
from splatproj.data.point_cloud import SparsePointCloud
from splatproj.model.gaussians import GaussianModel


def make_test_point_cloud(n=5):
    points = np.random.rand(n, 3).astype(np.float32)
    colors = np.random.randint(0, 255, size=(n, 3)).astype(np.uint8)
    return SparsePointCloud(points=points, colors=colors)


def test_gaussian_count_matches_point_cloud():
    cloud = make_test_point_cloud(n=10)
    model = GaussianModel(cloud)
    assert len(model) == 10


def test_means_initialized_from_point_cloud():
    cloud = make_test_point_cloud(n=5)
    model = GaussianModel(cloud)
    assert torch.allclose(model.means, torch.from_numpy(cloud.points))


def test_colors_normalized_to_zero_one_range():
    cloud = make_test_point_cloud(n=5)
    model = GaussianModel(cloud)
    assert model.colors.max() <= 1.0
    assert model.colors.min() >= 0.0


def test_parameters_require_grad():
    # This is the whole point of using nn.Parameter --- confirm gradients
    # are actually enabled, since Day 3's training loop depends on it.
    cloud = make_test_point_cloud(n=5)
    model = GaussianModel(cloud)
    assert model.means.requires_grad
    assert model.scales.requires_grad
    assert model.quats.requires_grad
    assert model.opacities.requires_grad
    assert model.colors.requires_grad


def test_quats_start_as_identity():
    cloud = make_test_point_cloud(n=3)
    model = GaussianModel(cloud)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(model.quats[0], expected)