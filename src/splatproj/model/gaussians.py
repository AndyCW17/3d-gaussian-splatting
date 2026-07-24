"""
The trainable Gaussian representation: one row per Gaussian, 
five attribute per row (position, scale, rotation, opacity, color).

Initialized directly from a SparsePointCloud rather than random positions
--- seeding Gaussian near real COLMAP points is a big part of why 3DGS
training converges quikly instead of blindly searching blindly through space.
(initializing the Gaussian object from nn.module inherently gives us trackable
gradients to optimize later in training)
"""


import torch
import torch.nn as nn
from splatproj.data.point_cloud import SparsePointCloud


class GaussianModel(nn.Module):
    def __init__(
        self,
        point_cloud: SparsePointCloud,
        initial_scale: float = 0.01,
        initial_opacity: float = 0.5,
    ):
        super().__init__()

        n = point_cloud.points.shape[0]

        # Positions: taken directly from the sparse point cloud.
        means = torch.from_numpy(point_cloud.points).float()
        self.means = nn.Parameter(means)

        # Scale: same small constant in all 3 local axes to start ---
        # training will stretch/shrink these per-Gaussian as needed.
        # Kept as plain positive floats for now; a stricter version would
        # store these in log-space to guarantee positivity during
        # optimization, but that's a refinement for Day 3, not today.
        raw_scale_init = torch.log(torch.tensor(initial_scale))
        raw_scales = torch.full((n, 3), raw_scale_init.item(), dtype=torch.float32)
        self.raw_scales = nn.Parameter(raw_scales)

        # Rotation: identity quaternion (w=1, x=y=z=0) --- "no rotation"
        # to start, same convention as our camera rotation code.
        quats = torch.zeros((n, 4), dtype=torch.float32)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)

        # Opacity: a single value per Gaussian, moderate to start so
        # gradients can push it toward more or less transparent.
        # sigmoid(0) == 0.5, so raw=0 happens to give our default of 0.5.
        raw_opacity_init = torch.logit(torch.tensor(initial_opacity))
        raw_opacities = torch.full((n,), raw_opacity_init.item(), dtype=torch.float32)
        self.raw_opacities = nn.Parameter(raw_opacities)

        # Color: taken directly from the point cloud's RGB, normalized
        # from [0, 255] to [0, 1]. This is SH degree 0 --- a single flat
        # color per Gaussian, no view-dependence yet. Higher-order SH
        # (color that changes with viewing angle) can be added later.
        colors = torch.from_numpy(point_cloud.colors).float() / 255.0
        self.colors = nn.Parameter(colors)


    @property
    def scales(self) -> torch.Tensor:
        """Actual, always-positive scale values used for rendering.
        Read-only --- computed fresh from raw_scales every access."""
        return torch.exp(self.raw_scales)


    @property
    def opacities(self) -> torch.Tensor:
        """Actual opacity values, always in (0, 1), used for rendering."""
        return torch.sigmoid(self.raw_opacities)


    def __len__(self) -> int:
        return self.means.shape[0]
    