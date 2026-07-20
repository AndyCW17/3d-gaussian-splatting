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
        scales = torch.full((n, 3), initial_scale, dtype=torch.float32)
        self.scales = nn.Parameter(scales)

        # Rotation: identity quaternion (w=1, x=y=z=0) --- "no rotation"
        # to start, same convention as our camera rotation code.
        quats = torch.zeros((n, 4), dtype=torch.float32)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)

        # Opacity: a single value per Gaussian, moderate to start so
        # gradients can push it toward more or less transparent.
        opacities = torch.full((n,), initial_opacity ,dtype=torch.float32)
        self.opacities = nn.Parameter(opacities)

        # Color: taken directly from the point cloud's RGB, normalized
        # from [0, 255] to [0, 1]. This is SH degree 0 --- a single flat
        # color per Gaussian, no view-dependence yet. Higher-order SH
        # (color that changes with viewing angle) can be added later.
        colors = torch.from_numpy(point_cloud.colors).float() / 255.0
        self.colors = nn.Parameter(colors)
    
    def __len__(self) -> int:
        return self.means.shape[0]
    