"""
Sparse 3D point cloud, as produced by COLMAP's structure-from-motion step.

Used later to initialize where our Gaussians start out in space --- rather
than starting from random positions, we seed each Gaussian near a real
point COLMAP already found, which is a big part of why 3DGS training
converges quickly.
"""


from dataclasses import dataclass
import numpy as np


@dataclass
class SparsePointCloud:
    points: np.ndarray  # (N, 3) float32, xyz positions
    colors: np.array # (N, 3) uint8, rgb colors


    def __post_init__(self):
        assert self.points.shape[1] == 3, f"points must be (N, 3), got {self.points.shape}"
        assert self.points.shape[0] == self.colors.shape[0], (
            f"points and colors must have the same N, "
            f"got {self.points.shape[0] and {self.colors.shape[0]}}"
        )