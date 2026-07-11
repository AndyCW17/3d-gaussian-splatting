"""
Camera representation for the splat pipeline.

Split into two pieces, matching how real cameras work:
  - intrinsics: properties of the camera itself (focal length, optical center)
  - extrinsics: where the camera is in the world (rotation + translation)

This file only defines the data structure. Projection math (turning a 3D
point into a 2D pixel coordinate) comes in the next chunk, once this is
tested and committed.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Intrinsics:
    """The camera's internal optics. Same for every photo taken with the
    same physical camera/lens, regardless of where it was standing."""

    fx: float   # focal length in pixels, x-axis
    fy: float   # focal length in pixels, y-axis
    cx: float   # optical center x (usually ~ image_width / 2)
    cy: float   # optical center y (usually ~ image_height / 2)
    width: int
    height: int

    def as_matrix(self) -> np.ndarray:
        """3x3 intrinsics matrix K, standard pinhole camera form:
            [ fx  0  cx ]
            [ 0  fy  cy ]
            [ 0   0   1 ]
        """
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


@dataclass
class Extrinsics:
    """Where the camera was for one specific photo, stored as a
    world-to-camera transform: p_camera = R @ p_world + t.
    This matches COLMAP's own convention, so we don't need to convert
    back and forth later."""

    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # shape (3,) translation vector

    def __post_init__(self):
        assert self.R.shape == (3, 3), f"R must be 3x3, got {self.R.shape}"
        assert self.t.shape == (3,), f"t must be shape (3,), got {self.t.shape}"


@dataclass
class Camera:
    """One posed photo: its intrinsics + extrinsics + a name tracing it
    back to the original uploaded image file."""

    intrinsics: Intrinsics
    extrinsics: Extrinsics
    image_name: str
