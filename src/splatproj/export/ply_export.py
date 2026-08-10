"""
Exports a trained GaussianModel to the standard 3DGS .ply format, the
same layout used by the original INRIA reference implementation and
loadable directly by mkkellogg's GaussianSplats3D three.js viewer.

Binary PLY, one "vertex" record per Gaussian, fields in this exact
order (order matters -- it's positional, not name-matched):
    x, y, z                     -- position (means)
    nx, ny, nz                  -- normals (unused by the renderer,
                                    included for format compatibility;
                                    zeros are fine)
    f_dc_0, f_dc_1, f_dc_2       -- SH degree-0 color coefficient
    f_rest_0 ... f_rest_44       -- higher-order SH coefficients
                                    (we only model degree 0, so these
                                    are all zero -- flat color, no
                                    view-dependent shading yet)
    opacity                      -- inverse-sigmoid(opacity), i.e.
                                    exactly our raw_opacities already
    scale_0, scale_1, scale_2    -- log(scale), i.e. exactly our
                                    raw_scales already
    rot_0, rot_1, rot_2, rot_3   -- quaternion (w, x, y, z)
"""

import numpy as np
import torch

# SH degree-0 basis function constant. Color and the SH DC coefficient
# are related by: color = 0.5 + SH_C0 * f_dc  ->  f_dc = (color - 0.5) / SH_C0
SH_C0 = 0.28209479177387814

N_SH_REST = 45  # 15 higher-order coefficients per RGB channel (degree <=3)


def export_ply(model, path: str) -> None:
    means = model.means.detach().cpu().numpy().astype(np.float32)
    n = means.shape[0]

    normals = np.zeros((n, 3), dtype=np.float32)

    colors = model.colors.detach().cpu().numpy().astype(np.float32)
    f_dc = (colors - 0.5) / SH_C0  # [N, 3]

    f_rest = np.zeros((n, N_SH_REST), dtype=np.float32)

    # Already in the exact space the format wants -- see module docstring.
    opacity = model.raw_opacities.detach().cpu().numpy().astype(np.float32).reshape(n, 1)
    scales = model.raw_scales.detach().cpu().numpy().astype(np.float32)

    rots = model.quats.detach().cpu().numpy().astype(np.float32)
    # Normalize defensively -- optimization can drift a quaternion
    # slightly off unit length, and a non-unit quaternion means a
    # slightly-wrong rotation in the viewer.
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)

    # Concatenate in the exact field order the header declares below.
    attributes = np.concatenate([means, normals, f_dc, f_rest, opacity, scales, rots], axis=1)

    header = _build_header(n)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(attributes.astype(np.float32).tobytes())


def _build_header(n_vertices: int) -> str:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_vertices}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
    ]
    lines += [f"property float f_rest_{i}" for i in range(N_SH_REST)]
    lines += [
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
        "end_header",
        "",  # trailing newline after end_header, required by the spec
    ]
    return "\n".join(lines)