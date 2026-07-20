"""
Wraps gsplat's rasterization kernel: given a GaussianModel and a Camera, 
renders an actual image. 

This is intentionally thin -- gsplat handles projection, per-tile depth
sorting, and alpha-compositing internally (the CUDA kernel work we're 
deliberately not reimplmenting). Our job is converting our own
Camera/GaussianModel representation into the tensor shapes gsplat
expects, and handing the results back in shape we understand.
"""


import torch
import gsplat
from splatproj.camera.camera import Camera
from splatproj.model.gaussians import GaussianModel


def camera_to_viewmat(camera: Camera) -> torch.Tensor:
    """ Builds gsplat's expected 4x4 world-to-camera matrix:
        [ R  t ]
        [ 0  1 ]
    """
    R = torch.from_numpy(camera.extrinsics.R).float()
    t = torch.from_numpy(camera.extrinsics.t).float()

    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat


def camera_to_K(camera: Camera) -> torch.Tensor:
    return torch.from_numpy(camera.intrinsics.as_matrix()).float()


def render(model: GaussianModel, camera: Camera, device: str = "cuda") -> torch.Tensor:
    """
    Renders one image of the given Gaussians from the given camera.

    Returns:
        (H, W, 3) tensor of rendered RGB values in [0, 1].
    """

    viewmat = camera_to_viewmat(camera).to(device).unsqueeze(0)
    K = camera_to_K(camera).to(device).unsqueeze(0)

    renders, alphas, meta = gsplat.rasterization(
        means=model.means.to(device),
        quats=model.quats.to(device),
        scales=model.scales.to(device),
        opacities=model.opacities.to(device),
        colors=model.colors.to(device),
        viewmats=viewmat,
        Ks=K,
        width=camera.intrinsics.width,
        height=camera.intrinsics.height,
    )

    # render shape:(1, H, W, 3) --- we only passed one camera, so drop
    # that leading "batch of cameras" dimension.
    return renders[0]