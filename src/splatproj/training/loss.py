"""
The training loss: a weighted combination of L1 (per-pixel accuracy) and
D-SSIM (structural/perceptual similarity), matching the original 3DGS
paper's choice: loss = 0.8 * L1 + 0.2 * D-SSIM.

Why not just L1 alone: L1 treats every pixel independently, so it can't
tell the difference between "slightly blurry but structurally correct"
and "sharp but shifted by one pixel" --- both might have similar L1 error,
but only one of those actually looks right to a human. SSIM compares
local windows of the image (structure, contrast, luminance) rather than
raw pixel values, which is a much closer match to perceived quality.
"""


import torch
import torch.nn.functional as F


def l1_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference per pixel. Both inputs (H, W, 3) in [0, 1]."""
    return torch.abs(rendered - target).mean()


def _gaussian_kernel(window_size: int, sigma: float, device) -> torch.Tensor:
    """1D gaussian kernel, used to build a 2d blur window for SSIM's local 
    windowed statistics (mean, variance, covariance)."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def ssim(rendered: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Structural Similarity Index between two images, computed with a
    Gaussian-weighted local window (standard SSIM formulation).

    Args:
        rendered, target: (H, W, 3) tensors in [0, 1]

    Returns:
        scalar SSIM value in [-1, 1], where 1 means identical.
    """
    device  = rendered.device

    # SSIM operates per-channel via 2D convolution, so reshape from
    # (H, W, 3) to (1, 3, H, W) --- the (batch, channels, H, W) format
    # torch's conv2d expects.
    r = rendered.permute(2, 0, 1).unsqueeze(0)
    t = target.permute(2, 0, 1).unsqueeze(0)

    kernel_1d = _gaussian_kernel(window_size, sigma=1.5, device=device)
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel = kernel_2d.expand(3, 1, window_size, window_size)

    pad = window_size // 2
    mu_r = F.conv2d(r, kernel, padding=pad, groups=3)
    mu_t = F.conv2d(t, kernel, padding=pad, groups=3)
    mu_r_sq, mu_t_sq, mu_rt = mu_r**2, mu_t**2, mu_r * mu_t

    sigma_r_sq = F.conv2d(r * r, kernel, padding=pad, groups=3) - mu_r_sq
    sigma_t_sq = F.conv2d(t * t, kernel, padding=pad, groups=3) - mu_t_sq
    sigma_rt = F.conv2d(r * t, kernel, padding=pad, groups=3) - mu_rt

    # Small stability constants (standard SSIM values for [0,1]-range images)
    c1, c2 = 0.01**2, 0.03**2

    numerator = (2 * mu_rt + c1) * (2 * sigma_rt + c2)
    denominator = (mu_r_sq + mu_t_sq + c1) * (sigma_r_sq + sigma_t_sq + c2)

    return (numerator / denominator).mean()


def training_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The combined loss used for optimization: 0.8 * L1 + 0.2 * D-SSIM."""
    l1 = l1_loss(rendered, target)
    d_ssim = 1.0 - ssim(rendered, target)
    return 0.8 * l1 + 0.2 * d_ssim