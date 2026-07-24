"""
Builds the training optimizer with per-parameter learning rates ---
different Gaussian attributes have different natural scales and need
different step sizes, so we give each its own learning rate rather than
one compromise value for everything.

Rates below are simplified, fixed versions of the original 3DGS paper's
choices (the paper also decays the position learning rate over training;
we start with a fixed rate for simplicity and can add decay later if needed).
"""


import torch
from splatproj.model.gaussians import GaussianModel


def build_optimizer(model: GaussianModel) -> torch.optim.Adam:
    param_groups = [
        # Position: smallest learning rate. A large step here could send
        # a Gaussian to a wildly wrong place in the scene, since this is 
        # a real-world splatial coordinate, not a bounded/normalized value.
        {"params": [model.means], "lr":1.6e-4, "name": "means"},

        # Scale and roatation: moderate rates --- these control shape, and 
        # neeed to adjust noticeably during training but shouldn't swing badly
        {"params": [model.scales], "lr":5e-3, "name": "scales"},
        {"params": [model.quats], "lr":5e-3, "name": "quats"},

        # Opacity and color: can tolerate larger, faster-moving updates,
        # since they're bounded/normalized quantities, not spatial coordinates
        {"params": [model.opacities], "lr":5e-2, "name": "opacities"},
        {"params": [model.colors], "lr":5e-2, "name": "colors"},
    ]
    return torch.optim.Adam(param_groups, eps=1e-15)
