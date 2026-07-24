"""
Save/load a GaussianModel's trained parameters to/from disk.

Thin wrapper around PyTorch's standard state_dict mechanism --- since
GaussianModel is an nn.Module, this is the normal, correct way to persist
its parameters (means, scales, quats, opacities, colors) as a single file.
"""

import torch
from splatproj.model.gaussians import GaussianModel


def save_checkpoint(model: GaussianModel, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: GaussianModel, path: str, device: str = "cuda") -> GaussianModel:
    """Loads saved parameters into an already-constructed GaussianModel
    (the model must already have the right number of Gaussians ---
    this only restores values, not the Gaussian count itself)."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model