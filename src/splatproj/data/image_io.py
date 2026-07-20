"""
Small utility for saving a rendered tensor as a viewable image file.
Kept separate from renderer.py --- rendering and file I/O are different
concerns, and this will get reused later (saving training progress
snapshots, saving final outputs for the UI).
"""


import torch
import numpy as np
from PIL import Image


def save_render(image: torch.Tensor, path: str) -> None:
    """
    Args:
        image: (H, W, 3) tensor with rgb values in [0, 1]
        path: where to write the .png file
    """
    assert image.ndim == 3 and image.shape[2] == 3, f"expected (H, W, 3), got {image.shape}"


    # Move to CPU, detach from the autograd graph (we're not backpropagating
    # through a saved file), clamp to a valid range, convert to uint8.
    array = image.detach().cpu().clamp(0.0, 1.0).numpy()
    array_uint8 = (array * 255).astype(np.uint8)

    Image.fromarray(array_uint8).save(path)