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


def load_target_image(path: str, target_width: int, target_height: int) -> torch.Tensor:
    """
    Loads a real photo and resizes it to match a (possibly downscaled)
    camera's resolution, so it can be directly compared against a render
    from that same camera.

    Returns:
        (H, W, 3) tensor, values in [0, 1] --- same convention as render().
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((target_width, target_height), Image.BILINEAR)

    array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array)