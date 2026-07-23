import numpy as np
import torch
from PIL import Image
from splatproj.data.image_io import load_target_image


def test_load_target_image_shape_and_range(tmp_path):
    # Create a small fake image on disk, since we don't want tests
    # depending on your actual photos existing.
    fake_img_path = tmp_path / "fake.jpg"
    array = np.random.randint(0, 255, size=(100, 200, 3), dtype=np.uint8)
    Image.fromarray(array).save(fake_img_path)

    loaded = load_target_image(str(fake_img_path), target_width=50, target_height=25)

    assert loaded.shape == (25, 50, 3)
    assert loaded.min() >= 0.0
    assert loaded.max() <= 1.0


def test_load_target_image_is_float_tensor(tmp_path):
    fake_img_path = tmp_path / "fake2.jpg"
    array = np.full((10, 10, 3), 128, dtype=np.uint8)
    Image.fromarray(array).save(fake_img_path)

    loaded = load_target_image(str(fake_img_path), target_width=10, target_height=10)

    assert loaded.dtype == torch.float32