import numpy as np
from PIL import Image
from splatproj.data.scene_dataset import SceneDataset


def test_dataset_len_and_getitem(tmp_path):
    # Build a minimal fake COLMAP export, same technique as test_colmap_loader.py
    sparse_dir = tmp_path / "sparse_txt"
    sparse_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    (sparse_dir / "cameras.txt").write_text(
        "1 PINHOLE 800 400 700.0 700.0 400.0 200.0\n"
    )
    (sparse_dir / "images.txt").write_text(
        "1 1.0 0.0 0.0 0.0 0.0 0.0 3.0 1 photo_a.jpg\n"
        "100.0 200.0 -1\n"
        "2 1.0 0.0 0.0 0.0 0.1 0.0 3.0 1 photo_b.jpg\n"
        "100.0 200.0 -1\n"
    )
    (sparse_dir / "points3D.txt").write_text(
        "1 0.0 0.0 0.0 255 0 0 0.5 1 0\n"
    )

    # Real (fake) image files matching the names above
    fake_array = np.random.randint(0, 255, size=(400, 800, 3), dtype=np.uint8)
    Image.fromarray(fake_array).save(images_dir / "photo_a.jpg")
    Image.fromarray(fake_array).save(images_dir / "photo_b.jpg")

    dataset, point_cloud = SceneDataset.from_colmap(
        str(sparse_dir), str(images_dir), target_width=400
    )

    assert len(dataset) == 2
    assert point_cloud.points.shape == (1, 3)

    camera, target = dataset[0]
    assert camera.intrinsics.width == 400  # downscaled from 800
    assert camera.intrinsics.height == 200  # aspect ratio preserved
    assert target.shape == (200, 400, 3)  # matches downscaled camera exactly