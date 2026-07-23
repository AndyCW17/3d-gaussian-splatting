"""
Pairs each posed Camera with its real photo, downscaled consistently, so
the training loop can pull (camera, target_image) pairs directly --- the
one thing it actually needs each step.
"""


"""
Pairs each posed Camera with its real photo, downscaled consistently, so
the training loop can pull (camera, target_image) pairs directly --- the
one thing it actually needs each step.
"""

from dataclasses import replace
from splatproj.camera.camera import Camera
from splatproj.data.colmap_loader import parse_cameras_txt, parse_images_txt, parse_points3d_txt
from splatproj.data.image_io import load_target_image
from splatproj.data.point_cloud import SparsePointCloud


class SceneDataset:
    def __init__(self, images_folder: str, cameras: list[Camera], target_width: int = 800):
        """
        Args:
            images_folder: folder containing the actual photo files
                            (camera.image_name is looked up relative to this)
            cameras: list of posed Camera objects, at their ORIGINAL
                      (full) resolution --- this class handles downscaling
            target_width: render/training resolution --- keeps training
                            fast; height is derived to preserve aspect ratio
        """
        self.images_folder = images_folder

        # Downscale every camera's intrinsics once, up front, rather than
        # repeating this calculation on every __getitem__ call.
        self.cameras: list[Camera] = []
        for cam in cameras:
            scale_factor = target_width / cam.intrinsics.width
            scaled_intrinsics = cam.intrinsics.scaled(scale_factor)
            self.cameras.append(replace(cam, intrinsics=scaled_intrinsics))

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx: int):
        """Returns (camera, target_image) --- exactly what one training
        step needs: a pose to render from, and real pixels to compare against."""
        camera = self.cameras[idx]
        image_path = f"{self.images_folder}/{camera.image_name}"
        target = load_target_image(
            image_path,
            target_width=camera.intrinsics.width,
            target_height=camera.intrinsics.height,
        )
        return camera, target

    @classmethod
    def from_colmap(cls, sparse_txt_folder: str, images_folder: str, target_width: int = 800):
        """Convenience constructor: runs the full Day 1 parsing pipeline
        and wraps the result as a dataset, in one call."""
        cameras_dict = parse_cameras_txt(f"{sparse_txt_folder}/cameras.txt")
        images = parse_images_txt(f"{sparse_txt_folder}/images.txt", cameras_dict)
        point_cloud = parse_points3d_txt(f"{sparse_txt_folder}/points3D.txt")

        dataset = cls(images_folder, images, target_width=target_width)
        return dataset, point_cloud