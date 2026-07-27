"""
Trains a Gaussian scene on real COLMAP data, saving periodic preview
renders and a final checkpoint.

Run inside Docker (needs GPU):
    PYTHONPATH=src python3 scripts/train_real_scene.py \
        data/raw/mug/sparse_txt data/raw/mug outputs/mug_training
"""


import os
import sys
from splatproj.data.scene_dataset import SceneDataset
from splatproj.data.image_io import save_render
from splatproj.model.gaussians import GaussianModel
from splatproj.model.renderer import render
from splatproj.training.optimizer import build_optimizer
from splatproj.training.train import train
from splatproj.training.checkpoint import save_checkpoint


def main():
    sparse_folder, images_folder, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/snapshots", exist_ok=True)

    dataset, point_cloud = SceneDataset.from_colmap(sparse_folder, images_folder, target_width=400)
    print(f"Loaded {len(dataset)} cameras, {point_cloud.points.shape[0]} points")

    model = GaussianModel(point_cloud, initial_scale=0.02).to("cuda")
    optimizer = build_optimizer(model)

    # Fixed camera used for every progress snapshot, so they're directly
    # comparable to each other over time --- watching the SAME view sharpen.
    snapshot_camera, _  = dataset[0]

    def on_iteration(iteration, loss_value, model):
        if iteration % 200 == 0:
            with_no_grad_image, _ = render(model, snapshot_camera, device="cuda")
            save_render(with_no_grad_image, f"{output_dir}/snapshots/iter_{iteration:05d}.png")

    num_iterations = 3000
    loss_history = train(
        model, dataset, optimizer, num_iterations,
        device="cuda", log_every=100, on_iteration=on_iteration,
    )

    print(f"scale range: {model.scales.min().item():.4f} to {model.scales.max().item():.4f}")
    print(f"opacity range: {model.opacities.min().item():.4f} to {model.opacities.max().item():.4f}")

    save_checkpoint(model, f"{output_dir}/final_checkpoint.pt")
    print(f"Done. Final Loss:{loss_history[-1]:.4f}")
    print(f"Checkpoint saved to {output_dir}/final_checkpoint.pt")
    print(f"Snapshots saved to {output_dir}/snapshots/")


if __name__ == "__main__":
    main()