"""
The actual reconstruction pipeline: COLMAP output + photos in, a viewer-
ready .ply file out. Both scripts/train_real_scene.py (CLI) and the
Gradio app call this -- neither should duplicate the training setup
itself, only decide how to invoke it and what to do with progress
updates.
"""

import os
import functools
from splatproj.data.scene_dataset import SceneDataset
from splatproj.model.gaussians import GaussianModel
from splatproj.training.optimizer import build_optimizer
from splatproj.training.train import train
from splatproj.training.checkpoint import save_checkpoint
from splatproj.training.density_control import (
    DensificationState,
    DensificationConfig,
    densify_and_prune,
    reset_opacities,
)
from splatproj.export.ply_export import export_ply


def run_pipeline(
    sparse_folder: str,
    images_folder: str,
    output_dir: str,
    num_iterations: int = 3000,
    target_width: int = 400,
    device: str = "cuda",
    on_iteration=None,
) -> str:
    """
    Runs dataset loading -> training (with densification + opacity
    reset) -> checkpoint + PLY export.

    on_iteration, if given, is forwarded straight to train() -- same
    injected-callback pattern as train.py itself, so this function
    doesn't need to know or care whether the caller is a plain CLI
    script (printing progress) or a Gradio app (updating a progress bar).

    Returns:
        path to the exported .ply file.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset, point_cloud = SceneDataset.from_colmap(sparse_folder, images_folder, target_width=target_width)

    model = GaussianModel(point_cloud, initial_scale=0.02).to(device)
    optimizer = build_optimizer(model)

    density_state = DensificationState(len(model), device=device)
    densify_config = DensificationConfig(scale_threshold=0.04)
    densify_fn = functools.partial(densify_and_prune, config=densify_config)

    loss_history = train(
        model, dataset, optimizer, num_iterations,
        device=device, log_every=100, on_iteration=on_iteration,
        density_state=density_state, densify_fn=densify_fn,
        densify_start=500, densify_end=int(num_iterations * 0.83), densify_every=100,
        opacity_reset_fn=reset_opacities,
        opacity_reset_start=500, opacity_reset_every=400,
    )

    save_checkpoint(model, f"{output_dir}/final_checkpoint.pt")

    ply_path = f"{output_dir}/scene.ply"
    export_ply(model, ply_path)

    return ply_path