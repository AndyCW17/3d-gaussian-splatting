"""
The training loop: repeatedly picks a random (camera, target photo) pair,
renders the current Gaussians from that camera, compares against the real
photo, and updates every Gaussian parameter to reduce the difference.
"""


import random
import torch
from splatproj.model.gaussians import GaussianModel
from splatproj.model.renderer import render
from splatproj.training.loss import training_loss
from splatproj.training.density_control import DensificationState


def train(
    model: GaussianModel,
    dataset,
    optimizer: torch.optim.Optimizer,
    num_iterations: int,
    device: str = "cuda",
    log_every: int = 50,
    on_iteration=None,
    density_state: DensificationState = None,
    densify_fn=None,
    densify_every: int = 100,
    densify_start: int = 500,
    densify_end: int = None,
    opacity_reset_fn=None,
    opacity_reset_every: int = 400,
    opacity_reset_start: int = 500,
) -> list[float]:
    """
    Runs the training loop.

    Returns:
        list of the loss values, one per iteration -- useful for plotting a
        loss curve or, in test, checking that loss actually went down.
    """
    loss_history = []
    if densify_end is None:
        densify_end = num_iterations

    for iteration in range(num_iterations):
        idx = random.randrange(len(dataset))
        camera, target = dataset[idx]
        target = target.to(device)

        rendered, meta = render(model, camera, device=device)
        loss = training_loss(rendered, target)

        optimizer.zero_grad()
        loss.backward()
        # Grads are populate but not yet consumed by optimizer .step() -- 
        # the only point where means2d.grad is both valid and still 
        # tied to this iteration's visibility (radii).
        if density_state is not None:
            density_state.update(meta["means2d"], meta["radii"])

        optimizer.step()

        if (
            densify_fn is not None
            and densify_start <= iteration < densify_end
            and iteration % densify_every == 0
            and iteration > 0
        ):
            densify_fn(model, optimizer, density_state)
            density_state.reset()

        if (
                    opacity_reset_fn is not None
                    and opacity_reset_start <= iteration 
                    and iteration % opacity_reset_every == 0
                    and iteration > 0
                ):
                    opacity_reset_fn(model, optimizer)
        
        
        loss_history.append(loss.item())
        if on_iteration is not None:
            on_iteration(iteration, loss.item(), model)

        if iteration % log_every == 0:
            print(f"iter {iteration:5d} | loss {loss.item():.4f}  |" 
                  f"n_gaussians {len(model):6d} | camera {camera.image_name}")

    return loss_history