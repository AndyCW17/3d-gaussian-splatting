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


def train(
    model: GaussianModel,
    dataset,
    optimizer: torch.optim.Optimizer,
    num_iterations: int,
    device: str = "cuda",
    log_every: int = 50,
) -> list[float]:
    """
    Runs the training loop.

    Returns:
        list of the loss values, one per iteration -- useful for plotting a
        loss curve or, in test, checking that loss actually went down.
    """
    loss_history = []

    for iteration in range(num_iterations):
        idx = random.randrange(len(dataset))
        camera, target = dataset[idx]
        target = target.to(device)

        rendered = render(model, camera, device=device)
        loss = training_loss(rendered, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if iteration % log_every == 0:
            print(f"iter {iteration:5d} | loss {loss.item():.4f} | "
                  f"camera {camera.image_name}")

    return loss_history