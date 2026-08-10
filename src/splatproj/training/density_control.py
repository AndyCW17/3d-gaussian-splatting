import torch
from dataclasses import dataclass


@dataclass
class DensificationConfig:
    grad_threshold: float = 0.00002     # avg screen-space grad above this = under-reconstructed
    scale_threshold: float = 0.04      # world-space size above this = "large" -> split, below = "small" -> clone
    opacity_prune_threshold: float = 0.005  # opacity below this = prune
    max_screen_size: float = 20.0      # projected radius above this = pathologically large -> prune


class DensificationState:
    """Accumulates per-Gaussian screen-space gradient norms between
    densification passes. A single iteration's gradient is noisy, so we
    average over a window of iterations before deciding what to grow.
    """

    def __init__(self, n_gaussians: int, device: str = "cuda"):
        self.grad_accum = torch.zeros(n_gaussians, device=device)
        self.denom = torch.zeros(n_gaussians, device=device)
        self.max_radii = torch.zeros(n_gaussians, device=device)

    def update(self, means2d: torch.Tensor, radii: torch.Tensor) -> None:
        """
        Call once per training iteration, right after loss.backward()
        and before optimizer.step() (grads are populated but not yet
        consumed).

        means2d: meta["means2d"] from render() -- shape [1, N, 2], grad
                 populated because renderer.py already calls retain_grad().
        radii:   meta["radii"] from render() -- shape [1, N, 2]. Zero
                 radius means gsplat culled that Gaussian this view --
                 its grad is meaningless and must be excluded.
        """
        grad = means2d.grad
        if grad is None:
            return
        # print(f"means2d.grad shape: {means2d.grad.shape}, radii shape: {radii.shape}")
        grad = grad[0]
        radii = radii[0]

        visible = (radii > 0).any(dim=-1) # use the visible splats only
        grad_norm = grad[visible].norm(dim=-1)

        self.grad_accum[visible] += grad_norm
        self.denom[visible] += 1

        radii_flat = radii[visible].float().max(dim=-1).values
        self.max_radii[visible] = torch.maximum(self.max_radii[visible], radii_flat)

    def get_average_grad(self) -> torch.Tensor:
        denom_safe = self.denom.clamp(min=1)
        return self.grad_accum / denom_safe

    def reset(self) -> None:
        self.grad_accum.zero_()
        self.denom.zero_()
        self.max_radii.zero_()
            

def select_densify_candidates(model, avg_grad, config: DensificationConfig):
    """Returns (clone_mask, split_mask) -- boolean tensors, shape [N].
    A Gaussian only qualifies if its accumulated screen-space gradient
    is high enough (it's under-reconstructing something). Among those,
    small ones get cloned (need more coverage), large ones get split
    (already covering too much space poorly).
    """

    high_grad = avg_grad > config.grad_threshold

    # max scale across the 3 axes -- if any axis is "large", treat the
    # whole Gaussian as large for this decision
    max_scale = model.scales.max(dim=-1).values

    clone_mask = high_grad & (max_scale <= config.scale_threshold)
    split_mask = high_grad & (max_scale > config.scale_threshold)

    return clone_mask, split_mask


def clone_gaussians(model, clone_mask):
    """Returns a dict of new parameter tensors for the cloned Gaussians
    only (not concatenated with the originals yet -- that happens later,
    once all clone+split+prune ops for this pass are computed)."""
    return {
        "means": model.means[clone_mask].clone(),
        "raw_scales": model.raw_scales[clone_mask].clone(),
        "quats": model.quats[clone_mask].clone(),
        "raw_opacities": model.raw_opacities[clone_mask].clone(),
        "colors": model.colors[clone_mask].clone(),
    }


def split_gaussians(model, split_mask, split_scale_factor=1.6):
    """Each selected Gaussian becomes 2 new ones: same rotation/opacity/
    color, but repositioned by sampling from its own covariance (so the
    2 new points land within where the original Gaussian actually was),
    and shrunk so together they don't just recreate the same blob."""
    n_split = split_mask.sum().item()
    if n_split == 0:
        return {k: torch.empty(0, *v.shape[1:], device=v.device)
                for k, v in [("means", model.means), ("raw_scales", model.raw_scales),
                             ("quats", model.quats), ("raw_opacities", model.raw_opacities),
                             ("colors", model.colors)]}

    orig_means = model.means[split_mask]        # [n_split, 3]
    orig_scales = model.scales[split_mask]       # [n_split, 3], already exp'd (real scale)
    orig_quats = model.quats[split_mask]
    orig_opacities = model.raw_opacities[split_mask]
    orig_colors = model.colors[split_mask]

    # Sample 2 offsets per split Gaussian from a standard normal, scaled
    # by that Gaussian's own extent -- this is why large Gaussians in
    # particular need splitting: their scale directly controls how far
    # apart the 2 children land.
    samples = torch.randn(2, n_split, 3, device=orig_means.device) * orig_scales.unsqueeze(0)
    new_means = (orig_means.unsqueeze(0) + samples).reshape(2 * n_split, 3)

    new_scale = orig_scales / split_scale_factor
    new_raw_scales = torch.log(new_scale).repeat(2, 1)

    new_quats = orig_quats.repeat(2, 1)
    new_raw_opacities = orig_opacities.repeat(2)
    new_colors = orig_colors.repeat(2, 1)

    return {
        "means": new_means, "raw_scales": new_raw_scales,
        "quats": new_quats, "raw_opacities": new_raw_opacities,
        "colors": new_colors,
    }


def select_prune_mask(model, meta_max_radii, config: DensificationConfig):
    """Gaussians to remove: opacity too low to matter, or screen-space
    size pathologically large (usually a sign of a degenerate Gaussian
    that ballooned rather than converged)."""
    low_opacity = model.opacities < config.opacity_prune_threshold
    too_large = meta_max_radii > config.max_screen_size
    prune_mask = low_opacity | too_large

    print(f"  prune breakdown: {low_opacity.sum().item()} low-opacity, "
        f"{too_large.sum().item()} too-large "
        f"(overlap: {(low_opacity & too_large).sum().item()})")
    return prune_mask


ATTR_NAMES = ["means", "raw_scales", "quats", "raw_opacities", "colors"]


def build_new_attributes(model, keep_mask, clone_dict, split_dict):
    """Concatenates: surviving originals + clones + split children,
    for all 5 attributes at once, keeping them in the same order so
    row i means the same Gaussian across every attribute tensor."""
    current = {
        "means": model.means, "raw_scales": model.raw_scales,
        "quats": model.quats, "raw_opacities": model.raw_opacities,
        "colors": model.colors,
    }
    new_attrs = {}
    for name in ATTR_NAMES:
        new_attrs[name] = torch.cat([
            current[name][keep_mask].detach(),
            clone_dict[name],
            split_dict[name],
        ], dim=0)
    return new_attrs


def densify_and_prune(model, optimizer, density_state, config: DensificationConfig):
    avg_grad = density_state.get_average_grad()

    # Debug: only look at Gaussians that were actually visible/updated this
    # window (denom > 0) -- otherwise we're diluting stats with Gaussians
    # that never got a chance to accumulate anything.
    seen = density_state.denom > 0
    if seen.any():
        seen_grads = avg_grad[seen]
        print(f"avg_grad (seen={seen.sum().item()}): "
            f"min={seen_grads.min().item():.8f} "
            f"mean={seen_grads.mean().item():.8f} "
            f"max={seen_grads.max().item():.8f}")

        seen_radii = density_state.max_radii[seen]
        print(f"max_radii (seen={seen.sum().item()}): "
            f"min={seen_radii.min().item():.4f} "
            f"mean={seen_radii.mean().item():.4f} "
            f"max={seen_radii.max().item():.4f}")

    clone_mask, split_mask = select_densify_candidates(model, avg_grad, config)
    prune_mask = select_prune_mask(model, density_state.max_radii, config)

    clone_dict = clone_gaussians(model, clone_mask)
    split_dict = split_gaussians(model, split_mask)

    # Gaussians that got split are replaced by their 2 children -- remove
    # the parent. Anything flagged for pruning is removed regardless.
    # Cloned Gaussians are NOT removed -- the original stays, the clone
    # is a new, separate row added on top.
    keep_mask = (~split_mask) & (~prune_mask)

    new_attrs = build_new_attributes(model, keep_mask, clone_dict, split_dict)

    n_before = len(model)
    n_after = new_attrs["means"].shape[0]
    print(f"densify: {n_before} -> {n_after} Gaussians "
          f"(+{clone_mask.sum().item()} cloned, "
          f"+{2*split_mask.sum().item()} from split, "
          f"-{(~keep_mask).sum().item()} removed)")

    n_new = clone_dict["means"].shape[0] + split_dict["means"].shape[0]

    for name in ATTR_NAMES:
        old_param = getattr(model, name)
        new_param = torch.nn.Parameter(new_attrs[name])

        _transplant_optimizer_state(optimizer, old_param, new_param, keep_mask, n_new)
        _replace_param_in_group(optimizer, old_param, new_param)

        setattr(model, name, new_param)

    # DensificationState's buffers are sized for the old Gaussian count --
    # resize for the new count. train.py calls density_state.reset() right
    # after this returns, so we don't need to preserve any values here,
    # just get the shapes right before the next update() call.
    density_state.grad_accum = torch.zeros(n_after, device=density_state.grad_accum.device)
    density_state.denom = torch.zeros(n_after, device=density_state.denom.device)
    density_state.max_radii = torch.zeros(n_after, device=density_state.max_radii.device)


def _transplant_optimizer_state(optimizer, old_param, new_param, keep_mask, n_new):
    """Moves Adam's momentum (exp_avg, exp_avg_sq) for surviving rows from
    old_param's optimizer state onto new_param's state. Newly added rows
    (clones/split children) get zero-initialized state -- they have no
    training history to draw momentum from."""
    old_state = optimizer.state.pop(old_param, None)

    if old_state is not None and "exp_avg" in old_state:
        kept_exp_avg = old_state["exp_avg"][keep_mask]
        kept_exp_avg_sq = old_state["exp_avg_sq"][keep_mask]

        new_rows_shape = (n_new,) + kept_exp_avg.shape[1:]
        zeros = lambda t: torch.zeros(new_rows_shape, device=t.device, dtype=t.dtype)

        optimizer.state[new_param] = {
            "step": old_state["step"],
            "exp_avg": torch.cat([kept_exp_avg, zeros(kept_exp_avg)], dim=0),
            "exp_avg_sq": torch.cat([kept_exp_avg_sq, zeros(kept_exp_avg_sq)], dim=0),
        }
    # else: no prior state yet (e.g. densify fires before any optimizer.step()
    # has run) -- leave empty, Adam lazily initializes it on the next step().

def _replace_param_in_group(optimizer, old_param, new_param):
    """Without this, optimizer.param_groups still points at the old
    (now-orphaned) tensor, and the new one silently never gets optimized."""
    for group in optimizer.param_groups:
        for idx, p in enumerate(group["params"]):
            if p is old_param:
                group["params"][idx] = new_param
                return
    raise ValueError("old_param not found in any optimizer param group")


def reset_opacities(model, optimizer, reset_value: float = 0.01):
    """Pushes every Gaussian's opacity down to reset_value. Doesn't change
    Gaussian count, so unlike densify_and_prune this doesn't need the
    param_groups surgery -- raw_opacities stays the same tensor object,
    just with new values written into it in place.
    """
    reset_raw = torch.logit(torch.tensor(reset_value, device=model.raw_opacities.device))
    with torch.no_grad():
        model.raw_opacities.fill_(reset_raw.item())