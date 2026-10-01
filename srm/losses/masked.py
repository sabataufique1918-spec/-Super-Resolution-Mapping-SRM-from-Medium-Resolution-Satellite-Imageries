"""Masked pixel losses.

Real reference imagery has holes. NEON flight lines leave large empty wedges
at tile corners, and those pixels are stored as the nodata sentinel -- zero,
in the SEN2NEON HR product. An unmasked L1 therefore hands the network a
strong, perfectly consistent training signal that says "the top-left corner of
a scene is black", and it learns exactly that: darkened borders in every
output, in places where nothing was ever observed.

So every pixel term takes a validity mask and averages over observed pixels
only. Two details matter more than they look:

* **Normalise by the number of valid pixels, not by the tensor size.** Dividing
  a masked sum by `numel` silently scales the loss down in proportion to how
  much of the patch is missing, which quietly reduces the learning rate on the
  tiles that are hardest.
* **Handle the all-invalid patch.** It should contribute exactly zero and a
  zero gradient, not a NaN that propagates into every weight on the next
  backward pass and ends the run.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def expand_mask(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
    """Broadcast a (B, H, W) or (B, 1, H, W) mask to `like`'s shape as float."""
    if mask is None:
        return None
    if mask.dim() == like.dim() - 1:
        mask = mask.unsqueeze(1)
    return mask.to(dtype=like.dtype, device=like.device).expand_as(like)


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of `values` over valid pixels; 0 when nothing is valid."""
    if mask is None:
        return values.mean()
    mask = expand_mask(mask, values)
    denom = mask.sum()
    if denom.item() == 0:
        # `values * mask` keeps the graph connected, so the optimiser sees a
        # real zero gradient rather than a detached constant.
        return (values * mask).sum()
    return (values * mask).sum() / denom


class MaskedL1Loss(nn.Module):
    """Mean absolute error over valid pixels.

    The specification's starting objective, and a good one: L1 is what SR
    models are normally tuned against, its gradient does not grow with the
    error, and on reflectance it is directly interpretable -- a loss of 0.01
    is a mean error of 0.01 reflectance units.
    """

    def forward(self, pred, target, mask=None):
        return masked_mean((pred - target).abs(), mask)


class MaskedL2Loss(nn.Module):
    """Mean squared error over valid pixels, for when PSNR is the target."""

    def forward(self, pred, target, mask=None):
        return masked_mean((pred - target) ** 2, mask)


class MaskedCharbonnierLoss(nn.Module):
    """Smooth L1 variant, ``sqrt((x-y)^2 + eps^2)``.

    Differentiable at zero, which L1 is not. In practice it converges slightly
    faster than L1 and is marginally more forgiving of the outliers that
    cross-sensor pairs are full of.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = float(eps) ** 2

    def forward(self, pred, target, mask=None):
        return masked_mean(torch.sqrt((pred - target) ** 2 + self.eps2), mask)


PIXEL_LOSSES = {
    "l1": MaskedL1Loss,
    "mae": MaskedL1Loss,
    "l2": MaskedL2Loss,
    "mse": MaskedL2Loss,
    "charbonnier": MaskedCharbonnierLoss,
}


def build_pixel_loss(name: str = "l1", **kwargs) -> nn.Module:
    key = str(name).lower()
    if key not in PIXEL_LOSSES:
        raise KeyError(f"unknown pixel loss {name!r}; available: {sorted(PIXEL_LOSSES)}")
    if key == "charbonnier":
        return MaskedCharbonnierLoss(**kwargs)
    return PIXEL_LOSSES[key]()
