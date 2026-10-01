"""Adversarial objectives for the perceptual training stage."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """Standard, hinge, or relativistic-average GAN loss.

    `ragan` is the ESRGAN default and the recommended one here. Asking "is the
    real image more realistic than the fake, on average" gives a gradient that
    stays informative long after a plain discriminator has saturated, which is
    what keeps texture improving instead of collapsing.
    """

    def __init__(self, mode: str = "ragan"):
        super().__init__()
        if mode not in ("standard", "hinge", "ragan"):
            raise ValueError(f"unknown gan mode '{mode}'")
        self.mode = mode

    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor):
        if self.mode == "hinge":
            return F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()
        if self.mode == "standard":
            return _bce(real_pred, True) + _bce(fake_pred, False)
        # relativistic average
        loss_real = _bce(real_pred - fake_pred.mean(), True)
        loss_fake = _bce(fake_pred - real_pred.mean(), False)
        return (loss_real + loss_fake) / 2.0

    def generator_loss(self, real_pred: torch.Tensor | None, fake_pred: torch.Tensor):
        if self.mode == "hinge":
            return -fake_pred.mean()
        if self.mode == "standard" or real_pred is None:
            return _bce(fake_pred, True)
        loss_real = _bce(real_pred - fake_pred.mean(), False)
        loss_fake = _bce(fake_pred - real_pred.mean(), True)
        return (loss_real + loss_fake) / 2.0


def _bce(logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def toggle_grad(module: nn.Module, requires_grad: bool) -> None:
    """Freeze/unfreeze a module so the two optimisers cannot fight."""
    for param in module.parameters():
        param.requires_grad_(requires_grad)
