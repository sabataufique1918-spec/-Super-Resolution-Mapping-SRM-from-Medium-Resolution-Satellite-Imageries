"""Composite objective assembled from the config."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.degradation import DegradationConfig
from .gan import GANLoss, toggle_grad
from .masked import (
    MaskedCharbonnierLoss,
    MaskedL1Loss,
    MaskedL2Loss,
    build_pixel_loss,
    expand_mask,
    masked_mean,
)
from .perceptual import VGGPerceptualLoss
from .spectral import (
    CharbonnierLoss,
    ConsistencyLoss,
    GradientLoss,
    IndexConsistencyLoss,
    SpectralAngleLoss,
)

__all__ = [
    "SRMLoss",
    "GANLoss",
    "MaskedL1Loss",
    "MaskedL2Loss",
    "MaskedCharbonnierLoss",
    "build_pixel_loss",
    "expand_mask",
    "masked_mean",
    "VGGPerceptualLoss",
    "CharbonnierLoss",
    "ConsistencyLoss",
    "SpectralAngleLoss",
    "IndexConsistencyLoss",
    "GradientLoss",
    "toggle_grad",
]


class SRMLoss(nn.Module):
    """Weighted sum of the enabled terms; returns the total and a breakdown.

    Every term is opt-in through its weight, so one config file can express
    the whole ladder from a pure-L1 PSNR model to a full perceptual GAN
    without touching code. The breakdown is logged per term, which is the only
    practical way to see a GAN stage going wrong before it ruins the weights.
    """

    def __init__(self, cfg: dict, degradation: DegradationConfig | None = None,
                 channels: int = 4, logger=None):
        super().__init__()
        loss_cfg = cfg.get("loss", {})
        self.weights: dict[str, float] = {}

        pixel_cfg = loss_cfg.get("pixel", {})
        # Masked, and L1 by default: the specification's starting objective.
        # `CharbonnierLoss` stays importable for the older configs, but nothing
        # constructs the *unmasked* variant any more -- an unmasked pixel term
        # on real reference imagery learns the nodata wedges as dark ground.
        self.pixel = build_pixel_loss(pixel_cfg.get("type", "l1"))
        self.weights["pixel"] = float(pixel_cfg.get("weight", 1.0))

        self.weights["sam"] = float(loss_cfg.get("spectral_angle", {}).get("weight", 0.0))
        self.sam = SpectralAngleLoss() if self.weights["sam"] > 0 else None

        self.weights["consistency"] = float(loss_cfg.get("consistency", {}).get("weight", 0.0))
        self.consistency = None
        if self.weights["consistency"] > 0:
            degradation = degradation or DegradationConfig(
                scale=cfg.get("scale", 4), bands=tuple(cfg.get("bands", ()))
            )
            self.consistency = ConsistencyLoss(degradation, channels=channels)

        self.weights["index"] = float(loss_cfg.get("index", {}).get("weight", 0.0))
        self.index = IndexConsistencyLoss() if self.weights["index"] > 0 else None

        self.weights["gradient"] = float(loss_cfg.get("gradient", {}).get("weight", 0.0))
        self.gradient = GradientLoss() if self.weights["gradient"] > 0 else None

        perceptual_cfg = loss_cfg.get("perceptual", {})
        self.weights["perceptual"] = float(perceptual_cfg.get("weight", 0.0))
        self.perceptual = None
        if self.weights["perceptual"] > 0:
            self.perceptual = VGGPerceptualLoss(
                layers=tuple(perceptual_cfg.get("layers", ("relu3_3", "relu4_3")))
            )
            if not self.perceptual.available:
                if logger:
                    logger.warning(
                        "perceptual loss disabled -- VGG19 weights unavailable (%s). "
                        "Training continues without it.", self.perceptual.reason,
                    )
                self.weights["perceptual"] = 0.0

        self.weights["gan"] = float(loss_cfg.get("gan", {}).get("weight", 0.0))
        self.gan = GANLoss(loss_cfg.get("gan", {}).get("type", "ragan"))

    @property
    def uses_gan(self) -> bool:
        return self.weights["gan"] > 0

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        lr: torch.Tensor | None = None,
        fake_pred: torch.Tensor | None = None,
        real_pred: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """`mask` is (B, H, W) or (B, 1, H, W), 1 where the reference is observed.

        Every term that compares SR against HR takes the mask and averages
        over observed pixels. The consistency term does not -- it compares
        against the *input*, which has its own validity -- and the GAN and
        perceptual terms operate on whole patches by construction, so they
        get the imagery unmodified.

        Earlier this zeroed the invalid pixels in both tensors and passed the
        result on. That was wrong twice over: it drove the normalised-index
        denominator to zero (returning `inf` for an entire run), and it
        manufactured a step edge around every nodata region for the gradient
        term to chase. Masking belongs inside each term, applied to its own
        per-pixel error.
        """
        parts: dict[str, torch.Tensor] = {"pixel": self.pixel(sr, hr, mask)}
        if self.sam is not None:
            parts["sam"] = self.sam(sr, hr, mask)
        if self.consistency is not None and lr is not None:
            parts["consistency"] = self.consistency(sr, lr)
        if self.index is not None:
            parts["index"] = self.index(sr, hr, mask)
        if self.gradient is not None:
            parts["gradient"] = self.gradient(sr, hr, mask)
        if self.perceptual is not None and self.weights["perceptual"] > 0:
            # VGG features are not separable per pixel, so this one genuinely
            # cannot honour a mask; the invalid area is zeroed in both so it
            # at least compares like with like.
            keep = expand_mask(mask, sr) if mask is not None else None
            parts["perceptual"] = self.perceptual(
                sr * keep if keep is not None else sr,
                hr * keep if keep is not None else hr,
            )
        if self.uses_gan and fake_pred is not None:
            parts["gan"] = self.gan.generator_loss(real_pred, fake_pred)

        total = sr.new_zeros(())
        for name, value in parts.items():
            total = total + self.weights.get(name, 0.0) * value
        breakdown = {name: float(value.detach()) for name, value in parts.items()}
        breakdown["total"] = float(total.detach())
        return total, breakdown
