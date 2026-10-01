"""Losses that hold the physics in place.

A pixel loss alone lets a network produce a sharp image whose reflectance is
subtly wrong -- and wrong reflectance breaks every downstream product that
matters (NDVI, classification, change detection). These three terms are what
turn "a sharper picture" into "a usable data product":

* **Spectral angle** penalises a change in the *shape* of the spectrum
  independently of its brightness, which is exactly the quantity a land-cover
  classifier keys on.
* **Wald consistency** requires that degrading the super-resolved image with
  the same forward sensor model returns the observed input. This is the only
  term that is checkable on real data with no high-resolution reference, and
  it is what keeps the result anchored to the observation.
* **Index consistency** protects NDVI/NDWI directly, because a band ratio can
  drift even when both bands are individually close.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.degradation import DegradationConfig, TorchDegradation


class CharbonnierLoss(nn.Module):
    """Differentiable-everywhere L1. Robust to the outliers L2 chases."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class SpectralAngleLoss(nn.Module):
    """Mean spectral angle between predicted and reference pixel spectra (radians).

    Scale invariant by construction: doubling a spectrum leaves the angle
    unchanged. That complements the pixel loss rather than duplicating it.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask=None):
        from .masked import masked_mean

        dot = (pred * target).sum(dim=1)
        norm = pred.norm(dim=1) * target.norm(dim=1)
        cosine = (dot / (norm + self.eps)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        # Averaged over observed pixels only. An unobserved pixel has no
        # spectrum to compare, and including it contributes a fixed angle
        # that dilutes the real signal.
        return masked_mean(torch.acos(cosine), mask)


class ConsistencyLoss(nn.Module):
    """Wald's protocol as a training signal: degrade(SR) must equal LR.

    Shares its kernel with the dataset's forward model, so the network is being
    asked to be consistent with the *same* sensor assumption its pairs were
    built under.
    """

    def __init__(self, degradation: DegradationConfig, channels: int = 4,
                 criterion: str = "l1"):
        super().__init__()
        self.operator = TorchDegradation(degradation, channels=channels)
        self.criterion = CharbonnierLoss() if criterion == "charbonnier" else nn.L1Loss()

    def forward(self, sr, lr):
        return self.criterion(self.operator(sr), lr)


class IndexConsistencyLoss(nn.Module):
    """Preserve normalised band ratios between SR and reference.

    Defaults to NDVI (B08, B04) and NDWI (B03, B08) on the standard four-band
    ordering B02, B03, B04, B08.
    """

    def __init__(self, pairs: tuple[tuple[int, int], ...] = ((3, 2), (1, 3)), eps: float = 1e-4):
        super().__init__()
        self.pairs = pairs
        self.eps = eps

    @staticmethod
    def _index(x, a: int, b: int, eps: float):
        """Normalised difference, bounded in [-1, 1] for any finite input.

        The bands are clamped non-negative *before* the ratio, not the
        denominator afterwards. Both choices avoid the division by zero, but
        only this one keeps the index in the range it is defined on:

        * ``(a - b) / (a + b + eps)`` on raw values is infinite when
          ``a + b == -eps``. A model clamped only at inference -- as the
          specified architecture is, because `clamp` has no gradient --
          emits negative reflectance early in training and lands on exactly
          that. It returned `inf` for a whole run with no other symptom.
        * flooring the denominator at eps removes the infinity but leaves the
          ratio unbounded: a numerator of 0.1 over a floor of 1e-4 is 1000.
          Measured on a real batch, the term reached 14.2 and took 83% of the
          objective, which would have dominated early training.

        With ``a, b >= 0`` the numerator cannot exceed the denominator, so
        the index lies in [-1, 1] by construction and the loss in [0, 2].
        That is also what the index physically means: NDVI and NDWI are
        defined on reflectances, and are undefined where a reflectance is
        negative. Gradients do not flow through the clamped region, which is
        correct -- the pixel term is what should be pulling those pixels back
        into range.
        """
        a_val = x[:, a].clamp(min=0.0)
        b_val = x[:, b].clamp(min=0.0)
        return (a_val - b_val) / (a_val + b_val + eps)

    def forward(self, pred, target, mask=None):
        from .masked import masked_mean

        total = pred.new_zeros(())
        for a, b in self.pairs:
            if max(a, b) >= pred.shape[1]:
                continue
            difference = (self._index(pred, a, b, self.eps)
                          - self._index(target, a, b, self.eps)).abs()
            total = total + masked_mean(difference, mask)
        return total / max(1, len(self.pairs))


class GradientLoss(nn.Module):
    """Match image gradients -- a cheap, artefact-free way to ask for edges.

    Unlike an adversarial term this cannot invent structure: it only rewards
    putting an edge where the reference actually has one.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        self.register_buffer("kx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("ky", sobel_x.t().contiguous().view(1, 1, 3, 3))

    def _grad(self, x):
        c = x.shape[1]
        kx = self.kx.to(x.dtype).expand(c, 1, 3, 3)
        ky = self.ky.to(x.dtype).expand(c, 1, 3, 3)
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, kx, groups=c), F.conv2d(padded, ky, groups=c)

    def forward(self, pred, target, mask=None):
        """Gradients are taken on the *unmasked* tensors, then masked.

        Zeroing invalid pixels before differentiating would manufacture a
        step edge all the way around every nodata region, and this term would
        then spend its weight teaching the model to reproduce edges that do
        not exist on the ground. Masking after the convolution compares
        gradients only where both images were actually observed.
        """
        from .masked import expand_mask, masked_mean

        px, py = self._grad(pred)
        tx, ty = self._grad(target)
        if mask is not None:
            # Erode by one pixel: a 3x3 Sobel at the edge of the valid region
            # still reaches into the invalid one.
            keep = expand_mask(mask, px)
            keep = -F.max_pool2d(-keep, kernel_size=3, stride=1, padding=1)
            mask = keep
        return masked_mean((px - tx).abs(), mask) + masked_mean((py - ty).abs(), mask)
