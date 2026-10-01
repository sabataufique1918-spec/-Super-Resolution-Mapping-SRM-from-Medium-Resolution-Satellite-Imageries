"""VGG feature loss, adapted for multispectral reflectance.

VGG was trained on 8-bit sRGB photographs, not on 4-band surface reflectance,
so feeding it raw reflectance is meaningless. The adapter below maps
(B04, B03, B02) to a display-like RGB with a fixed reflectance stretch, then
applies ImageNet normalisation. The stretch is fixed rather than per-batch:
a data-dependent stretch would make the loss non-stationary and let the model
lower it by changing global brightness.

The NIR band is not passed to VGG at all. Texture supervision from VGG is
about structure, and structure is shared across bands; the spectral terms in
`spectral.py` are what look after B08.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# VGG19 feature indices for the commonly used layer names.
VGG19_LAYERS = {
    "relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu3_4": 17,
    "relu4_3": 24, "relu4_4": 26, "relu5_3": 33, "relu5_4": 35,
}


class ReflectanceToRGB(nn.Module):
    """(B, C>=3, H, W) reflectance -> ImageNet-normalised 3-channel tensor."""

    def __init__(self, rgb_index: tuple[int, int, int] = (2, 1, 0), stretch: float = 0.35):
        super().__init__()
        self.rgb_index = rgb_index
        self.stretch = stretch
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        idx = [i for i in self.rgb_index if i < x.shape[1]]
        while len(idx) < 3:
            idx.append(idx[-1])
        rgb = x[:, idx[:3]]
        rgb = (rgb / self.stretch).clamp(0.0, 1.0)
        return (rgb - self.mean.to(rgb.dtype)) / self.std.to(rgb.dtype)


class VGGPerceptualLoss(nn.Module):
    """Weighted L1 between VGG19 activations of SR and reference.

    `available` is False when torchvision weights cannot be fetched (offline
    machine, no cache). The loss then returns zero instead of crashing a
    training run -- the caller logs the downgrade once.
    """

    def __init__(
        self,
        layers: tuple[str, ...] = ("relu3_3", "relu4_3"),
        weights: tuple[float, ...] | None = None,
        rgb_index: tuple[int, int, int] = (2, 1, 0),
        stretch: float = 0.35,
    ):
        super().__init__()
        self.layers = tuple(layers)
        self.weights = tuple(weights or (1.0,) * len(self.layers))
        self.adapter = ReflectanceToRGB(rgb_index, stretch)
        self.available = False
        self.reason = ""
        self.features: nn.Module | None = None
        try:
            from torchvision.models import VGG19_Weights, vgg19

            backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            max_idx = max(VGG19_LAYERS[name] for name in self.layers)
            self.features = backbone[: max_idx + 1].eval()
            for param in self.features.parameters():
                param.requires_grad_(False)
            self.available = True
        except Exception as exc:  # no network / no cached weights
            self.reason = f"{type(exc).__name__}: {exc}"

    def train(self, mode: bool = True):
        super().train(mode)
        if self.features is not None:
            self.features.eval()  # frozen backbone stays in eval forever
        return self

    def forward(self, pred, target):
        if not self.available or self.features is None:
            return pred.new_zeros(())
        wanted = {VGG19_LAYERS[name]: w for name, w in zip(self.layers, self.weights)}
        x = self.adapter(pred)
        total = pred.new_zeros(())
        with torch.no_grad():
            y = self.adapter(target)
            targets = {}
            for idx, layer in enumerate(self.features):
                y = layer(y)
                if idx in wanted:
                    targets[idx] = y.detach()
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in wanted:
                total = total + wanted[idx] * F.l1_loss(x, targets[idx])
        return total
