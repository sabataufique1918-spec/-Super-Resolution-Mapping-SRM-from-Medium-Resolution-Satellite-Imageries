"""Shared building blocks.

Two design choices run through all of them and both are deliberate:

* **No batch normalisation.** BN normalises away the absolute radiometry, and
  absolute reflectance is the physical quantity we are obliged to preserve.
  It also produces the well-known SR artefacts on smooth regions.
* **Residual in the reflectance domain.** Every network predicts a *residual*
  on top of a bicubic upsample of its input. The low-frequency content is then
  carried by an operator that is spectrally exact by construction, and the
  network only has to supply detail.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_ch: int, out_ch: int, kernel: int = 3, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=bias)


class ResidualBlock(nn.Module):
    """EDSR-style block: conv-ReLU-conv, scaled residual, no normalisation."""

    def __init__(self, features: int, res_scale: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = conv(features, features)
        self.conv2 = conv(features, features)
        self.res_scale = res_scale
        # Dropout is kept for MC-dropout uncertainty at inference, not for
        # regularisation -- hence the small default and 2d (channel) variant.
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv2(self.drop(F.relu(self.conv1(x), inplace=True)))
        return x + out * self.res_scale


class ChannelAttention(nn.Module):
    """Squeeze-excite over channels; cheap and it helps the NIR band behave."""

    def __init__(self, features: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, features // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(features, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))


class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int):
        super().__init__()
        self.conv = conv(in_ch, growth)

    def forward(self, x):
        return torch.cat([x, F.leaky_relu(self.conv(x), 0.2, inplace=True)], dim=1)


class ResidualDenseBlock(nn.Module):
    """RDB from ESRGAN: five dense layers with a scaled residual."""

    def __init__(self, features: int = 64, growth: int = 32, res_scale: float = 0.2):
        super().__init__()
        layers = []
        channels = features
        for _ in range(4):
            layers.append(DenseLayer(channels, growth))
            channels += growth
        self.dense = nn.Sequential(*layers)
        self.fuse = conv(channels, features)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.fuse(self.dense(x)) * self.res_scale


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, features: int = 64, growth: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualDenseBlock(features, growth, res_scale) for _ in range(3)])
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.blocks(x) * self.res_scale


class PixelShuffleUpsampler(nn.Sequential):
    """Sub-pixel upsampling by powers of two (and a direct x3 path)."""

    def __init__(self, features: int, scale: int, out_channels: int | None = None):
        layers: list[nn.Module] = []
        if scale & (scale - 1) == 0:  # 2, 4, 8, ...
            for _ in range(int(math.log2(scale))):
                layers += [conv(features, features * 4), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [conv(features, features * 9), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"unsupported scale {scale}; use 2, 3, 4 or 8")
        if out_channels is not None:
            layers.append(conv(features, out_channels))
        super().__init__(*layers)


def init_weights(module: nn.Module, scale: float = 0.1) -> None:
    """Kaiming init with a small gain.

    Residual branches initialised near zero make the network start out as
    (almost exactly) the bicubic baseline, so training never has to climb out
    of a worse-than-interpolation hole. It also keeps the GAN stage stable.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def bicubic_up(x: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)


class BaseSRModel(nn.Module):
    """Common contract: `forward(lr) -> sr` with a bicubic skip and a clamp."""

    def __init__(self, scale: int, in_channels: int, out_channels: int):
        super().__init__()
        self.scale = int(scale)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def residual(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = bicubic_up(x[:, : self.out_channels], self.scale)
        out = base + self.residual(x)
        # Reflectance cannot be negative; the upper bound leaves headroom for
        # bright roofs and specular water rather than clipping at 1.0.
        return out.clamp(0.0, 1.5)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_mc_dropout(self) -> int:
        """Put only the dropout layers in train mode, for MC-dropout sampling."""
        count = 0
        for m in self.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                m.train()
                count += 1
        return count
