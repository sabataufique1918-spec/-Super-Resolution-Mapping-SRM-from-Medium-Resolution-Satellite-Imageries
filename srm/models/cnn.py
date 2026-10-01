"""CNN generators: a bicubic control, SRCNN, EDSR and an RRDB (ESRGAN) trunk."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    RRDB,
    BaseSRModel,
    ChannelAttention,
    PixelShuffleUpsampler,
    ResidualBlock,
    bicubic_up,
    conv,
    init_weights,
)


class BicubicBaseline(BaseSRModel):
    """The control condition. Zero parameters, and it must be reported.

    An SR paper without an interpolation baseline on the same metric and the
    same test split is not making a measurable claim.
    """

    def __init__(self, scale: int = 4, in_channels: int = 4, out_channels: int = 4, **_):
        super().__init__(scale, in_channels, out_channels)
        self.register_parameter("_dummy", nn.Parameter(torch.zeros(1), requires_grad=False))

    def residual(self, x):
        return torch.zeros_like(bicubic_up(x[:, : self.out_channels], self.scale))


class SRCNN(BaseSRModel):
    """The 2014 three-layer network, as a weak learned baseline."""

    def __init__(self, scale: int = 4, in_channels: int = 4, out_channels: int = 4,
                 features: int = 64, **_):
        super().__init__(scale, in_channels, out_channels)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, features, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features // 2, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, out_channels, 5, padding=2),
        )
        init_weights(self, scale=0.1)

    def residual(self, x):
        return self.body(bicubic_up(x, self.scale))


class EDSR(BaseSRModel):
    """Enhanced Deep Residual network -- the PSNR-oriented workhorse.

    This is the recommended starting point for an SRM product: it is stable to
    train, spectrally faithful, and its errors are honest blur rather than
    invented structure. Use it when the output feeds a classifier or an index.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 4,
        out_channels: int = 4,
        features: int = 64,
        blocks: int = 16,
        res_scale: float = 0.1,
        dropout: float = 0.0,
        attention: bool = True,
        **_,
    ):
        super().__init__(scale, in_channels, out_channels)
        self.head = conv(in_channels, features)
        body: list[nn.Module] = [
            ResidualBlock(features, res_scale, dropout) for _ in range(blocks)
        ]
        if attention:
            body.append(ChannelAttention(features))
        body.append(conv(features, features))
        self.body = nn.Sequential(*body)
        self.tail = PixelShuffleUpsampler(features, scale, out_channels)
        init_weights(self, scale=0.1)

    def residual(self, x):
        head = self.head(x)
        return self.tail(head + self.body(head))


class RRDBNet(BaseSRModel):
    """ESRGAN generator trunk.

    Higher capacity than EDSR and the right generator for the adversarial
    stage, where the goal is perceptually plausible texture. Note the trade:
    everything it gains in apparent sharpness is detail the sensor never
    recorded, which is why `uncertainty.py` exists.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 4,
        out_channels: int = 4,
        features: int = 64,
        blocks: int = 12,
        growth: int = 32,
        res_scale: float = 0.2,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__(scale, in_channels, out_channels)
        self.head = conv(in_channels, features)
        self.body = nn.Sequential(*[RRDB(features, growth, res_scale) for _ in range(blocks)])
        self.body_fuse = conv(features, features)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.up = PixelShuffleUpsampler(features, scale)
        self.tail = nn.Sequential(
            conv(features, features),
            nn.LeakyReLU(0.2, inplace=True),
            conv(features, out_channels),
        )
        init_weights(self, scale=0.1)

    def residual(self, x):
        head = self.head(x)
        body = self.body_fuse(self.drop(self.body(head)))
        return self.tail(self.up(head + body))


class UNetDiscriminator(nn.Module):
    """Real-ESRGAN style U-Net discriminator with spectral normalisation.

    A per-pixel realism map rather than a single scalar. On satellite imagery
    this matters: a scene is a mosaic of regimes, and one global real/fake
    score lets the generator trade hallucinated urban texture against
    over-smoothed farmland and still satisfy the critic.
    """

    def __init__(self, in_channels: int = 4, features: int = 64, spectral: bool = True):
        super().__init__()
        norm = nn.utils.spectral_norm if spectral else (lambda m: m)
        self.conv0 = nn.Conv2d(in_channels, features, 3, 1, 1)
        self.down1 = norm(nn.Conv2d(features, features * 2, 4, 2, 1, bias=False))
        self.down2 = norm(nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False))
        self.down3 = norm(nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False))
        self.up1 = norm(nn.Conv2d(features * 8, features * 4, 3, 1, 1, bias=False))
        self.up2 = norm(nn.Conv2d(features * 4, features * 2, 3, 1, 1, bias=False))
        self.up3 = norm(nn.Conv2d(features * 2, features, 3, 1, 1, bias=False))
        self.out = nn.Sequential(
            norm(nn.Conv2d(features, features, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, 1, 3, 1, 1),
        )

    def forward(self, x):
        act = lambda t: F.leaky_relu(t, 0.2, inplace=True)  # noqa: E731
        x0 = act(self.conv0(x))
        x1 = act(self.down1(x0))
        x2 = act(self.down2(x1))
        x3 = act(self.down3(x2))
        u1 = act(self.up1(F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False))) + x2
        u2 = act(self.up2(F.interpolate(u1, size=x1.shape[-2:], mode="bilinear", align_corners=False))) + x1
        u3 = act(self.up3(F.interpolate(u2, size=x0.shape[-2:], mode="bilinear", align_corners=False))) + x0
        return self.out(u3)
