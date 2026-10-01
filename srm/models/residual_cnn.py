"""The reference architecture: a configurable residual CNN, 4 bands, x4.

This implements the specified network literally, layer for layer:

    input  [B, 4, 64, 64]
      -> 3x3 conv, 4 -> 64
      -> 8 x { 3x3 conv, ReLU, 3x3 conv, + local residual }
      -> 3x3 conv, + global feature skip from the head
      -> 2 x { 3x3 conv 64 -> 256, PixelShuffle(2), ReLU }
      -> 3x3 conv, 64 -> 4
    output [B, 4, 256, 256]

It deliberately does **not** inherit `BaseSRModel`. Every other generator in
this repo predicts a residual on top of `bicubic_up(x)`, which is a good
default for reflectance products but is not the specified network: here the
final convolution produces the image itself, and the only skip is the
*feature-domain* one at step 4. Mixing the two would mean reporting results
for an architecture nobody asked for.

The bicubic skip is still reachable through `global_skip="bicubic"` for a
controlled comparison, and the two variants differ in one important way that
is easy to get wrong:

* ``global_skip="none"`` (the specification, and the default) starts from a
  near-*black* image, because the output is whatever the last convolution
  emits. It therefore needs ordinary Kaiming initialisation and a few hundred
  steps before it beats interpolation.
* ``global_skip="bicubic"`` starts from bicubic and needs a small-gain
  initialisation instead, or it starts worse than the baseline.

`init_gain` follows that automatically, and the mismatch is why the two are
not interchangeable at the same learning rate.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import bicubic_up, conv

#: Roughly the mean surface reflectance of a mixed land scene. Used only as
#: the tail bias at initialisation, so a direct-output model starts as a
#: plausible flat grey rather than on the clamp floor.
TYPICAL_REFLECTANCE = 0.12

#: Gain applied to the second convolution of each residual block at init, so
#: the block starts near the identity. Purely an initialisation choice: it
#: does not change the architecture, and training is free to move away from it.
RESIDUAL_BRANCH_INIT = 0.1


class PlainResidualBlock(nn.Module):
    """conv - ReLU - conv with a local residual addition, as specified.

    No normalisation: batch norm would divide out the absolute radiometry,
    and absolute reflectance is the physical quantity this product exists to
    preserve. `res_scale` defaults to 1.0, which is the plain formulation in
    the specification; lowering it towards 0.1 gives the EDSR-style scaled
    residual, which is steadier when `blocks` is large.
    """

    def __init__(self, features: int = 64, res_scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.conv1 = conv(features, features)
        self.conv2 = conv(features, features)
        self.res_scale = float(res_scale)
        # Present only so MC-dropout sampling has something to sample; it is
        # off by default and is not a regulariser at these rates.
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv2(self.drop(F.relu(self.conv1(x), inplace=True)))
        return x + out * self.res_scale


class UpsampleStage(nn.Sequential):
    """conv 64 -> 64*4, PixelShuffle(2), ReLU -- one doubling."""

    def __init__(self, features: int = 64):
        super().__init__(
            conv(features, features * 4),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )


class ResidualSRCNN(nn.Module):
    """Configurable residual CNN for 4-band x4 super-resolution.

    Everything the specification calls out as tunable is a constructor
    argument, and all of it is written into the checkpoint by
    `registry.save_checkpoint`, so an inference run can rebuild the exact
    network it was trained as.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 4,
        out_channels: int = 4,
        features: int = 64,
        blocks: int = 8,
        res_scale: float = 1.0,
        dropout: float = 0.0,
        global_skip: str = "none",     # "none" (specified) | "bicubic"
        clamp: tuple[float, float] | None = (0.0, 1.5),
        **_,
    ):
        super().__init__()
        if scale & (scale - 1) or scale < 2:
            raise ValueError(f"scale must be a power of two >= 2, got {scale}")
        if global_skip not in ("none", "bicubic"):
            raise ValueError(f"global_skip must be 'none' or 'bicubic', got {global_skip!r}")

        self.scale = int(scale)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.features = int(features)
        self.n_blocks = int(blocks)
        self.global_skip = global_skip
        self.clamp = clamp

        # 2. head: 3x3, 4 -> 64
        self.head = conv(in_channels, features)
        # 3. the residual trunk
        self.body = nn.Sequential(
            *[PlainResidualBlock(features, res_scale, dropout) for _ in range(blocks)]
        )
        # 4. trunk-closing convolution; the skip itself is added in forward()
        self.body_fuse = conv(features, features)
        # 5-6. one upsampling stage per doubling: x4 -> two stages
        self.upsample = nn.Sequential(
            *[UpsampleStage(features) for _ in range(int(math.log2(scale)))]
        )
        # 7. tail: 3x3, 64 -> 4
        self.tail = conv(features, out_channels)

        self.apply_init()

    # -- initialisation ---------------------------------------------------- #

    @property
    def init_gain(self) -> float:
        """Small gain only when a bicubic skip carries the low frequencies.

        With `global_skip="none"` the tail *is* the image, so damping it by
        10x would start training from a near-black output and waste the first
        epochs climbing back to zero-mean.
        """
        return 0.1 if self.global_skip == "bicubic" else 1.0

    def apply_init(self) -> None:
        gain = self.init_gain
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                m.weight.data *= gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Each block computes `x + conv(relu(conv(x)))`. At res_scale 1.0 and
        # full Kaiming gain the branch is the same magnitude as the identity,
        # so eight of them compound to roughly 2^8 and the first forward pass
        # is wildly out of range. Damping the second convolution starts every
        # block near the identity, which is what lets a plain (unscaled)
        # residual stack train at all -- the depth is then free.
        for block in self.body:
            block.conv2.weight.data *= RESIDUAL_BRANCH_INIT

        # PixelShuffle checkerboarding comes from the four sub-pixel filters
        # starting out different. Initialising each group identically ("ICNR")
        # makes the first output an exact nearest-neighbour upsample instead.
        for stage in self.upsample:
            _icnr_(stage[0].weight, upscale=2)

        if self.global_skip == "none":
            # Without a bicubic skip the tail convolution *is* the image, and
            # a Kaiming-initialised tail emits large signed values. Clamped to
            # non-negative reflectance that lands every pixel on the floor,
            # where the clamp's gradient is zero and the model is dead on
            # arrival -- it cannot learn its way back off the boundary.
            # Damping the tail and biasing it to a typical reflectance starts
            # the output as a plausible flat grey instead.
            nn.init.zeros_(self.tail.bias)
            self.tail.weight.data *= 0.1
            self.tail.bias.data.fill_(TYPICAL_REFLECTANCE)

    # -- forward ----------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body_fuse(self.body(head))
        out = self.tail(self.upsample(head + body))       # global feature skip
        if self.global_skip == "bicubic":
            out = out + bicubic_up(x[:, : self.out_channels], self.scale)
        if self.clamp is not None and not self.training:
            # Reflectance cannot be negative, and the ceiling leaves headroom
            # above 1.0 for bright roofs and specular water.
            #
            # Applied at inference only. `clamp` has exactly zero gradient
            # outside its bounds, so clamping during training lets a batch
            # that lands on the boundary detach from the optimiser entirely --
            # for a direct-output model that is not a rare edge case, it is
            # how the first forward pass behaves. The loss is what keeps
            # training in range; the clamp enforces physics on the product.
            out = out.clamp(*self.clamp)
        return out

    # -- introspection ----------------------------------------------------- #

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_mc_dropout(self) -> int:
        count = 0
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
                count += 1
        return count

    def extra_repr(self) -> str:
        return (
            f"scale={self.scale}, in={self.in_channels}, out={self.out_channels}, "
            f"features={self.features}, blocks={self.n_blocks}, skip={self.global_skip}"
        )


@torch.no_grad()
def _icnr_(weight: torch.Tensor, upscale: int = 2) -> None:
    """Initialise a sub-pixel conv so PixelShuffle starts as nearest-neighbour."""
    out_channels, in_channels, kh, kw = weight.shape
    sub = out_channels // (upscale ** 2)
    if sub * upscale ** 2 != out_channels:
        return
    kernel = torch.zeros(sub, in_channels, kh, kw, device=weight.device, dtype=weight.dtype)
    nn.init.kaiming_normal_(kernel, a=0, mode="fan_in", nonlinearity="relu")
    weight.copy_(kernel.repeat_interleave(upscale ** 2, dim=0))
