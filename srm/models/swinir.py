"""A compact SwinIR: shifted-window transformer super-resolution.

Why a transformer here at all. Satellite scenes are unusually repetitive --
the same roof, the same furrow spacing, the same field boundary recurs across
a tile. Window attention can copy evidence from those repetitions, where a
convolution can only use its own receptive field. In practice this shows up as
better reconstruction of periodic structure (crop rows, building arrays) than
EDSR at comparable parameter count.

The shape here is the real SwinIR (RSTB of shifted-window blocks with a
convolutional residual), scaled down to train on a 4 GB GPU. Window masks are
built per forward pass so the model accepts any input size, which whole-scene
tiled inference needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BaseSRModel, PixelShuffleUpsampler, conv, init_weights


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, window*window, C)"""
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window * window, c)


def window_reverse(windows: torch.Tensor, window: int, h: int, w: int) -> torch.Tensor:
    """(B*nW, window*window, C) -> (B, H, W, C)"""
    b = windows.shape[0] // ((h // window) * (w // window))
    x = windows.view(b, h // window, w // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention inside a window, with relative position bias."""

    def __init__(self, dim: int, window: int, heads: int, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window = window
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # Learned bias per relative offset: the transformer equivalent of
        # telling the model that nearby pixels matter more than distant ones.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window), torch.arange(window), indexing="ij"))
        flat = torch.flatten(coords, 1)
        relative = flat[:, :, None] - flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window - 1
        relative[:, :, 1] += window - 1
        relative[:, :, 0] *= 2 * window - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, n, n)

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(out))


class SwinBlock(nn.Module):
    """LayerNorm -> (shifted) window attention -> LayerNorm -> MLP."""

    def __init__(self, dim: int, heads: int, window: int = 8, shift: int = 0,
                 mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window, heads, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def _attn_mask(self, h: int, w: int, device, dtype) -> torch.Tensor | None:
        """Block attention across the wrap-around seam created by the roll."""
        if self.shift == 0:
            return None
        img_mask = torch.zeros((1, h, w, 1), device=device)
        slices = (slice(0, -self.window), slice(-self.window, -self.shift), slice(-self.shift, None))
        count = 0
        for hs in slices:
            for ws in slices:
                img_mask[:, hs, ws, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0).to(dtype)

    def forward(self, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        h, w = hw
        b, n, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        mask = self._attn_mask(h, w, x.device, x.dtype)

        windows = window_partition(x, self.window)
        windows = self.attn(windows, mask)
        x = window_reverse(windows, self.window, h, w)

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = shortcut + x.view(b, n, c)
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):
    """Residual Swin Transformer Block: attention stack + a conv, residually added.

    The trailing convolution is what makes SwinIR work for restoration rather
    than classification: it reintroduces the locality and translation
    equivariance that pure attention gives up.
    """

    def __init__(self, dim: int, depth: int, heads: int, window: int = 8,
                 mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, window, shift=0 if i % 2 == 0 else window // 2,
                      mlp_ratio=mlp_ratio, drop=drop)
            for i in range(depth)
        ])
        self.conv = conv(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, (h, w))
        out = tokens.transpose(1, 2).view(b, c, h, w)
        return x + self.conv(out)


class SwinIRLite(BaseSRModel):
    """SwinIR at a size that fits a 4 GB GPU with 64x64 LR patches."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 4,
        out_channels: int = 4,
        features: int = 60,
        depths: tuple[int, ...] = (4, 4, 4, 4),
        heads: tuple[int, ...] = (4, 4, 4, 4),
        window: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__(scale, in_channels, out_channels)
        if features % max(heads) != 0:
            raise ValueError(f"features={features} must be divisible by heads={max(heads)}")
        self.window = window
        self.head = conv(in_channels, features)
        self.layers = nn.ModuleList([
            RSTB(features, depth, head_count, window, mlp_ratio, dropout)
            for depth, head_count in zip(depths, heads)
        ])
        self.norm = nn.LayerNorm(features)
        self.body_conv = conv(features, features)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.tail = PixelShuffleUpsampler(features, scale, out_channels)
        init_weights(self, scale=0.1)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        # Windows must tile the feature map exactly; reflect-pad and crop back.
        pad_h = (self.window - h % self.window) % self.window
        pad_w = (self.window - w % self.window) % self.window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feat = self.head(x)
        out = feat
        for layer in self.layers:
            out = layer(out)
        b, c, fh, fw = out.shape
        out = self.norm(out.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, c, fh, fw)
        out = feat + self.body_conv(self.drop(out))
        out = self.tail(out)

        if pad_h or pad_w:
            out = out[:, :, : h * self.scale, : w * self.scale]
        return out
