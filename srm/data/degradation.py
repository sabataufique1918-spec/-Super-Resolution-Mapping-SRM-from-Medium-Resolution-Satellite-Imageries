"""Forward sensor model: high-resolution scene -> medium-resolution observation.

Everything about the quality of an SRM model is decided here. A network trained
on pairs made with a plain `bicubic` downsample learns to invert *bicubic*, and
then fails on real Sentinel-2 imagery, whose blur comes from the MSI optics and
detector footprint. So the default forward model is built from the published
Sentinel-2 MTF instead:

    observation = decimate( PSF * scene ) + noise

with the PSF a Gaussian fitted to the band's MTF at Nyquist, and `decimate`
an area average over the detector footprint.

Because the same operator is needed inside the loss (Wald's protocol: the
super-resolved image must reproduce its own input when degraded again), the
torch implementation is differentiable and shares its kernel with the numpy
one used for dataset construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .. import S2_MTF_NYQUIST


def mtf_to_sigma(mtf_nyquist: float, scale: int = 1) -> float:
    """Gaussian sigma (in fine-grid pixels) reproducing a given MTF at Nyquist.

    A Gaussian PSF of width sigma has MTF(f) = exp(-2 pi^2 sigma^2 f^2). At the
    Nyquist frequency of the coarse grid (f = 0.5 cycles per coarse pixel) this
    inverts to sigma = sqrt(-2 ln MTF) / pi, expressed in coarse pixels, which
    we then rescale to the fine grid.

    >>> round(mtf_to_sigma(0.30, scale=4), 3)
    1.976
    """
    mtf_nyquist = float(np.clip(mtf_nyquist, 1e-4, 0.999))
    sigma_coarse = np.sqrt(-2.0 * np.log(mtf_nyquist)) / np.pi
    return float(sigma_coarse * scale)


def band_sigmas(bands: Sequence[str], scale: int = 4, default: float = 0.30) -> list[float]:
    """Per-band PSF widths; B08 is blurrier than B02, and the model should know."""
    return [mtf_to_sigma(S2_MTF_NYQUIST.get(b, default), scale) for b in bands]


def gaussian_kernel_2d(sigma: float, size: int | None = None, sigma_y: float | None = None,
                       theta: float = 0.0) -> np.ndarray:
    """Normalised (possibly anisotropic, rotated) Gaussian kernel."""
    sigma_x = max(float(sigma), 1e-3)
    sigma_y = max(float(sigma_y if sigma_y is not None else sigma), 1e-3)
    if size is None:
        size = int(2 * np.ceil(3 * max(sigma_x, sigma_y)) + 1)
    size = int(size) | 1  # force odd so the kernel has a true centre
    half = size // 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = xx * cos_t + yy * sin_t
    y_rot = -xx * sin_t + yy * cos_t
    kernel = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


@dataclass
class DegradationConfig:
    scale: int = 4
    mode: str = "mtf"          # mtf | bicubic | blind
    mtf_nyquist: float = 0.30
    kernel_size: int = 11
    noise_sigma: float = 0.002  # reflectance units
    shot_noise: bool = True     # signal-dependent component
    jitter: float = 0.15        # blind mode: relative sigma jitter
    anisotropy: float = 0.2     # blind mode: max sigma_y/sigma_x deviation
    per_band_mtf: bool = True
    bands: tuple[str, ...] = ("B02", "B03", "B04", "B08")

    @classmethod
    def from_dict(cls, data: dict, scale: int = 4, bands: Sequence[str] | None = None):
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        kwargs["scale"] = scale
        if bands is not None:
            kwargs["bands"] = tuple(bands)
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# numpy path: used to build the on-disk paired dataset
# --------------------------------------------------------------------------- #

def area_decimate(array: np.ndarray, scale: int) -> np.ndarray:
    """Average over scale x scale blocks -- the detector footprint integral."""
    c, h, w = array.shape
    h_crop, w_crop = (h // scale) * scale, (w // scale) * scale
    cropped = array[:, :h_crop, :w_crop]
    return cropped.reshape(c, h_crop // scale, scale, w_crop // scale, scale).mean(axis=(2, 4))


def blur(array: np.ndarray, kernels: Sequence[np.ndarray]) -> np.ndarray:
    """Per-band convolution with reflect padding (no dark halo at tile edges)."""
    from scipy.ndimage import convolve

    out = np.empty_like(array)
    for i in range(array.shape[0]):
        kernel = kernels[i % len(kernels)]
        out[i] = convolve(array[i], kernel, mode="reflect")
    return out


def degrade(
    hr: np.ndarray,
    cfg: DegradationConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """(C, H, W) fine-grid reflectance -> (C, H/s, W/s) sensor-like observation."""
    rng = rng or np.random.default_rng()
    hr = np.asarray(hr, dtype=np.float32)
    scale = cfg.scale

    if cfg.mode == "bicubic":
        lr = _bicubic_numpy(hr, 1.0 / scale)
    else:
        kernels = _build_kernels(cfg, rng)
        lr = area_decimate(blur(hr, kernels), scale)

    if cfg.noise_sigma > 0:
        sigma = cfg.noise_sigma
        if cfg.mode == "blind":
            sigma *= float(rng.uniform(0.5, 1.5))
        noise = rng.normal(0.0, 1.0, size=lr.shape).astype(np.float32)
        if cfg.shot_noise:
            # Photon noise grows with signal; keep a floor so dark water still
            # receives read noise rather than becoming artificially clean.
            gain = np.sqrt(np.clip(lr, 0.01, None) / 0.2)
            noise = noise * gain
        lr = lr + sigma * noise

    return np.clip(lr, 0.0, 1.5).astype(np.float32)


def _build_kernels(cfg: DegradationConfig, rng: np.random.Generator) -> list[np.ndarray]:
    if cfg.per_band_mtf:
        sigmas = band_sigmas(cfg.bands, cfg.scale, cfg.mtf_nyquist)
    else:
        sigmas = [mtf_to_sigma(cfg.mtf_nyquist, cfg.scale)] * len(cfg.bands)

    kernels = []
    for sigma in sigmas:
        sigma_x, sigma_y, theta = sigma, sigma, 0.0
        if cfg.mode == "blind":
            # Train for a PSF we only know approximately: jitter width,
            # anisotropy and orientation so the model cannot key on one kernel.
            sigma_x = sigma * float(rng.uniform(1 - cfg.jitter, 1 + cfg.jitter))
            sigma_y = sigma_x * float(rng.uniform(1 - cfg.anisotropy, 1 + cfg.anisotropy))
            theta = float(rng.uniform(0, np.pi))
        kernels.append(gaussian_kernel_2d(sigma_x, cfg.kernel_size, sigma_y, theta))
    return kernels


def _bicubic_numpy(array: np.ndarray, factor: float) -> np.ndarray:
    import cv2

    c, h, w = array.shape
    new_h, new_w = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    out = np.empty((c, new_h, new_w), dtype=np.float32)
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    for i in range(c):
        out[i] = cv2.resize(array[i], (new_w, new_h), interpolation=interp)
    return out


def upsample_bicubic(array: np.ndarray, scale: int) -> np.ndarray:
    """The baseline every SR model must beat, and the default display reference."""
    return _bicubic_numpy(np.asarray(array, dtype=np.float32), float(scale))


# --------------------------------------------------------------------------- #
# torch path: differentiable, used by the consistency loss and augmentation
# --------------------------------------------------------------------------- #

class TorchDegradation:
    """Differentiable blur + area-decimate operator.

    Held as a module-free helper so it can be applied inside a loss without
    appearing in the model's parameter list.
    """

    def __init__(self, cfg: DegradationConfig, channels: int = 4, device: str = "cpu"):
        import torch

        self.cfg = cfg
        self.scale = cfg.scale
        sigmas = (
            band_sigmas(cfg.bands, cfg.scale, cfg.mtf_nyquist)
            if cfg.per_band_mtf
            else [mtf_to_sigma(cfg.mtf_nyquist, cfg.scale)] * channels
        )
        kernels = [gaussian_kernel_2d(sigmas[i % len(sigmas)], cfg.kernel_size) for i in range(channels)]
        weight = np.stack(kernels)[:, None]  # (C, 1, k, k) for depthwise conv
        self.weight = torch.from_numpy(weight).to(device)
        self.channels = channels
        self.pad = cfg.kernel_size // 2

    def to(self, device):
        self.weight = self.weight.to(device)
        return self

    def __call__(self, hr):
        import torch.nn.functional as F

        if self.weight.device != hr.device or self.weight.dtype != hr.dtype:
            self.weight = self.weight.to(device=hr.device, dtype=hr.dtype)
        blurred = F.conv2d(
            F.pad(hr, (self.pad,) * 4, mode="reflect"),
            self.weight,
            groups=self.channels,
        )
        return F.avg_pool2d(blurred, self.scale)
