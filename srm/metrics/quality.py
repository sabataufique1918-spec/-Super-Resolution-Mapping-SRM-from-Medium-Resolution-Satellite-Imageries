"""Quality metrics for super-resolved reflectance imagery.

Three families, and a serious evaluation reports all three:

**Full reference** (PSNR, SSIM, RMSE) -- available only when true
high-resolution imagery exists. Necessary but not sufficient: PSNR rewards
blur, so a model can win on PSNR by refusing to sharpen.

**Spectral** (SAM, ERGAS, SCC, Q-index) -- imported from the pansharpening
literature, which has spent thirty years on exactly this question: did
resolution enhancement corrupt the radiometry? ERGAS in particular is the
standard summary number for a fused product.

**Reference free** (Wald consistency, sharpness gain) -- computable on real
Sentinel-2 with no reference at all, which makes them the only metrics you can
run over an operational output. Consistency error is the closest thing to a
hallucination detector available without ground truth.

Everything takes numpy (C, H, W) in reflectance units and returns plain floats.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

_EPS = 1e-10


def _prep(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    return x


def _valid_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = _prep(pred), _prep(target)
    mask = _valid_mask(pred, target)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((pred[:, mask] - target[:, mask]) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error in reflectance units.

    Reported alongside RMSE because the two disagree in a way that is
    diagnostic: RMSE >> MAE means the error is concentrated in a few pixels
    (usually edges or a misregistered feature) rather than spread evenly,
    which is the difference between a model that blurs and one that
    occasionally invents something.
    """
    pred, target = _prep(pred), _prep(target)
    mask = _valid_mask(pred, target)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(pred[:, mask] - target[:, mask])))


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB.

    `data_range` is the physical dynamic range of reflectance, not the range
    of the particular patch. Using a per-patch max, as some implementations do,
    makes dark scenes look artificially good and the numbers incomparable.
    """
    error = rmse(pred, target)
    if not np.isfinite(error) or error < _EPS:
        return float("inf")
    return float(20.0 * np.log10(data_range / error))


def ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    window: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean structural similarity over bands, Gaussian-weighted (Wang et al.)."""
    from scipy.ndimage import gaussian_filter

    pred, target = _prep(pred), _prep(target)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    truncate = ((window - 1) / 2) / sigma
    scores = []
    for band in range(pred.shape[0]):
        x = np.nan_to_num(pred[band])
        y = np.nan_to_num(target[band])
        kw = dict(sigma=sigma, truncate=truncate, mode="reflect")
        mu_x, mu_y = gaussian_filter(x, **kw), gaussian_filter(y, **kw)
        mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_xx = gaussian_filter(x * x, **kw) - mu_xx
        sigma_yy = gaussian_filter(y * y, **kw) - mu_yy
        sigma_xy = gaussian_filter(x * y, **kw) - mu_xy
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
        scores.append(float(np.mean(numerator / (denominator + _EPS))))
    return float(np.mean(scores))


def sam(pred: np.ndarray, target: np.ndarray, degrees: bool = True) -> float:
    """Spectral Angle Mapper: mean angle between pixel spectra.

    The single most important number for an SRM product that will feed a
    classifier. Below ~2 degrees the spectra are preserved for practical
    purposes; above ~5 the land-cover signature has been altered.
    """
    pred, target = _prep(pred), _prep(target)
    mask = _valid_mask(pred, target)
    if not mask.any():
        return float("nan")
    p, t = pred[:, mask], target[:, mask]
    dot = np.sum(p * t, axis=0)
    norm = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0)
    cosine = np.clip(dot / (norm + _EPS), -1.0, 1.0)
    angle = float(np.mean(np.arccos(cosine)))
    return float(np.degrees(angle)) if degrees else angle


def ergas(pred: np.ndarray, target: np.ndarray, scale: int = 4) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese (lower is better).

    Band-wise RMSE normalised by each band's mean, so a bright band cannot
    dominate, then scaled by the resolution ratio. Values under 3 are
    conventionally read as a good fusion result.
    """
    pred, target = _prep(pred), _prep(target)
    mask = _valid_mask(pred, target)
    if not mask.any():
        return float("nan")
    terms = []
    for band in range(pred.shape[0]):
        p, t = pred[band][mask], target[band][mask]
        mean = float(np.mean(t))
        if abs(mean) < _EPS:
            continue
        terms.append(float(np.mean((p - t) ** 2)) / (mean**2))
    if not terms:
        return float("nan")
    return float(100.0 / scale * np.sqrt(np.mean(terms)))


def scc(pred: np.ndarray, target: np.ndarray) -> float:
    """Spatial Correlation Coefficient: correlation of high-pass content.

    Measures whether the *detail* agrees, not whether the image agrees. A model
    that returns a blurred copy scores well on PSNR and badly here, which is
    precisely the discrimination PSNR fails to make.
    """
    from scipy.ndimage import convolve

    laplacian = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
    pred, target = _prep(pred), _prep(target)
    scores = []
    for band in range(pred.shape[0]):
        hp_p = convolve(np.nan_to_num(pred[band]), laplacian, mode="reflect").ravel()
        hp_t = convolve(np.nan_to_num(target[band]), laplacian, mode="reflect").ravel()
        if hp_p.std() < _EPS or hp_t.std() < _EPS:
            continue
        scores.append(float(np.corrcoef(hp_p, hp_t)[0, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def q_index(pred: np.ndarray, target: np.ndarray, block: int = 32) -> float:
    """Universal Image Quality Index (Wang & Bovik), averaged over blocks."""
    pred, target = _prep(pred), _prep(target)
    scores = []
    _, height, width = pred.shape
    for band in range(pred.shape[0]):
        for y in range(0, height - block + 1, block):
            for x in range(0, width - block + 1, block):
                p = np.nan_to_num(pred[band, y : y + block, x : x + block]).ravel()
                t = np.nan_to_num(target[band, y : y + block, x : x + block]).ravel()
                mu_p, mu_t = p.mean(), t.mean()
                var_p, var_t = p.var(), t.var()
                cov = float(np.mean((p - mu_p) * (t - mu_t)))
                denom = (var_p + var_t) * (mu_p**2 + mu_t**2)
                if denom < _EPS:
                    continue
                scores.append(4.0 * cov * mu_p * mu_t / denom)
    return float(np.mean(scores)) if scores else float("nan")


# --------------------------------------------------------------------------- #
# reference-free: what you can actually compute on an operational product
# --------------------------------------------------------------------------- #

def consistency_error(sr: np.ndarray, lr: np.ndarray, degradation) -> dict[str, float]:
    """Wald's protocol on real data: degrade the SR result and compare to input.

    High consistency error means the model moved energy around that the sensor
    never saw -- the operational definition of hallucination when no reference
    exists. Reported both as RMSE in reflectance and as a relative percentage.
    """
    from ..data.degradation import degrade

    sr_down = degrade(np.asarray(sr, dtype=np.float32), degradation)
    lr = _prep(lr)
    sr_down = _prep(sr_down)
    h = min(sr_down.shape[1], lr.shape[1])
    w = min(sr_down.shape[2], lr.shape[2])
    a, b = sr_down[:, :h, :w], lr[:, :h, :w]
    error = rmse(a, b)
    mean = float(np.nanmean(b))
    return {
        "consistency_rmse": error,
        "consistency_relative_pct": float(100.0 * error / (abs(mean) + _EPS)),
        "consistency_sam_deg": sam(a, b),
    }


def sharpness(image: np.ndarray) -> float:
    """Mean gradient magnitude -- a proxy for apparent spatial detail."""
    from scipy.ndimage import sobel

    image = _prep(image)
    values = []
    for band in range(image.shape[0]):
        x = np.nan_to_num(image[band])
        values.append(float(np.mean(np.hypot(sobel(x, axis=0), sobel(x, axis=1)))))
    return float(np.mean(values))


def sharpness_gain(sr: np.ndarray, lr_upsampled: np.ndarray) -> float:
    """Ratio of SR sharpness to the interpolation baseline. >1 means sharper.

    Report it next to consistency error, never alone: sharpness with poor
    consistency is invented detail, and that combination is the failure this
    project is most obliged to surface.
    """
    base = sharpness(lr_upsampled)
    return float(sharpness(sr) / (base + _EPS))


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

ALL_METRICS = ("psnr", "ssim", "sam", "ergas", "scc", "rmse", "mae", "q_index")

#: Units, direction and the assumption each metric carries. A number without
#: these is not reportable: "PSNR 36.2" means nothing until the data range is
#: stated, and "SAM 2.3" is ambiguous between degrees and radians.
METRIC_INFO: dict[str, dict[str, str]] = {
    "psnr":  {"unit": "dB", "better": "higher",
              "note": "computed against an explicit reflectance data range, not the patch maximum"},
    "ssim":  {"unit": "0-1", "better": "higher",
              "note": "Gaussian-weighted, mean over bands, same explicit data range"},
    "sam":   {"unit": "degrees", "better": "lower",
              "note": "mean spectral angle between per-pixel 4-band spectra"},
    "ergas": {"unit": "dimensionless", "better": "lower",
              "note": "scaled by the resolution ratio"},
    "scc":   {"unit": "correlation", "better": "higher",
              "note": "correlation of high-pass content, so it rewards real detail"},
    "rmse":  {"unit": "reflectance", "better": "lower", "note": "0-1 surface reflectance"},
    "mae":   {"unit": "reflectance", "better": "lower", "note": "0-1 surface reflectance"},
    "q_index": {"unit": "0-1", "better": "higher", "note": "Wang & Bovik, block-averaged"},
    "sharpness_gain": {"unit": "ratio vs bicubic", "better": "higher",
                       "note": "detail added relative to interpolation; not evidence it is correct"},
}


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    names: Sequence[str] = ALL_METRICS,
    data_range: float = 1.0,
    scale: int = 4,
) -> dict[str, float]:
    """Full-reference metric bundle."""
    out: dict[str, float] = {}
    for name in names:
        key = name.lower()
        if key == "psnr":
            out["psnr"] = psnr(pred, target, data_range)
        elif key == "ssim":
            out["ssim"] = ssim(pred, target, data_range)
        elif key == "sam":
            out["sam"] = sam(pred, target)
        elif key == "ergas":
            out["ergas"] = ergas(pred, target, scale)
        elif key == "scc":
            out["scc"] = scc(pred, target)
        elif key == "rmse":
            out["rmse"] = rmse(pred, target)
        elif key in ("mae", "l1"):
            out["mae"] = mae(pred, target)
        elif key in ("q", "q_index"):
            out["q_index"] = q_index(pred, target)
    return out


def aggregate(rows: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    """mean / std / median / p05 / p95 per metric across a test set.

    The spread matters as much as the mean: a model with good average PSNR and
    a long bad tail is a model that fails on a specific land-cover type, and
    the mean alone will never tell you that.
    """
    rows = list(rows)
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array(
            [row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=np.float64
        )
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "n": int(values.size),
        }
    return summary
