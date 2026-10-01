"""Spectral indices, and whether super-resolution keeps them honest.

An index is the usual end product of medium-resolution imagery, so "did SR
help?" is really "did SR give me a better NDVI map?". That has two parts and
they pull in opposite directions:

* **Sharper boundaries.** At 10 m, a pixel on a field edge mixes crop and road
  and its NDVI is a meaningless average of the two. SR should unmix it.
* **Unchanged values.** The NDVI of a pixel well inside a field must not move.
  If it does, every threshold and time series built on it breaks.

`index_fidelity` measures both at once.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Band positions in the standard four-band ordering B02, B03, B04, B08.
BLUE, GREEN, RED, NIR = 0, 1, 2, 3

INDEX_DEFINITIONS: dict[str, tuple[int, int, str]] = {
    "ndvi": (NIR, RED, "vegetation vigour"),
    "ndwi": (GREEN, NIR, "open water (McFeeters)"),
    "ndbi_proxy": (RED, NIR, "built-up proxy without SWIR"),
    "gndvi": (NIR, GREEN, "chlorophyll-sensitive vegetation"),
}


def normalised_difference(image: np.ndarray, a: int, b: int, eps: float = 1e-6) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    num = image[a] - image[b]
    den = image[a] + image[b]
    out = np.divide(num, den + eps, out=np.zeros_like(num), where=np.isfinite(den))
    return np.clip(out, -1.0, 1.0)


def compute_index(image: np.ndarray, name: str = "ndvi") -> np.ndarray:
    key = name.lower()
    if key not in INDEX_DEFINITIONS:
        raise KeyError(f"unknown index '{name}'. Available: {sorted(INDEX_DEFINITIONS)}")
    a, b, _ = INDEX_DEFINITIONS[key]
    return normalised_difference(image, a, b)


def compute_all(image: np.ndarray) -> dict[str, np.ndarray]:
    return {name: compute_index(image, name) for name in INDEX_DEFINITIONS}


def _edge_mask(reference: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    """Pixels near a boundary in the reference index, where mixing happens."""
    from scipy.ndimage import sobel

    magnitude = np.hypot(sobel(reference, axis=0), sobel(reference, axis=1))
    threshold = float(np.quantile(magnitude, quantile))
    return magnitude >= threshold


def index_fidelity(
    sr: np.ndarray,
    reference: np.ndarray,
    baseline: np.ndarray | None = None,
    names: Sequence[str] = ("ndvi", "ndwi"),
) -> dict[str, dict[str, float]]:
    """Per-index error, split into boundary pixels and field-interior pixels.

    The split is the point. A model that improves the interior while degrading
    boundaries (or the reverse) shows up as one number moving and the other
    not, which a single global RMSE would hide completely.
    """
    out: dict[str, dict[str, float]] = {}
    for name in names:
        ref = compute_index(reference, name)
        pred = compute_index(sr, name)
        h, w = min(ref.shape[0], pred.shape[0]), min(ref.shape[1], pred.shape[1])
        ref, pred = ref[:h, :w], pred[:h, :w]
        edges = _edge_mask(ref)
        interior = ~edges

        stats = {
            "rmse": _rmse(pred, ref),
            "rmse_boundary": _rmse(pred[edges], ref[edges]),
            "rmse_interior": _rmse(pred[interior], ref[interior]),
            "bias": float(np.mean(pred - ref)),
            "correlation": _corr(pred, ref),
        }
        if baseline is not None:
            base = compute_index(baseline, name)[:h, :w]
            stats["baseline_rmse"] = _rmse(base, ref)
            stats["baseline_rmse_boundary"] = _rmse(base[edges], ref[edges])
            stats["baseline_rmse_interior"] = _rmse(base[interior], ref[interior])
            stats["improvement_pct"] = float(
                100.0 * (stats["baseline_rmse"] - stats["rmse"]) / (stats["baseline_rmse"] + 1e-9)
            )
            stats["improvement_boundary_pct"] = float(
                100.0
                * (stats["baseline_rmse_boundary"] - stats["rmse_boundary"])
                / (stats["baseline_rmse_boundary"] + 1e-9)
            )
        out[name] = stats
    return out


def mixed_pixel_fraction(index_map: np.ndarray, low: float, high: float) -> float:
    """Fraction of pixels whose index sits between two class centres.

    A direct, interpretable measure of the problem SRM exists to solve: at
    10 m a large share of pixels are spectral mixtures that belong to no class.
    Super-resolution should shrink this fraction; if it does not, it has not
    unmixed anything.
    """
    values = np.asarray(index_map, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean((finite > low) & (finite < high)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or a[mask].std() < 1e-9 or b[mask].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])
