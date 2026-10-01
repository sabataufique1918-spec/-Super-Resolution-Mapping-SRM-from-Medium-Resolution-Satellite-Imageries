"""Raster I/O that keeps the geospatial contract intact.

Super-resolution changes the pixel grid, so every write has to rebuild the
affine transform. Getting that wrong yields an image that looks sharper and
sits in the wrong place, which is the most damaging failure mode for an SRM
product precisely because it is invisible in a side-by-side screenshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import S2_REFLECTANCE_SCALE


def _rasterio():
    try:
        import rasterio

        return rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rasterio is required for GeoTIFF I/O. Install with: pip install rasterio"
        ) from exc


@dataclass
class GeoMeta:
    """The minimum needed to write a georeferenced child of a source raster."""

    crs: Any = None
    transform: Any = None
    nodata: float | None = None
    band_names: list[str] = field(default_factory=list)

    def to_profile(self, height: int, width: int, count: int, dtype: str = "float32") -> dict:
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "height": int(height),
            "width": int(width),
            "count": int(count),
            "dtype": dtype,
            "compress": "deflate",
            "predictor": 3 if dtype.startswith("float") else 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        if self.crs is not None:
            profile["crs"] = self.crs
        if self.transform is not None:
            profile["transform"] = self.transform
        if self.nodata is not None:
            profile["nodata"] = self.nodata
        return profile


def scale_transform(transform: Any, factor: int | float) -> Any:
    """Return `transform` with the pixel size divided by `factor`.

    The upper-left corner is a pixel edge, not a centre, so it is preserved
    exactly and only the resolution terms change. That is what keeps an x4
    product pixel-aligned with its 10 m parent scene.
    """
    if transform is None:
        return None
    from affine import Affine

    return Affine(
        transform.a / factor,
        transform.b / factor,
        transform.c,
        transform.d / factor,
        transform.e / factor,
        transform.f,
    )


def read_raster(
    path: str | Path,
    bands: Sequence[int] | None = None,
    as_reflectance: bool = True,
) -> tuple[np.ndarray, GeoMeta]:
    """Read a raster as float32 (C, H, W).

    `as_reflectance` rescales integer Sentinel-2 digital numbers into 0-1
    physical units. Float inputs are assumed to be reflectance already and are
    passed through untouched.
    """
    rasterio = _rasterio()
    with rasterio.open(str(path)) as src:
        idx = list(bands) if bands is not None else list(range(1, src.count + 1))
        array = src.read(idx).astype(np.float32)
        descriptions = src.descriptions
        meta = GeoMeta(
            crs=src.crs,
            transform=src.transform,
            nodata=src.nodata,
            band_names=[descriptions[i - 1] or f"band_{i}" for i in idx],
        )
        integer_source = np.issubdtype(np.dtype(src.dtypes[0]), np.integer)
    if meta.nodata is not None:
        array = np.where(array == meta.nodata, np.nan, array)
    if as_reflectance and integer_source:
        array = array / S2_REFLECTANCE_SCALE
    return array, meta


def write_raster(
    path: str | Path,
    array: np.ndarray,
    meta: GeoMeta | None = None,
    scale_factor: int | float = 1,
    dtype: str = "float32",
    band_names: Sequence[str] | None = None,
) -> Path:
    """Write (C, H, W) to GeoTIFF, shrinking the pixel size by `scale_factor`."""
    rasterio = _rasterio()
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    count, height, width = array.shape
    meta = meta or GeoMeta()
    out_meta = GeoMeta(
        crs=meta.crs,
        transform=scale_transform(meta.transform, scale_factor),
        nodata=meta.nodata,
        band_names=list(band_names or meta.band_names),
    )
    profile = out_meta.to_profile(height, width, count, dtype=dtype)
    payload = array
    if dtype.startswith("uint"):
        payload = np.clip(array * S2_REFLECTANCE_SCALE, 0, np.iinfo(dtype).max)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(payload.astype(dtype))
        for i, name in enumerate(out_meta.band_names[:count], start=1):
            dst.set_band_description(i, str(name))
    return path


def pixel_size(meta: GeoMeta) -> float | None:
    """Ground sample distance in CRS units (metres for UTM), or None."""
    if meta.transform is None:
        return None
    return float(abs(meta.transform.a))


def to_uint8_rgb(
    array: np.ndarray,
    rgb_index: Sequence[int] = (2, 1, 0),
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """(C, H, W) reflectance -> (H, W, 3) uint8 for display, percentile-stretched.

    Stretch bounds are shared across the three bands so colour balance is not
    silently altered: a per-band stretch makes vegetation look different between
    the LR and SR panels even when the model changed nothing there.
    """
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    idx = [i for i in rgb_index if i < array.shape[0]]
    if not idx:
        idx = [0]
    while len(idx) < 3:
        idx.append(idx[-1])
    rgb = array[idx[:3]]
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros(rgb.shape[1:] + (3,), dtype=np.uint8)
    lo, hi = np.percentile(finite, percentiles)
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((np.nan_to_num(rgb) - lo) / (hi - lo), 0.0, 1.0)
    return (np.moveaxis(out, 0, -1) * 255.0).astype(np.uint8)
