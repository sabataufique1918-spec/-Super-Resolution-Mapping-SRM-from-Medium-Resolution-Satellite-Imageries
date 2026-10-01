"""Georeferencing: the invariants that fail silently.

A super-resolved scene with a dropped CRS or an unscaled transform still opens,
still looks right, and is worthless -- it will not overlay anything. Nothing
raises. These tests are the only thing standing between that and a shipped
product.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from affine import Affine  # noqa: E402
from rasterio.crs import CRS  # noqa: E402

from srm import S2_REFLECTANCE_SCALE  # noqa: E402
from srm.utils.geo import GeoMeta, pixel_size, read_raster, scale_transform, write_raster  # noqa: E402

# A plausible Sentinel-2 tile origin in UTM 31N, 10 m pixels.
TRANSFORM = Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 5800000.0)
UTM31N = CRS.from_epsg(32631)


@pytest.fixture
def meta() -> GeoMeta:
    return GeoMeta(crs=UTM31N, transform=TRANSFORM, nodata=None,
                   band_names=["B02", "B03", "B04", "B08"])


def test_scale_transform_shrinks_the_pixel_not_the_origin():
    """x4 super-resolution means 2.5 m pixels anchored at the same corner."""
    fine = scale_transform(TRANSFORM, 4)
    assert fine.a == pytest.approx(2.5)
    assert fine.e == pytest.approx(-2.5)
    assert fine.c == pytest.approx(TRANSFORM.c)   # upper-left easting unmoved
    assert fine.f == pytest.approx(TRANSFORM.f)   # upper-left northing unmoved


def test_scale_transform_by_one_is_the_identity():
    assert scale_transform(TRANSFORM, 1) == TRANSFORM


def test_scale_transform_tolerates_a_missing_transform():
    """Synthetic scenes have no georeferencing; that must not be fatal."""
    assert scale_transform(None, 4) is None


def test_roundtrip_preserves_crs_and_transform(tmp_path, meta, hr_scene):
    path = write_raster(tmp_path / "scene.tif", hr_scene, meta)
    array, back = read_raster(path)

    assert array.shape == hr_scene.shape
    assert back.crs == UTM31N
    # Affine iterates as the full 3x3, so compare the six affine coefficients.
    assert tuple(back.transform)[:6] == pytest.approx(tuple(TRANSFORM)[:6], abs=1e-9)


def test_roundtrip_preserves_pixel_values(tmp_path, meta, hr_scene):
    """float32 + deflate is lossless; a predictor mismatch would show up here."""
    path = write_raster(tmp_path / "scene.tif", hr_scene, meta)
    array, _ = read_raster(path)
    assert np.allclose(array, hr_scene, atol=1e-6)


def test_superresolved_output_carries_the_finer_transform(tmp_path, meta, hr_scene):
    """The end-to-end georeferencing contract of the whole project."""
    sr = np.repeat(np.repeat(hr_scene, 4, axis=1), 4, axis=2)
    path = write_raster(tmp_path / "sr.tif", sr, meta, scale_factor=4)
    _, back = read_raster(path)

    assert pixel_size(back) == pytest.approx(2.5)
    assert back.crs == UTM31N
    with rasterio.open(path) as src:
        # same ground footprint as the 10 m parent, four times the pixels
        assert src.bounds.left == pytest.approx(600000.0)
        assert src.bounds.top == pytest.approx(5800000.0)
        assert src.width == hr_scene.shape[2] * 4


def test_band_descriptions_survive_the_roundtrip(tmp_path, meta, hr_scene):
    """Losing band names turns a 4-band product into an unidentifiable stack."""
    path = write_raster(tmp_path / "named.tif", hr_scene, meta)
    _, back = read_raster(path)
    assert back.band_names == ["B02", "B03", "B04", "B08"]


def test_uint16_output_is_rescaled_to_sentinel2_digital_numbers(tmp_path, meta, hr_scene):
    """Reflectance goes out scaled by 10000 and must come back as reflectance."""
    path = write_raster(tmp_path / "dn.tif", hr_scene, meta, dtype="uint16")
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "uint16"
        raw = src.read(1)
    assert raw.max() > 100, "values look like they were written unscaled"

    array, _ = read_raster(path, as_reflectance=True)
    assert np.allclose(array, hr_scene, atol=1.0 / S2_REFLECTANCE_SCALE + 1e-6)


def test_float_input_is_not_rescaled(tmp_path, meta, hr_scene):
    """Only integer sources are digital numbers; floats are already physical."""
    path = write_raster(tmp_path / "f32.tif", hr_scene, meta)
    array, _ = read_raster(path, as_reflectance=True)
    assert array.max() < 1.5


def test_nodata_becomes_nan_on_read(tmp_path, hr_scene):
    """Cloud and edge fill must not be averaged into a metric as if it were data."""
    scene = hr_scene.copy()
    scene[:, :8, :] = -9999.0
    path = write_raster(tmp_path / "nodata.tif", scene,
                        GeoMeta(crs=UTM31N, transform=TRANSFORM, nodata=-9999.0))
    array, back = read_raster(path)
    assert back.nodata == -9999.0
    assert np.isnan(array[:, :8, :]).all()
    assert np.isfinite(array[:, 8:, :]).all()


def test_band_subset_selection(tmp_path, meta, hr_scene):
    array, back = read_raster(write_raster(tmp_path / "s.tif", hr_scene, meta), bands=[1, 4])
    assert array.shape[0] == 2
    assert back.band_names == ["B02", "B08"]


def test_writing_without_georeferencing_still_works(tmp_path, hr_scene):
    """Synthetic scenes have no CRS; the writer must not require one."""
    path = write_raster(tmp_path / "plain.tif", hr_scene)
    array, back = read_raster(path)
    assert array.shape == hr_scene.shape
    assert back.crs is None
    assert pixel_size(back) is None or pixel_size(back) == pytest.approx(1.0)


def test_two_dimensional_input_is_promoted_to_one_band(tmp_path, meta):
    """Index and uncertainty maps are single-band and arrive as (H, W)."""
    path = write_raster(tmp_path / "ndvi.tif", np.zeros((16, 16), dtype=np.float32), meta)
    array, _ = read_raster(path)
    assert array.shape == (1, 16, 16)
