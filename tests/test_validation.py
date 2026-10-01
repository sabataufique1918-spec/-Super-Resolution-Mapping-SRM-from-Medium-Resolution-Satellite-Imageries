"""Input admission control and the geographic contract of the output.

The failure this guards against is the worst one an SRM product has: an image
that looks sharper and sits in the wrong place, which a side-by-side
screenshot cannot reveal.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from srm.utils.geo import read_raster, scale_transform, write_raster
from srm.validation import inspect_raster, resolve_band_mapping

UTM17 = CRS.from_epsg(32617)


def _write(path, array, *, crs=UTM17, transform=Affine(10.0, 0, 539120.0, 0, -10.0, 4146000.0),
           nodata=None, names=None, dtype="uint16"):
    array = np.asarray(array)
    profile = {
        "driver": "GTiff", "height": array.shape[1], "width": array.shape[2],
        "count": array.shape[0], "dtype": dtype,
    }
    if crs is not None:
        profile["crs"] = crs
    if transform is not None:
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(array.astype(dtype))
        for i, name in enumerate(names or [], start=1):
            dst.set_band_description(i, name)
    return str(path)


@pytest.fixture
def scene(tmp_path):
    rng = np.random.default_rng(0)
    array = rng.integers(200, 4000, size=(12, 64, 64), dtype=np.uint16)
    names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
             "B8A", "B9", "B11", "B12"]
    return _write(tmp_path / "scene.tif", array, nodata=65535, names=names)


# --------------------------------------------------------------------------- #
# inspection
# --------------------------------------------------------------------------- #

def test_inspection_reports_the_geometry_the_dashboard_shows(scene):
    r = inspect_raster(scene)
    assert r.ok and not r.errors
    assert (r.width, r.height, r.band_count) == (64, 64, 12)
    assert r.epsg == 32617
    assert r.pixel_size == pytest.approx(10.0)
    assert r.valid_fraction == pytest.approx(1.0)
    assert r.ground_extent_m() == (640.0, 640.0)
    assert r.output_shape(4) == (256, 256)
    assert r.output_gsd(4) == pytest.approx(2.5)


def test_a_file_without_a_crs_is_refused_with_an_actionable_message(tmp_path):
    path = _write(tmp_path / "nocrs.tif", np.ones((4, 16, 16), dtype=np.uint16),
                  crs=None, transform=None)
    r = inspect_raster(path)
    assert not r.ok
    assert any("coordinate reference system" in e.lower() for e in r.errors)
    assert any("gdal_edit" in e for e in r.errors), "the message must say what to do"


def test_an_unreadable_file_is_reported_not_raised(tmp_path):
    path = tmp_path / "junk.tif"
    path.write_bytes(b"this is not a geotiff")
    r = inspect_raster(path)
    assert not r.ok and r.errors


def test_nodata_reduces_the_reported_valid_fraction(tmp_path):
    array = np.full((4, 32, 32), 1000, dtype=np.uint16)
    array[:, :24] = 65535                       # three quarters unobserved
    r = inspect_raster(_write(tmp_path / "holes.tif", array, nodata=65535))
    assert r.valid_fraction == pytest.approx(0.25, abs=0.02)
    assert any("carry data" in w for w in r.warnings)


def test_a_fully_empty_scene_is_refused(tmp_path):
    array = np.full((4, 16, 16), 65535, dtype=np.uint16)
    r = inspect_raster(_write(tmp_path / "empty.tif", array, nodata=65535))
    assert not r.ok
    assert any("nodata" in e.lower() for e in r.errors)


def test_unexpected_pixel_spacing_warns_but_does_not_block(tmp_path):
    path = _write(tmp_path / "coarse.tif", np.ones((4, 16, 16), dtype=np.uint16),
                  transform=Affine(20.0, 0, 500000.0, 0, -20.0, 4000000.0))
    r = inspect_raster(path)
    assert r.ok, "a 20 m scene is processable, just outside the trained range"
    assert any("20" in w and "trained" in w for w in r.warnings)


# --------------------------------------------------------------------------- #
# band mapping
# --------------------------------------------------------------------------- #

def test_named_bands_beat_position_on_a_12_band_product(scene):
    """The concrete trap: RGB+NIR are file bands 2,3,4,8 -- never 1,2,3,4."""
    r = inspect_raster(scene)
    indices, names, notes = resolve_band_mapping(r, ["B02", "B03", "B04", "B08"], 4)
    assert indices == [2, 3, 4, 8]
    assert names == ["B02", "B03", "B04", "B08"]
    assert any("by name" in n for n in notes)


def test_an_unnamed_12_band_stack_still_resolves_to_4_3_2_8(tmp_path):
    """SEN2NEON GeoTIFFs carry no band descriptions at all.

    Without a documented layout for the band count, the name lookup finds
    nothing and falls through to positions 1-4 -- the precise error the name
    lookup exists to prevent. Verified against the real files: every
    description is None.
    """
    path = _write(tmp_path / "unnamed12.tif",
                  np.ones((12, 16, 16), dtype=np.uint16))
    r = inspect_raster(path)
    assert all(n.startswith("band_") for n in r.band_names), "precondition: unnamed"

    indices, names, notes = resolve_band_mapping(r, ["B02", "B03", "B04", "B08"], 4)
    assert indices == [2, 3, 4, 8]
    assert names == ["B02", "B03", "B04", "B08"]
    assert any("names no bands" in n for n in notes), "the assumption must be stated"


def test_an_unnamed_13_band_l1c_stack_accounts_for_b10(tmp_path):
    """B10 present shifts B11/B12; the layout must follow the band count."""
    path = _write(tmp_path / "unnamed13.tif", np.ones((13, 16, 16), dtype=np.uint16))
    r = inspect_raster(path)
    indices, _, _ = resolve_band_mapping(r, ["B02", "B03", "B04", "B08"], 4)
    assert indices == [2, 3, 4, 8]
    # B11 sits at 12 in L1C but at 11 in the 12-band stack.
    assert resolve_band_mapping(r, ["B11"], 1)[0] == [12]


def test_positional_fallback_is_used_only_when_names_do_not_match_and_says_so(tmp_path):
    path = _write(tmp_path / "unnamed.tif", np.ones((6, 16, 16), dtype=np.uint16))
    r = inspect_raster(path)
    indices, _, notes = resolve_band_mapping(r, ["B02", "B03", "B04", "B08"], 4)
    assert indices == [1, 2, 3, 4]
    assert any("first 4" in n or "band mapping" in n for n in notes)


def test_too_few_bands_is_an_error_not_a_silent_truncation(tmp_path):
    path = _write(tmp_path / "three.tif", np.ones((3, 16, 16), dtype=np.uint16))
    r = inspect_raster(path)
    with pytest.raises(ValueError, match="needs 4 bands"):
        resolve_band_mapping(r, ["B02", "B03", "B04", "B08"], 4)


# --------------------------------------------------------------------------- #
# the georeferencing contract
# --------------------------------------------------------------------------- #

def test_super_resolved_output_covers_exactly_the_same_ground(tmp_path, scene):
    """Ground extent is invariant under super-resolution. This is the contract."""
    array, meta = read_raster(scene, bands=[2, 3, 4, 8])
    sr = np.repeat(np.repeat(array, 4, axis=1), 4, axis=2)
    out = write_raster(tmp_path / "sr.tif", sr, meta, scale_factor=4)

    with rasterio.open(scene) as a, rasterio.open(str(out)) as b:
        assert a.crs == b.crs
        # Bounds identical to well under a millimetre.
        assert np.allclose(a.bounds, b.bounds, atol=1e-6)
        assert b.width == a.width * 4 and b.height == a.height * 4
        assert b.res[0] == pytest.approx(a.res[0] / 4)
        # The upper-left corner is a pixel *edge* and must be preserved exactly.
        assert b.transform.c == a.transform.c
        assert b.transform.f == a.transform.f


def test_scale_transform_divides_resolution_and_keeps_the_origin():
    t = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    s = scale_transform(t, 4)
    assert (s.a, s.e) == (2.5, -2.5)
    assert (s.c, s.f) == (t.c, t.f)


def test_a_mismatched_transform_would_be_caught_by_the_bounds_check(tmp_path, scene):
    """Guard on the guard: if the transform were left unscaled, bounds diverge."""
    array, meta = read_raster(scene, bands=[2, 3, 4, 8])
    sr = np.repeat(np.repeat(array, 4, axis=1), 4, axis=2)
    wrong = write_raster(tmp_path / "wrong.tif", sr, meta, scale_factor=1)
    with rasterio.open(scene) as a, rasterio.open(str(wrong)) as b:
        assert not np.allclose(a.bounds, b.bounds, atol=1e-6)
