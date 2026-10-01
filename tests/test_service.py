"""The orchestration layer the dashboard renders.

The regression pinned here is one that was actually shipped and caught: the
reference raster was read with `range(1, n_bands + 1)`, so a 12-band NEON
reference was compared band-for-band as 1,2,3,4 against the model's
2,3,4,8. Spectral angle came out at 61.8 degrees instead of 0.68, and every
accuracy figure in the dashboard was wrong while looking entirely plausible.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS

from srm import service
from srm.models.registry import build_model, save_checkpoint

UTM17 = CRS.from_epsg(32617)
ORIGIN = Affine(10.0, 0.0, 539120.0, 0.0, -10.0, 4146000.0)

#: A dense-forest spectrum: blue, green, red low, NIR high. Positional band
#: confusion destroys this ordering, which is what makes it a good probe.
FOREST = {"B1": 0.11, "B2": 0.029, "B3": 0.054, "B4": 0.026, "B5": 0.20,
          "B6": 0.35, "B7": 0.40, "B8": 0.464, "B8A": 0.44, "B9": 0.30,
          "B11": 0.18, "B12": 0.08}
LAYOUT = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
          "B8A", "B9", "B11", "B12")


def _twelve_band(path, size, pixel, *, rng_scale=0.002, seed=0):
    """A 12-band unnamed stack with a realistic forest spectrum, like SEN2NEON."""
    rng = np.random.default_rng(seed)
    stack = np.stack([
        np.full((size, size), FOREST[b], dtype=np.float32)
        + rng.normal(0, rng_scale, (size, size)).astype(np.float32)
        for b in LAYOUT
    ])
    data = np.clip(stack * 10000, 0, 65535).astype(np.uint16)
    profile = {
        "driver": "GTiff", "height": size, "width": size, "count": 12,
        "dtype": "uint16", "crs": UTM17,
        "transform": Affine(pixel, 0.0, ORIGIN.c, 0.0, -pixel, ORIGIN.f),
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data)
    return str(path)


@pytest.fixture
def checkpoint(tmp_path):
    cfg = {
        "experiment": "t", "scale": 4, "seed": 1, "bands": ["B02", "B03", "B04", "B08"],
        "model": {"name": "residual_cnn", "features": 8, "blocks": 2},
        "data": {"patches": "none"},
    }
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "m.pt", build_model(cfg), cfg, epoch=3)
    return service.discover_checkpoints([tmp_path])[0]


@pytest.fixture
def scene(tmp_path):
    return _twelve_band(tmp_path / "lr.tif", 32, 10.0)


@pytest.fixture
def reference(tmp_path):
    return _twelve_band(tmp_path / "hr.tif", 128, 2.5, rng_scale=0.001, seed=1)


# --------------------------------------------------------------------------- #
# the regression
# --------------------------------------------------------------------------- #

def test_reference_bands_are_resolved_by_name_not_by_position(
        checkpoint, scene, reference):
    """The shipped bug. Positional reading gave SAM 61.8 deg; correct is < 5.

    Both rasters here carry the same forest spectrum, so a correctly mapped
    reference must agree with the input to a fraction of a degree. Reading
    the reference as bands 1-4 compares B02/B03/B04/B08 against
    B1/B2/B3/B4 and the angle explodes.
    """
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    evaluation = service.evaluate_against_reference(result, reference)

    assert evaluation["available"], evaluation.get("reason")
    sam = evaluation["metrics"]["bicubic"]["sam"]
    assert sam < 5.0, f"reference appears mis-mapped: SAM {sam:.2f} deg"
    assert "B02, B03, B04, B08" in evaluation["note"]


def test_reference_note_states_which_bands_were_used(checkpoint, scene, reference):
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    note = service.evaluate_against_reference(result, reference)["note"]
    assert "[2, 3, 4, 8]" in note
    assert "names no bands" in note, "an assumed layout must be declared"


# --------------------------------------------------------------------------- #
# evaluation contract
# --------------------------------------------------------------------------- #

def test_both_predictions_are_scored_on_the_same_pixels(checkpoint, scene, reference):
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    evaluation = service.evaluate_against_reference(result, reference)
    assert set(evaluation["metrics"]) == {"model", "bicubic"}
    for row in evaluation["metrics"].values():
        assert set(service.REPORT_METRICS).issubset(row)
    assert evaluation["data_range"] == 1.0, "PSNR/SSIM need an explicit range"
    assert evaluation["valid_pixels"] > 0


def test_a_non_overlapping_reference_is_refused_not_scored(checkpoint, scene, tmp_path):
    """Comparing rasters that share no ground would measure nothing real."""
    far = _twelve_band(tmp_path / "far.tif", 128, 2.5)
    with rasterio.open(far, "r+") as dst:
        dst.transform = Affine(2.5, 0.0, 1_000_000.0, 0.0, -2.5, 1_000_000.0)
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    evaluation = service.evaluate_against_reference(result, far)
    assert not evaluation["available"]
    assert "overlap" in evaluation["reason"].lower()


def test_no_reference_means_unavailable_with_a_reason(checkpoint, scene, tmp_path):
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    report = service.format_run_report(result)
    assert "Not available" in report
    assert "without a higher-resolution reference" in report


# --------------------------------------------------------------------------- #
# products
# --------------------------------------------------------------------------- #

def test_exported_geotiff_preserves_crs_and_ground_extent(checkpoint, scene, tmp_path):
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    written = service.write_outputs(result, tmp_path / "out")

    with rasterio.open(scene) as a, rasterio.open(written["enhanced_geotiff"]) as b:
        assert a.crs == b.crs
        assert np.allclose(a.bounds, b.bounds, atol=1e-6)
        assert (b.width, b.height) == (a.width * 4, a.height * 4)
        assert b.res[0] == pytest.approx(a.res[0] / 4)
        assert b.count == 4


def test_bicubic_is_exported_separately_and_labelled(checkpoint, scene, tmp_path):
    """The baseline is never substituted for the model; both are written."""
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    written = service.write_outputs(result, tmp_path / "out")
    assert "bicubic" in written["bicubic_geotiff"]
    assert written["bicubic_geotiff"] != written["enhanced_geotiff"]


def test_report_declares_ensemble_unavailable_for_one_checkpoint(
        checkpoint, scene, tmp_path):
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    report = service.format_run_report(result)
    assert "Ensemble uncertainty unavailable" in report


def test_report_warns_that_a_finer_grid_is_not_finer_detail(
        checkpoint, scene, tmp_path):
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    report = service.format_run_report(result)
    assert "not the same thing as finer resolved detail" in report


# --------------------------------------------------------------------------- #
# checkpoint listing and display
# --------------------------------------------------------------------------- #

def test_untrained_checkpoints_are_excluded_from_the_usable_list(tmp_path):
    cfg = {"experiment": "u", "scale": 4, "seed": 1,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 1},
           "data": {"patches": "none"}}
    save_checkpoint(tmp_path / "trained.pt", build_model(cfg), cfg, epoch=4)
    save_checkpoint(tmp_path / "untrained.pt", build_model(cfg), cfg, epoch=0)
    usable = service.trained_checkpoints(service.discover_checkpoints([tmp_path]))
    assert [c["epoch"] for c in usable] == [4]


def test_a_corrupt_checkpoint_is_reported_not_dropped(tmp_path):
    (tmp_path / "broken.pt").write_bytes(b"not a checkpoint")
    found = service.discover_checkpoints([tmp_path])
    assert len(found) == 1 and found[0]["error"]
    assert service.trained_checkpoints(found) == []


def test_display_stretch_is_shared_so_panels_are_comparable():
    """Per-panel stretching would show a difference the model did not make."""
    rng = np.random.default_rng(0)
    a = rng.random((4, 16, 16)).astype(np.float32) * 0.3
    b = a * 1.0
    bounds = service.shared_stretch(a)
    assert np.array_equal(service.to_rgb(a, bounds), service.to_rgb(b, bounds))
    # A brighter image under the same bounds must render brighter, not identical.
    assert service.to_rgb(a + 0.1, bounds).mean() > service.to_rgb(a, bounds).mean()


def test_colorize_marks_invalid_pixels_rather_than_drawing_them_as_zero():
    values = np.full((8, 8), 0.5, dtype=np.float32)
    values[0, 0] = np.nan
    rgb = service.colorize(values, vmax=1.0)
    assert rgb.shape == (8, 8, 3)
    assert not np.array_equal(rgb[0, 0], rgb[4, 4])


def test_exported_band_descriptions_name_the_actual_spectral_bands(
        checkpoint, scene, tmp_path):
    """"band_3" is the ambiguity the export exists to remove.

    The source file names no bands, so its descriptions read band_2/band_3/...
    The checkpoint knows the channels are B02/B03/B04/B08, and that is what a
    downstream reader needs to see.
    """
    result = service.run_super_resolution(
        scene, checkpoint, [2, 3, 4, 8], device="cpu", tile=32, overlap=0)
    assert result.product_band_names == ["B02", "B03", "B04", "B08"]

    written = service.write_outputs(result, tmp_path / "out")
    with rasterio.open(written["enhanced_geotiff"]) as src:
        assert list(src.descriptions) == ["B02", "B03", "B04", "B08"]
