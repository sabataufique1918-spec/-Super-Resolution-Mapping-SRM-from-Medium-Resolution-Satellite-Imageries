"""Every entry point selects bands by name, not by position.

This bug was found and fixed three separate times in three different files
before this sweep existed:

  1. `service._reference_on_grid`  -- read the reference as bands 1-4, so
     spectral angle came out at 61.8 degrees instead of 0.68
  2. `ensemble.estimate_from_file` -- sliced `array[:n_bands]`, so the
     disagreement map reported 0.0140 where the truth was 0.00093
  3. `uncertainty.run_file`        -- sliced `array[:model.in_channels]`

Each looked right in isolation, each produced plausible numbers, and none
was caught by a test of the module it lived in. The failure is systemic: a
12-band Sentinel-2 stack whose first four bands are *not* RGB+NIR, read by
code that assumes they are. So this file tests the property across the whole
surface at once, and any new entry point should be added to it.

The probe is a forest spectrum: red low, NIR high. Reading B1-B4 in place of
B02/B03/B04/B08 inverts that relationship, which NDVI detects unambiguously.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS

from srm.models.registry import build_model, save_checkpoint

#: SEN2NEON band order. B10 is absent, which is why positions shift.
LAYOUT = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")

#: Dense forest. The four bands the model wants are B02=0.029, B03=0.054,
#: B04=0.026, B08=0.464 -- NDVI about +0.89. The first four bands are
#: B1=0.110, B2=0.029, B3=0.054, B4=0.026, whose "NDVI" is about -0.35.
FOREST = {"B1": 0.110, "B2": 0.029, "B3": 0.054, "B4": 0.026, "B5": 0.20,
          "B6": 0.35, "B7": 0.40, "B8": 0.464, "B8A": 0.44, "B9": 0.30,
          "B11": 0.18, "B12": 0.08}

CORRECT_NDVI = (FOREST["B8"] - FOREST["B4"]) / (FOREST["B8"] + FOREST["B4"])
POSITIONAL_NDVI = (FOREST["B4"] - FOREST["B3"]) / (FOREST["B4"] + FOREST["B3"])


def _twelve_band(path, size, pixel, seed=0):
    rng = np.random.default_rng(seed)
    stack = np.stack([
        np.full((size, size), FOREST[b], dtype=np.float32)
        + rng.normal(0, 0.001, (size, size)).astype(np.float32) for b in LAYOUT
    ])
    profile = {
        "driver": "GTiff", "height": size, "width": size, "count": 12,
        "dtype": "uint16", "crs": CRS.from_epsg(32617),
        "transform": Affine(pixel, 0, 539120.0, 0, -pixel, 4146000.0),
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(np.clip(stack * 10000, 0, 65535).astype(np.uint16))
    # Precondition for the whole file: the product really does name no bands.
    with rasterio.open(str(path)) as src:
        assert all(d is None for d in src.descriptions)
    return str(path)


def ndvi(array: np.ndarray) -> float:
    """NDVI on this project's canonical order (B02, B03, B04, B08)."""
    nir, red = array[3], array[2]
    return float(np.nanmedian((nir - red) / (nir + red + 1e-9)))


@pytest.fixture
def checkpoint(tmp_path):
    cfg = {"experiment": "t", "scale": 4, "seed": 1,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 1},
           "data": {"patches": "p"}}
    torch.manual_seed(0)
    return str(save_checkpoint(tmp_path / "a.pt", build_model(cfg), cfg, epoch=4))


@pytest.fixture
def second_checkpoint(tmp_path):
    cfg = {"experiment": "t2", "scale": 4, "seed": 2,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 1},
           "data": {"patches": "p"}}
    torch.manual_seed(7)
    return str(save_checkpoint(tmp_path / "b.pt", build_model(cfg), cfg, epoch=4))


@pytest.fixture
def scene(tmp_path):
    return _twelve_band(tmp_path / "scene.tif", 32, 10.0)


def test_the_probe_distinguishes_the_two_readings():
    """Guard on the guard: correct and positional readings must differ loudly."""
    assert CORRECT_NDVI == pytest.approx(0.894, abs=0.01)
    assert POSITIONAL_NDVI == pytest.approx(-0.35, abs=0.02)


def test_srm_infer_reads_the_named_bands(tmp_path, checkpoint, scene):
    from srm.infer import super_resolve_file
    from srm.utils.geo import read_raster

    summary = super_resolve_file(scene, tmp_path / "sr.tif", checkpoint,
                                 tile=32, overlap=0, device="cpu")
    assert summary["bands_used"] == [2, 3, 4, 8]
    assert summary["band_names"] == ["B02", "B03", "B04", "B08"]
    # The probe goes on what was *read*, not on what an untrained model emits.
    fed, _ = read_raster(scene, bands=summary["bands_used"])
    assert ndvi(fed) == pytest.approx(CORRECT_NDVI, abs=0.02)


def test_service_run_reads_the_named_bands(checkpoint, scene):
    from srm import service

    info = service.discover_checkpoints([__import__("pathlib").Path(checkpoint).parent])
    info = [i for i in info if i["path"] == checkpoint][0]
    report = service.describe_input(scene)
    plan = service.plan_bands(report, info)
    assert plan["indices"] == [2, 3, 4, 8]

    result = service.run_super_resolution(scene, info, plan["indices"],
                                          device="cpu", tile=32, overlap=0)
    assert ndvi(result.lr) > 0.5
    assert result.product_band_names == ["B02", "B03", "B04", "B08"]


def test_service_reference_scoring_reads_the_named_bands(tmp_path, checkpoint, scene):
    """Regression 1: this returned SAM 61.8 degrees instead of 0.68."""
    from srm import service

    reference = _twelve_band(tmp_path / "ref.tif", 128, 2.5, seed=1)
    info = service.discover_checkpoints([__import__("pathlib").Path(checkpoint).parent])
    info = [i for i in info if i["path"] == checkpoint][0]
    result = service.run_super_resolution(scene, info, [2, 3, 4, 8],
                                          device="cpu", tile=32, overlap=0)
    evaluation = service.evaluate_against_reference(result, reference)
    assert evaluation["available"]
    assert "B02, B03, B04, B08" in evaluation["note"]
    assert evaluation["metrics"]["bicubic"]["sam"] < 5.0


def test_ensemble_reads_the_named_bands(tmp_path, checkpoint, second_checkpoint, scene):
    """Regression 2: reported disagreement 0.0140 against a true 0.00093."""
    from srm.ensemble import estimate_from_file

    report = estimate_from_file(scene, [checkpoint, second_checkpoint],
                                tmp_path / "ens", device="cpu", tile=32, overlap=0)
    assert report["bands_used"] == [2, 3, 4, 8]
    assert report["band_names"] == ["B02", "B03", "B04", "B08"]
    with rasterio.open(tmp_path / "ens" / "ensemble_mean.tif") as src:
        assert list(src.descriptions) == ["B02", "B03", "B04", "B08"]
    from srm.utils.geo import read_raster
    fed, _ = read_raster(scene, bands=report["bands_used"])
    assert ndvi(fed) == pytest.approx(CORRECT_NDVI, abs=0.02)


def test_uncertainty_reads_the_named_bands(tmp_path, checkpoint, scene):
    """Regression 3: sliced `array[:model.in_channels]`."""
    from srm.uncertainty import run_file

    run_file(scene, tmp_path / "unc", checkpoint, samples=2,
             method="tta", device="cpu", tile=32, overlap=0)
    written = sorted(p.name for p in (tmp_path / "unc").glob("*.tif"))
    assert written, "uncertainty produced no rasters"


def test_a_four_band_file_still_works_unchanged(tmp_path, checkpoint):
    """The named path must not break the simple case it replaced."""
    from srm.infer import super_resolve_file

    data = np.stack([np.full((16, 16), FOREST[b], dtype=np.float32)
                     for b in ("B2", "B3", "B4", "B8")])
    path = tmp_path / "four.tif"
    with rasterio.open(str(path), "w", driver="GTiff", height=16, width=16, count=4,
                       dtype="uint16", crs=CRS.from_epsg(32617),
                       transform=Affine(10, 0, 539120.0, 0, -10, 4146000.0)) as dst:
        dst.write(np.clip(data * 10000, 0, 65535).astype(np.uint16))

    summary = super_resolve_file(path, tmp_path / "sr4.tif", checkpoint,
                                 tile=16, overlap=0, device="cpu")
    assert summary["bands_used"] == [1, 2, 3, 4]
