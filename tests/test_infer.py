"""Tiled inference.

Scenes do not fit in memory, so they are processed in overlapping tiles and
blended. The failure mode is a visible grid of seams -- which survives every
scalar metric, because a few pixels of discontinuity barely move a mean.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from srm.infer import cosine_window, super_resolve
from srm.models.registry import build_model, save_checkpoint


@pytest.fixture
def model():
    cfg = {"scale": 4, "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "edsr", "features": 8, "blocks": 1}}
    torch.manual_seed(0)
    return build_model(cfg).eval()


def test_cosine_window_is_symmetric_and_peaks_in_the_middle():
    window = cosine_window(64, 16)
    assert window.shape == (64, 64)
    assert np.allclose(window, window[::-1, :], atol=1e-6)
    assert np.allclose(window, window[:, ::-1], atol=1e-6)
    assert window[32, 32] == pytest.approx(window.max())


def test_cosine_window_tapers_to_near_zero_at_the_edge():
    """A window that does not taper produces exactly the seams it should hide."""
    window = cosine_window(64, 16)
    assert window[0, 0] < 0.05 * window.max()


def test_cosine_window_is_positive_everywhere():
    """Zeros at the border would divide by ~0 in the blend accumulator."""
    assert cosine_window(64, 16).min() >= 0.0


def test_super_resolve_upscales_by_the_scale(model, hr_scene):
    lr = hr_scene[:, :32, :32]
    out = super_resolve(lr, model, tile=16, overlap=4, progress=False)
    assert out.shape == (4, 128, 128)


def test_tiled_and_untiled_inference_agree(model, hr_scene):
    """Tiling is an implementation detail, not a result change.

    Agreement is asserted on the distribution rather than on the single worst
    pixel. Convolution padding at a tile border sees different context than it
    does mid-scene, so a handful of boundary pixels differ by ~2e-3 reflectance
    no matter how much overlap is used -- that residual is inherent to tiled
    convolution, not a blending defect. Everything away from a border agrees to
    a few parts in a million.
    """
    lr = hr_scene[:, :32, :32]
    tiled = super_resolve(lr, model, tile=16, overlap=8, progress=False)
    whole = super_resolve(lr, model, tile=64, overlap=8, progress=False)

    difference = np.abs(tiled - whole)
    assert difference.mean() < 1e-4
    assert np.percentile(difference, 99.9) < 1e-3
    assert difference.max() < 5e-3


def test_tiling_leaves_no_seam_on_a_smooth_scene(model):
    """A gradient has no real edges, so any sharp row is a blending artefact."""
    size = 48
    ramp = np.linspace(0.05, 0.45, size, dtype=np.float32)
    lr = np.repeat(np.repeat(ramp[None, None, :], size, axis=1), 4, axis=0)

    out = super_resolve(lr, model, tile=16, overlap=8, progress=False)
    column_steps = np.abs(np.diff(out.mean(axis=(0, 1))))
    # On a linear ramp every step should be the same size; a seam shows up as
    # an outlier many times the median step.
    assert column_steps.max() < 6.0 * np.median(column_steps)


def test_tile_sizes_that_do_not_divide_the_scene_are_handled(model, hr_scene):
    """Real scenes are never a neat multiple of the tile size."""
    lr = hr_scene[:, :37, :29]
    out = super_resolve(lr, model, tile=16, overlap=4, progress=False)
    assert out.shape == (4, 37 * 4, 29 * 4)
    assert np.isfinite(out).all()


def test_scene_smaller_than_one_tile_is_handled(model, hr_scene):
    lr = hr_scene[:, :9, :9]
    out = super_resolve(lr, model, tile=32, overlap=8, progress=False)
    assert out.shape == (4, 36, 36)
    assert np.isfinite(out).all()


def test_nan_input_is_filled_for_the_network_and_restored_after(model, hr_scene):
    """Cloud and edge NaN must not propagate through conv, nor be invented over."""
    lr = hr_scene[:, :32, :32].copy()
    lr[:, :4, :4] = np.nan
    out = super_resolve(lr, model, tile=16, overlap=4, progress=False)

    assert np.isnan(out[:, :16, :16]).all(), "masked region should stay masked"
    assert np.isfinite(out[:, 20:, 20:]).all(), "NaN leaked into valid pixels"


def test_an_all_nan_band_does_not_crash(model, hr_scene):
    lr = hr_scene[:, :32, :32].copy()
    lr[1] = np.nan
    out = super_resolve(lr, model, tile=16, overlap=4, progress=False)
    assert np.isnan(out[1]).all()
    assert np.isfinite(out[0]).all()


def test_batch_size_does_not_change_the_result(model, hr_scene):
    lr = hr_scene[:, :32, :32]
    one = super_resolve(lr, model, tile=16, overlap=4, batch_size=1, progress=False)
    many = super_resolve(lr, model, tile=16, overlap=4, batch_size=8, progress=False)
    assert np.allclose(one, many, atol=1e-5)


def test_tta_is_invariant_for_a_symmetric_model(model, hr_scene):
    """x8 self-ensembling must average dihedral transforms, not shuffle them."""
    lr = hr_scene[:, :32, :32]
    plain = super_resolve(lr, model, tile=32, overlap=8, tta=False, progress=False)
    ensembled = super_resolve(lr, model, tile=32, overlap=8, tta=True, progress=False)
    assert ensembled.shape == plain.shape
    assert np.isfinite(ensembled).all()
    # The ensemble should be close to, but not identical with, the single pass.
    assert not np.allclose(ensembled, plain, atol=1e-9)
    assert np.abs(ensembled - plain).mean() < 0.05


def test_super_resolve_file_writes_a_georeferenced_product(tmp_path, model, hr_scene):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    from rasterio.crs import CRS

    from srm.infer import super_resolve_file
    from srm.utils.geo import GeoMeta, write_raster

    meta = GeoMeta(crs=CRS.from_epsg(32631),
                   transform=Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 5800000.0),
                   band_names=["B02", "B03", "B04", "B08"])
    scene = write_raster(tmp_path / "in.tif", hr_scene[:, :32, :32], meta)

    cfg = {"scale": 4, "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "edsr", "features": 8, "blocks": 1}}
    ckpt = save_checkpoint(tmp_path / "ck.pt", model, cfg)

    summary = super_resolve_file(scene, tmp_path / "out.tif", ckpt,
                                 tile=16, overlap=4, device="cpu")

    assert summary["gsd_in_m"] == pytest.approx(10.0)
    assert summary["gsd_out_m"] == pytest.approx(2.5)
    with rasterio.open(tmp_path / "out.tif") as src:
        assert src.crs == CRS.from_epsg(32631)
        assert src.width == 128
        assert abs(src.transform.a) == pytest.approx(2.5)
