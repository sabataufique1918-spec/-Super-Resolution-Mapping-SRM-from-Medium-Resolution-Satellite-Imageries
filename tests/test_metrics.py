"""Quality metrics, against cases whose answer is known independently.

These numbers are what every claim in every report rests on, so they are
checked against closed-form values and against each other's orderings rather
than against a previously recorded output.
"""
from __future__ import annotations

import numpy as np
import pytest

from srm.data.degradation import degrade, upsample_bicubic
from srm.metrics.quality import (
    aggregate,
    compute_metrics,
    consistency_error,
    ergas,
    psnr,
    rmse,
    sam,
    scc,
    sharpness,
    sharpness_gain,
    ssim,
)


# --------------------------------------------------------------------------- #
# identity cases: a perfect prediction must score perfectly
# --------------------------------------------------------------------------- #

def test_rmse_of_identical_images_is_zero(hr_scene):
    assert rmse(hr_scene, hr_scene) == pytest.approx(0.0, abs=1e-9)


def test_psnr_of_identical_images_is_infinite(hr_scene):
    assert not np.isfinite(psnr(hr_scene, hr_scene))


def test_ssim_of_identical_images_is_one(hr_scene):
    assert ssim(hr_scene, hr_scene) == pytest.approx(1.0, abs=1e-5)


def test_sam_of_identical_images_is_zero(hr_scene):
    """Zero to within the epsilon guard, not to machine precision.

    `sam` divides by `norm + _EPS`, so identical spectra give a cosine a hair
    under 1, and arccos(1 - d) ~ sqrt(2d) turns that into a small but non-zero
    angle. It is ~0.003 deg here against a ~2 deg threshold of interest, so the
    guard is free in practice -- but it is not exactly zero, and a test that
    demanded that would be wrong rather than strict.
    """
    assert sam(hr_scene, hr_scene) == pytest.approx(0.0, abs=0.01)


def test_ergas_of_identical_images_is_zero(hr_scene):
    assert ergas(hr_scene, hr_scene, scale=4) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# closed-form cases
# --------------------------------------------------------------------------- #

def test_rmse_matches_its_definition():
    target = np.zeros((1, 8, 8), dtype=np.float32)
    pred = np.full((1, 8, 8), 0.25, dtype=np.float32)
    assert rmse(pred, target) == pytest.approx(0.25)


def test_psnr_matches_its_definition():
    """PSNR = 20 log10(range / rmse); a 0.1 error at range 1.0 is exactly 20 dB."""
    target = np.zeros((1, 16, 16), dtype=np.float32)
    pred = np.full((1, 16, 16), 0.1, dtype=np.float32)
    assert psnr(pred, target, data_range=1.0) == pytest.approx(20.0, abs=1e-4)


def test_sam_is_scale_invariant():
    """Spectral angle measures shape, not brightness -- that is its whole point."""
    target = np.random.default_rng(0).random((4, 16, 16)).astype(np.float32) + 0.1
    assert sam(target * 2.0, target) == pytest.approx(0.0, abs=1e-3)


def test_sam_detects_a_spectral_shape_change():
    target = np.ones((2, 8, 8), dtype=np.float32)
    pred = target.copy()
    pred[1] = 0.0  # 45 degrees away from (1, 1)
    assert sam(pred, target) == pytest.approx(45.0, abs=1e-3)


def test_scc_is_one_for_a_perfect_prediction(hr_scene):
    assert scc(hr_scene, hr_scene) == pytest.approx(1.0, abs=1e-6)


def test_scc_falls_when_detail_is_uncorrelated(hr_scene, rng):
    """The metric that catches hallucination must actually catch it."""
    noise = rng.standard_normal(hr_scene.shape).astype(np.float32) * 0.05
    assert scc(hr_scene + noise, hr_scene) < scc(hr_scene, hr_scene)


# --------------------------------------------------------------------------- #
# orderings: a worse image must score worse
# --------------------------------------------------------------------------- #

def test_metrics_degrade_monotonically_with_added_noise(hr_scene, rng):
    clean = hr_scene
    mild = clean + 0.01 * rng.standard_normal(clean.shape).astype(np.float32)
    severe = clean + 0.05 * rng.standard_normal(clean.shape).astype(np.float32)

    assert psnr(mild, clean) > psnr(severe, clean)
    assert ssim(mild, clean) > ssim(severe, clean)
    assert rmse(mild, clean) < rmse(severe, clean)
    assert sam(mild, clean) < sam(severe, clean)
    assert ergas(mild, clean, 4) < ergas(severe, clean, 4)


def test_blur_costs_ssim_more_than_it_costs_psnr(hr_scene, degradation):
    """SSIM is structural; that is why PSNR alone is never sufficient."""
    blurred = upsample_bicubic(degrade(hr_scene, degradation), 4)
    assert ssim(blurred, hr_scene) < 1.0
    assert psnr(blurred, hr_scene) < psnr(hr_scene, hr_scene)


def test_sharpness_gain_is_one_against_itself(hr_scene):
    assert sharpness_gain(hr_scene, hr_scene) == pytest.approx(1.0, rel=1e-6)


def test_sharpness_gain_exceeds_one_for_a_sharper_image(hr_scene, degradation):
    blurred = upsample_bicubic(degrade(hr_scene, degradation), 4)
    assert sharpness(hr_scene) > sharpness(blurred)
    assert sharpness_gain(hr_scene, blurred) > 1.0


# --------------------------------------------------------------------------- #
# the Wald consistency check
# --------------------------------------------------------------------------- #

def test_consistency_is_near_zero_for_a_physically_consistent_result(hr_scene, degradation):
    """degrade(truth) reproduces the observation exactly, by construction."""
    lr = degrade(hr_scene, degradation)
    result = consistency_error(hr_scene, lr, degradation)
    assert result["consistency_rmse"] == pytest.approx(0.0, abs=1e-5)


def test_consistency_penalises_an_inconsistent_result(hr_scene, degradation, rng):
    """An output that cannot reproduce its own input is the failure this catches."""
    lr = degrade(hr_scene, degradation)
    tampered = hr_scene + 0.05 * rng.standard_normal(hr_scene.shape).astype(np.float32)
    honest = consistency_error(hr_scene, lr, degradation)
    dishonest = consistency_error(tampered, lr, degradation)
    assert dishonest["consistency_rmse"] > honest["consistency_rmse"]


# --------------------------------------------------------------------------- #
# the aggregation layer
# --------------------------------------------------------------------------- #

def test_compute_metrics_returns_every_requested_name(hr_scene, degradation):
    lr = degrade(hr_scene, degradation)
    sr = upsample_bicubic(lr, 4)
    names = ["psnr", "ssim", "sam", "ergas", "scc", "rmse"]
    result = compute_metrics(sr, hr_scene, names, data_range=1.0, scale=4)
    assert set(names).issubset(result)
    assert all(np.isfinite(result[n]) for n in names)


def test_aggregate_reports_spread_not_just_a_mean():
    rows = [{"psnr": 30.0}, {"psnr": 34.0}, {"psnr": 38.0}]
    summary = aggregate(rows)
    assert summary["psnr"]["mean"] == pytest.approx(34.0)
    assert summary["psnr"]["median"] == pytest.approx(34.0)
    assert summary["psnr"]["std"] > 0


def test_metrics_ignore_nan_padding(hr_scene):
    """Scene edges and cloud masks arrive as NaN; they must not poison a score."""
    target = hr_scene.copy()
    pred = hr_scene.copy()
    pred[:, :4, :] = np.nan
    target[:, :4, :] = np.nan
    assert rmse(pred, target) == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(sam(pred, target))
