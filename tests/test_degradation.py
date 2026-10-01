"""The forward observation model.

This is the load-bearing assumption of the whole project: training pairs are
manufactured by degrading, so if `degrade` is wrong, every metric in every
report is measuring the wrong thing while looking perfectly healthy.
"""
from __future__ import annotations

import numpy as np
import pytest

from srm import S2_MTF_NYQUIST
from srm.data.degradation import (
    DegradationConfig,
    area_decimate,
    band_sigmas,
    blur,
    degrade,
    gaussian_kernel_2d,
    mtf_to_sigma,
    upsample_bicubic,
)


def test_mtf_to_sigma_inverts_the_gaussian_mtf():
    """sigma must reproduce the requested MTF at the coarse grid's Nyquist."""
    for mtf in (0.15, 0.30, 0.50):
        sigma_coarse = mtf_to_sigma(mtf, scale=1)
        recovered = np.exp(-2.0 * np.pi**2 * sigma_coarse**2 * 0.5**2)
        assert recovered == pytest.approx(mtf, rel=1e-6)


def test_mtf_to_sigma_scales_to_the_fine_grid():
    assert mtf_to_sigma(0.30, scale=4) == pytest.approx(4 * mtf_to_sigma(0.30, scale=1))


def test_lower_mtf_means_more_blur():
    """A sensor that passes less at Nyquist has a wider PSF, not a narrower one."""
    assert mtf_to_sigma(0.20) > mtf_to_sigma(0.40)


def test_band_sigmas_follow_the_published_mtf_ordering():
    """B08 is the blurriest of the 10 m bands; the degradation must reflect that."""
    bands = ("B02", "B03", "B04", "B08")
    sigmas = dict(zip(bands, band_sigmas(bands, scale=4)))
    assert sigmas["B08"] == max(sigmas.values())
    assert sigmas["B02"] == min(sigmas.values())
    # and the ordering is exactly the inverse of the MTF table
    by_sigma = sorted(bands, key=lambda b: sigmas[b])
    by_mtf = sorted(bands, key=lambda b: -S2_MTF_NYQUIST[b])
    assert by_sigma == by_mtf


def test_gaussian_kernel_is_normalised_and_symmetric():
    kernel = gaussian_kernel_2d(1.5, size=9)
    assert kernel.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(kernel, kernel.T, atol=1e-12)
    assert np.allclose(kernel, kernel[::-1, :], atol=1e-12)


def test_blur_preserves_the_mean():
    """A normalised kernel with reflect padding must not change scene radiance."""
    scene = np.random.default_rng(0).random((4, 64, 64)).astype(np.float32)
    kernels = [gaussian_kernel_2d(1.2, size=9)] * 4
    assert blur(scene, kernels).mean() == pytest.approx(scene.mean(), rel=1e-4)


def test_area_decimate_is_the_detector_footprint_integral(hr_scene):
    """Block averaging, exactly -- not subsampling."""
    lr = area_decimate(hr_scene, 4)
    assert lr.shape == (4, 32, 32)
    assert lr[0, 0, 0] == pytest.approx(hr_scene[0, :4, :4].mean(), rel=1e-6)
    assert lr.mean() == pytest.approx(hr_scene.mean(), rel=1e-4)


def test_area_decimate_crops_rather_than_pads_a_ragged_scene():
    """A scene that is not a multiple of the scale must not invent edge data."""
    scene = np.ones((2, 10, 10), dtype=np.float32)
    assert area_decimate(scene, 4).shape == (2, 2, 2)


def test_degrade_reduces_resolution_by_exactly_the_scale(hr_scene, degradation):
    lr = degrade(hr_scene, degradation)
    assert lr.shape == (4, 32, 32)
    assert lr.dtype == np.float32


def test_degrade_is_radiometrically_unbiased(hr_scene, degradation):
    """Noise-free degradation must not shift the mean reflectance."""
    lr = degrade(hr_scene, degradation)
    assert lr.mean() == pytest.approx(hr_scene.mean(), abs=2e-3)


def test_degrade_removes_high_frequency_energy(hr_scene, degradation):
    """The point of a PSF: the observation is smoother than the scene."""
    lr = degrade(hr_scene, degradation)
    lr_detail = np.abs(np.diff(upsample_bicubic(lr, 4), axis=-1)).mean()
    hr_detail = np.abs(np.diff(hr_scene, axis=-1)).mean()
    assert lr_detail < hr_detail


def test_degrade_is_deterministic_given_a_seed(hr_scene):
    cfg = DegradationConfig(scale=4, noise_sigma=0.01, shot_noise=True)
    a = degrade(hr_scene, cfg, np.random.default_rng(7))
    b = degrade(hr_scene, cfg, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_noise_actually_perturbs_the_observation(hr_scene):
    """A noise_sigma that changes nothing would silently disable the term."""
    clean = degrade(hr_scene, DegradationConfig(scale=4, noise_sigma=0.0))
    noisy = degrade(hr_scene, DegradationConfig(scale=4, noise_sigma=0.02, shot_noise=False),
                    np.random.default_rng(3))
    assert not np.allclose(clean, noisy)
    assert np.std(noisy - clean) == pytest.approx(0.02, rel=0.35)


def test_degrade_never_emits_negative_reflectance(hr_scene):
    """Noise on dark water is the case that produces negatives if unclipped."""
    dark = np.full_like(hr_scene, 0.003)
    lr = degrade(dark, DegradationConfig(scale=4, noise_sigma=0.05, shot_noise=True),
                 np.random.default_rng(11))
    assert lr.min() >= 0.0


def test_per_band_mtf_blurs_bands_differently(hr_scene):
    """With per_band_mtf on, B08 must come out smoother than B02.

    Every band is given identical content here on purpose: the bands of a real
    scene differ in contrast, so comparing their gradients directly would
    measure the imagery rather than the PSF.
    """
    identical = np.repeat(hr_scene[:1], 4, axis=0)
    cfg = DegradationConfig(scale=4, noise_sigma=0.0, per_band_mtf=True,
                            bands=("B02", "B03", "B04", "B08"))
    lr = degrade(identical, cfg)
    detail = [float(np.abs(np.diff(lr[i], axis=-1)).mean()) for i in range(4)]
    assert detail[3] < detail[0], "B08 should be the blurriest of the 10 m bands"
    assert detail == sorted(detail, reverse=True), "detail must fall as MTF falls"


def test_per_band_mtf_off_treats_every_band_alike(hr_scene):
    """The flag has to actually do something, in both positions."""
    identical = np.repeat(hr_scene[:1], 4, axis=0)
    cfg = DegradationConfig(scale=4, noise_sigma=0.0, per_band_mtf=False,
                            bands=("B02", "B03", "B04", "B08"))
    lr = degrade(identical, cfg)
    assert np.allclose(lr[0], lr[3], atol=1e-6)


def test_upsample_bicubic_restores_the_shape_and_mean(hr_scene, degradation):
    lr = degrade(hr_scene, degradation)
    up = upsample_bicubic(lr, 4)
    assert up.shape == hr_scene.shape
    assert up.mean() == pytest.approx(lr.mean(), rel=1e-2)


def test_config_from_dict_ignores_unknown_keys():
    """YAML grows keys over time; an unknown one must not crash training."""
    cfg = DegradationConfig.from_dict(
        {"mode": "blind", "noise_sigma": 0.01, "some_future_option": True},
        scale=4, bands=("B02", "B03"),
    )
    assert cfg.mode == "blind"
    assert cfg.noise_sigma == 0.01
    assert cfg.bands == ("B02", "B03")
    assert cfg.scale == 4
