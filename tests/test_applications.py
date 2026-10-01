"""Downstream applications: indices, land cover, change detection.

These are what the product is *for*, so their invariants matter as much as the
reconstruction metrics. A super-resolved scene that scores well on PSNR but
shifts NDVI has failed at its actual job.
"""
from __future__ import annotations

import numpy as np
import pytest

from srm.applications import change_detection as cd
from srm.applications import indices, landcover
from srm.applications.indices import (
    GREEN,
    NIR,
    RED,
    compute_all,
    compute_index,
    mixed_pixel_fraction,
    normalised_difference,
)


# --------------------------------------------------------------------------- #
# spectral indices
# --------------------------------------------------------------------------- #

def test_ndvi_is_high_for_vegetation_and_low_for_water():
    """The sign convention, pinned. Getting this backwards is silent and fatal."""
    vegetation = np.zeros((4, 4, 4), dtype=np.float32)
    vegetation[RED], vegetation[NIR] = 0.04, 0.45

    water = np.zeros((4, 4, 4), dtype=np.float32)
    water[RED], water[NIR] = 0.05, 0.02

    assert compute_index(vegetation, "ndvi").mean() > 0.6
    assert compute_index(water, "ndvi").mean() < 0.0


def test_ndwi_is_high_for_water():
    water = np.zeros((4, 8, 8), dtype=np.float32)
    water[GREEN], water[NIR] = 0.09, 0.02
    assert compute_index(water, "ndwi").mean() > 0.5


def test_index_is_bounded():
    """A normalised difference cannot leave [-1, 1], including on zero pixels."""
    image = np.zeros((4, 16, 16), dtype=np.float32)
    image[:, :8] = np.random.default_rng(0).random((4, 8, 16)).astype(np.float32)
    for name in compute_all(image):
        values = compute_index(image, name)
        assert values.min() >= -1.0 and values.max() <= 1.0
        assert np.isfinite(values).all()


def test_index_of_an_all_zero_pixel_does_not_divide_by_zero():
    black = np.zeros((4, 4, 4), dtype=np.float32)
    assert np.isfinite(compute_index(black, "ndvi")).all()


def test_normalised_difference_matches_its_definition():
    image = np.zeros((4, 1, 1), dtype=np.float32)
    image[NIR], image[RED] = 0.6, 0.2
    assert normalised_difference(image, NIR, RED)[0, 0] == pytest.approx(0.5, abs=1e-5)


def test_unknown_index_fails_loudly():
    with pytest.raises(KeyError, match="unknown index"):
        compute_index(np.zeros((4, 4, 4), dtype=np.float32), "ndsi")


def test_mixed_pixel_fraction_counts_the_band_between_classes():
    values = np.array([0.0, 0.2, 0.25, 0.35, 0.9], dtype=np.float32)
    assert mixed_pixel_fraction(values, 0.10, 0.40) == pytest.approx(3 / 5)


def test_mixed_pixel_fraction_ignores_nan():
    values = np.array([0.2, np.nan, 0.9], dtype=np.float32)
    assert mixed_pixel_fraction(values, 0.10, 0.40) == pytest.approx(0.5)


def test_coarser_imagery_has_more_mixed_pixels(hr_scene, degradation):
    """The premise of the entire project, stated as a test.

    Averaging over a larger footprint pushes pixels towards the middle of the
    index range, because each one now contains several classes.
    """
    from srm.data.degradation import degrade, upsample_bicubic

    fine_ndvi = compute_index(hr_scene, "ndvi")
    coarse_ndvi = compute_index(upsample_bicubic(degrade(hr_scene, degradation), 4), "ndvi")

    low, high = np.quantile(fine_ndvi, [0.35, 0.65])
    assert mixed_pixel_fraction(coarse_ndvi, low, high) > mixed_pixel_fraction(fine_ndvi, low, high)


# --------------------------------------------------------------------------- #
# land cover
# --------------------------------------------------------------------------- #

def test_spatial_split_is_disjoint():
    """Random pixel splits leak neighbours across the boundary and inflate accuracy."""
    train, test = landcover.spatial_split((64, 64), axis=1, train_fraction=0.5)
    assert not (train & test).any()
    assert (train | test).all()


def test_classification_recovers_separable_classes():
    """A sanity floor: clean, well-separated spectra must classify near-perfectly.

    The classes vary along rows while `spatial_split` divides along columns, so
    both halves contain both classes. Laying the classes out along the split
    axis instead would train on one class and test on the other, which scores
    zero however good the classifier is.
    """
    size = 64
    image = np.zeros((4, size, size), dtype=np.float32)
    labels = np.zeros((size, size), dtype=np.uint8)

    image[:, : size // 2, :] = np.array([0.03, 0.05, 0.04, 0.42])[:, None, None]  # vegetation
    labels[: size // 2, :] = 2
    image[:, size // 2 :, :] = np.array([0.07, 0.09, 0.06, 0.02])[:, None, None]  # water
    labels[size // 2 :, :] = 1
    image += 0.002 * np.random.default_rng(0).standard_normal(image.shape).astype(np.float32)

    result = landcover.classify(image, labels, class_names=["other", "water", "vegetation"])
    assert result.overall_accuracy > 0.95
    assert result.kappa > 0.9


def test_classification_of_noise_is_near_chance():
    """The test must be able to fail; an always-high score would prove nothing."""
    rng = np.random.default_rng(0)
    image = rng.random((4, 64, 64)).astype(np.float32)
    labels = rng.integers(0, 2, (64, 64)).astype(np.uint8)
    result = landcover.classify(image, labels, class_names=["a", "b"])
    assert result.overall_accuracy < 0.75


def test_build_features_adds_texture_channels():
    image = np.zeros((4, 32, 32), dtype=np.float32)
    plain = landcover.build_features(image, with_texture=False)
    textured = landcover.build_features(image, with_texture=True)
    assert textured.shape[-1] > plain.shape[-1]


# --------------------------------------------------------------------------- #
# change detection
# --------------------------------------------------------------------------- #

def test_no_change_between_identical_scenes(hr_scene):
    """The false-positive floor. A detector that fires here is unusable."""
    mask, _ = cd.detect(hr_scene, hr_scene.copy())
    assert mask.mean() < 0.02


def test_a_real_change_is_detected(hr_scene):
    t2 = hr_scene.copy()
    t2[:, 40:60, 40:60] = np.array([0.30, 0.30, 0.30, 0.30])[:, None, None]

    mask, _ = cd.detect(hr_scene, t2)
    truth = np.zeros(hr_scene.shape[1:], dtype=bool)
    truth[40:60, 40:60] = True

    result = cd.score(mask, truth)
    assert result.recall > 0.8
    assert result.precision > 0.5


def test_change_vector_magnitude_is_zero_for_no_change(hr_scene):
    assert cd.change_vector_magnitude(hr_scene, hr_scene.copy()).max() == pytest.approx(0.0, abs=1e-5)


def test_otsu_separates_a_bimodal_distribution():
    """What matters is the partition, not where in the gap the cut lands.

    Every threshold between two well-separated clusters yields the identical
    partition and therefore the identical between-class variance, so the
    argmax is tied across the whole gap and lands on its lower edge. Asserting
    a specific position would be testing the tie-break, not the algorithm.
    """
    low = np.random.default_rng(0).normal(0.1, 0.02, 500)
    high = np.random.default_rng(1).normal(0.9, 0.02, 500)
    threshold = cd.otsu_threshold(np.concatenate([low, high]))

    assert low.min() < threshold < high.max()
    correct = (low < threshold).mean() + (high > threshold).mean()
    assert correct / 2 > 0.99


def test_speckle_below_the_minimum_object_size_is_removed():
    """Single-pixel noise is not change, and would otherwise dominate the count."""
    t1 = np.full((4, 64, 64), 0.2, dtype=np.float32)
    t2 = t1.copy()
    rng = np.random.default_rng(0)
    for _ in range(30):                               # isolated single pixels
        t2[:, rng.integers(0, 64), rng.integers(0, 64)] = 0.9
    t2[:, 10:20, 10:20] = 0.9                         # one genuine block

    mask, _ = cd.detect(t1, t2, min_object_px=4)
    from scipy import ndimage

    _, n_objects = ndimage.label(mask)
    assert n_objects <= 3, "speckle survived the minimum-object filter"


def test_score_is_perfect_against_its_own_truth():
    truth = np.zeros((32, 32), dtype=bool)
    truth[8:16, 8:16] = True
    result = cd.score(truth.copy(), truth)
    assert result.f1 == pytest.approx(1.0)
    assert result.iou == pytest.approx(1.0)


def test_score_of_an_empty_prediction_is_zero_not_nan():
    truth = np.zeros((32, 32), dtype=bool)
    truth[8:16, 8:16] = True
    result = cd.score(np.zeros((32, 32), dtype=bool), truth)
    assert result.recall == pytest.approx(0.0)
    assert np.isfinite(result.f1)
