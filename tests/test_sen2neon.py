"""SEN2NEON pairing, band mapping and the geographic split.

These are the checks that catch the failures which produce a model that looks
trained and is wrong: the wrong four bands, a split that leaks, a mask that
does not follow its imagery, and a pair that is not actually aligned.
"""
from __future__ import annotations

import numpy as np
import pytest

from srm.data.sen2neon import (
    RGB_NIR_BANDS,
    SEN2NEON_BAND_ORDER,
    IndexFilter,
    PairPatchSpec,
    TilePair,
    band_indices,
    geographic_split,
    iter_pair_patches,
    normalise_band,
    split_report,
    stratified_subset,
)


# --------------------------------------------------------------------------- #
# band mapping -- the "not the first four bands" trap
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("B4", "B04"), ("B04", "B04"), ("b8", "B08"), ("B8A", "B8A"),
    ("b8a", "B8A"), ("B11", "B11"), (" b2 ", "B02"),
])
def test_band_names_normalise_across_both_conventions(raw, expected):
    assert normalise_band(raw) == expected


def test_rgb_nir_maps_to_bands_4_3_2_8_not_1_2_3_4():
    """The single most damaging silent error this dataset invites.

    SEN2NEON stores B1..B12 with B10 absent. Reading the first four bands
    yields coastal-aerosol/blue/green/red and calls it blue/green/red/NIR.
    """
    assert band_indices(("B04", "B03", "B02", "B08")) == (4, 3, 2, 8)
    assert band_indices(RGB_NIR_BANDS) == (2, 3, 4, 8)
    assert band_indices(RGB_NIR_BANDS) != (1, 2, 3, 4)


def test_b10_is_absent_so_positional_indexing_shifts():
    """B11 sits at position 11, not 11-minus-nothing; names are the only safe key."""
    assert "B10" not in SEN2NEON_BAND_ORDER
    assert band_indices(("B11",)) == (11,)
    assert band_indices(("B12",)) == (12,)


def test_requesting_an_absent_band_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="B10"):
        band_indices(("B10",))


# --------------------------------------------------------------------------- #
# geographic split
# --------------------------------------------------------------------------- #

def _pairs(n_sites: int = 20, per_site: int = 5) -> list[TilePair]:
    out = []
    for s in range(n_sites):
        site = f"S{s:02d}"
        for t in range(per_site):
            out.append(TilePair(
                tile_id=f"2018_{site}_1__{t}_0", lr_path=f"lr/{site}_{t}.tif",
                hr_path=f"hr/{site}_{t}.tif", site=site, mosaic=f"2018_{site}_1",
                year=2018, lon=0.0, lat=0.0, crs="EPSG:32617",
                land_cover="Forest" if s % 2 else "Rural",
                cloud_score=0.93, hr_nodata_pct=1.0, temporal_days=10,
            ))
    return out


def test_no_site_appears_in_two_splits():
    """The leakage check. A shared site means val patches sit beside train ones."""
    splits = geographic_split(_pairs(), (0.70, 0.15, 0.15), seed=7)
    report = split_report(splits)
    assert report["leakage_free"], report["shared_sites"]
    for overlap in report["shared_sites"].values():
        assert overlap == []


def test_every_tile_lands_in_exactly_one_split():
    pairs = _pairs()
    splits = geographic_split(pairs, seed=3)
    seen = [p.tile_id for items in splits.values() for p in items]
    assert sorted(seen) == sorted(p.tile_id for p in pairs)
    assert len(seen) == len(set(seen))


def test_split_is_close_to_the_requested_fractions():
    splits = geographic_split(_pairs(40, 5), (0.70, 0.15, 0.15), seed=11)
    fractions = split_report(splits)["fractions"]
    assert fractions["train"] == pytest.approx(0.70, abs=0.08)
    assert fractions["val"] == pytest.approx(0.15, abs=0.08)
    assert fractions["test"] == pytest.approx(0.15, abs=0.08)


def test_tiles_of_one_mosaic_stay_together():
    """Adjacent crops share a border; separating them leaks context."""
    splits = geographic_split(_pairs(12, 8), seed=5)
    location = {}
    for split, items in splits.items():
        for pair in items:
            location.setdefault(pair.mosaic, set()).add(split)
    assert all(len(v) == 1 for v in location.values())


def test_split_is_deterministic_for_a_seed():
    a = geographic_split(_pairs(), seed=99)
    b = geographic_split(_pairs(), seed=99)
    assert {k: [p.tile_id for p in v] for k, v in a.items()} == \
           {k: [p.tile_id for p in v] for k, v in b.items()}


def test_too_few_groups_fails_loudly_instead_of_emptying_a_split():
    with pytest.raises(ValueError, match="geographic groups"):
        geographic_split(_pairs(n_sites=1, per_site=40))


def test_stratified_subset_spreads_over_sites():
    """Taking the first N rows would collect three sites alphabetically."""
    chosen = stratified_subset(_pairs(30, 10), limit=30, seed=1)
    assert len(chosen) == 30
    assert len({p.site for p in chosen}) == 30


def test_index_filter_rejects_cloud_nodata_and_stale_pairs():
    base = dict(tile_id="t", lr_path="a", hr_path="b", site="X", mosaic="m",
                year=2018, lon=0.0, lat=0.0, crs="EPSG:32617", land_cover="Forest")
    f = IndexFilter(min_cloud_score=0.90, max_hr_nodata_pct=20.0, max_temporal_days=40)
    assert f.accepts(TilePair(**base, cloud_score=0.95, hr_nodata_pct=2.0, temporal_days=5))
    assert not f.accepts(TilePair(**base, cloud_score=0.80, hr_nodata_pct=2.0, temporal_days=5))
    assert not f.accepts(TilePair(**base, cloud_score=0.95, hr_nodata_pct=45.0, temporal_days=5))
    assert not f.accepts(TilePair(**base, cloud_score=0.95, hr_nodata_pct=2.0, temporal_days=99))


# --------------------------------------------------------------------------- #
# patch pairing and masking
# --------------------------------------------------------------------------- #

def _synthetic_pair(size: int = 128, scale: int = 4, invalid_corner: int = 0) -> dict:
    rng = np.random.default_rng(0)
    hr = rng.random((4, size * scale, size * scale)).astype(np.float32) * 0.4 + 0.05
    lr = hr.reshape(4, size, scale, size, scale).mean(axis=(2, 4)).astype(np.float32)
    hr_valid = np.ones(hr.shape[-2:], dtype=bool)
    if invalid_corner:
        hr_valid[:invalid_corner, :invalid_corner] = False
        hr = np.where(hr_valid[None], hr, np.nan)
    return {"lr": lr, "hr": hr, "lr_valid": np.ones(lr.shape[-2:], dtype=bool),
            "hr_valid": hr_valid}


def test_patch_pairs_are_the_same_ground():
    """An LR offset of y corresponds to HR y*scale, and must stay that way."""
    spec = PairPatchSpec(lr_patch=32, scale=4, min_std=0.0)
    data = _synthetic_pair(size=128)
    rng = np.random.default_rng(0)
    patches = list(iter_pair_patches(data, spec, rng))
    assert patches
    for patch in patches:
        assert patch["lr"].shape == (4, 32, 32)
        assert patch["hr"].shape == (4, 128, 128)
        # The LR window is the block mean of its HR window by construction.
        block_mean = patch["hr"].reshape(4, 32, 4, 32, 4).mean(axis=(2, 4))
        assert np.allclose(block_mean, patch["lr"], atol=1e-5)


def test_patches_mostly_covered_by_nodata_are_rejected():
    spec = PairPatchSpec(lr_patch=32, scale=4, min_std=0.0, min_valid_fraction=0.9)
    # Invalidate the top-left 256x256 of a 512x512 HR raster: the first patch
    # window is entirely inside it.
    data = _synthetic_pair(size=128, invalid_corner=256)
    patches = list(iter_pair_patches(data, spec, np.random.default_rng(0)))
    for patch in patches:
        assert patch["mask"].mean() >= 0.9
    assert all(np.isfinite(p["hr"]).all() for p in patches), "NaN must not reach the shards"


def test_mask_marks_exactly_the_unobserved_pixels():
    spec = PairPatchSpec(lr_patch=32, scale=4, min_std=0.0, min_valid_fraction=0.0)
    data = _synthetic_pair(size=64, invalid_corner=40)
    patches = list(iter_pair_patches(data, spec, np.random.default_rng(0)))
    first = patches[0]
    assert not first["mask"][:40, :40].any()
    assert first["mask"][60:, 60:].all()
