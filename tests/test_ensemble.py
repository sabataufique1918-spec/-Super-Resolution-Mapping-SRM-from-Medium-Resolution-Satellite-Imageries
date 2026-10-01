"""Ensemble disagreement, and its refusal to produce a number it cannot justify.

The requirement being tested is as much about what the module *declines* to do
as what it computes: one checkpoint must yield "unavailable", never a
single-model quantity dressed up as an ensemble one.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from srm.ensemble import EnsembleResult, EnsembleUnavailable, check_ensemble, estimate_ensemble
from srm.models.registry import save_checkpoint


def _cfg(seed: int = 1337, scale: int = 4, bands=("B02", "B03", "B04", "B08")) -> dict:
    return {
        "experiment": f"t{seed}", "scale": scale, "seed": seed, "bands": list(bands),
        "model": {"name": "residual_cnn", "features": 8, "blocks": 2},
        "data": {"patches": "data/none"},
    }


def _checkpoint(tmp_path, name: str, seed: int, **kw):
    from srm.models.registry import build_model

    cfg = _cfg(seed=seed, **kw)
    torch.manual_seed(seed)
    model = build_model(cfg)
    # Nudge the weights so members genuinely differ, as independent runs would.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return str(save_checkpoint(tmp_path / f"{name}.pt", model, cfg, epoch=5))


# --------------------------------------------------------------------------- #
# refusal
# --------------------------------------------------------------------------- #

def test_a_single_checkpoint_is_unavailable_not_approximated(tmp_path):
    one = _checkpoint(tmp_path, "a", 1)
    with pytest.raises(EnsembleUnavailable, match="at least 2"):
        check_ensemble([one])


def test_the_same_checkpoint_twice_is_not_an_ensemble(tmp_path):
    """Duplicating a path would give exactly zero spread and read as certainty."""
    one = _checkpoint(tmp_path, "a", 1)
    with pytest.raises(EnsembleUnavailable, match="at least 2"):
        check_ensemble([one, one])


def test_members_disagreeing_on_scale_are_refused(tmp_path):
    a = _checkpoint(tmp_path, "a", 1, scale=4)
    b = _checkpoint(tmp_path, "b", 2, scale=2)
    with pytest.raises(EnsembleUnavailable, match="scale"):
        check_ensemble([a, b])


def test_members_disagreeing_on_band_order_are_refused(tmp_path):
    a = _checkpoint(tmp_path, "a", 1, bands=("B02", "B03", "B04", "B08"))
    b = _checkpoint(tmp_path, "b", 2, bands=("B04", "B03", "B02", "B08"))
    with pytest.raises(EnsembleUnavailable, match="band order"):
        check_ensemble([a, b])


def test_identical_seeds_warn_because_the_spread_understates(tmp_path, caplog):
    a = _checkpoint(tmp_path, "a", 7)
    b = _checkpoint(tmp_path, "b", 7)
    with caplog.at_level("WARNING"):
        check_ensemble([a, b])
    assert any("seed" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# the computation
# --------------------------------------------------------------------------- #

@pytest.fixture
def two_members(tmp_path):
    return [_checkpoint(tmp_path, "a", 1), _checkpoint(tmp_path, "b", 2)]


def test_disagreement_has_the_output_shape_and_is_non_negative(two_members):
    lr = np.random.default_rng(0).random((4, 16, 16)).astype(np.float32) * 0.3
    result = estimate_ensemble(lr, two_members, tile=16, overlap=0)

    assert result.mean.shape == (4, 64, 64)
    assert result.per_band_std.shape == (4, 64, 64)
    assert result.disagreement.shape == (64, 64)
    finite = result.disagreement[np.isfinite(result.disagreement)]
    assert (finite >= 0).all(), "a standard deviation cannot be negative"
    assert len(result.members) == 2


def test_disagreement_is_the_rms_of_the_per_band_spreads(two_members):
    """The documented aggregate, checked against its own definition."""
    lr = np.random.default_rng(1).random((4, 16, 16)).astype(np.float32) * 0.3
    result = estimate_ensemble(lr, two_members, tile=16, overlap=0)
    expected = np.sqrt(np.nanmean(result.per_band_std ** 2, axis=0))
    assert np.allclose(result.disagreement, expected, equal_nan=True, atol=1e-6)


def test_identical_members_produce_zero_disagreement(tmp_path):
    """The sanity anchor: no difference between members means no disagreement."""
    from srm.models.registry import build_model

    cfg = _cfg(seed=3)
    torch.manual_seed(3)
    model = build_model(cfg)
    a = str(save_checkpoint(tmp_path / "a.pt", model, cfg, epoch=5))
    b = str(save_checkpoint(tmp_path / "b.pt", model, cfg, epoch=5))

    lr = np.random.default_rng(2).random((4, 16, 16)).astype(np.float32) * 0.3
    result = estimate_ensemble(lr, [a, b], tile=16, overlap=0)
    assert float(np.nanmax(result.disagreement)) == pytest.approx(0.0, abs=1e-6)


def test_the_mean_lies_between_the_members(two_members):
    lr = np.random.default_rng(4).random((4, 16, 16)).astype(np.float32) * 0.3
    result = estimate_ensemble(lr, two_members, tile=16, overlap=0)
    assert np.isfinite(result.mean).all()
    assert result.mean.min() >= 0.0


def test_summary_states_units_and_the_aggregate_definition(two_members):
    """A map without a stated reduction and unit is not interpretable."""
    lr = np.random.default_rng(5).random((4, 16, 16)).astype(np.float32) * 0.3
    summary = estimate_ensemble(lr, two_members, tile=16, overlap=0).summary()
    assert summary["n_members"] == 2
    assert "reflectance" in summary["units"]
    assert "standard deviation" in summary["aggregate_definition"]
    for key in ("disagreement_mean", "disagreement_p95", "disagreement_max"):
        assert np.isfinite(summary[key])


def test_the_language_is_disagreement_not_confidence():
    """The labelling requirement, asserted rather than left to review."""
    text = EnsembleResult.AGGREGATE_DEFINITION.lower()
    assert "disagreement" in text
    for forbidden in ("probability", "confidence", "reliability", "accuracy"):
        assert forbidden not in text or "not" in text
    assert "not an error bar" in text
