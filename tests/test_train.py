"""The training loop, the loss assembly, and the LR-schedule guard.

The guard tests below exist because of a real failure: a demo run of 36 total
steps with `warmup_iters: 50` never left its warmup ramp, so the residual
branch stayed at its near-zero initialisation and the model was numerically
indistinguishable from bicubic -- which the generated report then presented as
a measured result, with no error anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from srm.train import MAX_WARMUP_FRACTION, MIN_USEFUL_STEPS, build_scheduler
from srm.losses import SRMLoss
from srm.data.degradation import DegradationConfig


def _lr_curve(total_steps: int, warmup: int, epochs: int = 1, base_lr: float = 3e-4):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=base_lr)
    cfg = {"train": {"epochs": epochs, "warmup_iters": warmup, "scheduler": "cosine"}}
    scheduler = build_scheduler(optimizer, cfg, total_steps // epochs)
    curve = []
    for _ in range(total_steps):
        curve.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return np.array(curve)


# --------------------------------------------------------------------------- #
# the schedule guard
# --------------------------------------------------------------------------- #

def test_warmup_longer_than_the_run_is_clamped():
    """The exact shape that produced a bicubic-equivalent checkpoint."""
    curve = _lr_curve(total_steps=36, warmup=50, epochs=6)
    assert curve.max() == pytest.approx(3e-4, rel=1e-6), "never reached base LR"
    assert curve[-1] < curve.max(), "cosine decay never engaged"


def test_the_clamp_respects_the_documented_fraction():
    total, base = 100, 3e-4
    curve = _lr_curve(total_steps=total, warmup=90, epochs=1)
    peak_at = int(np.argmax(curve))
    assert peak_at <= total * MAX_WARMUP_FRACTION + 1


def test_a_reasonable_warmup_is_left_alone():
    """The guard must not alter a schedule that was already sensible."""
    total, warmup = 1000, 100
    curve = _lr_curve(total_steps=total, warmup=warmup, epochs=1)
    assert int(np.argmax(curve)) == pytest.approx(warmup - 1, abs=1)


def test_warmup_ramps_from_near_zero_and_is_monotonic():
    curve = _lr_curve(total_steps=1000, warmup=100, epochs=1)
    ramp = curve[:100]
    assert ramp[0] < 0.02 * curve.max()
    assert np.all(np.diff(ramp) > 0)


def test_cosine_decays_but_never_reaches_zero():
    """A cosine that hits exactly zero wastes the tail of the run."""
    curve = _lr_curve(total_steps=1000, warmup=50, epochs=1)
    assert curve[-1] < 0.05 * curve.max()
    assert curve[-1] > 0


def test_a_too_short_run_is_reported(caplog):
    """Silence is the bug; the warning is the fix."""
    with caplog.at_level("WARNING"):
        _lr_curve(total_steps=MIN_USEFUL_STEPS // 2, warmup=4, epochs=1)
    assert any("optimiser steps" in record.getMessage() for record in caplog.records)


def test_an_adequate_run_warns_about_nothing(caplog):
    with caplog.at_level("WARNING"):
        _lr_curve(total_steps=2000, warmup=100, epochs=1)
    assert not caplog.records


def test_scheduler_none_holds_the_base_rate():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=1e-4)
    cfg = {"train": {"epochs": 1, "warmup_iters": 0, "scheduler": "none"}}
    scheduler = build_scheduler(optimizer, cfg, 50)
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


# --------------------------------------------------------------------------- #
# loss assembly
# --------------------------------------------------------------------------- #

def test_only_the_configured_terms_are_active():
    """One config file has to express the whole ladder from L1 to a full GAN."""
    cfg = {"scale": 4, "bands": ["B02", "B03", "B04", "B08"],
           "loss": {"pixel": {"weight": 1.0}, "spectral_angle": {"weight": 0.1}}}
    loss = SRMLoss(cfg, DegradationConfig(scale=4), channels=4)

    sr, hr = torch.rand(2, 4, 32, 32), torch.rand(2, 4, 32, 32)
    _, parts = loss(sr, hr)
    assert set(parts) == {"pixel", "sam", "total"}
    assert not loss.uses_gan


def test_a_perfect_prediction_gives_a_near_zero_loss():
    cfg = {"scale": 4, "bands": ["B02", "B03", "B04", "B08"],
           "loss": {"pixel": {"weight": 1.0}, "spectral_angle": {"weight": 0.1},
                    "gradient": {"weight": 0.05}}}
    loss = SRMLoss(cfg, DegradationConfig(scale=4), channels=4)
    hr = torch.rand(2, 4, 32, 32) * 0.4 + 0.05
    total, _ = loss(hr.clone(), hr)
    assert float(total) < 1e-3


def test_a_worse_prediction_costs_more():
    cfg = {"scale": 4, "bands": ["B02", "B03", "B04", "B08"],
           "loss": {"pixel": {"weight": 1.0}}}
    loss = SRMLoss(cfg, DegradationConfig(scale=4), channels=4)
    hr = torch.rand(2, 4, 32, 32) * 0.4 + 0.05
    mild, _ = loss(hr + 0.01, hr)
    severe, _ = loss(hr + 0.10, hr)
    assert float(severe) > float(mild)


def test_the_breakdown_sums_to_the_reported_total():
    """The per-term log is the only way to see a GAN stage going wrong."""
    cfg = {"scale": 4, "bands": ["B02", "B03", "B04", "B08"],
           "loss": {"pixel": {"weight": 1.0}, "spectral_angle": {"weight": 0.1},
                    "gradient": {"weight": 0.05}}}
    loss = SRMLoss(cfg, DegradationConfig(scale=4), channels=4)
    sr, hr = torch.rand(2, 4, 32, 32), torch.rand(2, 4, 32, 32)
    _, parts = loss(sr, hr)

    recomposed = sum(loss.weights[name] * value
                     for name, value in parts.items() if name != "total")
    assert recomposed == pytest.approx(parts["total"], rel=1e-5)


def test_the_loss_is_differentiable():
    cfg = {"scale": 4, "bands": ["B02", "B03", "B04", "B08"],
           "loss": {"pixel": {"weight": 1.0}, "spectral_angle": {"weight": 0.1},
                    "consistency": {"weight": 0.5}, "gradient": {"weight": 0.05}}}
    loss = SRMLoss(cfg, DegradationConfig(scale=4), channels=4)
    sr = (torch.rand(2, 4, 32, 32) * 0.4 + 0.05).requires_grad_(True)
    hr = torch.rand(2, 4, 32, 32) * 0.4 + 0.05
    lr = torch.rand(2, 4, 8, 8) * 0.4 + 0.05

    total, _ = loss(sr, hr, lr)
    total.backward()
    assert sr.grad is not None
    assert torch.isfinite(sr.grad).all()
    assert sr.grad.abs().sum() > 0


def test_consistency_term_prefers_a_physically_consistent_output():
    """Wald: degrading the SR result must reproduce the observed input."""
    from srm.data.degradation import degrade

    cfg = {"scale": 4, "bands": ["B02", "B03", "B04", "B08"],
           "loss": {"pixel": {"weight": 0.0}, "consistency": {"weight": 1.0}}}
    degradation = DegradationConfig(scale=4, noise_sigma=0.0, shot_noise=False)
    loss = SRMLoss(cfg, degradation, channels=4)

    hr = torch.rand(1, 4, 32, 32) * 0.3 + 0.1
    lr = torch.from_numpy(degrade(hr[0].numpy(), degradation)).unsqueeze(0)

    consistent, _ = loss(hr, hr, lr)
    inconsistent, _ = loss(hr + 0.1, hr, lr)
    assert float(inconsistent) > float(consistent)


# --------------------------------------------------------------------------- #
# the EMA horizon guard -- same family of trap as the warmup clamp above
# --------------------------------------------------------------------------- #

from srm.train import MAX_EMA_HORIZON_FRACTION, effective_ema_decay  # noqa: E402


def test_an_ema_horizon_longer_than_the_run_is_clamped(caplog):
    """decay 0.999 averages ~1000 steps; over a 200-step run that is the init.

    Left alone, the exported weights stay a blend of the initialisation and
    every validation number reads far worse than the model actually is --
    which looks like "the model did not learn" rather than like a bug.
    """
    with caplog.at_level("WARNING"):
        clamped = effective_ema_decay(0.999, total_steps=200)
    assert clamped < 0.999
    assert 1.0 / (1.0 - clamped) <= 200 * MAX_EMA_HORIZON_FRACTION + 1
    assert any("ema_decay" in r.message for r in caplog.records)


def test_a_sane_ema_decay_is_left_alone(caplog):
    with caplog.at_level("WARNING"):
        assert effective_ema_decay(0.999, total_steps=10_000) == 0.999
    assert not caplog.records


def test_ema_guard_tolerates_a_disabled_ema():
    assert effective_ema_decay(0.0, total_steps=500) == 0.0
