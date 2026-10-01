"""Masked pixel losses.

The property that matters: what happens in an invalid pixel must not reach
the gradient. Reference imagery stores its holes as a sentinel, and a loss
that sees them teaches the model to reproduce them.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from srm.losses.masked import (
    MaskedCharbonnierLoss,
    MaskedL1Loss,
    MaskedL2Loss,
    build_pixel_loss,
    expand_mask,
    masked_mean,
)

LOSSES = [MaskedL1Loss(), MaskedL2Loss(), MaskedCharbonnierLoss()]


@pytest.mark.parametrize("loss", LOSSES, ids=lambda m: type(m).__name__)
def test_invalid_pixels_do_not_affect_the_loss(loss):
    """The whole point: garbage in a masked-out pixel changes nothing."""
    torch.manual_seed(0)
    pred = torch.rand(2, 4, 16, 16)
    target = torch.rand(2, 4, 16, 16)
    mask = torch.ones(2, 16, 16)
    mask[:, :8] = 0.0

    before = loss(pred, target, mask)
    poisoned = target.clone()
    poisoned[:, :, :8] = 1e6           # absurd values, all inside the hole
    after = loss(pred, poisoned, mask)
    assert torch.allclose(before, after)


@pytest.mark.parametrize("loss", LOSSES, ids=lambda m: type(m).__name__)
def test_no_mask_matches_a_full_mask(loss):
    pred, target = torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8)
    assert torch.allclose(loss(pred, target), loss(pred, target, torch.ones(1, 8, 8)))


@pytest.mark.parametrize("loss", LOSSES, ids=lambda m: type(m).__name__)
def test_an_entirely_invalid_patch_is_zero_not_nan(loss):
    """A NaN here would poison every weight on the next backward pass."""
    pred = torch.rand(1, 4, 8, 8, requires_grad=True)
    value = loss(pred, torch.rand(1, 4, 8, 8), torch.zeros(1, 8, 8))
    assert torch.isfinite(value) and float(value) == 0.0
    value.backward()
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) == 0.0


def test_normalisation_is_by_valid_count_not_tensor_size():
    """Dividing by numel would scale the loss down on the hardest tiles."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    mask = torch.zeros(1, 4, 4)
    mask[:, 0, :] = 1.0                 # a quarter of the pixels valid
    # Every valid pixel has |0 - 1| = 1, so a correctly normalised mean is 1.0.
    assert float(MaskedL1Loss()(pred, target, mask)) == pytest.approx(1.0)


def test_masked_l1_equals_mae_over_the_valid_region():
    torch.manual_seed(1)
    pred, target = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
    mask = (torch.rand(1, 8, 8) > 0.5).float()
    keep = expand_mask(mask, pred).bool()
    expected = (pred - target).abs()[keep].mean()
    assert torch.allclose(MaskedL1Loss()(pred, target, mask), expected, atol=1e-6)


def test_mask_broadcasts_from_both_accepted_shapes():
    pred, target = torch.rand(2, 4, 8, 8), torch.rand(2, 4, 8, 8)
    flat = torch.ones(2, 8, 8)
    channelled = flat.unsqueeze(1)
    assert torch.allclose(MaskedL1Loss()(pred, target, flat),
                          MaskedL1Loss()(pred, target, channelled))


def test_default_pixel_loss_is_l1_as_specified():
    assert isinstance(build_pixel_loss(), MaskedL1Loss)
    assert isinstance(build_pixel_loss("l1"), MaskedL1Loss)
    assert isinstance(build_pixel_loss("charbonnier"), MaskedCharbonnierLoss)
    with pytest.raises(KeyError):
        build_pixel_loss("nonsense")


def test_gradients_flow_only_through_valid_pixels():
    pred = torch.rand(1, 1, 4, 4, requires_grad=True)
    mask = torch.zeros(1, 4, 4)
    mask[:, 0, 0] = 1.0
    MaskedL1Loss()(pred, torch.zeros(1, 1, 4, 4), mask).backward()
    assert float(pred.grad[0, 0, 0, 0]) != 0.0
    assert float(pred.grad[0, 0, 1:].abs().sum()) == 0.0


def test_composite_loss_accepts_and_honours_a_mask():
    from srm.losses import SRMLoss

    cfg = {
        "scale": 4, "bands": ["B02", "B03", "B04", "B08"],
        "loss": {"pixel": {"type": "l1", "weight": 1.0},
                 "spectral_angle": {"weight": 0.1}},
    }
    criterion = SRMLoss(cfg, channels=4)
    sr, hr = torch.rand(1, 4, 16, 16), torch.rand(1, 4, 16, 16)
    mask = torch.ones(1, 16, 16)
    mask[:, :8] = 0.0

    total, parts = criterion(sr, hr, mask=mask)
    poisoned = hr.clone()
    poisoned[:, :, :8] = 50.0
    total_poisoned, _ = criterion(sr, poisoned, mask=mask)
    assert torch.allclose(total, total_poisoned, atol=1e-5)
    assert "pixel" in parts and torch.isfinite(total)


def test_masked_mean_handles_a_none_mask():
    values = torch.rand(2, 3, 4, 4)
    assert torch.allclose(masked_mean(values, None), values.mean())


# --------------------------------------------------------------------------- #
# numerical safety of the auxiliary terms
# --------------------------------------------------------------------------- #

def test_index_loss_is_finite_for_negative_reflectance():
    """The shipped bug: an entire training run returned `inf` and nothing said so.

    The normalised index divided by `a + b + eps`. That is not a guard -- it
    only moves the singularity to `a + b == -eps`. A model clamped at
    inference but not during training (which the specified architecture is,
    because `clamp` has no gradient) emits negative reflectance early on and
    lands on it.
    """
    from srm.losses.spectral import IndexConsistencyLoss

    loss = IndexConsistencyLoss()
    target = torch.rand(2, 4, 8, 8) * 0.3
    for value in (-1e-4, -1e-6, 0.0, -1.0):
        pred = torch.full((2, 4, 8, 8), value)
        out = loss(pred, target)
        assert torch.isfinite(out), f"index loss is {out} at pred={value}"


def test_index_loss_survives_the_exact_singularity():
    """`a + b` exactly equal to the old `-eps`, which produced infinity."""
    from srm.losses.spectral import IndexConsistencyLoss

    loss = IndexConsistencyLoss(eps=1e-4)
    pred = torch.zeros(1, 4, 4, 4)
    pred[:, 3] = -5e-5          # a + b == -1e-4 exactly
    pred[:, 2] = -5e-5
    assert torch.isfinite(loss(pred, torch.rand(1, 4, 4, 4) * 0.3))


def test_gradient_loss_ignores_edges_manufactured_by_masking():
    """Zeroing invalid pixels creates a step edge that is not on the ground.

    A prediction that matches the reference everywhere it is observed must
    score ~0, even when the invalid region is filled with something wildly
    different.
    """
    from srm.losses.spectral import GradientLoss

    loss = GradientLoss()
    target = torch.rand(1, 4, 16, 16) * 0.3
    pred = target.clone()
    mask = torch.ones(1, 16, 16)
    mask[:, :6] = 0.0
    pred[:, :, :6] = 9.0        # garbage, entirely inside the hole

    assert float(loss(pred, target, mask)) == pytest.approx(0.0, abs=1e-5)
    assert float(loss(pred, target)) > 0.1, "unmasked, this must be penalised"


def test_sam_loss_is_masked():
    """Garbage inside the hole must not reach the angle.

    The floor here is not zero: `cosine` is clamped to 1 - 1e-7 before
    `acos`, so even identical spectra return about 4.5e-4 radians. The
    assertion is therefore against the loss on the valid region alone, not
    against zero.
    """
    from srm.losses.spectral import SpectralAngleLoss

    loss = SpectralAngleLoss()
    target = torch.rand(1, 4, 8, 8) * 0.3
    pred = target.clone()
    mask = torch.ones(1, 8, 8)
    mask[:, :4] = 0.0
    pred[:, :, :4] = 5.0

    masked = float(loss(pred, target, mask))
    valid_only = float(loss(pred[:, :, 4:], target[:, :, 4:]))
    assert masked == pytest.approx(valid_only, abs=1e-6)
    assert float(loss(pred, target)) > 10 * masked, "unmasked, this must be penalised"


def test_every_composite_term_stays_finite_on_a_negative_prediction():
    """The integration check: the full objective, a masked batch, bad output."""
    from srm.data.degradation import DegradationConfig
    from srm.losses import SRMLoss

    cfg = {
        "scale": 4, "bands": ["B02", "B03", "B04", "B08"],
        "loss": {"pixel": {"type": "l1", "weight": 1.0},
                 "spectral_angle": {"weight": 0.1},
                 "consistency": {"weight": 0.5},
                 "index": {"weight": 0.1},
                 "gradient": {"weight": 0.25}},
    }
    criterion = SRMLoss(cfg, DegradationConfig(scale=4, bands=tuple(cfg["bands"])),
                        channels=4)
    sr = torch.full((2, 4, 32, 32), -0.05, requires_grad=True)
    hr = torch.rand(2, 4, 32, 32) * 0.3
    lr = torch.rand(2, 4, 8, 8) * 0.3
    mask = torch.ones(2, 32, 32)
    mask[:, :10] = 0.0

    total, parts = criterion(sr, hr, lr, mask=mask)
    for name, value in parts.items():
        assert np.isfinite(value), f"{name} = {value}"
    total.backward()
    assert torch.isfinite(sr.grad).all()


def test_index_is_bounded_so_it_cannot_dominate_the_objective():
    """Finite is not enough -- it has to stay in the range the index is defined on.

    Flooring the denominator removes the infinity but leaves the ratio
    unbounded: 0.1 over a 1e-4 floor is 1000. Measured on a real batch that
    put the term at 14.2 raw and 83% of the total objective. Clamping the
    bands non-negative first bounds the index to [-1, 1] and the loss to
    [0, 2] by construction.
    """
    from srm.losses.spectral import IndexConsistencyLoss

    loss = IndexConsistencyLoss()
    torch.manual_seed(0)
    for pred in (torch.randn(2, 4, 16, 16) * 3.0,        # wild, both signs
                 torch.full((2, 4, 16, 16), -0.05),       # uniformly negative
                 torch.zeros(2, 4, 16, 16),               # exactly zero
                 torch.rand(2, 4, 16, 16) * 0.3):         # physical
        for a, b in loss.pairs:
            index = loss._index(pred, a, b, loss.eps)
            assert torch.isfinite(index).all()
            assert float(index.min()) >= -1.0 - 1e-6
            assert float(index.max()) <= 1.0 + 1e-6
        value = float(loss(pred, torch.rand(2, 4, 16, 16) * 0.3))
        assert 0.0 <= value <= 2.0 + 1e-6, value


def test_index_matches_the_textbook_ndvi_on_physical_input():
    """The safety work must not change the answer where the answer is defined."""
    from srm.losses.spectral import IndexConsistencyLoss

    loss = IndexConsistencyLoss(eps=1e-4)
    x = torch.zeros(1, 4, 2, 2)
    x[:, 2] = 0.026          # red
    x[:, 3] = 0.464          # nir
    ndvi = loss._index(x, 3, 2, loss.eps)      # (nir - red) / (nir + red)
    expected = (0.464 - 0.026) / (0.464 + 0.026 + 1e-4)
    assert float(ndvi.mean()) == pytest.approx(expected, rel=1e-4)
    assert float(ndvi.mean()) == pytest.approx(0.894, abs=0.002)
