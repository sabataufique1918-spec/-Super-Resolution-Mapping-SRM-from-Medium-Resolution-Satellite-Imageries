"""Architectures, the registry, and the checkpoint contract."""
from __future__ import annotations

import pytest
import torch

from srm.models.blocks import bicubic_up
from srm.models.registry import REGISTRY, build_model, load_checkpoint, save_checkpoint, ModelEMA


def _cfg(name: str, **model_kwargs) -> dict:
    return {
        "scale": 4,
        "bands": ["B02", "B03", "B04", "B08"],
        "model": {"name": name, "features": 8, "blocks": 1, **model_kwargs},
    }


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_model_upscales_by_exactly_the_scale(name):
    model = build_model(_cfg(name)).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 4, 16, 16))
    assert out.shape == (1, 4, 64, 64), f"{name} produced {tuple(out.shape)}"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_model_emits_physical_reflectance(name):
    """Negative reflectance is unphysical and breaks every downstream index."""
    model = build_model(_cfg(name)).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 4, 16, 16))
    assert out.min() >= 0.0
    assert out.max() <= 1.5


#: Models whose forward pass is `bicubic_up(x) + residual(x)`. `residual_cnn`
#: is deliberately absent: the specified architecture emits the image from its
#: final convolution, with only a feature-domain skip, so it has no bicubic
#: floor to start from.
BICUBIC_SKIP_MODELS = sorted(set(REGISTRY) - {"residual_cnn", "srmnet"})


@pytest.mark.parametrize("name", BICUBIC_SKIP_MODELS)
def test_residual_models_start_near_bicubic(name):
    """The deliberate near-zero residual init.

    For the bicubic-skip architectures, training should never have to climb
    out of a worse-than-interpolation hole, so an untrained model must be
    close to its skip. This is also exactly why a too-short run yields a
    checkpoint indistinguishable from bicubic -- see the schedule guard in
    test_train.py.
    """
    torch.manual_seed(0)
    model = build_model(_cfg(name)).eval()
    x = torch.rand(1, 4, 16, 16) * 0.3 + 0.05
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out, bicubic_up(x, 4).clamp(0, 1.5), atol=0.05)


@pytest.mark.parametrize("name", ["residual_cnn", "srmnet"])
def test_specified_model_does_not_start_at_bicubic(name):
    """The specified architecture has no bicubic skip, and must not fake one.

    This is the inverse of the test above and it is the point of the
    architecture: the output comes from the tail convolution. If this ever
    starts passing `allclose` against bicubic, someone has reintroduced the
    global image-domain skip and the reported results are for a different
    network than the one specified.
    """
    torch.manual_seed(0)
    model = build_model(_cfg(name)).eval()
    x = torch.rand(1, 4, 16, 16) * 0.3 + 0.05
    with torch.no_grad():
        out = model(x)
    assert not torch.allclose(out, bicubic_up(x, 4).clamp(0, 1.5), atol=0.05)


def test_specified_model_can_opt_into_the_bicubic_skip():
    """`global_skip="bicubic"` is available for a controlled comparison."""
    torch.manual_seed(0)
    model = build_model(_cfg("residual_cnn", global_skip="bicubic")).eval()
    x = torch.rand(1, 4, 16, 16) * 0.3 + 0.05
    with torch.no_grad():
        out = model(x)
    assert model.init_gain == 0.1, "the skip variant needs the damped init"
    assert torch.allclose(out, bicubic_up(x, 4).clamp(0, 1.5), atol=0.05)


def test_specified_model_matches_the_architecture_spec():
    """Layer-for-layer check against the required network.

    3x3 conv 4->64, 8 residual blocks, trunk conv + feature skip, two
    conv/PixelShuffle(2) stages, 3x3 conv 64->4, on a [B,4,64,64] input.
    """
    import torch.nn as nn

    from srm.models.residual_cnn import PlainResidualBlock

    model = build_model({
        "scale": 4, "bands": ["B02", "B03", "B04", "B08"],
        "model": {"name": "residual_cnn", "features": 64, "blocks": 8},
    }).eval()

    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    blocks = [m for m in model.modules() if isinstance(m, PlainResidualBlock)]
    shuffles = [m for m in model.modules() if isinstance(m, nn.PixelShuffle)]

    assert len(blocks) == 8
    assert len(shuffles) == 2 and all(s.upscale_factor == 2 for s in shuffles)
    # head + 2 per block + trunk fuse + 1 per upsample stage + tail
    assert len(convs) == 1 + 2 * 8 + 1 + 2 + 1 == 21
    assert all(c.kernel_size == (3, 3) for c in convs), "every convolution is 3x3"
    assert (convs[0].in_channels, convs[0].out_channels) == (4, 64)
    assert (convs[-1].in_channels, convs[-1].out_channels) == (64, 4)
    assert [(c.in_channels, c.out_channels) for c in convs if c.out_channels == 256] == [
        (64, 256), (64, 256)
    ]

    with torch.no_grad():
        out = model(torch.rand(2, 4, 64, 64))
    assert tuple(out.shape) == (2, 4, 256, 256)


def test_specified_model_block_count_and_width_are_configurable():

    from srm.models.residual_cnn import PlainResidualBlock

    model = build_model({
        "scale": 4, "bands": ["B02", "B03", "B04", "B08"],
        "model": {"name": "residual_cnn", "features": 32, "blocks": 4},
    })
    assert len([m for m in model.modules() if isinstance(m, PlainResidualBlock)]) == 4
    assert model.head.out_channels == 32
    with torch.no_grad():
        assert model(torch.rand(1, 4, 16, 16)).shape == (1, 4, 64, 64)


def test_bicubic_baseline_is_exactly_bicubic():
    """The control condition must be the control, with no learned component."""
    model = build_model(_cfg("bicubic")).eval()
    x = torch.rand(1, 4, 16, 16) * 0.5
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out, bicubic_up(x, 4).clamp(0, 1.5), atol=1e-6)
    assert model.n_params == 0


def test_esrgan_and_rrdb_are_the_same_trunk():
    """They differ by loss configuration, not by architecture."""
    assert REGISTRY["esrgan"] is REGISTRY["rrdb"]


def test_unknown_model_name_fails_loudly():
    with pytest.raises(KeyError, match="unknown model"):
        build_model(_cfg("definitely_not_a_model"))


def test_band_count_drives_the_channel_count():
    cfg = {"scale": 4, "bands": ["B02", "B03"], "model": {"name": "edsr", "features": 8,
                                                          "blocks": 1}}
    model = build_model(cfg).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 2, 8, 8)).shape == (1, 2, 32, 32)


def test_models_accept_non_square_input():
    """Real scenes are not square, and neither are their edge tiles."""
    model = build_model(_cfg("edsr")).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 4, 12, 20)).shape == (1, 4, 48, 80)


def test_checkpoint_rebuilds_the_model_without_being_told_the_architecture(tmp_path):
    """The property that prevents 'wrong config, silently wrong output'."""
    cfg = _cfg("edsr", features=8, blocks=2)
    model = build_model(cfg).eval()
    path = save_checkpoint(tmp_path / "ck.pt", model, cfg, epoch=3, metrics={"psnr": 31.5})

    restored, payload = load_checkpoint(path, device="cpu")
    assert payload["epoch"] == 3
    assert payload["metrics"]["psnr"] == 31.5

    x = torch.rand(1, 4, 16, 16)
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x), atol=1e-6)


def test_checkpoint_without_a_config_is_rejected(tmp_path):
    """Better a clear error than a silently mismatched architecture."""
    torch.save({"model": {}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="no embedded config"):
        load_checkpoint(tmp_path / "bad.pt")


def test_load_prefers_the_ema_weights_when_present(tmp_path):
    cfg = _cfg("edsr")
    model = build_model(cfg)
    ema = ModelEMA(model, decay=0.0)          # decay 0 -> EMA tracks the model exactly
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)                        # move the live weights away from the EMA
    ema.update(model)

    path = save_checkpoint(tmp_path / "ema.pt", model, cfg, ema=ema)
    _, payload = load_checkpoint(path, prefer_ema=True)
    assert payload["used_ema"] is True
    _, payload = load_checkpoint(path, prefer_ema=False)
    assert payload["used_ema"] is False


def test_ema_tracks_the_model_without_being_the_model(tmp_path):
    """A decayed average must lag, not alias."""
    cfg = _cfg("edsr")
    model = build_model(cfg)
    ema = ModelEMA(model, decay=0.9)
    before = {k: v.clone() for k, v in ema.module.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)

    after = ema.module.state_dict()
    moved = [k for k in after if not torch.allclose(after[k], before[k])]
    assert moved, "EMA did not follow the model at all"
    live = dict(model.named_parameters())
    for key in moved:
        if key in live:
            assert not torch.allclose(after[key], live[key].detach()), "EMA aliased the model"


def test_mc_dropout_can_be_enabled_without_enabling_batchnorm():
    """Uncertainty sampling needs dropout live and everything else frozen."""
    model = build_model(_cfg("edsr", dropout=0.1)).eval()
    count = model.enable_mc_dropout()
    assert count > 0
    dropouts = [m for m in model.modules()
                if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d))]
    assert all(m.training for m in dropouts)
    others = [m for m in model.modules()
              if not isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d))]
    assert not any(m.training for m in others)


def test_mc_dropout_actually_produces_variation():
    """A dropout that changes nothing would make the uncertainty map a lie."""
    torch.manual_seed(0)
    model = build_model(_cfg("edsr", dropout=0.3)).eval()
    model.enable_mc_dropout()
    x = torch.rand(1, 4, 16, 16)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# checkpoint provenance and compatibility
# --------------------------------------------------------------------------- #

def test_checkpoint_records_the_provenance_inference_needs(tmp_path):
    """Band order, scale and normalisation travel with the weights.

    A checkpoint whose band order is unknown is unusable: applying it to
    (B02,B03,B04,B08) when it was trained on (B04,B03,B02,B08) yields a
    plausible image with red and blue swapped and every index wrong.
    """
    from srm.models.registry import inspect_checkpoint

    cfg = {"experiment": "e", "scale": 4, "seed": 99,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 2},
           "data": {"patches": "data/patches_sen2neon"}}
    path = save_checkpoint(tmp_path / "c.pt", build_model(cfg), cfg, epoch=7, step=1400)

    info = inspect_checkpoint(path)
    assert info["bands"] == ["B02", "B03", "B04", "B08"]
    assert info["scale"] == 4
    assert info["seed"] == 99
    assert info["epoch"] == 7 and info["step"] == 1400
    assert info["architecture"] == "residual_cnn"
    assert info["reflectance_scale"] == 10000.0
    assert info["trained_on"] == "data/patches_sen2neon"


def test_compatibility_report_explains_rather_than_just_refusing(tmp_path):
    """"4 bands expected, 12 supplied" is actionable; "incompatible" is not."""
    from srm.models.registry import compatibility_report, inspect_checkpoint

    cfg = {"experiment": "e", "scale": 4, "seed": 1,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 1},
           "data": {"patches": "p"}}
    info = inspect_checkpoint(save_checkpoint(tmp_path / "c.pt", build_model(cfg), cfg, epoch=3))

    assert compatibility_report(info, n_bands=4, scale=4) == []
    assert any("12" in p for p in compatibility_report(info, n_bands=12, scale=4))
    assert any("x2" in p for p in compatibility_report(info, n_bands=4, scale=2))


def test_an_untrained_checkpoint_is_flagged(tmp_path):
    from srm.models.registry import compatibility_report, inspect_checkpoint

    cfg = {"experiment": "e", "scale": 4, "seed": 1,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 1},
           "data": {"patches": "p"}}
    info = inspect_checkpoint(save_checkpoint(tmp_path / "c.pt", build_model(cfg), cfg, epoch=0))
    assert any("no completed training epoch" in p for p in compatibility_report(info, 4))


# --------------------------------------------------------------------------- #
# raw vs EMA weight selection
# --------------------------------------------------------------------------- #

def _two_weight_checkpoint(tmp_path, prefer=None):
    """A checkpoint whose EMA weights differ markedly from its raw weights."""
    from srm.models.registry import ModelEMA

    cfg = {"experiment": "e", "scale": 4, "seed": 1,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 1},
           "data": {"patches": "p"}}
    torch.manual_seed(0)
    model = build_model(cfg)
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():                       # move the raw weights away
        for p in model.parameters():
            p.add_(0.5)
    path = save_checkpoint(tmp_path / "c.pt", model, cfg, epoch=5, ema=ema)
    if prefer is not None:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        payload["prefer"] = prefer
        torch.save(payload, path)
    return path, cfg


def test_an_unstamped_checkpoint_loads_the_trained_weights_not_the_ema(tmp_path):
    """The shipped bug: always preferring EMA exported a worse model silently.

    On a run still improving at the end, the EMA averages in strictly worse
    earlier weights. Measured on this project that was 1.35 dB of PSNR and a
    doubling of spectral angle, reported nowhere.
    """
    path, _ = _two_weight_checkpoint(tmp_path)
    model, payload = load_checkpoint(path)
    assert payload["used_ema"] is False
    raw = torch.load(str(path), map_location="cpu", weights_only=False)["model"]
    for name, param in model.state_dict().items():
        assert torch.allclose(param, raw[name])


def test_a_stamped_preference_is_honoured(tmp_path):
    path, _ = _two_weight_checkpoint(tmp_path, prefer="ema")
    _, payload = load_checkpoint(path)
    assert payload["used_ema"] is True

    path, _ = _two_weight_checkpoint(tmp_path, prefer="model")
    _, payload = load_checkpoint(path)
    assert payload["used_ema"] is False


def test_an_explicit_prefer_ema_overrides_the_stamp(tmp_path):
    """`scripts/stamp_weights.py` needs to score both regardless of the stamp."""
    path, _ = _two_weight_checkpoint(tmp_path, prefer="model")
    assert load_checkpoint(path, prefer_ema=True)[1]["used_ema"] is True
    assert load_checkpoint(path, prefer_ema=False)[1]["used_ema"] is False


def test_a_checkpoint_without_ema_never_claims_to_use_it(tmp_path):
    cfg = {"experiment": "e", "scale": 4, "seed": 1,
           "bands": ["B02", "B03", "B04", "B08"],
           "model": {"name": "residual_cnn", "features": 8, "blocks": 1},
           "data": {"patches": "p"}}
    path = save_checkpoint(tmp_path / "n.pt", build_model(cfg), cfg, epoch=2)
    assert load_checkpoint(path, prefer_ema=True)[1]["used_ema"] is False
