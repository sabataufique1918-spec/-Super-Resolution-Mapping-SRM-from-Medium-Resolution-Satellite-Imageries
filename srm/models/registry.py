"""Model factory + checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from .cnn import EDSR, SRCNN, BicubicBaseline, RRDBNet, UNetDiscriminator
from .residual_cnn import ResidualSRCNN
from .swinir import SwinIRLite

REGISTRY: dict[str, Callable[..., nn.Module]] = {
    # The specified reference architecture, and the default for this project.
    "residual_cnn": ResidualSRCNN,
    "srmnet": ResidualSRCNN,     # alias
    "bicubic": BicubicBaseline,
    "srcnn": SRCNN,
    "edsr": EDSR,
    "rrdb": RRDBNet,
    "esrgan": RRDBNet,      # same generator; the difference is the loss config
    "swinir": SwinIRLite,
}


def build_model(cfg: dict) -> nn.Module:
    """Instantiate the generator described by `cfg["model"]`."""
    model_cfg = dict(cfg.get("model", {}))
    name = str(model_cfg.pop("name", "edsr")).lower()
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(REGISTRY)}")
    model_cfg.setdefault("scale", cfg.get("scale", 4))
    bands = cfg.get("bands") or []
    if bands:
        model_cfg.setdefault("in_channels", len(bands))
        model_cfg.setdefault("out_channels", len(bands))
    # depths/heads arrive from YAML as lists; SwinIR wants tuples.
    for key in ("depths", "heads"):
        if isinstance(model_cfg.get(key), list):
            model_cfg[key] = tuple(model_cfg[key])
    return REGISTRY[name](**model_cfg)


def build_discriminator(cfg: dict) -> nn.Module:
    gan_cfg = cfg.get("loss", {}).get("gan", {})
    channels = len(cfg.get("bands") or []) or cfg.get("model", {}).get("out_channels", 4)
    return UNetDiscriminator(in_channels=channels, features=int(gan_cfg.get("d_features", 64)))


class ModelEMA:
    """Exponential moving average of the generator weights.

    The EMA copy is what gets exported. For SR it is reliably worth a few
    tenths of a dB over the raw weights and removes most of the epoch-to-epoch
    jitter in validation curves, which matters when you are comparing models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = _clone_eval(model)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_params = dict(self.module.named_parameters())
        for name, param in model.named_parameters():
            if name in ema_params and param.dtype.is_floating_point:
                ema_params[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        ema_buffers = dict(self.module.named_buffers())
        for name, buf in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buf)

    def state_dict(self):
        return self.module.state_dict()


def _clone_eval(model: nn.Module) -> nn.Module:
    import copy

    clone = copy.deepcopy(model).eval()
    for param in clone.parameters():
        param.requires_grad_(False)
    return clone


#: Everything an inference run needs to interpret the tensors a checkpoint
#: produces, stored explicitly rather than inferred from the config later.
def checkpoint_provenance(cfg: dict) -> dict:
    """Band order, normalisation, scale and seed, recorded with the weights.

    A checkpoint whose band order is unknown is unusable: feeding it
    (B02, B03, B04, B08) when it was trained on (B04, B03, B02, B08) produces
    a plausible-looking image with the red and blue channels swapped and
    every spectral index wrong. The same applies to the reflectance scaling.
    """
    from .. import S2_REFLECTANCE_SCALE

    return {
        "bands": [str(b).upper() for b in (cfg.get("bands") or [])],
        "scale": int(cfg.get("scale", 4)),
        "reflectance_scale": S2_REFLECTANCE_SCALE,
        "normalization": "reflectance = DN / reflectance_scale; no fitted statistics",
        "model": dict(cfg.get("model", {})),
        "seed": cfg.get("seed"),
        "experiment": cfg.get("experiment"),
        "data_source": cfg.get("data", {}).get("patches"),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: dict,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    optimizer=None,
    ema: ModelEMA | None = None,
    discriminator: nn.Module | None = None,
    scheduler=None,
    step: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "provenance": checkpoint_provenance(cfg),
        "epoch": int(epoch),
        "step": int(step),
        "metrics": metrics or {},
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    prefer_ema: bool | None = None,
    strict: bool = True,
) -> tuple[nn.Module, dict]:
    """Rebuild a model straight from its checkpoint, config included.

    Inference never has to be told which architecture to use -- the checkpoint
    already knows -- which removes a whole class of "wrong config, silently
    wrong output" mistakes.

    Which *weights* to use is decided the same way. `prefer_ema=None` (the
    default) honours the ``prefer`` key that `train.select_preferred_weights`
    stamps in after measuring both sets on the validation split. Passing True
    or False overrides that, which is what the comparison in
    `scripts/stamp_weights.py` needs.

    The old behaviour -- always take the EMA weights when present -- is what
    this replaces. On a run still improving at the end, the EMA averages in
    strictly worse earlier weights and exports a measurably worse model, with
    nothing anywhere to say so.
    """
    payload = torch.load(str(path), map_location=device, weights_only=False)
    cfg = payload.get("config")
    if cfg is None:
        raise ValueError(f"{path} has no embedded config; cannot rebuild the model")
    model = build_model(cfg)

    has_ema = bool(payload.get("ema"))
    if prefer_ema is None:
        # No stamp means the checkpoint predates the measured selection; fall
        # back to the raw weights, which are the ones that were trained.
        use_ema = has_ema and payload.get("prefer") == "ema"
    else:
        use_ema = bool(prefer_ema) and has_ema
    state = payload["ema"] if use_ema else payload["model"]

    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:  # only reachable with strict=False
        print(f"[load_checkpoint] missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    model.to(device).eval()
    payload["used_ema"] = use_ema
    # Checkpoints written before provenance was recorded still have the full
    # config, so it can be reconstructed rather than left absent.
    payload.setdefault("provenance", checkpoint_provenance(cfg))
    return model, payload


def inspect_checkpoint(path: str | Path) -> dict:
    """Read a checkpoint's metadata without building the model or loading weights.

    Used by the interface to populate the model picker: listing ten
    checkpoints should not cost ten model constructions.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    provenance = payload.get("provenance") or checkpoint_provenance(cfg)
    return {
        "path": str(path),
        "name": Path(path).parent.name + "/" + Path(path).stem,
        "experiment": cfg.get("experiment", "?"),
        "architecture": cfg.get("model", {}).get("name", "?"),
        "scale": int(provenance.get("scale", cfg.get("scale", 4))),
        "bands": list(provenance.get("bands", cfg.get("bands", []))),
        "epoch": int(payload.get("epoch", 0)),
        "step": int(payload.get("step", 0)),
        "metrics": payload.get("metrics", {}) or {},
        "has_ema": bool(payload.get("ema")),
        # Which weights a load will actually use, so the interface can say so
        # rather than leaving it to be inferred.
        "weights": ("ema" if (payload.get("ema") and payload.get("prefer") == "ema")
                    else "raw"),
        "weight_selection": payload.get("weight_selection"),
        "seed": provenance.get("seed"),
        "trained_on": provenance.get("data_source"),
        "reflectance_scale": provenance.get("reflectance_scale"),
    }


def compatibility_report(info: dict, n_bands: int, scale: int | None = None) -> list[str]:
    """Reasons this checkpoint cannot be applied to an input, or an empty list.

    Returning reasons rather than a boolean is deliberate: the interface has
    to *tell the user what is wrong*, and "4 bands expected, 12 supplied" is
    actionable where "incompatible" is not.
    """
    problems: list[str] = []
    expected = len(info.get("bands") or [])
    if expected and expected != n_bands:
        problems.append(
            f"checkpoint was trained on {expected} bands "
            f"({', '.join(info['bands'])}) but {n_bands} were selected"
        )
    if scale is not None and int(info.get("scale", 0)) != int(scale):
        problems.append(
            f"checkpoint is x{info.get('scale')} but x{scale} was requested"
        )
    if info.get("epoch", 0) < 1:
        problems.append("checkpoint records no completed training epoch")
    return problems


def describe(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{type(model).__name__}: {total/1e6:.2f}M params ({trainable/1e6:.2f}M trainable)"
