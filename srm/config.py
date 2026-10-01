"""YAML-backed configuration with dataclass defaults.

A config is a plain nested dict, validated and completed against the defaults
below. Keeping it a dict (rather than a rigid schema) means an experiment can
carry extra keys without a code change, while `require` still catches typos in
the keys the training loop actually reads.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": "edsr_x4",
    "seed": 1337,
    "device": "auto",  # auto | cuda | cpu
    "scale": 4,
    "bands": ["B02", "B03", "B04", "B08"],
    "data": {
        "patches": "data/patches",
        "train_shards": "train",
        "val_shards": "val",
        "test_shards": "test",
        "hr_patch": 192,          # HR patch side in pixels; LR is hr_patch // scale
        "augment": True,
        "on_the_fly_degradation": True,
        "degradation": {
            "mode": "mtf",        # mtf | bicubic | blind
            "mtf_nyquist": 0.30,
            "kernel_size": 11,
            "noise_sigma": 0.002,  # in reflectance units (0-1)
            "jitter": 0.15,        # random +/- fraction on sigma when mode=blind
        },
    },
    "model": {
        "name": "edsr",
        "in_channels": 4,
        "out_channels": 4,
        "features": 64,
        "blocks": 16,
        "res_scale": 0.1,
        "dropout": 0.0,           # >0 enables MC-dropout uncertainty
    },
    "loss": {
        # Masked L1 + Adam is the specified starting point. Every term is
        # masked against the reference's validity where it can be; see
        # `srm.losses.masked`.
        "pixel": {"type": "l1", "weight": 1.0},
        "spectral_angle": {"weight": 0.1},
        "consistency": {"weight": 0.5},   # Wald protocol: downsample(SR) ~= LR
        "perceptual": {"weight": 0.0, "layers": ["relu3_3", "relu4_3"]},
        "gan": {"weight": 0.0, "type": "ragan", "d_features": 64},
    },
    "train": {
        "epochs": 40,
        "batch_size": 16,
        "num_workers": 4,
        "lr": 2.0e-4,
        "lr_d": 1.0e-4,
        "betas": [0.9, 0.99],
        "scheduler": "cosine",
        "warmup_iters": 200,
        "grad_clip": 1.0,
        "amp": True,
        "ema_decay": 0.999,
        "log_every": 25,
        "val_every": 1,
        "save_every": 5,
        "out_dir": "checkpoints",
        "init_from": None,        # path to a PSNR-pretrained generator for GAN stage
        "monitor": "loss",        # validation metric that selects best.pt
        "early_stopping_patience": 0,   # 0 disables; counted in validations
        "resume": False,          # True -> this run's last.pt, or a path
        "val_max_batches": None,  # cap the per-epoch validation pass; None = all
    },
    "eval": {
        "metrics": ["psnr", "ssim", "sam", "ergas", "scc", "rmse", "mae"],
        "data_range": 1.0,
    },
    "infer": {
        "tile": 256,              # LR tile side
        "overlap": 32,            # LR pixels of overlap, cosine-blended
        "tta": False,             # x8 self-ensemble
        "batch_size": 4,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    """Load `path` on top of DEFAULTS, then apply dotted-key `overrides`.

    >>> cfg = load_config("configs/edsr_x4.yaml", **{"train.epochs": 1})
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    for dotted, value in overrides.items():
        set_by_path(cfg, dotted.replace("__", "."), value)
    cfg["_source"] = str(path) if path else "<defaults>"
    return cfg


def set_by_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def get_by_path(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def require(cfg: dict[str, Any], *dotted_keys: str) -> None:
    """Fail loudly at start-up rather than three hours into a training run."""
    missing = [k for k in dotted_keys if get_by_path(cfg, k, _MISSING) is _MISSING]
    if missing:
        raise KeyError(f"config is missing required keys: {missing} (source={cfg.get('_source')})")


_MISSING = object()


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dumpable = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dumpable, fh, sort_keys=False)


def resolve_device(spec: str = "auto") -> str:
    if spec != "auto":
        return spec
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
