"""Shared fixtures.

Everything here is deterministic and small. The suite is meant to run in
seconds on CPU so it can gate every commit; anything that needs a GPU or a
download belongs in the demo, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

from srm import S2_10M_BANDS
from srm.data.degradation import DegradationConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1337)


@pytest.fixture
def hr_scene(rng) -> np.ndarray:
    """A 4-band 128x128 reflectance scene with real structure in it.

    Random noise is a bad test image: it has no spatial correlation, so blur
    and decimation behave degenerately and edge-sensitive metrics say nothing.
    This builds smooth regions, hard edges and a gradient instead.
    """
    size = 128
    scene = np.zeros((4, size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    base = 0.12 + 0.05 * np.sin(2 * np.pi * xx / 37.0) * np.cos(2 * np.pi * yy / 41.0)
    for band in range(4):
        scene[band] = base * (0.8 + 0.15 * band)

    scene[:, 20:60, 20:60] += np.array([0.02, 0.05, 0.03, 0.30])[:, None, None]  # vegetation
    scene[:, 70:110, 30:50] += np.array([0.06, 0.04, 0.03, -0.04])[:, None, None]  # water
    scene[:, 90:96, :] += np.array([0.10, 0.10, 0.10, 0.08])[:, None, None]  # road
    scene += 0.002 * rng.standard_normal(scene.shape).astype(np.float32)
    return np.clip(scene, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def degradation() -> DegradationConfig:
    return DegradationConfig(
        scale=4, mode="mtf", kernel_size=9, noise_sigma=0.0, shot_noise=False,
        bands=S2_10M_BANDS,
    )


@pytest.fixture
def tiny_cfg(tmp_path) -> dict:
    """The smallest config that still exercises the real code paths."""
    return {
        "experiment": "pytest",
        "seed": 1337,
        "device": "cpu",
        "scale": 4,
        "bands": list(S2_10M_BANDS),
        "data": {"patches": str(tmp_path / "patches"), "hr_patch": 32, "augment": False},
        "model": {"name": "edsr", "features": 8, "blocks": 1, "attention": False},
        "loss": {"pixel": {"type": "charbonnier", "weight": 1.0}},
        "train": {
            "epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1e-4,
            "scheduler": "cosine", "warmup_iters": 0, "out_dir": str(tmp_path / "ckpt"),
        },
        "eval": {"metrics": ["psnr", "ssim", "sam"], "data_range": 1.0},
        "infer": {"tile": 32, "overlap": 8, "tta": False, "batch_size": 2},
    }
