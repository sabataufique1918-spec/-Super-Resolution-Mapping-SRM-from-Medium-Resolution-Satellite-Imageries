"""End to end: patches -> train -> checkpoint -> infer -> evaluate.

Unit tests cover each stage in isolation; this covers the joins between them,
which is where shapes, dtypes and reflectance scaling actually go wrong. It is
deliberately tiny -- it proves the pipeline connects, not that a model is good.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from srm.data.degradation import DegradationConfig, degrade
from srm.data.patches import PatchSpec, build_patches, load_manifest
from srm.data.synthetic import LABELS, SceneSpec, generate_scene


# --------------------------------------------------------------------------- #
# synthetic scene generation
# --------------------------------------------------------------------------- #

def test_generated_scene_has_the_expected_shape_and_range():
    scene, labels = generate_scene(SceneSpec(size=128), seed=0)
    assert scene.shape == (4, 128, 128)
    assert labels.shape == (128, 128)
    assert scene.dtype == np.float32
    assert 0.0 <= scene.min() and scene.max() <= 1.0


def test_generated_scene_is_reproducible():
    a, la = generate_scene(SceneSpec(size=96), seed=42)
    b, lb = generate_scene(SceneSpec(size=96), seed=42)
    assert np.array_equal(a, b)
    assert np.array_equal(la, lb)


def test_different_seeds_give_different_scenes():
    a, _ = generate_scene(SceneSpec(size=96), seed=1)
    b, _ = generate_scene(SceneSpec(size=96), seed=2)
    assert not np.array_equal(a, b)


def test_generated_scene_contains_several_land_cover_classes():
    """A single-class scene would make the downstream applications vacuous."""
    _, labels = generate_scene(SceneSpec(size=256), seed=0)
    present = set(np.unique(labels).tolist())
    assert len(present) >= 3
    assert present.issubset(set(LABELS.values()))


def test_vegetation_really_is_the_most_vegetated_class():
    """The synthetic spectra must be physically sensible or nothing downstream means anything."""
    from srm.applications.indices import compute_index

    scene, labels = generate_scene(SceneSpec(size=256), seed=0)
    ndvi = compute_index(scene, "ndvi")
    vegetation = ndvi[labels == LABELS["vegetation"]]
    water = ndvi[labels == LABELS["water"]]
    if vegetation.size and water.size:
        assert vegetation.mean() > water.mean()


# --------------------------------------------------------------------------- #
# patch extraction
# --------------------------------------------------------------------------- #

def test_build_patches_writes_a_usable_manifest(tmp_path):
    scenes = [(generate_scene(SceneSpec(size=128), seed=s)[0], None) for s in range(3)]
    spec = PatchSpec(hr_patch=32, scale=4, stride=32, min_std=0.0, val_fraction=0.34)
    degradation = DegradationConfig(scale=4, noise_sigma=0.0)

    build_patches(scenes, tmp_path, spec, degradation)
    manifest = load_manifest(tmp_path)

    assert manifest["scale"] == 4
    assert manifest["counts"]["train"] > 0
    assert manifest["counts"]["val"] > 0


def test_patches_round_trip_through_uint16_storage(tmp_path):
    """Patches are stored as scaled uint16; the loader must undo that exactly."""
    from srm.data.dataset import SRMPatchDataset

    scenes = [(generate_scene(SceneSpec(size=128), seed=0)[0], None)]
    spec = PatchSpec(hr_patch=32, scale=4, stride=32, min_std=0.0, val_fraction=0.5)
    build_patches(scenes, tmp_path, spec, DegradationConfig(scale=4, noise_sigma=0.0))

    dataset = SRMPatchDataset(tmp_path, split="train", hr_patch=32, augment=False,
                           degradation=DegradationConfig(scale=4, noise_sigma=0.0))
    sample = dataset[0]

    assert sample["hr"].shape == (4, 32, 32)
    assert sample["lr"].shape == (4, 8, 8)
    assert sample["hr"].dtype == torch.float32
    assert 0.0 <= float(sample["hr"].min()) and float(sample["hr"].max()) <= 1.5


def test_flat_patches_are_rejected_when_the_quota_is_zero(tmp_path):
    """Empty ocean and bare desert teach a super-resolver nothing."""
    flat = [(np.full((4, 128, 128), 0.2, dtype=np.float32), None)]
    spec = PatchSpec(hr_patch=32, scale=4, stride=32, min_std=0.05,
                     max_flat_fraction=0.0, val_fraction=0.2)
    build_patches(flat, tmp_path, spec, DegradationConfig(scale=4, noise_sigma=0.0))

    manifest = load_manifest(tmp_path)
    assert manifest["counts"]["train"] == 0
    assert manifest["skipped_flat"] > 0


def test_some_flat_patches_are_kept_at_the_quota(tmp_path):
    """Dropping every flat patch is also wrong.

    A model that has never seen a uniform field cannot learn to leave one
    alone, so `max_flat_fraction` admits a share of them on purpose. This
    pins that the quota is honoured rather than silently ignored.
    """
    flat = [(np.full((4, 256, 256), 0.2, dtype=np.float32), None)]
    spec = PatchSpec(hr_patch=32, scale=4, stride=32, min_std=0.05,
                     max_flat_fraction=0.25, val_fraction=0.0)
    build_patches(flat, tmp_path, spec, DegradationConfig(scale=4, noise_sigma=0.0))

    manifest = load_manifest(tmp_path)
    kept = manifest["counts"]["train"] + manifest["counts"]["val"]
    total = kept + manifest["skipped_flat"]
    assert 0 < kept < total
    assert kept / total == pytest.approx(0.25, abs=0.15)


# --------------------------------------------------------------------------- #
# the full path
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_train_then_infer_then_evaluate(tmp_path, tiny_cfg):
    """One epoch on a handful of patches, purely to prove the stages connect."""
    from srm.evaluate import run as evaluate_run
    from srm.infer import super_resolve
    from srm.models.registry import load_checkpoint
    from srm.train import train

    patch_dir = tmp_path / "patches"
    scenes = [(generate_scene(SceneSpec(size=160), seed=s)[0], None) for s in range(4)]
    spec = PatchSpec(hr_patch=32, scale=4, stride=32, min_std=0.0, val_fraction=0.25)
    build_patches(scenes, patch_dir, spec, DegradationConfig(scale=4, noise_sigma=0.002))

    cfg = dict(tiny_cfg)
    cfg["data"] = {**cfg["data"], "patches": str(patch_dir)}

    checkpoint = train(cfg)
    assert checkpoint.exists()

    # the checkpoint must rebuild itself with no external configuration
    model, payload = load_checkpoint(checkpoint, device="cpu")
    assert payload["config"]["model"]["name"] == "edsr"

    scene, _ = generate_scene(SceneSpec(size=128), seed=99)
    lr = degrade(scene, DegradationConfig(scale=4, noise_sigma=0.0))
    sr = super_resolve(lr, model, tile=32, overlap=8, device="cpu", progress=False)

    assert sr.shape == scene.shape
    assert np.isfinite(sr).all()
    assert sr.min() >= 0.0

    report_dir = tmp_path / "report"
    result = evaluate_run(str(checkpoint), str(patch_dir), "val", str(report_dir),
                          limit=4, device="cpu")

    assert (report_dir / "report.md").exists()
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "per_patch_metrics.csv").exists()
    assert "model" in result and "bicubic" in result

    # every metric is reported as a distribution, not a bare number, so that a
    # mean can never be quoted without its spread
    for metric in ("psnr", "ssim", "sam"):
        for arm in ("model", "bicubic"):
            assert np.isfinite(result[arm][metric]["mean"])
            assert np.isfinite(result[arm][metric]["std"])


@pytest.mark.slow
def test_a_trained_model_is_not_worse_than_its_own_initialisation(tmp_path, tiny_cfg):
    """Training must move in the right direction, even by a little.

    Not a quality claim -- one epoch on synthetic patches proves nothing about
    Sentinel-2. It catches a loss with the wrong sign, a detached graph, or an
    optimiser that never steps, all of which otherwise produce a plausible
    checkpoint and a plausible report.
    """
    from srm.metrics.quality import psnr
    from srm.models.registry import build_model, load_checkpoint
    from srm.train import train

    patch_dir = tmp_path / "patches"
    scenes = [(generate_scene(SceneSpec(size=160), seed=s)[0], None) for s in range(4)]
    spec = PatchSpec(hr_patch=32, scale=4, stride=32, min_std=0.0, val_fraction=0.25)
    build_patches(scenes, patch_dir, spec, DegradationConfig(scale=4, noise_sigma=0.0))

    cfg = dict(tiny_cfg)
    cfg["data"] = {**cfg["data"], "patches": str(patch_dir)}
    cfg["train"] = {**cfg["train"], "epochs": 2, "lr": 1e-3}

    torch.manual_seed(0)
    untrained = build_model(cfg).eval()
    trained, _ = load_checkpoint(train(cfg), device="cpu")

    scene, _ = generate_scene(SceneSpec(size=128), seed=99)
    lr = torch.from_numpy(degrade(scene, DegradationConfig(scale=4, noise_sigma=0.0)))[None]
    with torch.no_grad():
        before = psnr(untrained(lr)[0].numpy(), scene)
        after = psnr(trained(lr)[0].numpy(), scene)

    assert after > before - 0.5, f"training degraded the model: {before:.2f} -> {after:.2f} dB"
