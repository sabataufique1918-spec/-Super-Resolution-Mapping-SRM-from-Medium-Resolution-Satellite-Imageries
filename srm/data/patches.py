"""Turn scenes into a paired patch dataset on disk.

Two ways to obtain training pairs, both handled by the same code path because
the only thing that differs is which raster you call "HR":

**Wald / scale-transfer pairs** (what you use with Sentinel-2 alone). The real
10 m scene plays the role of HR; its 40 m degradation plays LR. The model
learns a x4 step at 40->10 m and is then applied at 10->2.5 m. This assumes the
statistics of the x4 step are scale invariant over that range -- the standard
assumption in pansharpening and SR literature, and the reason validation
against true high-resolution imagery is not optional.

**Native pairs** (what you use when you have real high-resolution reference
data: PlanetScope 3 m, NAIP 1 m, WorldView, or the synthetic 2.5 m scenes in
this repo). HR is the fine raster, LR its 10 m degradation. No scale-transfer
assumption, so this is the honest setting for reporting accuracy.

Patches are stored HR-only by default and degraded on the fly at training
time, which halves the disk footprint and lets a blind degradation draw a new
PSF every epoch. Precomputed LR is stored only when the pairs are real.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .. import S2_REFLECTANCE_SCALE
from .degradation import DegradationConfig, degrade

_UINT16_MAX = 65535


@dataclass
class PatchSpec:
    hr_patch: int = 192
    scale: int = 4
    stride: int | None = None        # defaults to hr_patch (no overlap)
    min_std: float = 0.010           # reject near-flat patches
    max_flat_fraction: float = 0.25  # but keep a quota of them, they are real
    shard_size: int = 256
    val_fraction: float = 0.12
    store_lr: bool = False

    def __post_init__(self):
        if self.hr_patch % self.scale:
            raise ValueError(f"hr_patch {self.hr_patch} must be divisible by scale {self.scale}")
        if self.stride is None:
            self.stride = self.hr_patch


def iter_patches(
    scene: np.ndarray,
    spec: PatchSpec,
    valid: np.ndarray | None = None,
) -> Iterable[np.ndarray]:
    """Yield (C, P, P) HR crops from one scene, skipping masked or flat ones."""
    _, height, width = scene.shape
    size, stride = spec.hr_patch, int(spec.stride or spec.hr_patch)
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            patch = scene[:, y : y + size, x : x + size]
            if valid is not None and not valid[y : y + size, x : x + size].all():
                continue  # any cloud/nodata in the window disqualifies it
            if not np.isfinite(patch).all():
                continue
            yield patch


def _is_informative(patch: np.ndarray, spec: PatchSpec, rng: np.random.Generator) -> bool:
    """Keep textured patches; admit flat ones only at a quota.

    Uniform water and uniform fields dominate most scenes. Training on them
    unfiltered teaches the model that the safe answer is always "blur", but
    dropping them entirely leaves it unable to leave smooth areas alone.
    """
    if float(patch.std()) >= spec.min_std:
        return True
    return bool(rng.random() < spec.max_flat_fraction)


def build_patches(
    scenes: Sequence[tuple[np.ndarray, np.ndarray | None]],
    out_dir: str | Path,
    spec: PatchSpec | None = None,
    degradation: DegradationConfig | None = None,
    bands: Sequence[str] = ("B02", "B03", "B04", "B08"),
    seed: int = 1337,
    logger=None,
) -> dict:
    """Write train/ and val/ shards of HR patches (optionally with LR).

    `scenes` is a sequence of (image (C, H, W) reflectance, valid mask or None).
    Splitting is done per scene, not per patch, so a val patch never overlaps a
    train patch -- the most common way an SR benchmark silently inflates PSNR.
    """
    spec = spec or PatchSpec()
    degradation = degradation or DegradationConfig(scale=spec.scale, bands=tuple(bands))
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)

    n_scenes = len(scenes)
    n_val = max(1, int(round(spec.val_fraction * n_scenes))) if n_scenes > 1 else 0
    val_ids = set(rng.choice(n_scenes, size=n_val, replace=False).tolist()) if n_val else set()

    buffers: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    lr_buffers: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    shard_no = {"train": 0, "val": 0}
    skipped = 0

    for idx, (scene, valid) in enumerate(scenes):
        split = "val" if idx in val_ids else "train"
        for patch in iter_patches(np.asarray(scene, dtype=np.float32), spec, valid):
            if not _is_informative(patch, spec, rng):
                skipped += 1
                continue
            buffers[split].append(_to_uint16(patch))
            if spec.store_lr:
                lr_buffers[split].append(_to_uint16(degrade(patch, degradation, rng)))
            counts[split] += 1
            if len(buffers[split]) >= spec.shard_size:
                _flush(out_dir, split, shard_no, buffers, lr_buffers, spec)
    for split in ("train", "val"):
        if buffers[split]:
            _flush(out_dir, split, shard_no, buffers, lr_buffers, spec)

    manifest = {
        "bands": list(bands),
        "scale": spec.scale,
        "patch_spec": asdict(spec),
        "degradation": asdict(degradation),
        "counts": counts,
        "skipped_flat": skipped,
        "n_scenes": n_scenes,
        "n_val_scenes": len(val_ids),
        "reflectance_scale": S2_REFLECTANCE_SCALE,
        "storage": "uint16 scaled by reflectance_scale",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if logger:
        logger.info(
            "patches: %d train / %d val from %d scenes (%d flat patches dropped) -> %s",
            counts["train"], counts["val"], n_scenes, skipped, out_dir,
        )
    return manifest


def _to_uint16(patch: np.ndarray) -> np.ndarray:
    return np.clip(patch * S2_REFLECTANCE_SCALE, 0, _UINT16_MAX).astype(np.uint16)


def _flush(out_dir, split, shard_no, buffers, lr_buffers, spec) -> None:
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"shard_{shard_no[split]:05d}.npz"
    payload = {"hr": np.stack(buffers[split])}
    if spec.store_lr and lr_buffers[split]:
        payload["lr"] = np.stack(lr_buffers[split])
    np.savez_compressed(path, **payload)
    buffers[split].clear()
    lr_buffers[split].clear()
    shard_no[split] += 1


def load_manifest(patch_dir: str | Path) -> dict:
    path = Path(patch_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest at {path}. Build a dataset first: python scripts/prepare_dataset.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def scenes_from_geotiffs(
    paths: Sequence[str | Path],
    bands: Sequence[int] | None = None,
    use_scl: bool = True,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Read scene rasters for `build_patches`, honouring the SCL cloud mask.

    Accepts both the 6-band Process API product (bands + SCL + dataMask) and a
    plain 4-band raster.
    """
    from ..utils.geo import read_raster
    from .copernicus import scl_valid_mask

    out: list[tuple[np.ndarray, np.ndarray | None]] = []
    for path in paths:
        array, _ = read_raster(path, bands=bands)
        if use_scl and array.shape[0] >= 6:
            valid = scl_valid_mask(array[4]) & (array[5] > 0.5)
            array = array[:4]
        else:
            valid = None
            array = array[:4] if array.shape[0] > 4 else array
        out.append((np.nan_to_num(array).astype(np.float32), valid))
    return out
