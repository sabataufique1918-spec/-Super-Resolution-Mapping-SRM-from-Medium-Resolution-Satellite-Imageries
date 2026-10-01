"""Torch datasets over the patch shards."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .. import S2_REFLECTANCE_SCALE
from .degradation import DegradationConfig, degrade
from .patches import load_manifest


class SRMPatchDataset(Dataset):
    """Paired (LR, HR) patches, with the LR side made on the fly by default.

    Handles both dataset kinds this repo produces:

    * **simulated** -- HR-only shards, LR synthesised per epoch by the
      degradation model. Every pixel is valid, so no mask is stored.
    * **native** -- real LR/HR pairs (SEN2NEON), where the shards also carry a
      per-pixel ``mask`` marking where the HR reference was actually observed.
      That mask is returned in the batch and consumed by the masked losses;
      dropping it would train the model to reproduce NEON's empty flight-line
      wedges as dark ground.

    Augmentation is restricted to the dihedral group -- flips and 90-degree
    rotations. Anything else (scaling, shear, colour jitter) would break the
    geometric or radiometric premise of the task: the model must not learn that
    reflectance values are arbitrary, because the spectral losses depend on
    them meaning something. The mask is transformed with the imagery, or it
    would stop marking the pixels it describes.
    """

    def __init__(
        self,
        patch_dir: str | Path,
        split: str = "train",
        scale: int = 4,
        hr_patch: int | None = None,
        augment: bool = True,
        on_the_fly: bool = True,
        degradation: DegradationConfig | None = None,
        bands: Sequence[str] = ("B02", "B03", "B04", "B08"),
        seed: int = 0,
        cache: bool = True,
    ):
        self.dir = Path(patch_dir)
        self.split = split
        self.scale = scale
        self.augment = augment and split == "train"
        self.on_the_fly = on_the_fly
        self.hr_patch = hr_patch
        self.bands = tuple(bands)
        self.degradation = degradation or DegradationConfig(scale=scale, bands=self.bands)
        self.cache = cache
        self._cached: dict[int, dict[str, np.ndarray]] = {}
        # Deterministic per-epoch randomness without sharing state across workers.
        self._seed = seed

        self.shards = sorted((self.dir / split).glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(
                f"no shards in {self.dir / split}. Run scripts/prepare_dataset.py first."
            )
        self.index: list[tuple[int, int]] = []
        self._lengths: list[int] = []
        for shard_id, shard in enumerate(self.shards):
            with np.load(shard) as data:
                n = int(data["hr"].shape[0])
                self._has_lr = "lr" in data
            self._lengths.append(n)
            self.index.extend((shard_id, i) for i in range(n))

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, shard_id: int) -> dict[str, np.ndarray]:
        if shard_id in self._cached:
            return self._cached[shard_id]
        with np.load(self.shards[shard_id]) as data:
            payload = {k: data[k] for k in data.files}
        if self.cache:
            self._cached[shard_id] = payload
        return payload

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_id, row = self.index[idx]
        payload = self._shard(shard_id)
        hr = payload["hr"][row].astype(np.float32) / S2_REFLECTANCE_SCALE
        mask = payload["mask"][row].astype(np.float32) if "mask" in payload else None

        lr = None
        if not self.on_the_fly and "lr" in payload:
            lr = payload["lr"][row].astype(np.float32) / S2_REFLECTANCE_SCALE

        if self.hr_patch and hr.shape[-1] > self.hr_patch:
            hr, lr, mask = self._random_crop(hr, lr, mask, self.hr_patch, idx)

        if self.augment:
            hr, lr, mask = self._augment(hr, lr, mask, idx)

        if lr is None:
            rng = np.random.default_rng(self._seed + idx) if self.split != "train" else None
            lr = degrade(hr, self.degradation, rng)

        batch = {
            "lr": torch.from_numpy(np.ascontiguousarray(lr)),
            "hr": torch.from_numpy(np.ascontiguousarray(hr)),
        }
        # A simulated pair has no holes, so an all-ones mask is the honest
        # value and keeps the batch schema identical across dataset kinds --
        # the training loop then needs no branch.
        batch["mask"] = (
            torch.from_numpy(np.ascontiguousarray(mask)) if mask is not None
            else torch.ones(hr.shape[-2:], dtype=torch.float32)
        )
        return batch

    def _random_crop(self, hr, lr, mask, size: int, idx: int):
        rng = np.random.default_rng(None if self.split == "train" else self._seed + idx)
        max_off = hr.shape[-1] - size
        # Align the crop to the scale grid so the LR/HR pixel lattices stay in
        # phase; the LR window is the same ground, divided by the scale.
        oy = int(rng.integers(0, max_off // self.scale + 1)) * self.scale
        ox = int(rng.integers(0, max_off // self.scale + 1)) * self.scale
        hr = hr[:, oy : oy + size, ox : ox + size]
        if mask is not None:
            mask = mask[oy : oy + size, ox : ox + size]
        if lr is not None:
            ly, lx, ls = oy // self.scale, ox // self.scale, size // self.scale
            lr = lr[:, ly : ly + ls, lx : lx + ls]
        return hr, lr, mask

    def _augment(self, hr, lr, mask, idx: int):
        rng = np.random.default_rng(None if self.split == "train" else self._seed + idx)
        if rng.random() < 0.5:
            hr = hr[:, :, ::-1]
            lr = lr[:, :, ::-1] if lr is not None else None
            mask = mask[:, ::-1] if mask is not None else None
        if rng.random() < 0.5:
            hr = hr[:, ::-1, :]
            lr = lr[:, ::-1, :] if lr is not None else None
            mask = mask[::-1, :] if mask is not None else None
        k = int(rng.integers(0, 4))
        if k:
            hr = np.rot90(hr, k, axes=(1, 2))
            lr = np.rot90(lr, k, axes=(1, 2)) if lr is not None else None
            mask = np.rot90(mask, k, axes=(0, 1)) if mask is not None else None
        return (
            np.ascontiguousarray(hr),
            np.ascontiguousarray(lr) if lr is not None else None,
            np.ascontiguousarray(mask) if mask is not None else None,
        )


class InferencePatchDataset(Dataset):
    """Overlapping LR tiles of one scene, for tiled whole-raster inference."""

    def __init__(self, lr: np.ndarray, tile: int = 256, overlap: int = 32):
        self.lr = np.asarray(lr, dtype=np.float32)
        _, height, width = self.lr.shape
        self.tile = min(tile, height, width)
        self.overlap = min(overlap, max(0, self.tile // 2 - 1))
        step = max(1, self.tile - self.overlap)
        self.positions: list[tuple[int, int]] = []
        ys = list(range(0, max(1, height - self.tile + 1), step))
        xs = list(range(0, max(1, width - self.tile + 1), step))
        if ys[-1] != height - self.tile and height > self.tile:
            ys.append(height - self.tile)
        if xs[-1] != width - self.tile and width > self.tile:
            xs.append(width - self.tile)
        for y in ys:
            for x in xs:
                self.positions.append((y, x))

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        y, x = self.positions[idx]
        tile = self.lr[:, y : y + self.tile, x : x + self.tile]
        return torch.from_numpy(np.ascontiguousarray(tile)), y, x


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader | None, dict]:
    """Construct train/val loaders from a config dict."""
    from ..utils.seed import worker_init_fn

    patch_dir = cfg["data"]["patches"]
    manifest = load_manifest(patch_dir)
    scale = int(cfg["scale"])
    if int(manifest.get("scale", scale)) != scale:
        raise ValueError(
            f"config scale={scale} but dataset was built at scale={manifest['scale']}"
        )
    bands = cfg.get("bands") or manifest["bands"]
    if [str(b).upper() for b in bands] != [str(b).upper() for b in manifest["bands"]]:
        raise ValueError(
            f"config bands {list(bands)} do not match the dataset's {manifest['bands']}. "
            "The shards store bands in the dataset's order; training on a different "
            "assignment would silently swap channels."
        )
    degradation = DegradationConfig.from_dict(
        cfg["data"].get("degradation", {}), scale=scale, bands=bands
    )

    # Native pairs are real observations. Regenerating the LR side from the HR
    # one would throw away the actual Sentinel-2 measurement and train on a
    # simulation instead -- the single most expensive mistake available here.
    native = manifest.get("pairs_kind") == "native"
    on_the_fly = False if native else cfg["data"].get("on_the_fly_degradation", True)

    common = dict(
        patch_dir=patch_dir,
        scale=scale,
        hr_patch=cfg["data"].get("hr_patch"),
        on_the_fly=on_the_fly,
        degradation=degradation,
        bands=bands,
        seed=cfg.get("seed", 1337),
    )
    train_set = SRMPatchDataset(split=cfg["data"].get("train_shards", "train"),
                                augment=cfg["data"].get("augment", True), **common)
    try:
        val_set = SRMPatchDataset(split=cfg["data"].get("val_shards", "val"),
                                  augment=False, **common)
    except FileNotFoundError:
        val_set = None

    workers = int(cfg["train"].get("num_workers", 0))
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn if workers else None,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=max(1, int(cfg["train"]["batch_size"]) // 2),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        if val_set is not None
        else None
    )
    return train_loader, val_loader, manifest


def build_test_loader(cfg: dict, batch_size: int | None = None):
    """The held-out split, or None when the dataset has no `test/` shards.

    Kept separate from `build_dataloaders` on purpose: the test split must not
    be reachable from the training loop, so no checkpoint can be selected
    against it.
    """
    manifest = load_manifest(cfg["data"]["patches"])
    bands = cfg.get("bands") or manifest["bands"]
    try:
        dataset = SRMPatchDataset(
            patch_dir=cfg["data"]["patches"],
            split=cfg["data"].get("test_shards", "test"),
            scale=int(cfg["scale"]),
            hr_patch=cfg["data"].get("hr_patch"),
            augment=False,
            on_the_fly=(
                False if manifest.get("pairs_kind") == "native"
                else cfg["data"].get("on_the_fly_degradation", True)
            ),
            degradation=DegradationConfig.from_dict(
                cfg["data"].get("degradation", {}), scale=int(cfg["scale"]), bands=bands
            ),
            bands=bands,
            seed=cfg.get("seed", 1337),
        )
    except FileNotFoundError:
        return None, manifest
    loader = DataLoader(
        dataset,
        batch_size=batch_size or max(1, int(cfg["train"]["batch_size"]) // 2),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return loader, manifest
