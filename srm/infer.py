"""Whole-scene inference: tiled, blended, georeferenced.

A Sentinel-2 tile is 10980x10980 and will not fit on any GPU, so inference is
tiled. Two details make the difference between a usable product and a visibly
gridded one:

* **Overlap with a cosine-tapered weight.** Each tile is weighted by a raised
  cosine that falls to zero at its border, and the accumulated weights are
  normalised. Seams disappear instead of being averaged into a visible grid.
* **Context, not just coverage.** Tiles overlap so every output pixel is
  produced from a position well inside some tile, where the network has full
  receptive-field context rather than reflect-padded edge.

The output GeoTIFF carries the source CRS with the pixel size divided by the
scale factor, so it overlays the input exactly in any GIS.

Usage:
    python -m srm.infer --checkpoint checkpoints/edsr_x4/best.pt \\
        --input data/raw/scene.tif --output outputs/scene_sr.tif --tta
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .config import resolve_device
from .data.dataset import InferencePatchDataset
from .models.registry import load_checkpoint
from .utils.geo import pixel_size, read_raster, write_raster
from .utils.logging import get_logger
from .validation import inspect_raster, resolve_band_mapping

log = get_logger("srm.infer")


def cosine_window(size: int, overlap: int) -> np.ndarray:
    """2-D separable taper: 1 in the middle, 0 at the edges of the overlap band."""
    ramp = np.ones(size, dtype=np.float32)
    if overlap > 0:
        taper = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, overlap, dtype=np.float32)))
        ramp[:overlap] = taper
        ramp[-overlap:] = taper[::-1]
    window = np.outer(ramp, ramp).astype(np.float32)
    # A strictly zero border would leave un-covered pixels at the scene edge,
    # where no neighbouring tile contributes any weight.
    return np.maximum(window, 1e-3)


def _tta_forward(model, tile: torch.Tensor) -> torch.Tensor:
    """x8 self-ensemble over the dihedral group.

    Averaging the eight symmetries costs 8x compute and reliably buys
    0.1-0.3 dB, because SR errors are not symmetric under flips and rotations
    so they partially cancel.
    """
    outputs = []
    for k in range(4):
        rotated = torch.rot90(tile, k, dims=(-2, -1))
        for flip in (False, True):
            inp = torch.flip(rotated, dims=(-1,)) if flip else rotated
            out = model(inp)
            if flip:
                out = torch.flip(out, dims=(-1,))
            outputs.append(torch.rot90(out, -k, dims=(-2, -1)))
    return torch.stack(outputs).mean(dim=0)


@torch.no_grad()
def super_resolve(
    lr: np.ndarray,
    model,
    scale: int | None = None,
    tile: int = 256,
    overlap: int = 32,
    batch_size: int = 4,
    tta: bool = False,
    device: str = "cpu",
    progress: bool = True,
) -> np.ndarray:
    """(C, H, W) reflectance -> (C, H*s, W*s), tiled and blended."""
    model.eval().to(device)
    scale = int(scale or getattr(model, "scale", 4))
    lr = np.asarray(lr, dtype=np.float32)
    nan_mask = ~np.isfinite(lr)
    if nan_mask.any():
        # The network cannot consume NaN. Fill with the band mean, restore after.
        for band in range(lr.shape[0]):
            band_mask = nan_mask[band]
            if band_mask.any():
                lr[band][band_mask] = float(np.nanmean(lr[band])) if np.isfinite(lr[band]).any() else 0.0

    channels, height, width = lr.shape
    dataset = InferencePatchDataset(lr, tile=tile, overlap=overlap)
    out_h, out_w = height * scale, width * scale
    accum = np.zeros((channels, out_h, out_w), dtype=np.float32)
    weights = np.zeros((1, out_h, out_w), dtype=np.float32)
    window = cosine_window(dataset.tile * scale, dataset.overlap * scale)[None]

    n_tiles = len(dataset)
    start = time.time()
    for begin in range(0, n_tiles, batch_size):
        chunk = [dataset[i] for i in range(begin, min(begin + batch_size, n_tiles))]
        batch = torch.stack([c[0] for c in chunk]).to(device)
        sr = _tta_forward(model, batch) if tta else model(batch)
        sr = sr.float().cpu().numpy()
        for (_, y, x), patch in zip(chunk, sr):
            oy, ox = y * scale, x * scale
            ph, pw = patch.shape[-2:]
            accum[:, oy : oy + ph, ox : ox + pw] += patch * window[:, :ph, :pw]
            weights[:, oy : oy + ph, ox : ox + pw] += window[:, :ph, :pw]
        if progress and n_tiles > batch_size:
            done = min(begin + batch_size, n_tiles)
            log.info("tiles %d/%d (%.1fs)", done, n_tiles, time.time() - start)

    result = accum / np.maximum(weights, 1e-6)
    if nan_mask.any():
        upscaled_mask = np.repeat(np.repeat(nan_mask, scale, axis=1), scale, axis=2)
        result[upscaled_mask[:, :out_h, :out_w]] = np.nan
    return result.astype(np.float32)


def super_resolve_file(
    input_path: str | Path,
    output_path: str | Path,
    checkpoint: str | Path,
    tile: int = 256,
    overlap: int = 32,
    batch_size: int = 4,
    tta: bool = False,
    device: str = "auto",
    bands: tuple[int, ...] | None = None,
    save_baseline: bool = False,
) -> dict:
    """Super-resolve a GeoTIFF and write a georeferenced result."""
    device = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device=device)
    cfg = payload["config"]
    scale = int(cfg.get("scale", getattr(model, "scale", 4)))

    # Refuse before reading a multi-gigabyte scene, not after.
    report = inspect_raster(input_path)
    if not report.ok:
        raise ValueError(
            f"{input_path} cannot be processed:\n  - " + "\n  - ".join(report.errors)
        )
    for warning in report.warnings:
        log.warning("%s", warning)

    if bands is None:
        # Select by the band names the checkpoint was trained on, falling back
        # to file order only when the names cannot be matched. Positional
        # selection on a 12-band product silently substitutes the wrong bands.
        trained_on = (payload.get("provenance") or {}).get("bands") or cfg.get("bands")
        bands, resolved, notes = resolve_band_mapping(report, trained_on, model.in_channels)
        for note in notes:
            log.info("%s", note)
    else:
        bands = list(bands)
        resolved = [report.band_names[i - 1] for i in bands if i <= report.band_count]

    array, meta = read_raster(input_path, bands=bands)
    if array.shape[0] != model.in_channels:
        raise ValueError(
            f"{input_path}: selected {array.shape[0]} bands but the model expects "
            f"{model.in_channels}. Adjust the band mapping."
        )

    gsd_in = pixel_size(meta)
    log.info("input %s | %s | %s m GSD | model %s x%d%s",
             Path(input_path).name, array.shape, gsd_in, cfg.get("model", {}).get("name"),
             scale, " +TTA" if tta else "")

    started = time.time()
    sr = super_resolve(array, model, scale, tile, overlap, batch_size, tta, device)
    elapsed = time.time() - started

    band_names = meta.band_names or [f"band_{i+1}" for i in range(sr.shape[0])]
    write_raster(output_path, sr, meta, scale_factor=scale, band_names=band_names)

    baseline_path = None
    if save_baseline:
        from .data.degradation import upsample_bicubic

        baseline_path = str(Path(output_path).with_name(Path(output_path).stem + "_bicubic.tif"))
        write_raster(baseline_path, upsample_bicubic(array, scale), meta,
                     scale_factor=scale, band_names=band_names)

    extent = report.ground_extent_m()
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "baseline": baseline_path,
        "checkpoint": str(checkpoint),
        "model": cfg.get("model", {}).get("name"),
        "used_ema": payload.get("used_ema", False),
        "epoch": payload.get("epoch"),
        "scale": scale,
        "bands_used": list(bands),
        "band_names": list(resolved),
        "input_shape": list(array.shape),
        "output_shape": list(sr.shape),
        "gsd_in_m": gsd_in,
        "gsd_out_m": (gsd_in / scale) if gsd_in else None,
        "crs": report.crs,
        # The whole point of dividing the transform rather than resampling:
        # the product covers exactly the same ground as its parent.
        "ground_extent_m": list(extent) if extent else None,
        "valid_fraction": round(report.valid_fraction, 6),
        "seconds": round(elapsed, 2),
        "megapixels_per_second": (
            round(sr.shape[-1] * sr.shape[-2] / 1e6 / elapsed, 3) if elapsed > 0 else None
        ),
        "tta": tta,
        "warnings": report.warnings,
    }
    log.info("wrote %s | %s m GSD | %.1fs", output_path, summary["gsd_out_m"], elapsed)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Super-resolve a GeoTIFF scene")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tta", action="store_true", help="x8 self-ensemble")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-baseline", action="store_true",
                        help="also write the bicubic upsample for comparison")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    import json

    args = parse_args(argv)
    summary = super_resolve_file(
        args.input, args.output, args.checkpoint,
        tile=args.tile, overlap=args.overlap, batch_size=args.batch_size,
        tta=args.tta, device=args.device, save_baseline=args.save_baseline,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
