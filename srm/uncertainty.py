"""Per-pixel uncertainty for super-resolved imagery.

Some of the detail in an SR product was measured and some was inferred, and a
GeoTIFF gives you no way to tell them apart. That is the central scientific
problem with generative enhancement of Earth observation data: the output is
a hypothesis rendered with the same visual authority as an observation.

This module estimates three separable quantities and combines them:

1. **Epistemic spread** -- how much the answer changes under perturbations the
   answer should not depend on. Two estimators, both reference-free:
   *MC-dropout* (needs `model.dropout > 0`) samples the model's own posterior;
   *TTA disagreement* measures variance across the eight dihedral symmetries
   and works on any trained model. High spread means the model itself is
   undecided, which is the honest definition of "invented".

2. **Observation inconsistency** -- Wald residual, `degrade(SR) - LR`, mapped
   back to the fine grid. Non-zero means the model added or moved energy the
   sensor never recorded. Unlike the spread, this is anchored to data.

3. **Detail attribution** -- how much high-frequency content is present in the
   SR result beyond what bicubic interpolation of the same input produces.
   This is the *amount* of detail that had to come from the prior.

The combined reliability map is what should ship alongside any SRM product,
and what a downstream analyst should threshold on before trusting a small
feature. A bright, sharp building in a low-reliability pixel is a suggestion,
not a measurement.

Usage:
    python -m srm.uncertainty --checkpoint checkpoints/edsr_x4/best.pt \\
        --input data/raw/scene.tif --outdir outputs/scene_uncertainty --samples 8
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import resolve_device
from .data.degradation import DegradationConfig, degrade, upsample_bicubic
from .infer import super_resolve
from .models.registry import load_checkpoint
from .utils.geo import read_raster, write_raster
from .utils.logging import get_logger
from .validation import inspect_raster, resolve_band_mapping

log = get_logger("srm.uncertainty")


@dataclass
class UncertaintyMaps:
    sr: np.ndarray                  # (C, H*s, W*s) best estimate
    spread: np.ndarray              # (H*s, W*s) ensemble std, reflectance units
    inconsistency: np.ndarray       # (H*s, W*s) Wald residual magnitude
    inferred_detail: np.ndarray     # (H*s, W*s) high-frequency energy added
    reliability: np.ndarray         # (H*s, W*s) in [0, 1]; 1 = trustworthy

    def summary(self) -> dict[str, float]:
        return {
            "spread_mean": float(np.nanmean(self.spread)),
            "spread_p95": float(np.nanpercentile(self.spread, 95)),
            "inconsistency_mean": float(np.nanmean(self.inconsistency)),
            "inconsistency_p95": float(np.nanpercentile(self.inconsistency, 95)),
            "inferred_detail_mean": float(np.nanmean(self.inferred_detail)),
            "reliability_mean": float(np.nanmean(self.reliability)),
            "fraction_low_reliability": float(np.nanmean(self.reliability < 0.5)),
            "fraction_high_reliability": float(np.nanmean(self.reliability >= 0.8)),
        }


def _dihedral(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(x, k, axes=(-2, -1))
    return np.flip(out, axis=-1) if flip else out


def _undihedral(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.flip(x, axis=-1) if flip else x
    return np.rot90(out, -k, axes=(-2, -1))


@torch.no_grad()
def sample_predictions(
    lr: np.ndarray,
    model,
    samples: int = 8,
    method: str = "auto",
    device: str = "cpu",
    **infer_kwargs,
) -> np.ndarray:
    """Return (N, C, H*s, W*s) predictions from the chosen stochastic source.

    `method="auto"` prefers MC-dropout when the model has dropout layers and
    falls back to TTA disagreement, so this works on any checkpoint.
    """
    model.eval().to(device)
    has_dropout = any(
        isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)) and m.p > 0
        for m in model.modules()
    )
    if method == "auto":
        method = "mc_dropout" if has_dropout else "tta"
    if method == "mc_dropout" and not has_dropout:
        raise ValueError(
            "mc_dropout requested but the model has no active dropout. "
            "Train with model.dropout > 0, or use method='tta'."
        )

    predictions = []
    if method == "mc_dropout":
        n_enabled = model.enable_mc_dropout()
        log.info("MC-dropout over %d layers, %d samples", n_enabled, samples)
        for _ in range(samples):
            predictions.append(super_resolve(lr, model, device=device, progress=False,
                                             **infer_kwargs))
        model.eval()
    else:
        transforms = [(k, f) for k in range(4) for f in (False, True)][:max(2, samples)]
        log.info("TTA disagreement over %d symmetries", len(transforms))
        for k, flip in transforms:
            transformed = np.ascontiguousarray(_dihedral(lr, k, flip))
            out = super_resolve(transformed, model, device=device, progress=False,
                                **infer_kwargs)
            predictions.append(np.ascontiguousarray(_undihedral(out, k, flip)))
    return np.stack(predictions)


def _high_frequency_energy(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Per-pixel high-pass magnitude, averaged over bands."""
    from scipy.ndimage import gaussian_filter

    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    energy = np.zeros(image.shape[-2:], dtype=np.float32)
    for band in range(image.shape[0]):
        low = gaussian_filter(image[band], sigma=sigma, mode="reflect")
        energy += np.abs(image[band] - low)
    return energy / image.shape[0]


def _normalise(x: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Scale to [0, 1] by a high percentile, so a few outliers cannot flatten it."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    hi = float(np.percentile(finite, percentile))
    if hi <= 0:
        return np.zeros_like(x)
    return np.clip(x / hi, 0.0, 1.0)


def estimate(
    lr: np.ndarray,
    model,
    degradation: DegradationConfig,
    samples: int = 8,
    method: str = "auto",
    device: str = "cpu",
    weights: tuple[float, float, float] = (0.45, 0.40, 0.15),
    **infer_kwargs,
) -> UncertaintyMaps:
    """Run the full uncertainty estimation for one scene."""
    scale = int(getattr(model, "scale", degradation.scale))
    preds = sample_predictions(lr, model, samples, method, device, **infer_kwargs)
    sr = preds.mean(axis=0)
    # Spread across samples, averaged over bands: one map per pixel.
    spread = preds.std(axis=0).mean(axis=0)

    # Wald residual, computed on the LR grid then expanded to the fine grid.
    # Nearest-neighbour expansion is correct here: the residual is genuinely
    # only known at LR resolution, and interpolating it would imply otherwise.
    sr_down = degrade(sr, degradation)
    h = min(sr_down.shape[1], lr.shape[1])
    w = min(sr_down.shape[2], lr.shape[2])
    residual = np.abs(sr_down[:, :h, :w] - np.nan_to_num(lr[:, :h, :w])).mean(axis=0)
    inconsistency = np.repeat(np.repeat(residual, scale, axis=0), scale, axis=1)
    inconsistency = _pad_to(inconsistency, sr.shape[-2:])

    baseline = upsample_bicubic(np.nan_to_num(lr), scale)
    inferred = np.clip(
        _high_frequency_energy(sr) - _high_frequency_energy(baseline), 0.0, None
    )
    inferred = _pad_to(inferred, sr.shape[-2:])

    w_spread, w_incons, w_inferred = weights
    risk = (
        w_spread * _normalise(spread)
        + w_incons * _normalise(inconsistency)
        + w_inferred * _normalise(inferred)
    ) / max(1e-6, sum(weights))
    reliability = np.clip(1.0 - risk, 0.0, 1.0).astype(np.float32)

    return UncertaintyMaps(
        sr=sr.astype(np.float32),
        spread=spread.astype(np.float32),
        inconsistency=inconsistency.astype(np.float32),
        inferred_detail=inferred.astype(np.float32),
        reliability=reliability,
    )


def _pad_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    h = min(array.shape[0], shape[0])
    w = min(array.shape[1], shape[1])
    out[:h, :w] = array[:h, :w]
    if h < shape[0]:
        out[h:, :w] = out[h - 1, :w]
    if w < shape[1]:
        out[:, w:] = out[:, w - 1 : w]
    return out


def run_file(
    input_path: str | Path,
    outdir: str | Path,
    checkpoint: str | Path,
    samples: int = 8,
    method: str = "auto",
    device: str = "auto",
    tile: int = 256,
    overlap: int = 32,
) -> dict:
    device = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device=device)
    cfg = payload["config"]
    scale = int(cfg.get("scale", 4))
    degradation = DegradationConfig.from_dict(
        cfg.get("data", {}).get("degradation", {}), scale=scale,
        bands=tuple(cfg.get("bands", ())),
    )

    # Bands by name, not by position: `array[:in_channels]` on a 12-band
    # Sentinel-2 stack takes B1-B4, not RGB+NIR, and every map below would
    # then describe the wrong spectra.
    raster = inspect_raster(input_path)
    if not raster.ok:
        raise ValueError(
            f"{input_path} cannot be processed:\n  - " + "\n  - ".join(raster.errors)
        )
    trained = (payload.get("provenance") or {}).get("bands") or cfg.get("bands")
    band_idx, _, notes = resolve_band_mapping(raster, trained, model.in_channels)
    for note in notes:
        log.info("%s", note)
    array, meta = read_raster(input_path, bands=band_idx)
    maps = estimate(array, model, degradation, samples=samples, method=method,
                    device=device, tile=tile, overlap=overlap)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    band_names = meta.band_names[: maps.sr.shape[0]] or None
    write_raster(outdir / "sr.tif", maps.sr, meta, scale_factor=scale, band_names=band_names)
    for name, data, description in (
        ("uncertainty_spread.tif", maps.spread, "ensemble std (reflectance)"),
        ("uncertainty_inconsistency.tif", maps.inconsistency, "Wald residual (reflectance)"),
        ("uncertainty_inferred_detail.tif", maps.inferred_detail, "high-freq energy added"),
        ("reliability.tif", maps.reliability, "0-1, higher is more trustworthy"),
    ):
        write_raster(outdir / name, data[None], meta, scale_factor=scale,
                     band_names=[description])

    summary = maps.summary()
    summary.update({
        "input": str(input_path),
        "checkpoint": str(checkpoint),
        "method": method,
        "samples": samples,
        "scale": scale,
    })
    (outdir / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    log.info("uncertainty -> %s | reliability mean %.3f | %.1f%% of pixels below 0.5",
             outdir, summary["reliability_mean"], 100 * summary["fraction_low_reliability"])
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Per-pixel SR uncertainty maps")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--method", default="auto", choices=["auto", "mc_dropout", "tta"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = run_file(args.input, args.outdir, args.checkpoint, samples=args.samples,
                       method=args.method, device=args.device, tile=args.tile,
                       overlap=args.overlap)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
