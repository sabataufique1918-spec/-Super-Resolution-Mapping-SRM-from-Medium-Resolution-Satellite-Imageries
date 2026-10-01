"""Ensemble disagreement across independently trained checkpoints.

What this measures, stated precisely because the distinction is the whole
point: given N models trained on the same task from different random seeds,
this reports **how much they disagree with each other** at every pixel.

What it does *not* measure: whether any of them is right. Models that share an
architecture, a training set and a degradation assumption share their blind
spots, so they can agree confidently and be wrong together -- systematically
so, over land cover that was rare in training. Low disagreement is therefore
necessary for trust, not sufficient for it.

That is why every output here is named `disagreement` and never
`confidence`, `reliability` or `probability`. A map labelled "95% confident"
invites an analyst to treat an invented building as a measurement. A map
labelled "the models disagree by 0.02 reflectance here" does not.

With a single checkpoint there is nothing to compare and the honest result is
no map at all -- `estimate_ensemble` raises, and the caller is expected to
report "ensemble disagreement unavailable" rather than substituting a
single-model proxy.

For the reference-free single-model estimators (MC-dropout spread, TTA
spread, Wald observation inconsistency), see `srm.uncertainty`. Those measure
different things and are not interchangeable with this.

Usage:
    python -m srm.ensemble --checkpoints ckpt_a.pt ckpt_b.pt ckpt_c.pt \\
        --input scene.tif --outdir outputs/scene_ensemble
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import resolve_device
from .infer import super_resolve
from .models.registry import inspect_checkpoint, load_checkpoint
from .utils.geo import read_raster, write_raster
from .utils.logging import get_logger
from .validation import inspect_raster, resolve_band_mapping

log = get_logger("srm.ensemble")

#: Raised instead of returning a degraded stand-in, so a caller cannot
#: accidentally present a single-model quantity as an ensemble one.
class EnsembleUnavailable(RuntimeError):
    """Fewer than two compatible checkpoints were supplied."""


@dataclass
class EnsembleResult:
    """Ensemble mean and the spread around it.

    `per_band_std` is the raw quantity: the population standard deviation
    across members, per band, in reflectance units. Everything else is a
    documented reduction of it.
    """

    mean: np.ndarray            # (C, H, W) ensemble mean prediction
    per_band_std: np.ndarray    # (C, H, W) std across members, reflectance units
    disagreement: np.ndarray    # (H, W) aggregate, see `AGGREGATE_DEFINITION`
    max_band_std: np.ndarray    # (H, W) worst single band, for a strict threshold
    members: list[dict]         # provenance of each checkpoint used
    seconds: float

    #: Written into every report and shown in the interface, because an
    #: aggregate map with an undocumented reduction is not interpretable.
    AGGREGATE_DEFINITION = (
        "Root-mean-square of the per-band standard deviations across ensemble "
        "members, in surface reflectance units (0-1). One value per pixel; "
        "0 means the members produced identical values there. This is "
        "inter-model disagreement, not an error bar against ground truth."
    )

    def summary(self) -> dict:
        finite = self.disagreement[np.isfinite(self.disagreement)]
        if finite.size == 0:
            return {"n_members": len(self.members), "valid_pixels": 0}
        return {
            "n_members": len(self.members),
            "aggregate_definition": self.AGGREGATE_DEFINITION,
            "units": "surface reflectance (0-1)",
            "disagreement_mean": float(finite.mean()),
            "disagreement_median": float(np.median(finite)),
            "disagreement_p95": float(np.percentile(finite, 95)),
            "disagreement_max": float(finite.max()),
            "per_band_std_mean": [
                float(np.nanmean(self.per_band_std[c])) for c in range(self.per_band_std.shape[0])
            ],
            "seconds": round(self.seconds, 2),
        }


def check_ensemble(paths: Sequence[str | Path]) -> list[dict]:
    """Validate that these checkpoints can be ensembled, or say why not.

    Members must agree on scale and band order. Averaging a x4 model with a
    x2 one, or one trained on a different band assignment, produces a mean
    that is meaningless and a spread that looks like uncertainty but is
    really a configuration mismatch.
    """
    unique = list(dict.fromkeys(str(p) for p in paths))
    if len(unique) < 2:
        raise EnsembleUnavailable(
            f"ensemble disagreement needs at least 2 distinct checkpoints, got "
            f"{len(unique)}. Train a second model with a different seed:\n"
            f"  python -m srm.train --config <cfg> --set seed=2024 "
            f"experiment=<name>_s2024"
        )
    infos = [inspect_checkpoint(p) for p in unique]
    scales = {i["scale"] for i in infos}
    if len(scales) > 1:
        raise EnsembleUnavailable(f"members disagree on scale: {sorted(scales)}")
    bands = {tuple(i["bands"]) for i in infos}
    if len(bands) > 1:
        raise EnsembleUnavailable(f"members disagree on band order: {sorted(bands)}")
    seeds = [i.get("seed") for i in infos]
    if len(set(seeds)) == 1 and seeds[0] is not None:
        # Not fatal -- the weights may still differ -- but it means the spread
        # is not sampling the thing it is supposed to sample.
        log.warning(
            "all %d members report seed=%s. Ensemble disagreement is only "
            "meaningful across *independently* trained models; identical seeds "
            "understate it.", len(infos), seeds[0],
        )
    return infos


def estimate_ensemble(
    lr: np.ndarray,
    checkpoints: Sequence[str | Path],
    device: str = "cpu",
    tile: int = 256,
    overlap: int = 32,
    batch_size: int = 4,
    tta: bool = False,
    progress: bool = False,
) -> EnsembleResult:
    """Run every checkpoint over the same scene and measure the spread."""
    infos = check_ensemble(checkpoints)
    started = time.time()

    predictions: list[np.ndarray] = []
    for info in infos:
        model, payload = load_checkpoint(info["path"], device=device)
        log.info("member %s (%s, epoch %d, seed %s)",
                 info["name"], info["architecture"], info["epoch"], info["seed"])
        predictions.append(
            super_resolve(lr, model, scale=info["scale"], tile=tile, overlap=overlap,
                          batch_size=batch_size, tta=tta, device=device, progress=progress)
        )
        del model, payload

    stack = np.stack(predictions)                     # (N, C, H, W)
    mean = np.nanmean(stack, axis=0)
    # Population std (ddof=0): the spread of the members we actually have, not
    # an estimate of the spread of a hypothetical infinite ensemble. With 2-5
    # members the distinction is large and ddof=1 would overstate it.
    per_band_std = np.nanstd(stack, axis=0)
    disagreement = np.sqrt(np.nanmean(per_band_std ** 2, axis=0))
    max_band_std = np.nanmax(per_band_std, axis=0)

    # Pixels no member could predict stay unknown rather than becoming zeros,
    # which would read on a map as perfect agreement.
    invalid = ~np.isfinite(stack).all(axis=0).all(axis=0)
    disagreement[invalid] = np.nan
    max_band_std[invalid] = np.nan

    return EnsembleResult(
        mean=mean.astype(np.float32),
        per_band_std=per_band_std.astype(np.float32),
        disagreement=disagreement.astype(np.float32),
        max_band_std=max_band_std.astype(np.float32),
        members=infos,
        seconds=time.time() - started,
    )


def estimate_from_file(
    input_path: str | Path,
    checkpoints: Sequence[str | Path],
    outdir: str | Path,
    bands: Sequence[int] | None = None,
    device: str = "auto",
    tile: int = 256,
    overlap: int = 32,
    tta: bool = False,
) -> dict:
    """Ensemble a GeoTIFF and write the mean, the per-band spread and a report.

    Bands are resolved by **name** against the members' training order, like
    every other entry point. Reading the file and slicing `array[:n_bands]`
    takes B1-B4 from a 12-band Sentinel-2 stack, which is not RGB+NIR: the
    members then run on the wrong spectra and the disagreement map describes
    a scene nobody asked about. Measured on one tile, that slicing reported
    mean disagreement 0.0140 where the correct bands give 0.00093.
    """
    device = resolve_device(device)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    infos = check_ensemble(checkpoints)
    scale = infos[0]["scale"]
    trained_bands = list(infos[0].get("bands") or [])
    n_bands = len(trained_bands) or 4

    report = inspect_raster(input_path)
    if not report.ok:
        raise ValueError(
            f"{input_path} cannot be processed:\n  - " + "\n  - ".join(report.errors)
        )
    if bands is None:
        bands, resolved, notes = resolve_band_mapping(report, trained_bands, n_bands)
        for note in notes:
            log.info("%s", note)
    else:
        bands = list(bands)
        resolved = [report.band_names[i - 1] for i in bands if i <= report.band_count]

    array, meta = read_raster(input_path, bands=list(bands))
    if array.shape[0] != n_bands:
        raise ValueError(
            f"selected {array.shape[0]} bands but the members expect {n_bands}"
        )

    result = estimate_ensemble(array, checkpoints, device=device, tile=tile,
                               overlap=overlap, tta=tta)

    # The members' band order is the authoritative answer to what each channel
    # holds. The source file's own descriptions are often absent (SEN2NEON
    # names none of its bands), and "band_3" for a channel that is B04 is
    # exactly the ambiguity a georeferenced export should remove.
    names = [str(b) for b in (infos[0].get("bands") or [])][:n_bands]
    if len(names) != n_bands:
        names = meta.band_names[:n_bands] or [f"band_{i+1}" for i in range(n_bands)]
    write_raster(outdir / "ensemble_mean.tif", result.mean, meta,
                 scale_factor=scale, band_names=names)
    write_raster(outdir / "ensemble_disagreement.tif", result.disagreement[None], meta,
                 scale_factor=scale, band_names=["ensemble_disagreement_reflectance"])
    write_raster(outdir / "ensemble_disagreement_per_band.tif", result.per_band_std, meta,
                 scale_factor=scale, band_names=[f"{n}_std" for n in names])

    report_out = {
        "input": str(input_path),
        "outdir": str(outdir),
        "bands_used": list(bands),
        "band_names": list(resolved),
        "crs": report.crs,
        "members": [
            {k: m[k] for k in ("path", "name", "experiment", "architecture",
                               "epoch", "seed", "bands", "scale")}
            for m in result.members
        ],
        "summary": result.summary(),
        "interpretation": (
            "This is disagreement between independently trained models, not a "
            "probability that the prediction is correct. Members sharing an "
            "architecture and training set share blind spots and can agree "
            "while all being wrong."
        ),
    }
    (outdir / "ensemble_report.json").write_text(
        json.dumps(report_out, indent=2), encoding="utf-8")
    log.info("wrote %s (%d members, bands %s, %.1fs)",
             outdir, len(result.members), list(bands), result.seconds)
    return report_out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ensemble disagreement across independently trained checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="two or more independently trained checkpoints")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        report = estimate_from_file(
            args.input, args.checkpoints, args.outdir,
            device=args.device, tile=args.tile, overlap=args.overlap, tta=args.tta,
        )
    except EnsembleUnavailable as exc:
        raise SystemExit(f"ensemble disagreement unavailable: {exc}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
