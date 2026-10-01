"""Orchestration layer between the interface and the pipeline.

`app/app.py` is a view: it collects inputs, calls into here, and renders what
comes back. Every model construction, raster read, reprojection and metric
computation lives on this side, for two reasons.

* The dashboard and the CLI must produce the same numbers. They do, because
  both call `super_resolve`, `compute_metrics` and `write_raster` through the
  same functions rather than each having a copy.
* Streamlit re-runs its script top to bottom on every widget interaction. Any
  work that lives in the view is work that happens again when someone moves a
  slider. Everything expensive is a plain function here, so the view can cache
  it or stash the result in session state and re-render for free.

Nothing here imports Streamlit.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import resolve_device
from .data.degradation import upsample_bicubic
from .ensemble import EnsembleUnavailable, EnsembleResult, estimate_ensemble
from .infer import super_resolve
from .metrics.quality import METRIC_INFO, compute_metrics, consistency_error, sharpness_gain
from .models.registry import inspect_checkpoint, load_checkpoint
from .utils.geo import GeoMeta, read_raster, scale_transform, write_raster
from .utils.logging import get_logger
from .validation import InputReport, inspect_raster, resolve_band_mapping

log = get_logger("srm.service")

#: Metrics the evaluation panel reports, in display order.
REPORT_METRICS = ("mae", "rmse", "psnr", "ssim", "sam")


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #

def discover_checkpoints(roots: Sequence[str | Path]) -> list[dict]:
    """Every readable checkpoint under `roots`, newest first.

    A file that cannot be opened is reported with its error rather than
    dropped, so a corrupt checkpoint shows up in the picker as a problem
    instead of silently not existing.
    """
    found: list[dict] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.pt")):
            try:
                info = inspect_checkpoint(path)
            except Exception as exc:
                found.append({
                    "path": str(path), "name": f"{path.parent.name}/{path.stem}",
                    "error": f"{type(exc).__name__}: {exc}", "epoch": 0,
                    "bands": [], "scale": 0, "architecture": "?", "metrics": {},
                })
                continue
            info["error"] = None
            info["mtime"] = path.stat().st_mtime
            found.append(info)
    found.sort(key=lambda i: i.get("mtime", 0), reverse=True)
    return found


def trained_checkpoints(infos: Sequence[dict]) -> list[dict]:
    """Only those that record a completed epoch and loaded cleanly.

    AI enhancement requires a genuinely trained checkpoint. An untrained or
    unreadable file is excluded here rather than being offered and then
    silently producing a bicubic-like result.
    """
    return [i for i in infos if not i.get("error") and int(i.get("epoch", 0)) >= 1]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def stage_upload(data: bytes, filename: str, directory: str | Path) -> Path:
    """Write an uploaded file to disk; rasterio needs a real path, not a buffer."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(filename).name
    path.write_bytes(data)
    return path


def describe_input(path: str | Path) -> InputReport:
    return inspect_raster(path)


def plan_bands(report: InputReport, checkpoint: dict) -> dict:
    """Default band mapping for this file and this checkpoint, plus the reasoning."""
    expected = len(checkpoint.get("bands") or []) or 4
    try:
        indices, names, notes = resolve_band_mapping(report, checkpoint.get("bands"), expected)
        return {"indices": indices, "names": names, "notes": notes, "error": None,
                "expected": expected}
    except ValueError as exc:
        return {"indices": [], "names": [], "notes": [], "error": str(exc),
                "expected": expected}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    """Everything one press of Run produced. Held in session state and re-rendered."""

    lr: np.ndarray                     # (C, h, w) input reflectance
    sr: np.ndarray                     # (C, h*s, w*s) model output
    bicubic: np.ndarray                # (C, h*s, w*s) labelled baseline
    meta: GeoMeta
    report: InputReport
    checkpoint: dict
    scale: int
    band_indices: list[int]
    band_names: list[str]
    device: str
    seconds_model: float
    seconds_bicubic: float
    consistency: dict[str, float]
    sharpness_gain: float
    ensemble: EnsembleResult | None = None
    ensemble_error: str | None = None
    evaluation: dict | None = None
    written: dict[str, str] = field(default_factory=dict)

    @property
    def output_gsd(self) -> float | None:
        return self.report.output_gsd(self.scale)

    @property
    def product_band_names(self) -> list[str]:
        """Band descriptions written into the exported GeoTIFF.

        The checkpoint's band order is the authoritative answer to "what is
        in channel 2". Falling back to the source file's descriptions gives
        `band_3` for a product whose third channel is B04, which is exactly
        the ambiguity the export is supposed to remove.
        """
        trained = self.checkpoint.get("bands") or []
        if len(trained) == self.sr.shape[0]:
            return [str(b) for b in trained]
        return [str(n) for n in self.band_names] or [
            f"band_{i + 1}" for i in range(self.sr.shape[0])
        ]

    def run_summary(self) -> dict:
        extent = self.report.ground_extent_m()
        return {
            "input": self.report.path,
            "checkpoint": self.checkpoint.get("path"),
            "experiment": self.checkpoint.get("experiment"),
            "architecture": self.checkpoint.get("architecture"),
            "epoch": self.checkpoint.get("epoch"),
            "seed": self.checkpoint.get("seed"),
            "scale": self.scale,
            "bands_selected": self.band_indices,
            "band_names": self.band_names,
            "device": self.device,
            "input_shape": list(self.lr.shape),
            "output_shape": list(self.sr.shape),
            "crs": self.report.crs,
            "input_gsd_m": self.report.pixel_size,
            "output_gsd_m": self.output_gsd,
            "ground_extent_m": list(extent) if extent else None,
            "valid_fraction": round(self.report.valid_fraction, 6),
            "seconds_model": round(self.seconds_model, 3),
            "seconds_bicubic": round(self.seconds_bicubic, 3),
            "sharpness_gain_vs_bicubic": round(self.sharpness_gain, 4),
            "wald_consistency": self.consistency,
            "input_warnings": self.report.warnings,
        }


def run_super_resolution(
    input_path: str | Path,
    checkpoint: dict,
    band_indices: Sequence[int],
    device: str = "auto",
    tile: int = 128,
    overlap: int = 16,
    batch_size: int = 4,
    tta: bool = False,
    ensemble_paths: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    """Read, super-resolve, and produce the labelled bicubic control beside it.

    Raises `ValueError` with an actionable message rather than returning a
    partial result: the interface prints the message, and no panel is drawn
    from data that does not exist.
    """
    def say(message: str) -> None:
        log.info("%s", message)
        if progress:
            progress(message)

    report = inspect_raster(input_path)
    if not report.ok:
        raise ValueError("This file cannot be processed:\n- " + "\n- ".join(report.errors))

    device = resolve_device(device)
    say(f"loading {Path(checkpoint['path']).name} on {device}")
    model, payload = load_checkpoint(checkpoint["path"], device=device)
    scale = int(checkpoint.get("scale") or getattr(model, "scale", 4))

    say(f"reading {report.width}x{report.height} px, bands {list(band_indices)}")
    lr, meta = read_raster(input_path, bands=list(band_indices))
    if lr.shape[0] != model.in_channels:
        raise ValueError(
            f"{lr.shape[0]} bands were selected but this checkpoint expects "
            f"{model.in_channels}. Adjust the band mapping in the sidebar."
        )

    say(f"super-resolving {report.megapixels:.2f} Mpx in {tile}px tiles")
    started = time.perf_counter()
    sr = super_resolve(lr, model, scale=scale, tile=tile, overlap=overlap,
                       batch_size=batch_size, tta=tta, device=device, progress=False)
    seconds_model = time.perf_counter() - started

    started = time.perf_counter()
    baseline = upsample_bicubic(lr, scale)
    seconds_bicubic = time.perf_counter() - started

    result = RunResult(
        lr=lr, sr=sr, bicubic=baseline, meta=meta, report=report,
        checkpoint=checkpoint, scale=scale,
        band_indices=list(band_indices), band_names=list(meta.band_names),
        device=device, seconds_model=seconds_model, seconds_bicubic=seconds_bicubic,
        consistency=consistency_error(sr, lr, _degradation_for(payload["config"], scale)),
        sharpness_gain=float(sharpness_gain(sr, baseline)),
    )

    if ensemble_paths:
        try:
            say(f"running {len(ensemble_paths)} ensemble members")
            result.ensemble = estimate_ensemble(
                lr, ensemble_paths, device=device, tile=tile, overlap=overlap,
                batch_size=batch_size, tta=tta,
            )
        except EnsembleUnavailable as exc:
            result.ensemble_error = str(exc)
    return result


def _degradation_for(cfg: dict, scale: int):
    from .data.degradation import DegradationConfig

    return DegradationConfig.from_dict(
        cfg.get("data", {}).get("degradation", {}), scale=scale,
        bands=tuple(cfg.get("bands", ())),
    )


# --------------------------------------------------------------------------- #
# evaluation against an uploaded reference
# --------------------------------------------------------------------------- #

def evaluate_against_reference(
    result: RunResult,
    reference_path: str | Path,
    data_range: float = 1.0,
) -> dict:
    """Score the model and the bicubic control against a real reference raster.

    The reference is resampled onto the *output* grid by its own
    georeferencing, not by array position. Two rasters of the same scene can
    differ in size, origin and even CRS; comparing them index-by-index would
    measure misregistration and report it as model error.

    Only pixels valid in all three rasters are scored, and both predictions
    are scored on exactly that set, so the comparison is like for like.
    """
    grid_transform = scale_transform(result.meta.transform, result.scale)
    if result.meta.crs is None or grid_transform is None:
        return {"available": False,
                "reason": "The input has no CRS or transform, so a reference "
                          "cannot be aligned to the output grid."}

    reference, ref_report = _reference_on_grid(
        reference_path,
        crs=result.meta.crs,
        transform=grid_transform,
        shape=result.sr.shape[-2:],
        band_names=(result.checkpoint.get("bands")
                    or [str(n) for n in result.band_names]),
    )
    if reference is None:
        return {"available": False, "reason": ref_report}

    valid = (
        np.isfinite(reference).all(axis=0)
        & np.isfinite(result.sr).all(axis=0)
        & np.isfinite(result.bicubic).all(axis=0)
    )
    coverage = float(valid.mean())
    if coverage < 0.01:
        return {"available": False,
                "reason": f"The reference overlaps only {coverage:.2%} of the output "
                          "extent. Supply a reference covering the same ground."}

    def masked(array: np.ndarray) -> np.ndarray:
        return np.where(valid[None], array, np.nan)

    target = masked(reference)
    rows = {}
    for label, prediction in (("model", result.sr), ("bicubic", result.bicubic)):
        row = compute_metrics(masked(prediction), target, REPORT_METRICS,
                              data_range, result.scale)
        rows[label] = row

    return {
        "available": True,
        "reference": str(reference_path),
        "data_range": data_range,
        "valid_fraction": coverage,
        "valid_pixels": int(valid.sum()),
        "metrics": rows,
        "units": {k: METRIC_INFO[k] for k in REPORT_METRICS if k in METRIC_INFO},
        "timing": {
            "model_seconds": round(result.seconds_model, 3),
            "bicubic_seconds": round(result.seconds_bicubic, 3),
            "device": result.device,
        },
        "note": ref_report,
    }


def _reference_on_grid(
    path: str | Path,
    crs: Any,
    transform: Any,
    shape: tuple[int, int],
    band_names: Sequence[str],
) -> tuple[np.ndarray | None, str]:
    """Resample a reference raster onto the output grid. Returns (array, note).

    The reference's bands are resolved **by name**, exactly as the input's
    are. A reference is usually the same kind of multi-band product as the
    input -- a 12-band SEN2NEON HR tile, say -- so reading its first four
    bands positionally compares blue/green/red/NIR against
    coastal-aerosol/blue/green/red and reports the mismatch as model error.
    On a forest tile that shows up as a spectral angle near 60 degrees where
    the true value is under 5.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    from . import S2_REFLECTANCE_SCALE

    n_bands = len(band_names)
    report = inspect_raster(path)
    if not report.ok:
        return None, "Reference rejected: " + "; ".join(report.errors)
    if report.band_count < n_bands:
        return None, (f"The reference has {report.band_count} bands but the output "
                      f"has {n_bands}. Supply a matching reference.")
    if report.crs is None:
        return None, "The reference has no CRS, so it cannot be aligned."

    try:
        indices, resolved, band_notes = resolve_band_mapping(report, band_names, n_bands)
    except ValueError as exc:
        return None, f"Reference band mapping failed: {exc}"

    with rasterio.open(str(path)) as src:
        source = src.read(indices).astype(np.float32)
        if src.nodata is not None:
            source = np.where(source == float(src.nodata), np.nan, source)
        if np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
            source = source / S2_REFLECTANCE_SCALE
        destination = np.full((n_bands, *shape), np.nan, dtype=np.float32)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            # Nearest keeps the reference's own values intact. Any smoothing
            # resampler would blur the very detail the model is being scored
            # on and flatter it.
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    same_crs = str(src.crs) == str(crs)
    note = (
        f"Reference bands {indices} ({', '.join(resolved)}) resampled onto the "
        f"{shape[1]}x{shape[0]} output grid by its georeferencing "
        f"({'same CRS' if same_crs else f'reprojected from {src.crs}'}, "
        "nearest neighbour)."
    )
    if any("names no bands" in n or "Falling back" in n for n in band_notes):
        note += " " + " ".join(band_notes)
    return destination, note


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #

def write_outputs(
    result: RunResult,
    outdir: str | Path,
    stem: str | None = None,
    include_baseline: bool = True,
) -> dict[str, str]:
    """Write the georeferenced products and the run report to `outdir`."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = stem or Path(result.report.path).stem
    names = result.product_band_names
    written: dict[str, str] = {}

    path = outdir / f"{stem}_sr_x{result.scale}.tif"
    write_raster(path, result.sr, result.meta, scale_factor=result.scale, band_names=names)
    written["enhanced_geotiff"] = str(path)

    if include_baseline:
        path = outdir / f"{stem}_bicubic_x{result.scale}.tif"
        write_raster(path, result.bicubic, result.meta, scale_factor=result.scale,
                     band_names=names)
        written["bicubic_geotiff"] = str(path)

    if result.ensemble is not None:
        path = outdir / f"{stem}_ensemble_disagreement.tif"
        write_raster(path, result.ensemble.disagreement[None], result.meta,
                     scale_factor=result.scale,
                     band_names=["ensemble_disagreement_reflectance"])
        written["uncertainty_raster"] = str(path)

    summary: dict[str, Any] = {"run": result.run_summary()}
    if result.evaluation:
        summary["evaluation"] = result.evaluation
    if result.ensemble is not None:
        summary["ensemble"] = result.ensemble.summary()
    elif result.ensemble_error:
        summary["ensemble"] = {"available": False, "reason": result.ensemble_error}
    path = outdir / f"{stem}_report.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written["report_json"] = str(path)

    path = outdir / f"{stem}_report.md"
    path.write_text(format_run_report(result), encoding="utf-8")
    written["report_markdown"] = str(path)

    result.written = written
    return written


def format_run_report(result: RunResult) -> str:
    """A Markdown record of one run, stating what was and was not measured."""
    summary = result.run_summary()
    extent = summary["ground_extent_m"]
    lines = [
        f"# Super-resolution run -- {Path(result.report.path).name}",
        "",
        "## Input",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| File | `{Path(result.report.path).name}` |",
        f"| Dimensions | {result.report.width} x {result.report.height} px, "
        f"{result.report.band_count} bands |",
        f"| CRS | {result.report.crs or 'none'} |",
        f"| Pixel spacing | {_fmt(result.report.pixel_size)} m |",
        f"| Ground extent | {_fmt(extent[0] if extent else None, 0)} x "
        f"{_fmt(extent[1] if extent else None, 0)} m |",
        f"| Valid data | {result.report.valid_fraction:.2%} |",
        f"| Bands used | {', '.join(str(b) for b in result.band_indices)} |",
        "",
        "## Model",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Checkpoint | `{result.checkpoint.get('name')}` |",
        f"| Architecture | {result.checkpoint.get('architecture')} |",
        f"| Trained epochs | {result.checkpoint.get('epoch')} |",
        f"| Seed | {result.checkpoint.get('seed')} |",
        f"| Scale | x{result.scale} |",
        f"| Output | {result.sr.shape[-1]} x {result.sr.shape[-2]} px at "
        f"{_fmt(result.output_gsd)} m |",
        f"| Device | {result.device} |",
        f"| Time | {result.seconds_model:.2f} s (bicubic control "
        f"{result.seconds_bicubic:.2f} s) |",
        "",
    ]

    evaluation = result.evaluation
    lines += ["## Accuracy against the reference", ""]
    if evaluation and evaluation.get("available"):
        model, bicubic = evaluation["metrics"]["model"], evaluation["metrics"]["bicubic"]
        lines += [
            f"Scored on {evaluation['valid_pixels']:,} pixels valid in the input, "
            f"the output and the reference ({evaluation['valid_fraction']:.1%} of the "
            "output extent). Data range for PSNR and SSIM: "
            f"{evaluation['data_range']} (surface reflectance).",
            "",
            "| Metric | Unit | Bicubic | Model | Better |",
            "|---|---|---:|---:|:--:|",
        ]
        for key in REPORT_METRICS:
            if key not in model:
                continue
            info = METRIC_INFO.get(key, {"unit": "", "better": "lower"})
            m, b = model[key], bicubic.get(key)
            if b is None:
                lines.append(f"| {key} | {info['unit']} | - | {m:.4f} | |")
                continue
            wins = (m > b) if info["better"] == "higher" else (m < b)
            lines.append(
                f"| {key} | {info['unit']} | {b:.4f} | {m:.4f} | "
                f"{'**model**' if wins else 'bicubic'} |"
            )
    else:
        reason = (evaluation or {}).get("reason", "No reference image was supplied.")
        lines += [
            f"**Not available.** {reason}",
            "",
            "Reference-based accuracy cannot be computed without a higher-resolution "
            "reference covering the same ground. The reference-free indicators below "
            "describe the output but do not establish that it is correct.",
        ]

    lines += [
        "",
        "## Reference-free indicators",
        "",
        "| Indicator | Value | Meaning |",
        "|---|---:|---|",
        f"| Sharpness gain vs bicubic | {result.sharpness_gain:.4f} | "
        "Above 1 means detail was added. It does not mean the detail is correct. |",
    ]
    for key, value in result.consistency.items():
        lines.append(
            f"| {key} | {value:.5f} | Wald residual: re-degrading the output should "
            "reproduce the input. |"
        )

    lines += ["", "## Ensemble disagreement", ""]
    if result.ensemble is not None:
        s = result.ensemble.summary()
        lines += [
            f"{s['n_members']} independently trained checkpoints.",
            "",
            f"- Definition: {EnsembleResult.AGGREGATE_DEFINITION}",
            f"- Mean: {s['disagreement_mean']:.5f} reflectance",
            f"- 95th percentile: {s['disagreement_p95']:.5f} reflectance",
            f"- Maximum: {s['disagreement_max']:.5f} reflectance",
            "",
            "This is disagreement between models, not a probability that the "
            "prediction is correct.",
        ]
    else:
        lines.append(
            "**Ensemble uncertainty unavailable.** "
            + (result.ensemble_error or "Fewer than two compatible checkpoints were selected.")
        )

    lines += [
        "",
        "## Interpretation",
        "",
        f"The output is written on a {_fmt(result.output_gsd)} m pixel grid. "
        "A finer grid is not the same thing as finer resolved detail: the model "
        "produces a plausible estimate of sub-pixel structure, and only comparison "
        "against a true higher-resolution reference can establish how much of it "
        "is real. Where no reference exists, treat the output as a hypothesis.",
        "",
    ]
    if result.report.warnings:
        lines += ["### Input warnings", ""] + [f"- {w}" for w in result.report.warnings] + [""]
    return "\n".join(lines)


def _fmt(value: float | None, places: int = 2) -> str:
    return f"{value:.{places}f}" if value is not None and np.isfinite(value) else "n/a"


# --------------------------------------------------------------------------- #
# display
# --------------------------------------------------------------------------- #

def shared_stretch(
    reference: np.ndarray,
    rgb_index: Sequence[int] = (2, 1, 0),
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[float, float]:
    """Percentile bounds computed once, to be reused by every panel.

    Each panel stretching itself would make the input and the output differ in
    colour even where the model changed nothing, which flatters the result.
    A comparison is only fair on one scale.
    """
    array = np.asarray(reference, dtype=np.float32)
    idx = [i for i in rgb_index if i < array.shape[0]] or [0]
    finite = array[idx][np.isfinite(array[idx])]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(finite, percentiles))
    return lo, (hi if hi > lo else lo + 1e-6)


def to_rgb(
    array: np.ndarray,
    bounds: tuple[float, float],
    rgb_index: Sequence[int] = (2, 1, 0),
) -> np.ndarray:
    """(C, H, W) reflectance -> (H, W, 3) uint8 using *given* bounds.

    Display stretching is kept strictly out of the scientific data path: this
    is only ever used for on-screen panels and the PNG preview. Exported
    GeoTIFFs carry unstretched reflectance.
    """
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    idx = [i for i in rgb_index if i < array.shape[0]] or [0]
    while len(idx) < 3:
        idx.append(idx[-1])
    lo, hi = bounds
    out = np.clip((np.nan_to_num(array[idx[:3]]) - lo) / (hi - lo), 0.0, 1.0)
    return (np.moveaxis(out, 0, -1) * 255.0).astype(np.uint8)


def colorize(
    values: np.ndarray,
    vmax: float | None = None,
    cmap: str = "magma",
) -> np.ndarray:
    """Single-band map -> (H, W, 3) uint8, NaN rendered as transparent-dark.

    Used for the disagreement map. The scale bound is returned to the caller
    through `vmax` so the legend can state it -- a heatmap without a stated
    range is decoration.
    """
    import matplotlib

    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if vmax is None:
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    vmax = max(float(vmax), 1e-9)
    normalised = np.clip(np.nan_to_num(array) / vmax, 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](normalised)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(array)] = 20
    return rgb


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as PNG for the download button."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()
