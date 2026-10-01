"""Accuracy assessment and validation reporting.

What a defensible evaluation of an SRM product has to contain, and what this
module produces:

* **A baseline on the same split.** Bicubic upsampling of the same input,
  scored with the same metrics. Without it, a PSNR figure means nothing.
* **Both fidelity and detail metrics.** PSNR/SSIM/RMSE reward conservatism;
  SCC and sharpness gain reward detail. A model has to be read on both axes
  at once, because it is always possible to win either one alone.
* **Spectral integrity.** SAM and ERGAS, because a downstream classifier
  consumes reflectance, not appearance.
* **Reference-free consistency**, so the same protocol can be run over real
  operational output where no high-resolution truth exists.
* **The distribution, not just the mean.** p05/p95 across the test set, since
  SR failures concentrate in specific land cover rather than spreading evenly.

Usage:
    python -m srm.evaluate --checkpoint checkpoints/edsr_x4/best.pt \\
        --patches data/patches --split val --report outputs/report
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import resolve_device
from .data.degradation import DegradationConfig, degrade, upsample_bicubic
from .metrics.quality import (
    METRIC_INFO,
    aggregate,
    compute_metrics,
    consistency_error,
    sharpness_gain,
)
from .models.registry import load_checkpoint
from .utils.logging import get_logger

log = get_logger("srm.evaluate")

HIGHER_IS_BETTER = {"psnr", "ssim", "scc", "q_index", "sharpness_gain"}


def _align(pred: np.ndarray, target: np.ndarray):
    h = min(pred.shape[-2], target.shape[-2])
    w = min(pred.shape[-1], target.shape[-1])
    return pred[..., :h, :w], target[..., :h, :w]


@torch.no_grad()
def evaluate_samples(
    samples: Sequence[dict],
    model,
    degradation: DegradationConfig,
    metric_names: Sequence[str],
    device: str = "cpu",
    scale: int = 4,
    data_range: float = 1.0,
    include_baseline: bool = True,
) -> dict:
    """Score `{lr, hr, mask, site}` samples, model against the bicubic control.

    Three things this does that `evaluate_pairs` does not:

    * **Masks the reference.** Invalid HR pixels become NaN, and every metric
      already ignores non-finite pixels, so scores are computed over observed
      ground only. Without this a tile that is 20% empty scores the model on
      its ability to reproduce zeros.
    * **Times the model.** Processing time is a reported quantity, and the
      bicubic control is timed on the same machine for a fair comparison.
    * **Keeps the geographic label** with each row, so results can be broken
      down by area afterwards.
    """
    model.eval().to(device)
    model_rows, baseline_rows = [], []
    model_seconds = baseline_seconds = 0.0

    for sample in samples:
        lr, hr = sample["lr"], sample["hr"]
        mask = sample.get("mask")
        site = sample.get("site", "all")

        tensor = torch.from_numpy(np.ascontiguousarray(lr))[None].to(device)
        started = time.perf_counter()
        sr = model(tensor)[0].float().cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        model_seconds += time.perf_counter() - started

        sr, hr_c = _align(sr, hr)
        target = hr_c
        if mask is not None:
            mask_c = mask[..., : target.shape[-2], : target.shape[-1]]
            target = np.where(mask_c[None] > 0.5, target, np.nan)

        row = compute_metrics(sr, target, metric_names, data_range, scale)
        started = time.perf_counter()
        baseline = upsample_bicubic(lr, scale)
        baseline_seconds += time.perf_counter() - started
        baseline, _ = _align(baseline, target)
        row["sharpness_gain"] = sharpness_gain(sr, baseline)
        row.update(consistency_error(sr, lr, degradation))
        row["_site"] = site
        model_rows.append(row)

        if include_baseline:
            base_row = compute_metrics(baseline, target, metric_names, data_range, scale)
            base_row["sharpness_gain"] = 1.0
            base_row.update(consistency_error(baseline, lr, degradation))
            base_row["_site"] = site
            baseline_rows.append(base_row)

    n = max(len(model_rows), 1)
    result = {
        "model": aggregate(_numeric(model_rows)),
        "n": len(model_rows),
        "timing": {
            "model_seconds_total": round(model_seconds, 4),
            "model_ms_per_patch": round(1000 * model_seconds / n, 3),
            "bicubic_seconds_total": round(baseline_seconds, 4),
            "bicubic_ms_per_patch": round(1000 * baseline_seconds / n, 3),
            "device": device,
        },
        "_rows": model_rows,
    }
    if include_baseline:
        result["bicubic"] = aggregate(_numeric(baseline_rows))
        result["by_area"] = stratify(model_rows, baseline_rows)
    return result


def _numeric(rows: Sequence[dict]) -> list[dict]:
    """Drop the bookkeeping keys before aggregation."""
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def stratify(model_rows: Sequence[dict], baseline_rows: Sequence[dict]) -> dict:
    """Per-geographic-area metrics.

    A single mean over a test set hides the failure mode that matters: a
    model can gain 0.4 dB overall while losing 1 dB on every built-up tile,
    because natural land cover dominates the set. Reporting by area is the
    only way that shows up.
    """
    areas = sorted({str(row.get("_site", "all")) for row in model_rows})
    out: dict[str, dict] = {}
    for area in areas:
        m = [r for r in model_rows if str(r.get("_site", "all")) == area]
        b = [r for r in baseline_rows if str(r.get("_site", "all")) == area]
        out[area] = {
            "n": len(m),
            "model": aggregate(_numeric(m)),
            "bicubic": aggregate(_numeric(b)) if b else {},
        }
    return out


def load_samples(
    patch_dir: str | Path,
    split: str = "val",
    degradation: DegradationConfig | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read `{lr, hr, mask, site}` records from the stored shards.

    Where the shards hold a real LR side (native pairs) it is used as stored.
    Where they do not, the LR is regenerated with a fixed RNG seed, so an
    evaluation is reproducible and two models are compared on byte-identical
    inputs.
    """
    from . import S2_REFLECTANCE_SCALE
    from .data.patches import load_manifest

    patch_dir = Path(patch_dir)
    manifest = load_manifest(patch_dir)
    degradation = degradation or DegradationConfig.from_dict(
        manifest.get("degradation", {}), scale=int(manifest["scale"]),
        bands=tuple(manifest["bands"]),
    )
    samples: list[dict] = []
    rng = np.random.default_rng(12345)
    for shard in sorted((patch_dir / split).glob("shard_*.npz")):
        with np.load(shard) as data:
            hr_stack = data["hr"]
            lr_stack = data["lr"] if "lr" in data else None
            masks = data["mask"] if "mask" in data else None
            sites = data["site"] if "site" in data else None
            for i in range(hr_stack.shape[0]):
                hr = hr_stack[i].astype(np.float32) / S2_REFLECTANCE_SCALE
                lr = (
                    lr_stack[i].astype(np.float32) / S2_REFLECTANCE_SCALE
                    if lr_stack is not None
                    else degrade(hr, degradation, rng)
                )
                samples.append({
                    "lr": lr,
                    "hr": hr,
                    "mask": masks[i].astype(bool) if masks is not None else None,
                    "site": str(sites[i]) if sites is not None else "all",
                })
                if limit and len(samples) >= limit:
                    return samples
    return samples


def format_report(result: dict, title: str, extra: dict | None = None) -> str:
    """Markdown table comparing the model against the bicubic control."""
    model = result.get("model", {})
    baseline = result.get("bicubic", {})
    lines = [
        f"# {title}",
        "",
        f"Test patches: **{result.get('n', 0)}**",
        "",
        "| Metric | Bicubic | Model | Delta | Better |",
        "|---|---:|---:|---:|:--:|",
    ]
    for key in sorted(model):
        m = model[key]["mean"]
        b = baseline.get(key, {}).get("mean")
        if b is None:
            lines.append(f"| {key} | - | {m:.4f} | - | |")
            continue
        delta = m - b
        if key in HIGHER_IS_BETTER:
            better = "model" if delta > 0 else "bicubic"
        else:
            better = "model" if delta < 0 else "bicubic"
        mark = "**model**" if better == "model" else "bicubic"
        lines.append(f"| {key} | {b:.4f} | {m:.4f} | {delta:+.4f} | {mark} |")

    lines += ["", "## Spread across the test set (model)", "",
              "| Metric | mean | std | p05 | median | p95 |", "|---|---:|---:|---:|---:|---:|"]
    for key in sorted(model):
        s = model[key]
        lines.append(
            f"| {key} | {s['mean']:.4f} | {s['std']:.4f} | {s['p05']:.4f} | "
            f"{s['median']:.4f} | {s['p95']:.4f} |"
        )

    timing = result.get("timing")
    if timing:
        lines += [
            "", "## Processing time", "",
            f"Measured on **{timing.get('device')}**, forward pass only "
            "(patch already in memory, no I/O).", "",
            "| | total (s) | per patch (ms) |", "|---|---:|---:|",
            f"| model | {timing['model_seconds_total']:.3f} | "
            f"{timing['model_ms_per_patch']:.2f} |",
            f"| bicubic | {timing['bicubic_seconds_total']:.3f} | "
            f"{timing['bicubic_ms_per_patch']:.2f} |",
        ]

    by_area = result.get("by_area")
    if by_area and len(by_area) > 1:
        lines += [
            "", "## By geographic area", "",
            "Each area is a separate NEON site, held out as a whole. A model "
            "that wins on average can still lose on a land-cover type that is "
            "rare in training, and only this table shows it.", "",
            "| Area | n | PSNR bicubic | PSNR model | delta | SAM model (deg) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        def cell(value: float | None, fmt: str = ".3f") -> str:
            return f"{value:{fmt}}" if value is not None else "-"

        for area in sorted(by_area):
            entry = by_area[area]
            mp = entry["model"].get("psnr", {}).get("mean")
            bp = entry.get("bicubic", {}).get("psnr", {}).get("mean")
            ms = entry["model"].get("sam", {}).get("mean")
            if mp is None:
                continue
            delta = cell(mp - bp, "+.3f") if bp is not None else "-"
            lines.append(
                f"| {area} | {entry['n']} | {cell(bp)} | {cell(mp)} | {delta} | {cell(ms)} |"
            )

    lines += [
        "",
        "## Units and direction",
        "",
        "| Metric | Unit | Better | Note |",
        "|---|---|---|---|",
    ]
    for key in sorted(model):
        info = METRIC_INFO.get(key)
        if info:
            lines.append(f"| {key} | {info['unit']} | {info['better']} | {info['note']} |")
    lines += [
        "",
        "## How to read this",
        "",
        "- **psnr / ssim / rmse** -- pixel fidelity. A model that simply blurs can win",
        "  these, so they are never sufficient on their own.",
        "- **sam** (degrees) -- spectral distortion. Under ~2 deg the land-cover",
        "  signature is preserved; above ~5 deg downstream classification is affected.",
        "- **ergas** -- the standard summary for a resolution-fused product; under 3",
        "  is conventionally read as good.",
        "- **scc** -- correlation of high-frequency content. This is the metric that",
        "  separates real reconstruction from smoothing.",
        "- **sharpness_gain** -- apparent detail relative to bicubic. Above 1 is",
        "  sharper, but only meaningful when read together with the next row.",
        "- **consistency_rmse** -- Wald residual, reference-free. Detail that raises",
        "  sharpness_gain while also raising this was invented, not recovered.",
    ]
    if extra:
        lines += ["", "## Run", "", "```json", json.dumps(extra, indent=2), "```"]
    return "\n".join(lines) + "\n"


def run(
    checkpoint: str | Path,
    patch_dir: str | Path,
    split: str = "val",
    report_dir: str | Path = "outputs/report",
    limit: int | None = None,
    device: str = "auto",
) -> dict:
    device = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device=device)
    cfg = payload["config"]
    scale = int(cfg.get("scale", 4))
    degradation = DegradationConfig.from_dict(
        cfg.get("data", {}).get("degradation", {}), scale=scale,
        bands=tuple(cfg.get("bands", ())),
    )
    metric_names = cfg.get("eval", {}).get("metrics", ["psnr", "ssim", "sam", "ergas", "scc", "rmse"])

    samples = load_samples(patch_dir, split, degradation, limit)
    if not samples:
        raise SystemExit(
            f"no patches in {Path(patch_dir) / split}. Reference-based evaluation "
            "needs a split with stored pairs; build one first."
        )
    log.info("evaluating %s on %d %s patches", Path(checkpoint).name, len(samples), split)
    result = evaluate_samples(samples, model, degradation, metric_names, device, scale,
                              float(cfg.get("eval", {}).get("data_range", 1.0)))

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = result.pop("_rows")
    _write_csv(report_dir / "per_patch_metrics.csv",
               [{**{k: v for k, v in r.items() if not k.startswith("_")},
                 "site": r.get("_site", "all")} for r in rows])

    extra = {
        "checkpoint": str(checkpoint),
        "model": cfg.get("model", {}).get("name"),
        "used_ema": payload.get("used_ema", False),
        "epoch": payload.get("epoch"),
        "scale": scale,
        "split": split,
        "device": device,
    }
    title = f"SRM evaluation -- {cfg.get('experiment', 'model')} (x{scale})"
    (report_dir / "report.md").write_text(format_report(result, title, extra), encoding="utf-8")
    (report_dir / "metrics.json").write_text(
        json.dumps({"result": result, "run": extra}, indent=2), encoding="utf-8"
    )
    log.info("report -> %s", report_dir / "report.md")

    for key in ("psnr", "ssim", "sam", "ergas", "scc"):
        if key in result.get("model", {}):
            m = result["model"][key]["mean"]
            b = result.get("bicubic", {}).get(key, {}).get("mean", float("nan"))
            log.info("  %-8s model=%.4f  bicubic=%.4f", key, m, b)
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index"] + fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"index": i, **row})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an SRM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patches", default="data/patches")
    parser.add_argument("--split", default="val")
    parser.add_argument("--report", default="outputs/report")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args.checkpoint, args.patches, args.split, args.report, args.limit, args.device)


if __name__ == "__main__":
    main()
