"""Comparison figures.

One rule runs through all of these: **the same stretch for every panel.** A
figure where the SR panel was stretched independently from the bicubic panel
is not evidence of anything, and it is the single most common way SR results
are oversold. `to_uint8_rgb` bounds are computed once, on the reference, and
reused everywhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .geo import to_uint8_rgb


def _shared_stretch(reference: np.ndarray, percentiles=(2.0, 98.0)):
    """Return a function applying one fixed stretch to every panel."""
    ref = np.asarray(reference, dtype=np.float32)
    finite = ref[np.isfinite(ref)]
    lo, hi = (np.percentile(finite, percentiles) if finite.size else (0.0, 1.0))
    if hi <= lo:
        hi = lo + 1e-6

    def apply(image: np.ndarray, rgb_index=(2, 1, 0)) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        idx = [i for i in rgb_index if i < image.shape[0]] or [0]
        while len(idx) < 3:
            idx.append(idx[-1])
        rgb = np.nan_to_num(image[idx[:3]])
        out = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
        return (np.moveaxis(out, 0, -1) * 255.0).astype(np.uint8)

    return apply


def comparison_panel(
    panels: Sequence[tuple[str, np.ndarray]],
    out_path: str | Path,
    reference_for_stretch: np.ndarray | None = None,
    title: str = "",
    zoom: tuple[int, int, int] | None = None,
    dpi: int = 140,
) -> Path:
    """Write a row of RGB panels under one shared stretch.

    `zoom` is (y, x, size) in the coordinates of the *largest* panel; every
    panel is cropped to the same ground footprint, scaling the window by its
    own pixel size so the comparison is of the same place.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = list(panels)
    reference = reference_for_stretch if reference_for_stretch is not None else panels[-1][1]
    stretch = _shared_stretch(reference)
    max_side = max(p.shape[-1] for _, p in panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, image) in zip(axes, panels):
        img = image
        if zoom is not None:
            y, x, size = zoom
            factor = image.shape[-1] / max_side
            yy, xx, ss = int(y * factor), int(x * factor), max(4, int(size * factor))
            img = image[:, yy : yy + ss, xx : xx + ss]
        ax.imshow(stretch(img), interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def map_panel(
    maps: Sequence[tuple[str, np.ndarray, str]],
    out_path: str | Path,
    title: str = "",
    dpi: int = 140,
) -> Path:
    """Row of single-band maps with colourbars. Each entry is (name, array, cmap)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = list(maps)
    fig, axes = plt.subplots(1, len(maps), figsize=(4.6 * len(maps), 4.4))
    if len(maps) == 1:
        axes = [axes]
    for ax, (name, data, cmap) in zip(axes, maps):
        data = np.asarray(data, dtype=np.float32)
        finite = data[np.isfinite(data)]
        vmin, vmax = (np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def metric_bars(
    comparison: dict[str, dict[str, float]],
    out_path: str | Path,
    title: str = "Model vs bicubic baseline",
    dpi: int = 140,
) -> Path:
    """Grouped bars per metric. Values are normalised per metric so PSNR (dB)
    and SAM (degrees) can share an axis without one flattening the other."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = sorted(set(comparison.get("model", {})) & set(comparison.get("bicubic", {})))
    if not metrics:
        raise ValueError("comparison needs 'model' and 'bicubic' entries with shared metrics")
    model_vals = [comparison["model"][m] for m in metrics]
    base_vals = [comparison["bicubic"][m] for m in metrics]
    scales = [max(abs(a), abs(b), 1e-9) for a, b in zip(model_vals, base_vals)]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(1.5 * len(metrics) + 3, 4.4))
    ax.bar(x - 0.2, [v / s for v, s in zip(base_vals, scales)], 0.4, label="bicubic")
    ax.bar(x + 0.2, [v / s for v, s in zip(model_vals, scales)], 0.4, label="model")
    for i, (b, m) in enumerate(zip(base_vals, model_vals)):
        ax.text(i - 0.2, b / scales[i] + 0.02, f"{b:.3g}", ha="center", fontsize=8)
        ax.text(i + 0.2, m / scales[i] + 0.02, f"{m:.3g}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("value / per-metric max")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def find_interesting_window(image: np.ndarray, size: int = 96) -> tuple[int, int, int]:
    """Locate the most structurally busy window, for an honest zoom crop.

    Picking the zoom location by hand is how SR figures get cherry-picked.
    Choosing it by gradient energy is at least a stated, reproducible rule.
    """
    from scipy.ndimage import sobel, uniform_filter

    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    band = image[min(2, image.shape[0] - 1)]
    energy = np.hypot(sobel(band, axis=0), sobel(band, axis=1))
    smoothed = uniform_filter(energy, size=size, mode="constant")
    half = size // 2
    interior = smoothed[half : band.shape[0] - half, half : band.shape[1] - half]
    if interior.size == 0:
        return 0, 0, min(size, band.shape[0], band.shape[1])
    flat = int(np.argmax(interior))
    y = flat // interior.shape[1]
    x = flat % interior.shape[1]
    return y, x, size
