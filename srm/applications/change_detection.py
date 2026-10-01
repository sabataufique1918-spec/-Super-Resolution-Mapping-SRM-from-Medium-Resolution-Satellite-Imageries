"""Bi-temporal change detection on super-resolved imagery.

The operational claim for SRM in disaster response and urban monitoring is
that it surfaces changes too small to see at 10 m: a collapsed building, a
newly built row of houses, a flooded field corner. This module tests that
claim directly, by running the same detector over bicubic and SR inputs and
scoring both against a known change mask.

Change Vector Analysis is used rather than a learned detector on purpose. It
has no parameters to tune per input, so any difference in the result is
attributable to the imagery rather than to the detector having been fitted
more favourably to one of them.

Detection is scored **per change object**, not per pixel. Pixel F1 is
dominated by the large changes that were already visible at 10 m; what the
argument for SRM actually rests on is whether the *small* objects are found
at all, so the report breaks recall down by object size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ChangeResult:
    precision: float
    recall: float
    f1: float
    iou: float
    threshold: float
    n_detected: int
    n_true: int
    object_recall: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
            "threshold": self.threshold,
            "n_detected_px": self.n_detected,
            "n_true_px": self.n_true,
            "object_recall_by_size": self.object_recall,
        }


def change_vector_magnitude(t1: np.ndarray, t2: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Euclidean distance between the two dates in spectral space.

    `normalise` applies per-band mean/std standardisation computed jointly
    over both dates, which removes the global brightness difference caused by
    differing sun angle and atmosphere. Without it, CVA mostly detects the
    illumination change.
    """
    t1 = np.nan_to_num(np.asarray(t1, dtype=np.float32))
    t2 = np.nan_to_num(np.asarray(t2, dtype=np.float32))
    h = min(t1.shape[-2], t2.shape[-2])
    w = min(t1.shape[-1], t2.shape[-1])
    t1, t2 = t1[..., :h, :w], t2[..., :h, :w]

    if normalise:
        both = np.concatenate([t1, t2], axis=-1)
        mean = both.mean(axis=(-2, -1), keepdims=True)
        std = both.std(axis=(-2, -1), keepdims=True) + 1e-6
        t1 = (t1 - mean) / std
        t2 = (t2 - mean) / std
    return np.sqrt(((t2 - t1) ** 2).sum(axis=0))


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method: split the histogram to maximise between-class variance."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    hist, edges = np.histogram(finite, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(finite.mean())
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return float(finite.mean())
    cum_mean = np.cumsum(hist * centres)
    mean_bg = np.divide(cum_mean, weight_bg, out=np.zeros_like(cum_mean), where=weight_bg > 0)
    mean_fg = np.divide(
        cum_mean[-1] - cum_mean, weight_fg, out=np.zeros_like(cum_mean), where=weight_fg > 0
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -np.inf
    return float(centres[int(np.argmax(between))])


def detect(
    t1: np.ndarray,
    t2: np.ndarray,
    threshold: float | None = None,
    min_object_px: int = 4,
) -> tuple[np.ndarray, float]:
    """Binary change mask via CVA + Otsu, with speckle removed."""
    from scipy import ndimage

    magnitude = change_vector_magnitude(t1, t2)
    threshold = otsu_threshold(magnitude) if threshold is None else float(threshold)
    mask = magnitude > threshold

    if min_object_px > 1 and mask.any():
        labelled, count = ndimage.label(mask)
        if count:
            sizes = ndimage.sum_labels(mask, labelled, index=np.arange(1, count + 1))
            keep = np.zeros(count + 1, dtype=bool)
            keep[1:] = sizes >= min_object_px
            mask = keep[labelled]
    return mask, threshold


def _object_recall(
    predicted: np.ndarray,
    truth: np.ndarray,
    size_bins: Sequence[tuple[str, int, int]],
    overlap: float = 0.25,
) -> dict[str, float]:
    """Fraction of true change objects hit, broken down by object area.

    An object counts as detected when `overlap` of its pixels are flagged.
    Requiring a full match would punish a detector for imprecise boundaries
    on an object it clearly found.
    """
    from scipy import ndimage

    labelled, count = ndimage.label(truth)
    if count == 0:
        return {name: float("nan") for name, _, _ in size_bins}

    results: dict[str, list[bool]] = {name: [] for name, _, _ in size_bins}
    for obj_id in range(1, count + 1):
        obj = labelled == obj_id
        area = int(obj.sum())
        hit = bool((predicted & obj).sum() >= overlap * area)
        for name, lo, hi in size_bins:
            if lo <= area < hi:
                results[name].append(hit)
                break
    return {
        name: (float(np.mean(hits)) if hits else float("nan"))
        for name, hits in results.items()
    }


DEFAULT_SIZE_BINS = (
    ("tiny_lt_1_lr_px", 0, 16),        # smaller than one 10 m pixel at x4
    ("small_1_to_4_lr_px", 16, 64),
    ("medium_4_to_25_lr_px", 64, 400),
    ("large_gt_25_lr_px", 400, 10**9),
)


def score(
    predicted: np.ndarray,
    truth: np.ndarray,
    threshold: float = float("nan"),
    size_bins: Sequence[tuple[str, int, int]] = DEFAULT_SIZE_BINS,
) -> ChangeResult:
    h = min(predicted.shape[0], truth.shape[0])
    w = min(predicted.shape[1], truth.shape[1])
    predicted = predicted[:h, :w].astype(bool)
    truth = truth[:h, :w].astype(bool)

    tp = float((predicted & truth).sum())
    fp = float((predicted & ~truth).sum())
    fn = float((~predicted & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    union = tp + fp + fn

    return ChangeResult(
        precision=precision,
        recall=recall,
        f1=f1,
        iou=(tp / union) if union else 0.0,
        threshold=threshold,
        n_detected=int(predicted.sum()),
        n_true=int(truth.sum()),
        object_recall=_object_recall(predicted, truth, size_bins),
    )


def compare(
    sr_t1: np.ndarray,
    sr_t2: np.ndarray,
    base_t1: np.ndarray,
    base_t2: np.ndarray,
    truth: np.ndarray,
    min_object_px: int = 4,
) -> dict:
    """Run the same detector on SR and bicubic inputs and score both."""
    sr_mask, sr_thr = detect(sr_t1, sr_t2, min_object_px=min_object_px)
    base_mask, base_thr = detect(base_t1, base_t2, min_object_px=min_object_px)

    sr_result = score(sr_mask, truth, sr_thr)
    base_result = score(base_mask, truth, base_thr)
    return {
        "bicubic": base_result.as_dict(),
        "super_resolved": sr_result.as_dict(),
        "delta": {
            "f1": sr_result.f1 - base_result.f1,
            "iou": sr_result.iou - base_result.iou,
            "recall": sr_result.recall - base_result.recall,
            "precision": sr_result.precision - base_result.precision,
            "object_recall_by_size": {
                name: sr_result.object_recall.get(name, float("nan"))
                - base_result.object_recall.get(name, float("nan"))
                for name in sr_result.object_recall
            },
        },
        "_masks": {"super_resolved": sr_mask, "bicubic": base_mask},
    }
