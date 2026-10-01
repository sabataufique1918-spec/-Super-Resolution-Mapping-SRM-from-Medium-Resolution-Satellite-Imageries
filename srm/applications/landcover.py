"""Does super-resolution actually improve land-cover mapping?

This is the test that matters. PSNR is a proxy; classification accuracy on the
same ground truth, from LR-upsampled input versus SR input, is the thing an
operational user cares about.

The protocol is deliberately strict:

* One classifier, one hyperparameter set, trained separately on each input so
  neither representation gets an advantage from the fit.
* Training and test pixels come from **spatially disjoint halves** of the
  scene. Random pixel splits leak across the split through spatial
  autocorrelation and inflate every accuracy figure in the remote-sensing
  literature that uses them.
* Accuracy is reported overall, per class, and as Cohen's kappa, with the
  confusion matrix kept so errors can be attributed.

A small-feature breakdown is included too, because the whole argument for SRM
rests on narrow classes (roads, buildings) that are sub-pixel at 10 m.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .indices import compute_all


@dataclass
class ClassificationResult:
    overall_accuracy: float
    kappa: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_iou: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0

    def as_dict(self) -> dict:
        return {
            "overall_accuracy": self.overall_accuracy,
            "kappa": self.kappa,
            "per_class_f1": self.per_class_f1,
            "per_class_iou": self.per_class_iou,
            "confusion": self.confusion,
            "class_names": self.class_names,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def build_features(image: np.ndarray, with_texture: bool = True) -> np.ndarray:
    """(C, H, W) -> (H*W, F): reflectance, indices, and local texture.

    Texture is included because it is the only feature that can benefit from
    extra spatial resolution. A classifier on reflectance alone would show
    almost no gain from SR even if the SR result were perfect, which would
    make the comparison uninformative.
    """
    from scipy.ndimage import uniform_filter

    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    layers = [image[b] for b in range(image.shape[0])]
    layers += list(compute_all(image).values())

    if with_texture:
        for band in (2, 3):  # red and NIR carry most of the structural signal
            if band < image.shape[0]:
                x = image[band]
                mean = uniform_filter(x, size=5, mode="reflect")
                sq = uniform_filter(x * x, size=5, mode="reflect")
                layers.append(np.sqrt(np.clip(sq - mean * mean, 0, None)))  # local std
                layers.append(x - mean)                                     # local contrast

    stacked = np.stack(layers)
    return stacked.reshape(stacked.shape[0], -1).T.astype(np.float32)


def spatial_split(shape: tuple[int, int], axis: int = 1, train_fraction: float = 0.5):
    """Disjoint left/right (or top/bottom) halves as boolean pixel masks."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if axis == 1:
        mask[:, : int(width * train_fraction)] = True
    else:
        mask[: int(height * train_fraction), :] = True
    return mask.ravel(), (~mask).ravel()


def classify(
    image: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str] | None = None,
    max_train: int = 60_000,
    seed: int = 0,
    n_estimators: int = 120,
) -> ClassificationResult:
    """Train a random forest on one half of the scene, score it on the other."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

    labels = np.asarray(labels)
    height, width = labels.shape
    features = build_features(image)
    if features.shape[0] != height * width:
        raise ValueError(
            f"image {image.shape[-2:]} and labels {labels.shape} describe different grids"
        )
    flat_labels = labels.ravel()
    train_mask, test_mask = spatial_split((height, width))

    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(train_mask)
    if train_idx.size > max_train:
        train_idx = rng.choice(train_idx, max_train, replace=False)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size > max_train:
        test_idx = rng.choice(test_idx, max_train, replace=False)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=18, n_jobs=-1,
        random_state=seed, class_weight="balanced_subsample",
    )
    model.fit(features[train_idx], flat_labels[train_idx])
    predicted = model.predict(features[test_idx])
    truth = flat_labels[test_idx]

    present = sorted(set(truth.tolist()) | set(predicted.tolist()))
    names = [
        (class_names[c] if class_names is not None and c < len(class_names) else f"class_{c}")
        for c in present
    ]
    cm = confusion_matrix(truth, predicted, labels=present)
    f1 = f1_score(truth, predicted, labels=present, average=None, zero_division=0)
    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)

    return ClassificationResult(
        overall_accuracy=float((predicted == truth).mean()),
        kappa=float(cohen_kappa_score(truth, predicted)),
        per_class_f1={n: float(v) for n, v in zip(names, f1)},
        per_class_iou={
            n: float(i / u) if u > 0 else float("nan")
            for n, i, u in zip(names, intersection, union)
        },
        confusion=cm.astype(int).tolist(),
        class_names=names,
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
    )


def compare(
    hr_labels: np.ndarray,
    sr_image: np.ndarray,
    baseline_image: np.ndarray,
    hr_image: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    seed: int = 0,
) -> dict:
    """Classify from SR, from bicubic, and (as a ceiling) from true HR.

    The HR row is the upper bound the SR result is trying to approach. Without
    it, a small SR improvement over bicubic is impossible to put in context:
    it could be most of the available headroom, or almost none of it.
    """
    results = {
        "bicubic": classify(baseline_image, hr_labels, class_names, seed=seed),
        "super_resolved": classify(sr_image, hr_labels, class_names, seed=seed),
    }
    if hr_image is not None:
        results["true_hr_ceiling"] = classify(hr_image, hr_labels, class_names, seed=seed)

    out = {k: v.as_dict() for k, v in results.items()}
    base = results["bicubic"]
    sr = results["super_resolved"]
    delta = {
        "overall_accuracy": sr.overall_accuracy - base.overall_accuracy,
        "kappa": sr.kappa - base.kappa,
        "per_class_f1": {
            name: sr.per_class_f1.get(name, 0.0) - base.per_class_f1.get(name, 0.0)
            for name in set(sr.per_class_f1) | set(base.per_class_f1)
        },
    }
    if "true_hr_ceiling" in results:
        ceiling = results["true_hr_ceiling"].overall_accuracy
        headroom = ceiling - base.overall_accuracy
        delta["headroom_recovered_pct"] = float(
            100.0 * delta["overall_accuracy"] / headroom if abs(headroom) > 1e-9 else float("nan")
        )
    out["delta_sr_vs_bicubic"] = delta
    return out
