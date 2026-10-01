"""End-to-end pipeline on procedural data: build, train, evaluate, apply, report.

This is the one command that demonstrates the whole system without any
credentials or downloads. It runs the exact code paths the real pipeline uses
-- same degradation model, same training loop, same metrics, same GeoTIFF
writer -- on synthetic scenes, and writes a consolidated report.

    python scripts/run_demo.py                     # ~3-6 min on CPU, faster on GPU
    python scripts/run_demo.py --epochs 20 --n-scenes 40 --model swinir

What it produces in outputs/demo/:
    report.md                     the consolidated findings
    figures/*.png                 LR / bicubic / SR / reference comparisons
    test_scene_sr.tif             a georeferenced 2.5 m product
    uncertainty/*.tif             per-pixel reliability maps
    eval/report.md                the full accuracy assessment

The numbers it prints are real measurements of a real model. They are not
evidence about Sentinel-2, because the input is synthetic. That claim needs
the real-data path in README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm import evaluate, uncertainty  # noqa: E402
from srm.applications import change_detection as cd  # noqa: E402
from srm.applications import indices, landcover  # noqa: E402
from srm.config import load_config, resolve_device  # noqa: E402
from srm.data.degradation import DegradationConfig, degrade, upsample_bicubic  # noqa: E402
from srm.data.patches import PatchSpec, build_patches  # noqa: E402
from srm.data.synthetic import (  # noqa: E402
    LABELS,
    SceneSpec,
    generate_bitemporal,
    generate_scene,
    write_scene_geotiff,
)
from srm.infer import super_resolve, super_resolve_file  # noqa: E402
from srm.models.registry import load_checkpoint  # noqa: E402
from srm.train import train  # noqa: E402
from srm.utils.logging import get_logger  # noqa: E402
from srm.utils.seed import seed_everything  # noqa: E402
from srm.utils.viz import comparison_panel, find_interesting_window, map_panel, metric_bars  # noqa: E402

log = get_logger("srm.demo")
CLASS_NAMES = [k for k, _ in sorted(LABELS.items(), key=lambda kv: kv[1])]


def step(index: int, total: int, message: str) -> None:
    log.info("[%d/%d] %s", index, total, message)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the full SRM pipeline on synthetic data")
    parser.add_argument("--out", default="outputs/demo")
    parser.add_argument("--config", default="configs/demo_fast.yaml")
    parser.add_argument("--model", default=None, help="override model.name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-scenes", type=int, default=16)
    parser.add_argument("--scene-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse an existing checkpoint from a previous run")
    args = parser.parse_args(argv)

    started = time.time()
    seed_everything(args.seed)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    total_steps = 8

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
        cfg["experiment"] = f"demo_{args.model}"
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    scale = int(cfg["scale"])
    device = resolve_device(cfg["device"])
    degradation = DegradationConfig.from_dict(cfg["data"]["degradation"], scale=scale,
                                              bands=cfg["bands"])

    # ---------------------------------------------------------------- 1. data
    step(1, total_steps, f"generating {args.n_scenes} training scenes at {args.scene_size}px")
    scenes = []
    for i in range(args.n_scenes):
        rng = np.random.default_rng(args.seed + i)
        spec = SceneSpec(
            size=args.scene_size,
            n_fields=int(rng.integers(18, 60)),
            n_urban_blocks=int(rng.integers(0, 7)),
            road_density=float(rng.uniform(0.5, 1.7)),
            river=bool(rng.random() < 0.6),
            texture=float(rng.uniform(0.3, 0.9)),
        )
        scenes.append((generate_scene(spec, seed=args.seed + i)[0], None))

    step(2, total_steps, "building paired patch shards")
    manifest = build_patches(
        scenes, cfg["data"]["patches"],
        spec=PatchSpec(hr_patch=int(cfg["data"]["hr_patch"]), scale=scale, min_std=0.008),
        degradation=degradation, seed=args.seed, logger=log,
    )
    if manifest["counts"]["train"] == 0:
        raise SystemExit("no training patches produced; increase --n-scenes or --scene-size")

    # ------------------------------------------------------------- 2. training
    ckpt_dir = Path(cfg["train"]["out_dir"]) / cfg["experiment"]
    checkpoint = ckpt_dir / "best.pt"
    if args.skip_train and checkpoint.exists():
        step(3, total_steps, f"reusing checkpoint {checkpoint}")
    else:
        step(3, total_steps, f"training {cfg['model']['name']} for {cfg['train']['epochs']} epochs on {device}")
        checkpoint = train(cfg)

    # ----------------------------------------------------------- 3. evaluation
    step(4, total_steps, "accuracy assessment on the held-out split")
    eval_result = evaluate.run(checkpoint, cfg["data"]["patches"], "val",
                               out / "eval", limit=64, device=device)

    # -------------------------------------------- 4. whole-scene inference
    step(5, total_steps, "super-resolving a held-out test scene as a GeoTIFF")
    test_hr, test_labels = generate_scene(
        SceneSpec(size=args.test_size, n_fields=40, n_urban_blocks=4, road_density=1.3),
        seed=args.seed + 9_999,   # a scene the model has never seen
    )
    test_lr = degrade(test_hr, degradation)

    scenes_dir = out / "scenes"
    write_scene_geotiff(scenes_dir / "test_reference_2p5m.tif", test_hr, gsd=2.5)
    write_scene_geotiff(scenes_dir / "test_input_10m.tif", test_lr, gsd=10.0)

    infer_summary = super_resolve_file(
        scenes_dir / "test_input_10m.tif", out / "test_scene_sr.tif", checkpoint,
        tile=int(cfg["infer"]["tile"]), overlap=int(cfg["infer"]["overlap"]),
        device=device, save_baseline=True,
    )

    model, _ = load_checkpoint(checkpoint, device=device)
    test_sr = super_resolve(test_lr, model, scale, tile=int(cfg["infer"]["tile"]),
                            overlap=int(cfg["infer"]["overlap"]), device=device, progress=False)
    test_bicubic = upsample_bicubic(test_lr, scale)

    # ---------------------------------------------------------- 5. uncertainty
    step(6, total_steps, "estimating per-pixel uncertainty")
    maps = uncertainty.estimate(test_lr, model, degradation, samples=6, method="auto",
                                device=device, tile=int(cfg["infer"]["tile"]),
                                overlap=int(cfg["infer"]["overlap"]))
    unc_dir = out / "uncertainty"
    unc_dir.mkdir(parents=True, exist_ok=True)
    from srm.utils.geo import GeoMeta, write_raster
    from affine import Affine
    from rasterio.crs import CRS

    lr_meta = GeoMeta(crs=CRS.from_epsg(32643),
                      transform=Affine(10.0, 0, 500000.0, 0, -10.0, 4500000.0))
    for name, data in (("reliability.tif", maps.reliability),
                       ("uncertainty_spread.tif", maps.spread),
                       ("uncertainty_inconsistency.tif", maps.inconsistency),
                       ("uncertainty_inferred_detail.tif", maps.inferred_detail)):
        write_raster(unc_dir / name, data[None], lr_meta, scale_factor=scale)
    unc_summary = maps.summary()

    # --------------------------------------------------------- 6. applications
    step(7, total_steps, "downstream applications: indices, land cover, change detection")
    index_result = indices.index_fidelity(test_sr, test_hr, test_bicubic, ("ndvi", "ndwi"))
    mixed = {
        "reference_2p5m": indices.mixed_pixel_fraction(indices.compute_index(test_hr, "ndvi"), 0.10, 0.40),
        "bicubic_from_10m": indices.mixed_pixel_fraction(indices.compute_index(test_bicubic, "ndvi"), 0.10, 0.40),
        "super_resolved": indices.mixed_pixel_fraction(indices.compute_index(test_sr, "ndvi"), 0.10, 0.40),
    }
    lc_result = landcover.compare(test_labels, test_sr, test_bicubic, test_hr,
                                  CLASS_NAMES, seed=args.seed)

    t1, t2, change_truth = generate_bitemporal(
        SceneSpec(size=args.test_size, n_fields=36, n_urban_blocks=3), seed=args.seed + 4242
    )
    lr1, lr2 = degrade(t1, degradation), degrade(t2, degradation)
    sr1 = super_resolve(lr1, model, scale, device=device, progress=False)
    sr2 = super_resolve(lr2, model, scale, device=device, progress=False)
    cd_result = cd.compare(sr1, sr2, upsample_bicubic(lr1, scale), upsample_bicubic(lr2, scale),
                           change_truth)
    cd_masks = cd_result.pop("_masks")

    # -------------------------------------------------------------- 7. figures
    step(8, total_steps, "writing figures and the consolidated report")
    fig_dir = out / "figures"
    zoom = find_interesting_window(test_hr, size=max(64, args.test_size // 6))
    comparison_panel(
        [("input 10 m (nearest)", np.repeat(np.repeat(test_lr, scale, 1), scale, 2)),
         ("bicubic x4", test_bicubic),
         (f"{cfg['model']['name']} SR x4 (2.5 m)", test_sr),
         ("reference 2.5 m", test_hr)],
        fig_dir / "scene_comparison.png", reference_for_stretch=test_hr,
        title="Full test scene", dpi=130,
    )
    comparison_panel(
        [("input 10 m (nearest)", np.repeat(np.repeat(test_lr, scale, 1), scale, 2)),
         ("bicubic x4", test_bicubic),
         (f"{cfg['model']['name']} SR x4", test_sr),
         ("reference 2.5 m", test_hr)],
        fig_dir / "zoom_comparison.png", reference_for_stretch=test_hr,
        title=f"Zoom at the highest-gradient window (y={zoom[0]}, x={zoom[1]}, {zoom[2]}px)",
        zoom=zoom, dpi=150,
    )
    map_panel(
        [("reliability (1 = trusted)", maps.reliability, "RdYlGn"),
         ("ensemble spread", maps.spread, "magma"),
         ("Wald inconsistency", maps.inconsistency, "magma"),
         ("inferred detail", maps.inferred_detail, "viridis")],
        fig_dir / "uncertainty_maps.png", title="Per-pixel uncertainty decomposition",
    )
    map_panel(
        [("NDVI reference 2.5 m", indices.compute_index(test_hr, "ndvi"), "RdYlGn"),
         ("NDVI bicubic", indices.compute_index(test_bicubic, "ndvi"), "RdYlGn"),
         ("NDVI super-resolved", indices.compute_index(test_sr, "ndvi"), "RdYlGn"),
         ("NDVI error (SR - ref)",
          indices.compute_index(test_sr, "ndvi") - indices.compute_index(test_hr, "ndvi"), "coolwarm")],
        fig_dir / "ndvi_comparison.png", title="NDVI fidelity",
    )
    map_panel(
        [("true change", change_truth.astype(np.float32), "gray"),
         ("detected from bicubic", cd_masks["bicubic"].astype(np.float32), "gray"),
         ("detected from SR", cd_masks["super_resolved"].astype(np.float32), "gray")],
        fig_dir / "change_detection.png", title="Bi-temporal change detection",
    )
    try:
        metric_bars(
            {"model": {k: v["mean"] for k, v in eval_result["model"].items()
                       if k in ("psnr", "ssim", "sam", "ergas", "scc")},
             "bicubic": {k: v["mean"] for k, v in eval_result["bicubic"].items()
                         if k in ("psnr", "ssim", "sam", "ergas", "scc")}},
            fig_dir / "metrics.png",
            title=f"{cfg['experiment']}: model vs bicubic (val split)",
        )
    except ValueError as exc:
        log.warning("metric bar chart skipped: %s", exc)

    payload = {
        "config": cfg["experiment"],
        "model": cfg["model"]["name"],
        "device": device,
        "scale": scale,
        "dataset": manifest["counts"],
        "evaluation": {
            "model": {k: v["mean"] for k, v in eval_result["model"].items()},
            "bicubic": {k: v["mean"] for k, v in eval_result["bicubic"].items()},
            "spread": {k: {kk: v[kk] for kk in ("p05", "median", "p95")}
                       for k, v in eval_result["model"].items()},
        },
        "inference": infer_summary,
        "uncertainty": unc_summary,
        "indices": index_result,
        "mixed_pixel_fraction_ndvi": mixed,
        "landcover": lc_result,
        "change_detection": cd_result,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float),
                                      encoding="utf-8")
    (out / "report.md").write_text(render_report(payload), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  Demo complete in {payload['runtime_seconds']}s on {device}")
    print("=" * 74)
    ev = payload["evaluation"]
    for key in ("psnr", "ssim", "sam", "ergas", "scc"):
        if key in ev["model"]:
            print(f"  {key:<6} model {ev['model'][key]:>9.4f}   bicubic {ev['bicubic'][key]:>9.4f}")
    print(f"  land-cover OA   SR {lc_result['super_resolved']['overall_accuracy']:.4f}"
          f"   bicubic {lc_result['bicubic']['overall_accuracy']:.4f}"
          f"   HR ceiling {lc_result['true_hr_ceiling']['overall_accuracy']:.4f}")
    print(f"  change F1       SR {cd_result['super_resolved']['f1']:.4f}"
          f"   bicubic {cd_result['bicubic']['f1']:.4f}")
    print(f"  mean reliability  {unc_summary['reliability_mean']:.3f}")
    print(f"\n  Report:  {out / 'report.md'}")
    print(f"  Figures: {fig_dir}")
    print(f"  Product: {out / 'test_scene_sr.tif'}  (2.5 m, georeferenced)\n")


def render_report(p: dict) -> str:
    ev = p["evaluation"]
    lc = p["landcover"]
    cdr = p["change_detection"]
    unc = p["uncertainty"]

    def row(key: str) -> str:
        m, b = ev["model"].get(key), ev["bicubic"].get(key)
        if m is None or b is None:
            return ""
        better = "model" if ((m > b) == (key in evaluate.HIGHER_IS_BETTER)) else "bicubic"
        return f"| {key} | {b:.4f} | {m:.4f} | {m - b:+.4f} | {better} |\n"

    lines = [
        f"# SRM demo report -- {p['model']} x{p['scale']}",
        "",
        f"Run on **{p['device']}** in {p['runtime_seconds']}s. "
        f"Dataset: {p['dataset']['train']} train / {p['dataset']['val']} val patches.",
        "",
        "> **Synthetic input.** These scenes are procedurally generated, so the",
        "> numbers below measure the pipeline, not Sentinel-2 performance. They are",
        "> reproducible and the code path is identical to the real-data one; to make",
        "> a claim about Sentinel-2, run the same steps on downloaded imagery.",
        "",
        "## 1. Reconstruction accuracy (held-out val split)",
        "",
        "| Metric | Bicubic | Model | Delta | Better |",
        "|---|---:|---:|---:|:--:|",
    ]
    lines += [row(k).rstrip("\n") for k in ("psnr", "ssim", "sam", "ergas", "scc", "rmse")
              if row(k)]
    lines += [
        "",
        "PSNR and SSIM measure fidelity; SAM and ERGAS measure spectral integrity;",
        "SCC measures whether the high-frequency detail is genuinely correlated with",
        "the reference rather than merely present.",
        "",
        "## 2. Spectral index fidelity",
        "",
        "| Index | RMSE (SR) | RMSE (bicubic) | Improvement | Boundary improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in p["indices"].items():
        lines.append(
            f"| {name.upper()} | {stats['rmse']:.4f} | {stats['baseline_rmse']:.4f} | "
            f"{stats.get('improvement_pct', float('nan')):+.1f}% | "
            f"{stats.get('improvement_boundary_pct', float('nan')):+.1f}% |"
        )
    mixed = p["mixed_pixel_fraction_ndvi"]
    lines += [
        "",
        "Fraction of pixels whose NDVI falls between class centres (0.10-0.40), i.e.",
        "spectral mixtures belonging to no class -- the problem SRM exists to reduce:",
        "",
        f"- reference 2.5 m: **{mixed['reference_2p5m']:.3f}**",
        f"- bicubic from 10 m: **{mixed['bicubic_from_10m']:.3f}**",
        f"- super-resolved: **{mixed['super_resolved']:.3f}**",
        "",
        "## 3. Land-cover classification",
        "",
        "Random forest, trained and tested on spatially disjoint halves of the scene.",
        "",
        "| Input | Overall accuracy | Kappa |",
        "|---|---:|---:|",
        f"| bicubic from 10 m | {lc['bicubic']['overall_accuracy']:.4f} | {lc['bicubic']['kappa']:.4f} |",
        f"| super-resolved | {lc['super_resolved']['overall_accuracy']:.4f} | {lc['super_resolved']['kappa']:.4f} |",
        f"| true 2.5 m (ceiling) | {lc['true_hr_ceiling']['overall_accuracy']:.4f} | {lc['true_hr_ceiling']['kappa']:.4f} |",
        "",
    ]
    headroom = lc["delta_sr_vs_bicubic"].get("headroom_recovered_pct")
    if headroom is not None and np.isfinite(headroom):
        lines += [
            f"Super-resolution recovered **{headroom:.1f}%** of the accuracy gap between",
            "10 m input and true high-resolution imagery.",
            "",
        ]
    lines += ["Per-class F1 change (SR minus bicubic):", ""]
    for name, delta in sorted(lc["delta_sr_vs_bicubic"]["per_class_f1"].items(),
                              key=lambda kv: -kv[1]):
        lines.append(f"- `{name}`: {delta:+.4f}")

    lines += [
        "",
        "## 4. Change detection",
        "",
        "Identical CVA + Otsu detector on both inputs, so any difference is",
        "attributable to the imagery rather than to detector tuning.",
        "",
        "| Input | Precision | Recall | F1 | IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("bicubic", "bicubic from 10 m"), ("super_resolved", "super-resolved")):
        r = cdr[key]
        lines.append(
            f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['iou']:.4f} |"
        )
    lines += ["", "Object-level recall by change size (the claim that matters for SRM):", "",
              "| Object size | Bicubic | SR |", "|---|---:|---:|"]
    for size_key in cdr["super_resolved"]["object_recall_by_size"]:
        b = cdr["bicubic"]["object_recall_by_size"].get(size_key)
        s = cdr["super_resolved"]["object_recall_by_size"].get(size_key)
        fmt = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
        lines.append(f"| {size_key} | {fmt(b)} | {fmt(s)} |")

    lines += [
        "",
        "## 5. Uncertainty",
        "",
        f"- mean reliability: **{unc['reliability_mean']:.3f}** (1.0 = fully trusted)",
        f"- pixels below 0.5 reliability: **{100 * unc['fraction_low_reliability']:.1f}%**",
        f"- pixels at or above 0.8: **{100 * unc['fraction_high_reliability']:.1f}%**",
        f"- ensemble spread, mean {unc['spread_mean']:.5f} / p95 {unc['spread_p95']:.5f} reflectance",
        f"- Wald inconsistency, mean {unc['inconsistency_mean']:.5f} / p95 {unc['inconsistency_p95']:.5f}",
        "",
        "Low-reliability pixels are where the model's answer changed under",
        "perturbations it should be invariant to, or where degrading the result no",
        "longer reproduces the observed input. Fine features there are model",
        "suggestions and must not be treated as measurements.",
        "",
        "## 6. Output product",
        "",
        f"- `{p['inference']['output']}` -- {p['inference']['gsd_out_m']} m GSD, "
        f"{p['inference']['output_shape']}, CRS preserved from the input",
        f"- inference time: {p['inference']['seconds']}s",
        "",
        "## Figures",
        "",
        "| File | Shows |",
        "|---|---|",
        "| `figures/scene_comparison.png` | whole test scene, four panels, one shared stretch |",
        "| `figures/zoom_comparison.png` | highest-gradient window, chosen by rule not by hand |",
        "| `figures/uncertainty_maps.png` | reliability and its three components |",
        "| `figures/ndvi_comparison.png` | NDVI from each input, plus the SR error map |",
        "| `figures/change_detection.png` | truth vs detections from bicubic and SR |",
        "| `figures/metrics.png` | metric comparison against the baseline |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
