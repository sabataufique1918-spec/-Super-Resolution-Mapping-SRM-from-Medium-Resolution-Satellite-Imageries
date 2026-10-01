"""Turn downloaded SEN2NEON tiles into a geographically split patch dataset.

    python scripts/prepare_sen2neon.py \
        --root ~/srm-data/sen2neon \
        --out data/patches_sen2neon

What this does that the synthetic path does not:

* Selects RGB+NIR **by documented band name** (B04/B03/B02/B08 -> file bands
  4, 3, 2, 8), never by position.
* Splits 70/15/15 by **NEON site**, so no two splits share a location. Tiles
  from one mosaic overlap in context and repeat observations of a site recur
  across years; both stay together.
* Stores the HR **validity mask** with every patch, so the loss can ignore
  NEON's empty flight-line wedges instead of learning them.
* Writes the split report next to the shards, including the leakage check.

The split is computed over the tiles that are actually on disk, so a partial
download still produces a coherent dataset -- but the fractions are only
meaningful once enough sites are present, and the script says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm.data.sen2neon import (  # noqa: E402
    RGB_NIR_BANDS,
    IndexFilter,
    PairPatchSpec,
    band_indices,
    build_pair_patches,
    geographic_split,
    load_index,
    split_report,
)
from srm.utils.logging import get_logger  # noqa: E402
from srm.utils.seed import seed_everything  # noqa: E402

log = get_logger("srm.prepare_sen2neon")

DEFAULT_ROOT = Path.home() / "srm-data" / "sen2neon"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build a SEN2NEON patch dataset")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default="data/patches_sen2neon")
    parser.add_argument("--bands", nargs=4, default=list(RGB_NIR_BANDS),
                        metavar=("BLUE", "GREEN", "RED", "NIR"),
                        help="band names in the project's canonical order")
    parser.add_argument("--lr-patch", type=int, default=64,
                        help="LR patch side; HR is this times --scale")
    parser.add_argument("--stride", type=int, default=None,
                        help="LR stride between patches (default: no overlap)")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--min-valid", type=float, default=0.90,
                        help="minimum observed fraction of the HR window")
    parser.add_argument("--min-std", type=float, default=0.004)
    parser.add_argument("--group-by", choices=["site", "mosaic"], default="site")
    parser.add_argument("--fractions", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--min-cloud-score", type=float, default=0.90)
    parser.add_argument("--max-nodata-pct", type=float, default=20.0)
    parser.add_argument("--max-temporal-days", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    root = Path(args.root)
    out = Path(args.out)

    log.info("bands %s -> file band indices %s (1-based)",
             list(args.bands), list(band_indices(args.bands)))

    filters = IndexFilter(args.min_cloud_score, args.max_nodata_pct, args.max_temporal_days)
    pairs = load_index(root, filters=filters, require_files=True)
    if not pairs:
        raise SystemExit(
            f"no downloaded pairs found under {root}.\n"
            f"  python scripts/download_sen2neon.py --root {root} --limit 150"
        )
    log.info("%d downloaded pairs across %d sites",
             len(pairs), len({p.site for p in pairs}))

    splits = geographic_split(pairs, tuple(args.fractions), args.seed, args.group_by)
    report = split_report(splits)
    log.info("split by %s: %s  (%s)", args.group_by, report["counts"], report["fractions"])
    log.info("sites per split: %s", report["n_sites"])
    if not report["leakage_free"]:
        raise SystemExit(f"geographic leakage detected: {report['shared_sites']}")
    log.info("leakage check: no site appears in more than one split")

    spec = PairPatchSpec(
        lr_patch=args.lr_patch,
        scale=args.scale,
        stride=args.stride,
        min_valid_fraction=args.min_valid,
        min_std=args.min_std,
    )
    manifest = build_pair_patches(
        splits, root, out, spec=spec, bands=args.bands, seed=args.seed, logger=log
    )
    manifest["split_report"] = report
    manifest["index_filter"] = {
        "min_cloud_score": args.min_cloud_score,
        "max_hr_nodata_pct": args.max_nodata_pct,
        "max_temporal_days": args.max_temporal_days,
    }
    manifest["group_by"] = args.group_by
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    counts = manifest["counts"]
    print(f"\n{sum(counts.values())} patches "
          f"({args.lr_patch}x{args.lr_patch} -> {spec.hr_patch}x{spec.hr_patch}) -> {out}")
    for split in ("train", "val", "test"):
        print(f"  {split:<6} {counts.get(split, 0):>6} patches  "
              f"{report['n_sites'].get(split, 0):>3} sites")
    if manifest["failures"]:
        print(f"\n{len(manifest['failures'])} tiles were skipped; see manifest.json")
    if counts.get("train", 0) == 0:
        raise SystemExit(
            "no training patches survived filtering. Lower --min-valid or --min-std."
        )
    print("\nNext:\n  python -m srm.train --config configs/sen2neon_x4.yaml")


if __name__ == "__main__":
    main()
