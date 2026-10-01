"""Download Sentinel-2 L2A imagery from the Copernicus Data Space Ecosystem.

Setup (once):

1. Register free at https://dataspace.copernicus.eu
2. Create an OAuth client at
   https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
3. Put the credentials in the environment:

       PowerShell:  $env:CDSE_CLIENT_ID="..."; $env:CDSE_CLIENT_SECRET="..."
       bash:        export CDSE_CLIENT_ID=...  CDSE_CLIENT_SECRET=...

Then:

    python scripts/download_copernicus.py --list-aois
    python scripts/download_copernicus.py --all --out data/raw
    python scripts/download_copernicus.py --bbox 73.78 18.46 73.98 18.60 \
        --start 2024-01-01 --end 2024-03-31 --name pune --out data/raw

Pick areas interactively at https://browser.dataspace.copernicus.eu and copy
the bounding box from the URL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm.data.copernicus import (  # noqa: E402
    DEFAULT_AOIS,
    AOI,
    CopernicusError,
    fetch_s2_patch,
    get_token,
    load_s2_geotiff,
    search_products,
)
from srm.utils.logging import get_logger  # noqa: E402

log = get_logger("srm.download")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 L2A via Copernicus Data Space")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--all", action="store_true", help="download every built-in AOI")
    parser.add_argument("--list-aois", action="store_true")
    parser.add_argument("--aoi", nargs="*", default=[], help="built-in AOI names to fetch")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    parser.add_argument("--start", default="2024-05-01")
    parser.add_argument("--end", default="2024-09-30")
    parser.add_argument("--name", default="custom")
    parser.add_argument("--max-cloud", type=float, default=10.0)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--search-only", action="store_true",
                        help="list matching products without downloading (no credentials needed)")
    parser.add_argument("--min-valid", type=float, default=0.95,
                        help="reject a download with less than this fraction of clear pixels")
    args = parser.parse_args(argv)

    if args.list_aois:
        print(f"{'name':<22} {'bbox':<44} window")
        for aoi in DEFAULT_AOIS:
            bbox = ", ".join(f"{v:.3f}" for v in aoi.bbox)
            print(f"{aoi.name:<22} [{bbox}]  {aoi.start} .. {aoi.end}")
        return

    if args.bbox:
        targets = [AOI(args.name, tuple(args.bbox), args.start, args.end, args.max_cloud)]
    elif args.aoi:
        by_name = {a.name: a for a in DEFAULT_AOIS}
        missing = [n for n in args.aoi if n not in by_name]
        if missing:
            raise SystemExit(f"unknown AOI(s) {missing}. Use --list-aois.")
        targets = [by_name[n] for n in args.aoi]
    elif args.all:
        targets = list(DEFAULT_AOIS)
    else:
        raise SystemExit("pick one of --all, --aoi NAME..., --bbox ..., or --list-aois")

    if args.search_only:
        for aoi in targets:
            products = search_products(aoi.bbox, aoi.start, aoi.end, aoi.max_cloud)
            print(f"\n{aoi.name}: {len(products)} products")
            for product in products[:10]:
                print(f"  {product.get('ContentDate', {}).get('Start', '?')[:10]}  {product.get('Name')}")
        return

    try:
        token = get_token()
    except CopernicusError as exc:
        raise SystemExit(f"\n{exc}\n") from exc

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, failed = [], []

    for aoi in targets:
        path = out_dir / f"s2_{aoi.name}.tif"
        try:
            fetch_s2_patch(aoi, path, resolution_m=args.resolution, token=token)
            # Verify before keeping it: a cloudy scene in the training set is
            # worse than no scene, because the model learns cloud texture.
            _, valid, meta = load_s2_geotiff(path, min_valid_fraction=args.min_valid)
            log.info("%s -> %s (%.1f%% clear, CRS %s)",
                     aoi.name, path.name, 100 * float(valid.mean()), meta.crs)
            written.append(str(path))
        except CopernicusError as exc:
            log.warning("%s skipped: %s", aoi.name, exc)
            path.unlink(missing_ok=True)
            failed.append({"aoi": aoi.name, "reason": str(exc)})

    manifest = out_dir / "download_manifest.json"
    manifest.write_text(json.dumps({
        "aois": [a.as_dict() for a in targets],
        "written": written,
        "failed": failed,
        "resolution_m": args.resolution,
    }, indent=2), encoding="utf-8")

    print(f"\n{len(written)} scene(s) written to {out_dir}, {len(failed)} skipped.")
    if written:
        print("Next: python scripts/prepare_dataset.py --source geotiff "
              f'--glob "{out_dir}/s2_*.tif" --out data/patches')


if __name__ == "__main__":
    main()
