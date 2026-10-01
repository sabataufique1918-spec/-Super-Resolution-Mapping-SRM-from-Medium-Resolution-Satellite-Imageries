"""Fetch a configurable subset of SEN2NEON (real Sentinel-2 10 m / NEON 2.5 m pairs).

The full release is 2,269 pairs and roughly 60 GB with the canonical 2.5 m
product. Nothing here needs all of it: a few hundred tiles spread over many
NEON sites is a better training set than a thousand tiles from three sites,
because the geographic split is what makes the validation honest.

    # index only, no imagery -- inspect what is available first
    python scripts/download_sen2neon.py --metadata-only

    # 150 pairs spread across as many sites as possible (~4 GB)
    python scripts/download_sen2neon.py --limit 150

    # everything from two named sites
    python scripts/download_sen2neon.py --sites HARV JERC --limit 0

Downloads resume: files already present and the right size are skipped, so
re-running after an interruption costs only the HEAD requests.

Storage note: point --root somewhere that is not inside a synced folder.
Several GB of GeoTIFFs in OneDrive/Dropbox will be uploaded continuously and
can be locked mid-write.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm.data.sen2neon import (  # noqa: E402
    HR_DIR,
    LR_DIR,
    METADATA,
    REPO_ID,
    IndexFilter,
    geographic_split,
    load_index,
    split_report,
    stratified_subset,
)
from srm.utils.logging import get_logger  # noqa: E402

log = get_logger("srm.download")

DEFAULT_ROOT = Path.home() / "srm-data" / "sen2neon"


def _hub():
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required for this script.\n"
            "  pip install huggingface_hub"
        ) from exc


def fetch(relative: str, root: Path, retries: int = 3) -> Path:
    """Download one repo file into `root`, skipping it if already complete."""
    hf_hub_download = _hub()
    target = root / relative
    if target.is_file() and target.stat().st_size > 0:
        return target
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=relative,
                local_dir=str(root),
            )
            return target
        except Exception as exc:  # network flakiness is the norm, not the exception
            last = exc
            if attempt < retries:
                wait = 2 ** attempt
                log.warning("retry %d/%d for %s in %ds (%s)", attempt, retries,
                            relative, wait, type(exc).__name__)
                time.sleep(wait)
    raise RuntimeError(f"failed to download {relative}: {last}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download a SEN2NEON subset")
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help=f"destination directory (default: {DEFAULT_ROOT})")
    parser.add_argument("--limit", type=int, default=150,
                        help="number of LR/HR pairs; 0 means every matching tile")
    parser.add_argument("--sites", nargs="*", default=None,
                        help="restrict to these NEON site codes, e.g. HARV JERC")
    parser.add_argument("--metadata-only", action="store_true",
                        help="fetch the index and report what is available, then stop")
    parser.add_argument("--min-cloud-score", type=float, default=0.90)
    parser.add_argument("--max-nodata-pct", type=float, default=20.0)
    parser.add_argument("--max-temporal-days", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    log.info("index -> %s", root / METADATA)
    fetch(METADATA, root)
    fetch("README.md", root)
    fetch("DATASET_RELEASE_NOTES.md", root)

    filters = IndexFilter(
        min_cloud_score=args.min_cloud_score,
        max_hr_nodata_pct=args.max_nodata_pct,
        max_temporal_days=args.max_temporal_days,
    )
    everything = load_index(root, filters=IndexFilter(0.0, 100.0, 10**6))
    eligible = load_index(root, filters=filters)
    log.info("index: %d tiles total, %d pass the quality filter "
             "(cloud>=%.2f, hr_nodata<=%.0f%%, dt<=%dd)",
             len(everything), len(eligible), filters.min_cloud_score,
             filters.max_hr_nodata_pct, filters.max_temporal_days)

    if args.sites:
        wanted = {s.upper() for s in args.sites}
        eligible = [p for p in eligible if p.site in wanted]
        log.info("restricted to %s: %d tiles", sorted(wanted), len(eligible))
        if not eligible:
            raise SystemExit(f"no tiles for sites {sorted(wanted)}")

    chosen = eligible if args.limit <= 0 else stratified_subset(eligible, args.limit, args.seed)
    sites = sorted({p.site for p in chosen})
    log.info("selected %d pairs across %d sites: %s",
             len(chosen), len(sites), ", ".join(sites))

    if args.metadata_only:
        print(f"\n{len(eligible)} eligible tiles across "
              f"{len({p.site for p in eligible})} sites.")
        print(f"Would download {len(chosen)} pairs (~{len(chosen) * 27 / 1000:.1f} GB).")
        print("Re-run without --metadata-only to fetch them.")
        return

    # Preview the split now rather than after a multi-GB download, so an
    # unusable selection (too few sites for a 70/15/15) fails in seconds.
    try:
        report = split_report(geographic_split(chosen, seed=args.seed))
        log.info("projected split: %s across %s sites",
                 report["counts"], report["n_sites"])
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\nRaise --limit, or drop --sites, so more NEON sites are included."
        )

    started = time.time()
    for i, pair in enumerate(chosen, start=1):
        fetch(pair.lr_path, root)
        fetch(pair.hr_path, root)
        if i % 10 == 0 or i == len(chosen):
            rate = i / max(time.time() - started, 1e-6)
            log.info("%d/%d pairs (%.1f/min, %.0fs elapsed)",
                     i, len(chosen), rate * 60, time.time() - started)

    on_disk = sum(f.stat().st_size for f in root.rglob("*.tif"))
    print(f"\n{len(chosen)} pairs across {len(sites)} sites in {root}")
    print(f"  {LR_DIR}/  {HR_DIR}/   {on_disk / 1e9:.2f} GB")
    print("\nNext:")
    print(f"  python scripts/prepare_sen2neon.py --root {root} --out data/patches_sen2neon")


if __name__ == "__main__":
    main()
