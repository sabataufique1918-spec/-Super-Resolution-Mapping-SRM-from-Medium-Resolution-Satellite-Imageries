"""Build a paired patch dataset.

Three sources, all producing the same on-disk format:

    # synthetic scenes -- no credentials needed, runs anywhere
    python scripts/prepare_dataset.py --source synthetic --n-scenes 24 \
        --out data/patches

    # real Sentinel-2 already downloaded as GeoTIFFs
    python scripts/prepare_dataset.py --source geotiff --glob "data/raw/*.tif" \
        --out data/patches

    # real high-resolution reference imagery (PlanetScope/NAIP/WorldView),
    # which gives native 10 m -> 2.5 m pairs with no scale-transfer assumption
    python scripts/prepare_dataset.py --source geotiff --glob "data/hr/*.tif" \
        --out data/patches_native --hr-patch 256
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm.data.degradation import DegradationConfig  # noqa: E402
from srm.data.patches import PatchSpec, build_patches, scenes_from_geotiffs  # noqa: E402
from srm.data.synthetic import SceneSpec, generate_scene  # noqa: E402
from srm.utils.logging import get_logger  # noqa: E402
from srm.utils.seed import seed_everything  # noqa: E402

log = get_logger("srm.prepare")


def synthetic_scenes(n_scenes: int, size: int, seed: int):
    scenes = []
    for i in range(n_scenes):
        # Vary the landscape per scene so the model sees agriculture, dense
        # urban and water-dominated regimes rather than one synthetic texture.
        rng = np.random.default_rng(seed + i)
        spec = SceneSpec(
            size=size,
            n_fields=int(rng.integers(20, 70)),
            n_urban_blocks=int(rng.integers(0, 9)),
            road_density=float(rng.uniform(0.4, 1.8)),
            river=bool(rng.random() < 0.6),
            haze=float(rng.uniform(0.0, 0.010)),
            texture=float(rng.uniform(0.3, 0.9)),
        )
        image, _ = generate_scene(spec, seed=seed + i)
        scenes.append((image, None))
        log.info("synthetic scene %d/%d (%dx%d)", i + 1, n_scenes, size, size)
    return scenes


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build an SRM patch dataset")
    parser.add_argument("--source", choices=["synthetic", "geotiff"], default="synthetic")
    parser.add_argument("--glob", default="data/raw/*.tif",
                        help="input pattern when --source geotiff")
    parser.add_argument("--out", default="data/patches")
    parser.add_argument("--n-scenes", type=int, default=24)
    parser.add_argument("--scene-size", type=int, default=1024)
    parser.add_argument("--hr-patch", type=int, default=192)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--min-std", type=float, default=0.010)
    parser.add_argument("--val-fraction", type=float, default=0.12)
    parser.add_argument("--store-lr", action="store_true",
                        help="precompute the LR side (use for real, non-simulated pairs)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    seed_everything(args.seed)

    if args.source == "synthetic":
        scenes = synthetic_scenes(args.n_scenes, args.scene_size, args.seed)
    else:
        paths = sorted(glob(args.glob))
        if not paths:
            raise SystemExit(
                f"no files matched {args.glob!r}. Download some first:\n"
                "  python scripts/download_copernicus.py --list-aois"
            )
        log.info("reading %d GeoTIFF scenes", len(paths))
        scenes = scenes_from_geotiffs(paths)

    spec = PatchSpec(
        hr_patch=args.hr_patch,
        scale=args.scale,
        stride=args.stride,
        min_std=args.min_std,
        val_fraction=args.val_fraction,
        store_lr=args.store_lr,
    )
    manifest = build_patches(
        scenes,
        args.out,
        spec=spec,
        degradation=DegradationConfig(scale=args.scale),
        seed=args.seed,
        logger=log,
    )
    print(
        f"\n{manifest['counts']['train']} train / {manifest['counts']['val']} val patches "
        f"at {args.hr_patch}px HR (x{args.scale}) -> {args.out}"
    )
    if manifest["counts"]["train"] == 0:
        raise SystemExit(
            "no training patches survived filtering. Lower --min-std or use larger scenes."
        )


if __name__ == "__main__":
    main()
