"""Decide raw-vs-EMA weights for an existing checkpoint, by measurement.

`srm.train` now does this automatically at the end of a run. This script
applies the same selection to checkpoints produced before that existed, or
re-runs it after the validation split changes.

    python scripts/stamp_weights.py --checkpoint checkpoints/sen2neon_x4/best.pt

Why it matters: a checkpoint stores both the trained weights and their
exponential moving average, and the loader has to pick one. Always picking
the EMA -- the previous behaviour -- is wrong whenever the run was still
improving when it ended, because the average then includes strictly worse
earlier weights. Measured on this project that cost 1.35 dB of PSNR and
doubled spectral angle, with nothing anywhere reporting it.

The script scores both on the validation split, prints the comparison, and
writes the winner into the checkpoint's ``prefer`` key.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm.config import resolve_device  # noqa: E402
from srm.train import select_preferred_weights  # noqa: E402
from srm.utils.logging import get_logger  # noqa: E402

log = get_logger("srm.stamp")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Choose raw or EMA weights by measurement")
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="one or more checkpoints to stamp")
    parser.add_argument("--patches", default=None,
                        help="override the validation dataset (default: the "
                             "one recorded in the checkpoint's config)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="cap the validation pass; default is the whole split")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    for path in args.checkpoint:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"no such checkpoint: {path}")
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = payload.get("config")
        if cfg is None:
            raise SystemExit(f"{path} has no embedded config")
        if args.patches:
            cfg = {**cfg, "data": {**cfg.get("data", {}), "patches": args.patches}}

        log.info("scoring %s on %s", path, cfg["data"]["patches"])
        result = select_preferred_weights(path, cfg, device, args.max_batches)

        print(f"\n{path}")
        if "model" not in result:
            print(f"  {result.get('reason', result)}")
            continue
        monitor = result["monitor"]
        raw, ema = result["model"], result["ema"]
        keys = [k for k in ("loss", "psnr", "ssim", "sam", "mae") if k in raw]
        width = max(len(k) for k in keys) + 2
        print(f"  {'metric':<{width}}{'raw':>12}{'ema':>12}")
        for key in keys:
            mark = "  <- selection metric" if key == monitor else ""
            print(f"  {key:<{width}}{raw[key]:>12.5f}{ema[key]:>12.5f}{mark}")
        print(f"  -> stamped prefer = {result['chosen']!r}")


if __name__ == "__main__":
    main()
