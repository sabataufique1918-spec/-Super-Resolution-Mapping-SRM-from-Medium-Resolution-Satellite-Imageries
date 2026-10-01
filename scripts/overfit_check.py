"""Can this model learn at all? Overfit a handful of real patches and see.

The cheapest way to find out whether a training setup is wired correctly is
to ask it to memorise a tiny batch. A model that cannot drive the loss down
on eight patches it sees a thousand times will never learn the real dataset,
and the failure shows up in a minute instead of two hours.

It catches, specifically:

* a mask that is inverted, so the loss is computed on the holes
* an LR/HR pairing that is off by a crop, so no consistent mapping exists
* a learning rate or initialisation that cannot move the output at all
* an output head that is clamped or scaled wrongly

Passing does **not** mean the model generalises. It means the plumbing works.

    python scripts/overfit_check.py --config configs/sen2neon_x4.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm.config import load_config, resolve_device  # noqa: E402
from srm.data.dataset import build_dataloaders  # noqa: E402
from srm.data.degradation import DegradationConfig, upsample_bicubic  # noqa: E402
from srm.losses import SRMLoss  # noqa: E402
from srm.models.registry import build_model, describe  # noqa: E402
from srm.utils.logging import get_logger  # noqa: E402
from srm.utils.seed import seed_everything  # noqa: E402

log = get_logger("srm.overfit")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Overfit a tiny batch as a wiring check")
    parser.add_argument("--config", default="configs/sen2neon_x4.yaml")
    parser.add_argument("--patches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 1337)))
    device = resolve_device(args.device if args.device != "auto" else cfg.get("device", "auto"))

    cfg["train"]["batch_size"] = args.patches
    cfg["train"]["num_workers"] = 0
    cfg["data"]["augment"] = False          # memorising a moving target is not the test
    train_loader, _, manifest = build_dataloaders(cfg)
    batch = next(iter(train_loader))

    lr_img = batch["lr"].to(device)
    hr_img = batch["hr"].to(device)
    mask = batch["mask"].to(device)
    log.info("batch: lr %s hr %s mask valid %.1f%%",
             tuple(lr_img.shape), tuple(hr_img.shape), 100 * float(mask.mean()))

    model = build_model(cfg).to(device)
    log.info("%s", describe(model))
    criterion = SRMLoss(
        cfg,
        DegradationConfig.from_dict(cfg["data"].get("degradation", {}),
                                    scale=int(cfg["scale"]), bands=cfg["bands"]),
        channels=len(cfg["bands"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # The bar to clear: bicubic interpolation of the same input. A model that
    # memorises eight patches must beat it comfortably; one that cannot is
    # broken, not merely undertrained.
    with torch.no_grad():
        # upsample_bicubic works on one (C, H, W) image, so the batch is
        # upsampled image by image and restacked.
        baseline = torch.from_numpy(
            np.stack([upsample_bicubic(img, int(cfg["scale"]))
                      for img in lr_img.cpu().numpy()])
        ).to(device)
        baseline_loss = float(criterion(baseline, hr_img, lr_img, mask=mask)[0])

    model.train()
    first = last = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = criterion(model(lr_img), hr_img, lr_img, mask=mask)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        first = value if first is None else first
        last = value
        if step % max(1, args.steps // 10) == 0 or step == 1:
            log.info("step %4d  loss %.6f", step, value)

    ratio = last / max(first, 1e-12)
    print("\n" + "=" * 62)
    print(f"  initial loss   {first:.6f}")
    print(f"  final loss     {last:.6f}   ({100 * (1 - ratio):.1f}% reduction)")
    print(f"  bicubic loss   {baseline_loss:.6f}")
    print("=" * 62)

    problems = []
    if not np.isfinite(last):
        problems.append("loss became non-finite -- check the mask and the data range")
    if ratio > 0.5:
        problems.append(
            f"loss fell by only {100 * (1 - ratio):.1f}%. A correctly wired model "
            f"memorises {args.patches} patches easily; suspect the LR/HR pairing, "
            "the mask polarity, or the learning rate."
        )
    if np.isfinite(baseline_loss) and last > baseline_loss:
        problems.append(
            f"final loss {last:.6f} is still worse than bicubic {baseline_loss:.6f}. "
            "The model has not learned anything useful."
        )
    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nPASSED -- the training path is wired correctly.")
    print("This says nothing about generalisation; it says the plumbing works.")


if __name__ == "__main__":
    main()
