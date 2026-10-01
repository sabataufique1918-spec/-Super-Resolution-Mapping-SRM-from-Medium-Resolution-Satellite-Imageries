"""Training loop: PSNR stage, optional adversarial stage, EMA, checkpoints.

Recommended two-stage recipe, and the configs in `configs/` implement it:

1. Train EDSR (or RRDB) with pixel + spectral losses only. This converges
   reliably and gives a spectrally faithful model.
2. Fine-tune from that checkpoint with perceptual + adversarial terms added
   (`train.init_from`). Starting a GAN from scratch on satellite imagery
   reliably produces colour drift and texture that has no basis in the data.

Usage:
    python -m srm.train --config configs/edsr_x4.yaml
    python -m srm.train --config configs/esrgan_x4.yaml --set train.epochs=20
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config, resolve_device, save_config, set_by_path
from .data.dataset import build_dataloaders
from .data.degradation import DegradationConfig
from .losses import SRMLoss, toggle_grad
from .metrics.quality import compute_metrics
from .models.registry import (
    ModelEMA,
    build_discriminator,
    build_model,
    describe,
    save_checkpoint,
)
from .utils.logging import RunLogger, get_logger
from .utils.seed import seed_everything

log = get_logger("srm.train")

#: Validation metrics where a larger value is a better model. Everything else
#: -- loss, RMSE, MAE, SAM, ERGAS -- is minimised. Getting this backwards
#: silently selects the worst epoch, so selection reads it rather than
#: assuming a direction per metric name at the call site.
HIGHER_IS_BETTER = {"psnr", "ssim", "scc", "q_index", "sharpness_gain"}

# Ceiling on the warmup ramp, as a fraction of the run's total optimiser steps.
# A warmup that swallows the whole run is silently catastrophic: the LR never
# reaches base, cosine decay never engages, and because every model here is
# `bicubic_up(x) + residual(x)` with a near-zero-initialised residual, the
# result is a checkpoint indistinguishable from bicubic that the report then
# presents as a measurement. Short runs on small datasets hit this without any
# error, so the ramp is clamped and the clamp is logged.
MAX_WARMUP_FRACTION = 0.25

# Below this many optimiser steps, nothing meaningful is learned regardless of
# the schedule, and any comparison against the bicubic baseline is noise.
MIN_USEFUL_STEPS = 100

# An EMA with decay d averages roughly the last 1/(1-d) steps. Two distinct
# things go wrong when that horizon is long relative to the run:
#
#   * early on, the average still carries the initialisation;
#   * throughout a run that is *still improving*, it averages in strictly
#     worse earlier weights, so the exported model is behind the one that was
#     actually trained.
#
# The second is the one that bites here, and it bites even when the first has
# faded. Measured on this project at 4140 total steps with decay 0.999
# (horizon 1000, a quarter of the run): the EMA weights scored 36.45 dB where
# the raw weights scored 37.80 dB, and spectral angle doubled from 1.62 to
# 3.27 degrees. The EMA turned a model that beats bicubic into one that
# appears not to.
#
# So the horizon is capped at a tenth of the run, and -- because no constant
# is right for every run -- the choice between raw and EMA weights is settled
# by measurement at the end of training rather than assumed. See
# `select_preferred_weights`.
MAX_EMA_HORIZON_FRACTION = 0.1


def effective_ema_decay(requested: float, total_steps: int) -> float:
    """Clamp the EMA horizon to a sane share of the run, and say so."""
    if requested <= 0 or total_steps <= 1:
        return requested
    horizon = 1.0 / max(1e-9, 1.0 - requested)
    limit = max(1.0, total_steps * MAX_EMA_HORIZON_FRACTION)
    if horizon <= limit:
        return requested
    clamped = 1.0 - 1.0 / limit
    log.warning(
        "ema_decay=%.5f averages the last ~%.0f steps, more than %.0f%% of this "
        "run's %d. The average would carry the initialisation early on and lag "
        "the trajectory throughout, exporting a model worse than the one "
        "trained. Clamped to %.5f (~%.0f steps).",
        requested, horizon, 100 * MAX_EMA_HORIZON_FRACTION, total_steps, clamped, limit,
    )
    return clamped


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    total = max(1, int(cfg["train"]["epochs"]) * max(1, steps_per_epoch))
    requested = int(cfg["train"].get("warmup_iters", 0))
    warmup = min(requested, int(total * MAX_WARMUP_FRACTION))
    if warmup < requested:
        log.warning(
            "warmup_iters=%d exceeds %.0f%% of this run's %d total steps; clamped to %d. "
            "The run is short for the configured schedule -- raise train.epochs or "
            "enlarge the dataset before trusting the numbers.",
            requested, 100 * MAX_WARMUP_FRACTION, total, warmup,
        )
    if total < MIN_USEFUL_STEPS:
        log.warning(
            "only %d optimiser steps in this run (%d steps/epoch x %d epochs). "
            "Expect the model to stay near its bicubic initialisation; treat any "
            "comparison against the baseline as a smoke test, not a result.",
            total, steps_per_epoch, int(cfg["train"]["epochs"]),
        )
    kind = cfg["train"].get("scheduler", "cosine")

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "none":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        if kind == "cosine":
            # Floor at 1% of base LR: a cosine that reaches exactly zero wastes
            # the last few hundred steps.
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "step":
            return 0.5 ** int(progress * 4)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, loader, cfg: dict, device: str, max_batches: int | None = None,
             criterion=None) -> dict:
    """Metrics on the validation split, computed on CPU in numpy.

    Deliberately reuses the same metric code as `evaluate.py` rather than a
    fast torch approximation, so the number printed during training is the
    number that ends up in the report.

    Invalid reference pixels are set to NaN before scoring. The metric
    functions already ignore non-finite pixels, so PSNR and friends are
    measured over observed ground only -- otherwise a tile that is 20% empty
    would score the model on its ability to output zeros there.

    `loss` is reported alongside, on the same masked basis as training, and is
    what checkpoint selection uses by default.
    """
    if loader is None:
        return {}
    model.eval()
    rows = []
    losses: list[float] = []
    names = cfg["eval"]["metrics"]
    data_range = float(cfg["eval"].get("data_range", 1.0))
    scale = int(cfg["scale"])
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        lr = batch["lr"].to(device, non_blocking=True)
        hr_t = batch["hr"].to(device, non_blocking=True)
        mask_t = batch.get("mask")
        mask_t = mask_t.to(device, non_blocking=True) if mask_t is not None else None
        sr_t = model(lr)
        if criterion is not None:
            total, _ = criterion(sr_t, hr_t, lr, mask=mask_t)
            losses.append(float(total))

        sr = sr_t.float().cpu().numpy()
        hr = batch["hr"].numpy()
        mask = batch["mask"].numpy() if "mask" in batch else None
        for j in range(sr.shape[0]):
            target = hr[j]
            if mask is not None:
                target = np.where(mask[j][None] > 0.5, target, np.nan)
            rows.append(compute_metrics(sr[j], target, names, data_range, scale))
    model.train()
    if not rows:
        return {}
    out = {k: float(np.mean([r[k] for r in rows if np.isfinite(r.get(k, np.nan))]))
           for k in rows[0]}
    if losses:
        out["loss"] = float(np.mean(losses))
    return out


def train(cfg: dict) -> Path:
    seed_everything(int(cfg.get("seed", 1337)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["train"]["out_dir"]) / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run = RunLogger(out_dir, name=cfg["experiment"])
    save_config(cfg, out_dir / "config.yaml")

    train_loader, val_loader, manifest = build_dataloaders(cfg)
    log.info("dataset: %d train / %d val patches",
             len(train_loader.dataset), len(val_loader.dataset) if val_loader else 0)

    model = build_model(cfg).to(device)
    log.info("%s on %s", describe(model), device)

    init_from = cfg["train"].get("init_from")
    if init_from:
        payload = torch.load(str(init_from), map_location=device, weights_only=False)
        state = payload.get("ema") or payload["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("initialised from %s (missing=%d unexpected=%d)",
                 init_from, len(missing), len(unexpected))

    degradation = DegradationConfig.from_dict(
        cfg["data"].get("degradation", {}), scale=int(cfg["scale"]),
        bands=cfg.get("bands") or manifest["bands"],
    )
    criterion = SRMLoss(cfg, degradation, channels=len(cfg["bands"]), logger=log).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["train"]["lr"]),
        betas=tuple(cfg["train"].get("betas", (0.9, 0.99))),
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    total_steps = max(1, int(cfg["train"]["epochs"]) * max(1, len(train_loader)))
    ema = ModelEMA(
        model,
        effective_ema_decay(float(cfg["train"].get("ema_decay", 0.999)), total_steps),
    )

    discriminator = optimizer_d = None
    if criterion.uses_gan:
        discriminator = build_discriminator(cfg).to(device)
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=float(cfg["train"].get("lr_d", 1e-4)),
            betas=tuple(cfg["train"].get("betas", (0.9, 0.99))),
        )
        log.info("adversarial stage enabled (%s)", cfg["loss"]["gan"].get("type", "ragan"))

    use_amp = bool(cfg["train"].get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    log_every = int(cfg["train"].get("log_every", 25))

    # Selection metric. Validation *loss* is the default because it is the
    # quantity the run is actually minimising and it is masked; PSNR remains
    # available for comparison with the older runs.
    monitor = str(cfg["train"].get("monitor", "loss"))
    lower_is_better = monitor not in HIGHER_IS_BETTER
    patience = int(cfg["train"].get("early_stopping_patience", 0))

    best_score = float("inf") if lower_is_better else -float("inf")
    best: dict[str, float] = {}
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    step = 0
    start_epoch = 1
    stale = 0
    epochs = int(cfg["train"]["epochs"])
    epoch = 0  # bound even if the loop never runs (a fully-resumed checkpoint)

    if cfg["train"].get("resume"):
        resumed = _resume(cfg["train"]["resume"], last_path, model, optimizer, ema,
                          scheduler, discriminator, device)
        if resumed:
            start_epoch = resumed["epoch"] + 1
            step = resumed.get("step", 0)
            best = resumed.get("metrics", {}) or {}
            if monitor in best:
                best_score = best[monitor]
            log.info("resumed from epoch %d (step %d, best %s=%s)",
                     resumed["epoch"], step, monitor, best.get(monitor))
            if start_epoch > epochs:
                log.warning("checkpoint is already at epoch %d of %d; nothing to do",
                            resumed["epoch"], epochs)

    model.train()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        running: dict[str, float] = {}
        for batch in train_loader:
            lr_img = batch["lr"].to(device, non_blocking=True)
            hr_img = batch["hr"].to(device, non_blocking=True)
            mask_img = batch.get("mask")
            if mask_img is not None:
                mask_img = mask_img.to(device, non_blocking=True)

            # ---- generator ----
            if discriminator is not None:
                toggle_grad(discriminator, False)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                sr = model(lr_img)
                fake_pred = real_pred = None
                if discriminator is not None:
                    fake_pred = discriminator(sr)
                    real_pred = discriminator(hr_img).detach()
                loss, parts = criterion(sr, hr_img, lr_img, fake_pred, real_pred,
                                        mask=mask_img)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)

            # ---- discriminator ----
            # Both optimisers share one GradScaler, so `update()` is called
            # exactly once per iteration, after every `step()`.
            if discriminator is not None:
                toggle_grad(discriminator, True)
                optimizer_d.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    d_real = discriminator(hr_img)
                    d_fake = discriminator(sr.detach())
                    loss_d = criterion.gan.discriminator_loss(d_real, d_fake)
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                parts["d_loss"] = float(loss_d.detach())

            scaler.update()
            scheduler.step()
            ema.update(model)

            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            step += 1

            if step % log_every == 0:
                averaged = {k: v / log_every for k, v in running.items()}
                averaged["lr"] = scheduler.get_last_lr()[0]
                run.log_scalars(step, averaged, prefix="train/")
                log.info(
                    "epoch %d step %d | %s",
                    epoch, step,
                    " ".join(f"{k}={v:.4f}" for k, v in averaged.items()),
                )
                running = {}

        if val_loader is not None and epoch % int(cfg["train"].get("val_every", 1)) == 0:
            # Checkpoint selection only needs enough validation patches to rank
            # epochs reliably, and the numpy SSIM/ERGAS pass is expensive. The
            # reportable figures come from `srm.evaluate` on the held-out test
            # split, over every patch, at the end.
            metrics = validate(ema.module, val_loader, cfg, device,
                               max_batches=cfg["train"].get("val_max_batches"),
                               criterion=criterion)
            run.log_scalars(step, metrics, prefix="val/")
            log.info("epoch %d validation | %s | %.1fs", epoch,
                     " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                     time.time() - epoch_start)

            score = metrics.get(monitor)
            if score is None:
                raise KeyError(
                    f"train.monitor='{monitor}' is not among the validation metrics "
                    f"{sorted(metrics)}. Add it to eval.metrics, or use 'loss'."
                )
            improved = (score < best_score) if lower_is_better else (score > best_score)
            if improved:
                best_score, best, stale = score, metrics, 0
                save_checkpoint(best_path, model, cfg, epoch, metrics, ema=ema,
                                discriminator=discriminator, step=step)
                log.info("new best val %s=%.5f -> %s", monitor, score, best_path.name)
            else:
                stale += 1
                log.info("no improvement in val %s for %d epoch(s) (best %.5f)",
                         monitor, stale, best_score)
                if patience and stale >= patience:
                    log.info("early stopping: %d epochs without improvement", stale)
                    save_checkpoint(last_path, model, cfg, epoch, best,
                                    optimizer=optimizer, ema=ema,
                                    discriminator=discriminator, step=step)
                    break

        if epoch % int(cfg["train"].get("save_every", 5)) == 0 or epoch == epochs:
            save_checkpoint(last_path, model, cfg, epoch, best, optimizer=optimizer,
                            ema=ema, discriminator=discriminator, step=step)

    save_checkpoint(last_path, model, cfg, epoch, best, optimizer=optimizer,
                    ema=ema, discriminator=discriminator, step=step)

    # Decide raw-vs-EMA by measurement rather than by assumption, and record
    # the answer in the checkpoint so every later load agrees with it.
    final = best_path if best_path.exists() else last_path
    try:
        select_preferred_weights(final, cfg, device,
                                 max_batches=cfg["train"].get("val_max_batches"))
    except Exception as exc:  # never lose a finished run to a reporting step
        log.warning("weight selection failed (%s); the checkpoint keeps its "
                    "raw weights by default", exc)

    run.close()
    log.info("done. best=%s metrics=%s", final,
             {k: round(v, 4) for k, v in best.items() if np.isfinite(v)})
    return final


def select_preferred_weights(
    checkpoint_path: str | Path,
    cfg: dict,
    device: str = "cpu",
    max_batches: int | None = None,
) -> dict:
    """Score a checkpoint's raw and EMA weights on validation, stamp the winner.

    Averaging is supposed to help, and usually does -- but whether it helps on
    *this* run is an empirical question with a cheap answer, and getting it
    wrong is expensive and silent. So both sets of weights are scored on the
    validation split and the better one is recorded in the checkpoint under
    ``prefer``, which `load_checkpoint` then honours.

    This replaces the previous behaviour of always preferring EMA weights when
    present, which on a still-improving run exported a model measurably worse
    than the one that had been trained.
    """
    from .data.dataset import build_dataloaders
    from .models.registry import build_model

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not payload.get("ema"):
        payload["prefer"] = "model"
        torch.save(payload, checkpoint_path)
        return {"prefer": "model", "reason": "no EMA weights in this checkpoint"}

    _, val_loader, _ = build_dataloaders(cfg)
    if val_loader is None:
        return {"prefer": payload.get("prefer", "ema"),
                "reason": "no validation split; leaving the preference unchanged"}

    monitor = str(cfg["train"].get("monitor", "loss"))
    lower_is_better = monitor not in HIGHER_IS_BETTER
    criterion = SRMLoss(
        cfg,
        DegradationConfig.from_dict(
            cfg["data"].get("degradation", {}), scale=int(cfg["scale"]),
            bands=cfg.get("bands") or [],
        ),
        channels=len(cfg.get("bands") or []) or 4,
    ).to(device)

    scores: dict[str, dict] = {}
    for key, state_key in (("model", "model"), ("ema", "ema")):
        model = build_model(payload["config"]).to(device)
        model.load_state_dict(payload[state_key])
        model.eval()
        scores[key] = validate(model, val_loader, cfg, device,
                               max_batches=max_batches, criterion=criterion)

    a, b = scores["model"].get(monitor), scores["ema"].get(monitor)
    if a is None or b is None:
        return {"prefer": "ema", "reason": f"{monitor} unavailable for one of the two"}
    ema_wins = (b < a) if lower_is_better else (b > a)
    prefer = "ema" if ema_wins else "model"

    payload["prefer"] = prefer
    payload["weight_selection"] = {
        "monitor": monitor,
        "model": {k: v for k, v in scores["model"].items()},
        "ema": {k: v for k, v in scores["ema"].items()},
        "chosen": prefer,
    }
    torch.save(payload, checkpoint_path)
    log.info("weight selection on val %s: raw=%.5f ema=%.5f -> using %s weights",
             monitor, a, b, prefer)
    return payload["weight_selection"]


def _resume(resume, last_path: Path, model, optimizer, ema, scheduler,
            discriminator, device: str) -> dict | None:
    """Restore weights, optimiser and EMA from a checkpoint.

    `train.resume: true` picks up this experiment's own `last.pt`; a path
    resumes from somewhere else. Missing files are not an error -- the first
    run of a config that has resuming switched on should simply start.
    """
    path = last_path if resume is True else Path(str(resume))
    if not path.is_file():
        log.info("resume requested but %s does not exist; starting fresh", path)
        return None
    payload = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if payload.get("ema"):
        ema.module.load_state_dict(payload["ema"])
    if payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if discriminator is not None and payload.get("discriminator"):
        discriminator.load_state_dict(payload["discriminator"])
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an SRM model")
    parser.add_argument("--config", default="configs/edsr_x4.yaml")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="override config entries, e.g. --set train.epochs=5 model.features=32",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    import yaml

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got '{pair}'")
        key, raw = pair.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    train(cfg)


if __name__ == "__main__":
    main()
