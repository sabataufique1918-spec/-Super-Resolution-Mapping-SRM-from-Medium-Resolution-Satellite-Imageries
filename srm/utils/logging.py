"""Console + CSV + TensorBoard logging with a single call site."""
from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "srm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class RunLogger:
    """Writes scalars to a CSV (always) and TensorBoard (when installed)."""

    def __init__(self, out_dir: str | Path, name: str = "run", use_tensorboard: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / f"{name}_metrics.csv"
        self.log = get_logger("srm.run")
        self._fields: list[str] = []
        self._t0 = time.time()
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.dir / "tb"))
            except Exception:  # tensorboard is optional
                self.writer = None

    def log_scalars(self, step: int, scalars: dict[str, Any], prefix: str = "") -> None:
        row: dict[str, Any] = {"step": step, "elapsed_s": round(time.time() - self._t0, 2)}
        for key, value in scalars.items():
            row[f"{prefix}{key}" if prefix else key] = _to_float(value)
        self._append_csv(row)
        if self.writer is not None:
            for key, value in row.items():
                if key == "step":
                    continue
                val = _to_float(value)
                if isinstance(val, float):
                    self.writer.add_scalar(key, val, step)

    def log_line(self, msg: str) -> None:
        self.log.info(msg)

    def _append_csv(self, row: dict[str, Any]) -> None:
        new_fields = [k for k in row if k not in self._fields]
        if new_fields:
            # The column set grew (validation metrics appear only at epoch end).
            # Rewrite the header and pad earlier rows so the CSV stays parseable.
            old_rows: list[dict[str, Any]] = []
            if self.csv_path.exists():
                with open(self.csv_path, newline="", encoding="utf-8") as fh:
                    old_rows = list(csv.DictReader(fh))
            self._fields += new_fields
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fields)
                writer.writeheader()
                for old in old_rows:
                    writer.writerow({k: old.get(k, "") for k in self._fields})
                writer.writerow({k: row.get(k, "") for k in self._fields})
            return
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self._fields).writerow(
                {k: row.get(k, "") for k in self._fields}
            )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def _to_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
