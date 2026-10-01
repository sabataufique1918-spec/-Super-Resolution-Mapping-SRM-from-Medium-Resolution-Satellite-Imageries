"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 1337, deterministic: bool = False) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct but reproducible stream."""
    base = int(np.random.get_state()[1][0])
    np.random.seed((base + worker_id) % (2**32))
    random.seed((base + worker_id) % (2**32))
