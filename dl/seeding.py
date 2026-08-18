"""Per-fold deterministic seeding for python/numpy/torch (+MPS)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def fold_seed(random_state: int, fold_idx: int) -> int:
    return random_state + fold_idx
