"""Per-fold train/val/test assembly + leakage-safe normalization + torch Dataset.

Fold membership comes directly from baseline.folds.load_fold_assignment
(unchanged) -- identical subject-level 60/20/20 split to the classical
pipeline, so whole subjects never cross splits. Normalization is the DL
analogue of the classical pipeline's `StandardScaler.fit(X_train)`: per-
channel mean/std pooled over all TRAIN windows and all timesteps, applied
unchanged to validation/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from baseline.config import LABEL_TO_INT
from baseline.folds import load_fold_assignment

from .windows import RawWindowTable, select_foot_channels


@dataclass
class FoldSplitData:
    X: np.ndarray  # (n, C, T) float32, normalized
    y: np.ndarray  # (n,) int64
    subject_id: np.ndarray
    window_idx: np.ndarray


class WindowDataset(Dataset):
    def __init__(self, split: FoldSplitData) -> None:
        self.X = torch.from_numpy(split.X)
        self.y = torch.from_numpy(split.y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def fit_normalizer(train_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over training windows and timesteps, shape (1, C, 1).

    Returned so they can be stored alongside a trained model: inference on new
    recordings must reuse the training statistics, never recompute its own.
    """
    mean = train_X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = train_X.std(axis=(0, 2), keepdims=True).astype(np.float32)
    return mean, np.where(std < 1e-8, 1.0, std).astype(np.float32)


def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _fit_normalize(train_X: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean, std = fit_normalizer(train_X)
    return [apply_normalizer(arr, mean, std) for arr in (train_X, *others)]


def assemble_fold_splits(fold_idx: int, foot: str, table: RawWindowTable) -> dict[str, FoldSplitData]:
    foot_data = select_foot_channels(table, foot)  # (N, C, T) float32

    assignment = load_fold_assignment(fold_idx)
    split_map = dict(zip(assignment["subject_id"], assignment["split"]))
    window_splits = np.array([split_map.get(sid) for sid in table.subject_id])
    y_all = np.array([LABEL_TO_INT[lbl] for lbl in table.label], dtype=np.int64)

    masks = {name: (window_splits == name) for name in ("train", "validation", "test")}
    train_X, val_X, test_X = (foot_data[masks[name]] for name in ("train", "validation", "test"))
    train_X_n, val_X_n, test_X_n = _fit_normalize(train_X, val_X, test_X)

    return {
        "train": FoldSplitData(train_X_n, y_all[masks["train"]], table.subject_id[masks["train"]], table.window_idx[masks["train"]]),
        "validation": FoldSplitData(val_X_n, y_all[masks["validation"]], table.subject_id[masks["validation"]], table.window_idx[masks["validation"]]),
        "test": FoldSplitData(test_X_n, y_all[masks["test"]], table.subject_id[masks["test"]], table.window_idx[masks["test"]]),
    }


def concat_splits(a: FoldSplitData, b: FoldSplitData) -> FoldSplitData:
    return FoldSplitData(
        X=np.concatenate([a.X, b.X], axis=0),
        y=np.concatenate([a.y, b.y], axis=0),
        subject_id=np.concatenate([a.subject_id, b.subject_id], axis=0),
        window_idx=np.concatenate([a.window_idx, b.window_idx], axis=0),
    )
