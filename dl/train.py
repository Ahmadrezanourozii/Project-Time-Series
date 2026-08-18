"""Per-fold training: grid search (dropout x lr) with early stopping on
validation ROC-AUC, selection of the winning combo, and a refit on
train+validation for the winning epoch count -- the DL analogue of
GridSearchCV(refit=True) on a PredefinedSplit(train, validation).
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from baseline.folds import FoldResult

from .config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE_BY_MODEL,
    DROPOUT_GRID,
    EARLY_STOP_PATIENCE,
    FOOT_NCHANNELS,
    LR_GRID,
    MAX_EPOCHS_BY_MODEL,
    NUM_WORKERS,
    WEIGHT_DECAY,
)
from .datasets import FoldSplitData, WindowDataset, assemble_fold_splits, concat_splits
from .models import build_model
from .seeding import fold_seed, set_all_seeds


def _release_device_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def _predict_scores(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    scores = []
    tensor = torch.from_numpy(X)
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start : start + batch_size].to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        scores.append(probs.cpu().numpy())
    return np.concatenate(scores)


def _train_to_convergence(
    model_name: str,
    in_channels: int,
    dropout: float,
    lr: float,
    train_split: FoldSplitData,
    val_split: FoldSplitData | None,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int | None,
) -> tuple[nn.Module, int, float]:
    """Train one (dropout, lr) config.

    If val_split is given: early-stop on validation ROC-AUC, return the
    best-epoch model, the epoch it occurred at, and that best AUC. If
    val_split is None (the train+validation refit stage, where there is no
    held-out set left to monitor): train for exactly `max_epochs` epochs and
    return the final model.
    """
    set_all_seeds(seed)
    generator = torch.Generator().manual_seed(seed)

    model = build_model(model_name, in_channels=in_channels, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        WindowDataset(train_split),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        drop_last=True,  # avoid a trailing batch of size 1 breaking BatchNorm1d
    )

    best_state = copy.deepcopy(model.state_dict())
    best_auc = -np.inf
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        if val_split is None:
            continue  # refit stage: run the fixed epoch count, no monitoring

        val_scores = _predict_scores(model, val_split.X, device)
        val_auc = roc_auc_score(val_split.y, val_scores)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if patience is not None and epochs_since_improve >= patience:
                break

    if val_split is None:
        return model, max_epochs, float("nan")

    model.load_state_dict(best_state)
    return model, best_epoch, best_auc


def run_fold(fold_idx: int, foot: str, model_name: str, table) -> FoldResult:
    splits = assemble_fold_splits(fold_idx, foot, table)
    in_channels = FOOT_NCHANNELS[foot]
    device = DEVICE_BY_MODEL[model_name]
    base_seed = fold_seed(42, fold_idx)
    test_split = splits["test"]

    checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_{foot}_fold{fold_idx}.pt"
    if checkpoint_path.exists():
        # Resuming an interrupted sweep: this fold's grid-search + refit
        # already ran to completion previously and its winning weights were
        # saved. Skip retraining entirely -- dropout is irrelevant here since
        # nn.Dropout has no learned parameters and model.eval() (used for
        # prediction) disables it regardless of the configured rate.
        model = build_model(model_name, in_channels=in_channels, dropout=0.0).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        y_score = _predict_scores(model, test_split.X, device)
        y_pred = (y_score >= 0.5).astype(int)
        return FoldResult(
            fold_idx=fold_idx,
            foot=foot,
            model_name=model_name,
            subject_id=test_split.subject_id,
            window_idx=test_split.window_idx,
            y_true=test_split.y,
            y_pred=y_pred,
            y_score=y_score,
            best_params={"note": "resumed from checkpoint; original grid-search params not preserved"},
        )

    grid = list(itertools.product(DROPOUT_GRID[model_name], LR_GRID))

    best_combo: tuple[float, float] | None = None
    best_combo_auc = -np.inf
    best_combo_epoch = 1

    for dropout, lr in grid:
        combo_model, epoch, val_auc = _train_to_convergence(
            model_name=model_name,
            in_channels=in_channels,
            dropout=dropout,
            lr=lr,
            train_split=splits["train"],
            val_split=splits["validation"],
            device=device,
            seed=base_seed,
            max_epochs=MAX_EPOCHS_BY_MODEL[model_name],
            patience=EARLY_STOP_PATIENCE,
        )
        if val_auc > best_combo_auc:
            best_combo_auc = val_auc
            best_combo = (dropout, lr)
            best_combo_epoch = epoch
        del combo_model
        _release_device_memory(device)

    dropout, lr = best_combo
    trainval_split = concat_splits(splits["train"], splits["validation"])
    refit_model, _, _ = _train_to_convergence(
        model_name=model_name,
        in_channels=in_channels,
        dropout=dropout,
        lr=lr,
        train_split=trainval_split,
        val_split=None,
        device=device,
        seed=base_seed,
        max_epochs=max(best_combo_epoch, 1),
        patience=None,
    )
    _release_device_memory(device)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(refit_model.state_dict(), checkpoint_path)

    y_score = _predict_scores(refit_model, test_split.X, device)
    y_pred = (y_score >= 0.5).astype(int)

    return FoldResult(
        fold_idx=fold_idx,
        foot=foot,
        model_name=model_name,
        subject_id=test_split.subject_id,
        window_idx=test_split.window_idx,
        y_true=test_split.y,
        y_pred=y_pred,
        y_score=y_score,
        best_params={
            "dropout": dropout,
            "lr": lr,
            "best_epoch": best_combo_epoch,
            "val_auc": best_combo_auc,
        },
    )
