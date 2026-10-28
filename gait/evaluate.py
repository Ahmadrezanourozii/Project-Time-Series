"""Subject-wise cross-validation for subject-level (one row per subject) models.

Two protocols, both strictly subject-disjoint:
  predefined  the course's five folds; train+validation fit, test predicted,
              pooled over the five disjoint test sets (comparable with Mamba)
  repeated    stratified 5-fold CV repeated with different shuffles; averages
              out the luck of one particular split, which with 100 subjects is
              as large as most model differences
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOLD_JSON = PROJECT_ROOT / "metadata" / "fold_assignments.json"
LABELS_CSV = PROJECT_ROOT / "metadata" / "labels.csv"


def load_labels(subjects: list[str]) -> np.ndarray:
    labels = pd.read_csv(LABELS_CSV).set_index("subject_id")["label"]
    return (labels.loc[subjects] == "PD").astype(int).to_numpy()


def predefined_oof(make_model: Callable, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Out-of-fold PD scores over the five course folds (train+validation -> test)."""
    folds = json.loads(FOLD_JSON.read_text())
    index = {s: i for i, s in enumerate(X.index)}
    oof = np.full(len(y), np.nan)
    for fold in folds.values():
        fit = [index[s] for s in fold["train"] + fold["validation"]]
        test = [index[s] for s in fold["test"]]
        model = make_model().fit(X.iloc[fit], y[fit])
        oof[test] = model.predict_proba(X.iloc[test])[:, 1]
    assert not np.isnan(oof).any()
    return oof


def repeated_oof(make_model: Callable, X: pd.DataFrame, y: np.ndarray, repeats: int = 10,
                 seed: int = 0) -> np.ndarray:
    """(repeats, n_subjects) out-of-fold scores, one row per reshuffled 5-fold CV."""
    out = np.empty((repeats, len(y)))
    for r in range(repeats):
        for fit, test in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            model = make_model().fit(X.iloc[fit], y[fit])
            out[r, test] = model.predict_proba(X.iloc[test])[:, 1]
    return out


def summarize(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy/AUC of one score vector, or mean +/- sd over repeated rows."""
    scores = np.atleast_2d(scores)
    acc = np.array([np.mean((s >= threshold).astype(int) == y) for s in scores])
    auc = np.array([roc_auc_score(y, s) for s in scores])
    bacc = np.array([0.5 * (np.mean(s[y == 1] >= threshold) + np.mean(s[y == 0] < threshold)) for s in scores])
    return {
        "acc": float(acc.mean()), "acc_sd": float(acc.std()),
        "bacc": float(bacc.mean()), "auc": float(auc.mean()), "auc_sd": float(auc.std()),
    }
