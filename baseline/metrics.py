"""Manual confusion-matrix metrics and bootstrapped Mean +/- Std reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import N_BOOTSTRAP, POSITIVE_LABEL_INT, RANDOM_STATE


def confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = POSITIVE_LABEL_INT
) -> tuple[int, int, int, int]:
    """Manual TP/FP/TN/FN counts from boolean masks (PD=1 is positive)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    return tp, fp, tn, fn


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Accuracy, Sensitivity, Specificity (manual from confusion matrix), AUC,
    plus Precision/Recall/F1 both PD-positive ('binary') and class-weighted.

    Weighted recall is mathematically identical to accuracy; both are kept so
    tables can show either convention explicitly.
    """
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan

    # PD-positive (binary) precision/recall/F1 from the same counts
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

    # Class-weighted averages: HC treated as its own positive class, weighted
    # by true-class support (n_pd, n_hc)
    n_pd, n_hc = tp + fn, tn + fp
    precision_hc = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    recall_hc = specificity
    f1_hc = (
        2 * precision_hc * recall_hc / (precision_hc + recall_hc)
        if (precision_hc + recall_hc) > 0
        else np.nan
    )
    precision_w = (n_pd * precision + n_hc * precision_hc) / n
    recall_w = (n_pd * recall + n_hc * recall_hc) / n
    f1_w = (n_pd * f1 + n_hc * f1_hc) / n

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_w": precision_w,
        "recall_w": recall_w,
        "f1_w": f1_w,
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap (resample with replacement) Accuracy/Sensitivity/Specificity/AUC.

    Resamples that lack both classes (metrics undefined) are redrawn rather
    than counted, so exactly n_boot valid rows are always returned.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    rows = []
    attempts = 0
    max_attempts = n_boot * 20
    while len(rows) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.randint(0, n, size=n)
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        rows.append(compute_point_metrics(yt, yp, ys))

    if len(rows) < n_boot:
        raise RuntimeError(f"Only {len(rows)}/{n_boot} valid bootstrap resamples after {attempts} attempts")
    return pd.DataFrame(rows)


def summarize_bootstrap(boot_df: pd.DataFrame) -> dict[str, str]:
    """Format bootstrap results as 'Mean +/- Std' strings.

    Accuracy/Sensitivity/Specificity as percentages rounded to the nearest
    integer (e.g. '77 ± 6'); AUC on the 0-1 scale with 2 decimals.
    """
    out: dict[str, str] = {}
    pct_cols = ["accuracy", "sensitivity", "specificity", "precision", "recall", "precision_w", "recall_w"]
    for col in pct_cols:
        if col not in boot_df:
            continue
        mean_pct = boot_df[col].mean() * 100
        std_pct = boot_df[col].std() * 100
        out[col] = f"{mean_pct:.0f} ± {std_pct:.0f}"
    for col in ["auc", "f1", "f1_w"]:
        if col not in boot_df:
            continue
        out[col] = f"{boot_df[col].mean():.2f} ± {boot_df[col].std():.2f}"
    return out
