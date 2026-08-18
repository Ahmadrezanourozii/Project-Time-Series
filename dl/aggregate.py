"""Subject-level aggregation of pooled window-level test predictions.

The pooled FoldResult covers every subject's windows exactly once (disjoint
test folds), so grouping by subject yields one decision per subject (n=100).
The returned FoldResult is metric-compatible with the window-level one: the
existing bootstrap then resamples subjects instead of windows.
"""

from __future__ import annotations

import numpy as np

from baseline.folds import FoldResult


def aggregate_subjects(pooled: FoldResult, rule: str = "mean_prob", thresholds: dict | None = None) -> FoldResult:
    """One row per subject. rule: 'mean_prob' (mean window score vs threshold)
    or 'majority' (majority vote of window predictions; score kept as mean
    probability so AUC stays defined).

    thresholds maps subject_id -> decision threshold (the value calibrated on
    that subject's fold's validation split); missing entries fall back to 0.5.
    """
    subjects = np.unique(pooled.subject_id)
    y_true, y_pred, y_score = [], [], []
    for subject in subjects:
        mask = pooled.subject_id == subject
        true_vals = np.unique(pooled.y_true[mask])
        if len(true_vals) != 1:
            raise ValueError(f"Subject {subject} has inconsistent window labels")
        mean_score = float(np.mean(pooled.y_score[mask]))
        if rule == "mean_prob":
            pred = int(mean_score >= (thresholds or {}).get(subject, 0.5))
        elif rule == "majority":
            pred = int(np.mean(pooled.y_pred[mask]) >= 0.5)
        else:
            raise ValueError(f"Unknown rule: {rule}")
        y_true.append(int(true_vals[0]))
        y_pred.append(pred)
        y_score.append(mean_score)

    return FoldResult(
        fold_idx=-1,
        foot=pooled.foot,
        model_name=f"{pooled.model_name}_subject_{rule}",
        subject_id=subjects,
        window_idx=np.zeros(len(subjects), dtype=int),
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_score=np.asarray(y_score),
        best_params=pooled.best_params,
    )
