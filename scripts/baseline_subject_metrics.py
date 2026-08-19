"""Recompute every Deliverable-2 baseline at subject level, and add the leaky
window-level protocol, so all models are compared like-for-like with Mamba.

The pooled prediction CSVs from Deliverable 2 already cover all 100 subjects
exactly once (disjoint test folds), so subject-level metrics follow by
averaging each subject's window scores -- the same rule used for Mamba.

The leaky reference protocol is re-fitted here from the cached feature matrix
using random window splits (subjects shared between train and test).

Writes outputs/mamba/final/baseline_comparison.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.metrics import bootstrap_metrics, compute_point_metrics, confusion_counts, summarize_bootstrap

PRED_DIRS = {
    "random_forest": Path("outputs/baseline/predictions"),
    "svm_rbf": Path("outputs/baseline/predictions"),
    "resnet": Path("outputs/dl_v1_step2.5s/predictions"),
    "lstm": Path("outputs/dl_v1_step2.5s/predictions"),
}
FEET = ["left", "right", "both"]
OUT_PATH = Path("outputs/mamba/final/baseline_comparison.json")


def metrics(y_true, y_pred, y_score, n_boot=1000) -> dict:
    point = compute_point_metrics(y_true, y_pred, y_score)
    boot = summarize_bootstrap(bootstrap_metrics(y_true, y_pred, y_score, n_boot=n_boot))
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    return {
        "point": {k: (None if np.isnan(v) else round(float(v), 4)) for k, v in point.items()},
        "bootstrap": boot,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n": int(len(y_true)),
    }


def subject_level(df: pd.DataFrame, decision_col: str) -> dict:
    """Aggregate window predictions to one decision per subject (mean score).

    SVM scores are signed decision-function values, so the positive-class
    cut is 0 rather than 0.5; `decision_col` carries that threshold.
    """
    threshold = 0.0 if decision_col == "decision_function" else 0.5
    grouped = df.groupby("subject_id").agg(y_true=("y_true", "first"), score=("y_score", "mean"))
    y_pred = (grouped["score"].to_numpy() >= threshold).astype(int)
    return metrics(grouped["y_true"].to_numpy(), y_pred, grouped["score"].to_numpy())


def leaky_window_baselines() -> dict:
    """RF and SVM-RBF under random window splits (the inflated protocol)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC

    from baseline.config import FOOT_CONFIGS

    wf = pd.read_csv("outputs/baseline/cache/window_features_v3.csv")
    y = (wf["label"] == "PD").astype(int).to_numpy()
    out: dict[str, dict] = {}

    for foot in FEET:
        channels = set(FOOT_CONFIGS[foot])
        cols = [c for c in wf.columns if c.split("__")[0] in channels]
        X = wf[cols].to_numpy(dtype=float)

        for name in ("random_forest", "svm_rbf"):
            preds = np.zeros(len(y), dtype=int)
            scores = np.zeros(len(y), dtype=float)
            for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
                scaler = RobustScaler().fit(X[train_idx])
                Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
                if name == "random_forest":
                    clf = RandomForestClassifier(
                        n_estimators=500, max_depth=5, min_samples_leaf=5, max_features="sqrt",
                        random_state=42, n_jobs=-1, class_weight="balanced",
                    ).fit(Xtr, y[train_idx])
                    scores[test_idx] = clf.predict_proba(Xte)[:, 1]
                    preds[test_idx] = (scores[test_idx] >= 0.5).astype(int)
                else:
                    clf = SVC(kernel="rbf", C=0.1, gamma=0.001, class_weight="balanced", random_state=42).fit(
                        Xtr, y[train_idx]
                    )
                    scores[test_idx] = clf.decision_function(Xte)
                    preds[test_idx] = (scores[test_idx] >= 0).astype(int)
            out.setdefault(name, {})[foot] = metrics(y, preds, scores)
            print(f"  leaky {name:14s} {foot:6s} acc={out[name][foot]['point']['accuracy']:.3f}")
    return out


def main() -> None:
    results: dict = {"subject_wise": {}, "leaky_window": {}}

    print("Subject-wise protocol (headline):")
    for model, pred_dir in PRED_DIRS.items():
        for foot in FEET:
            path = pred_dir / f"{model}_{foot}_pooled.csv"
            if not path.exists():
                print(f"  MISSING {path}")
                continue
            df = pd.read_csv(path)
            decision_col = "decision_function" if model == "svm_rbf" else "proba"
            results["subject_wise"].setdefault(model, {})[foot] = {
                "window": metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df["y_score"].to_numpy()),
                "subject": subject_level(df, decision_col),
            }
            sub = results["subject_wise"][model][foot]["subject"]["point"]
            win = results["subject_wise"][model][foot]["window"]["point"]
            print(
                f"  {model:14s} {foot:6s} window acc={win['accuracy']:.3f} auc={win['auc']:.3f} | "
                f"subject acc={sub['accuracy']:.3f} f1_w={sub['f1_w']:.3f} auc={sub['auc']:.3f}"
            )

    print("\nLeaky window protocol (reference only):")
    results["leaky_window"] = leaky_window_baselines()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
