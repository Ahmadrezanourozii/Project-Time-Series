"""Significance tests for Table~\\ref{tab:main}.

Two questions, both answered on the 100 pooled test subjects:

1. Is each model better than chance?  Subject-level accuracy is a count out of
   100 independent subjects, so a one-sided exact binomial test against
   $p_0 = 0.5$ applies; the AUC is tested against $0.5$ with Mann-Whitney U.

2. Is Mamba better than a given baseline?  Both models see the same subjects,
   so the comparison is paired:

* accuracy -- exact McNemar test on the subjects the two models classify
  differently (binomial, two-sided);
* ROC-AUC  -- DeLong test for two correlated ROC curves, plus a paired
  bootstrap CI over subjects for the difference.

Writes outputs/mamba/final/significance.json and prints a compact table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASELINES = {
    "svm_rbf": (Path("outputs/baseline/predictions"), 0.0),
    "resnet": (Path("outputs/dl_v1_step2.5s/predictions"), 0.5),
    "lstm": (Path("outputs/dl_v1_step2.5s/predictions"), 0.5),
    "random_forest": (Path("outputs/baseline/predictions"), 0.5),
}
FEET = ["left", "right", "both"]
MAMBA_DIR = Path("outputs/mamba/final/primary_model/predictions")
OUT_PATH = Path("outputs/mamba/final/significance.json")
N_BOOT = 10000
SEED = 0


def subject_table(path: Path, threshold: float) -> pd.DataFrame:
    """One row per subject: true label, mean window score, hard decision."""
    df = pd.read_csv(path)
    g = df.groupby("subject_id").agg(y_true=("y_true", "first"), score=("y_score", "mean"))
    g["y_pred"] = (g["score"].to_numpy() >= threshold).astype(int)
    return g.sort_index()


# --- DeLong's test for two correlated ROC curves -------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong(y_true: np.ndarray, scores: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (AUCs, covariance matrix) via the fast DeLong implementation."""
    pos = np.asarray(scores)[:, y_true == 1]
    neg = np.asarray(scores)[:, y_true == 0]
    m, n, k = pos.shape[1], neg.shape[1], len(scores)

    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.concatenate([p, q])) for p, q in zip(pos, neg)])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n            # structural components, positives
    v10 = 1.0 - (tz[:, m:] - ty) / m      # structural components, negatives
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, cov.reshape(k, k)


def delong_p(y_true: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    aucs, cov = delong(y_true, [a, b])
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0] - aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0] - aucs[1]), float(2 * stats.norm.sf(abs(z)))


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[int, int, float]:
    """Exact (binomial) McNemar test on the discordant subjects."""
    a_right, b_right = pred_a == y_true, pred_b == y_true
    b01 = int(np.sum(a_right & ~b_right))   # only A correct
    b10 = int(np.sum(~a_right & b_right))   # only B correct
    if b01 + b10 == 0:
        return b01, b10, 1.0
    return b01, b10, float(stats.binomtest(b01, b01 + b10, 0.5).pvalue)


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, score))


def paired_bootstrap(y_true, m, b, n_boot=N_BOOT, seed=SEED) -> dict:
    """Percentile CIs for the accuracy and AUC differences, resampling subjects."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    d_acc, d_auc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        d_acc.append(np.mean(m["y_pred"][idx] == yt) - np.mean(b["y_pred"][idx] == yt))
        d_auc.append(auc(yt, m["score"][idx]) - auc(yt, b["score"][idx]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"acc_ci": q(d_acc), "auc_ci": q(d_auc)}


MODELS_ALL = {
    "svm_rbf": (Path("outputs/baseline/predictions"), 0.0),
    "resnet": (Path("outputs/dl_v1_step2.5s/predictions"), 0.5),
    "lstm": (Path("outputs/dl_v1_step2.5s/predictions"), 0.5),
    "random_forest": (Path("outputs/baseline/predictions"), 0.5),
}


def against_chance(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    """Is this row better than guessing?  Exact binomial on the number of
    correctly classified subjects, and Mann-Whitney U on the scores."""
    n_correct = int(np.sum(y_pred == y_true))
    n = len(y_true)
    p_acc = float(stats.binomtest(n_correct, n, 0.5, alternative="greater").pvalue)
    u = stats.mannwhitneyu(score[y_true == 1], score[y_true == 0], alternative="greater")
    return {
        "n_correct": n_correct, "n": n, "accuracy": round(n_correct / n, 4),
        "binomial_p": float(f"{p_acc:.3g}"),
        "auc": round(auc(y_true, score), 4), "mannwhitney_p": float(f"{u.pvalue:.3g}"),
    }


def chance_table() -> dict:
    """Every cell of Table~\\ref{tab:main}: model x foot, tested against chance."""
    out: dict[str, dict] = {}
    print("Against chance (one-sided exact binomial on accuracy, "
          "Mann-Whitney U on scores):")
    print(f"{'model':14s} {'foot':6s} {'correct':>8s} {'acc':>6s} {'p(acc)':>9s} "
          f"{'auc':>6s} {'p(auc)':>9s}")
    for name, (pred_dir, threshold) in MODELS_ALL.items():
        for foot in FEET:
            base = subject_table(pred_dir / f"{name}_{foot}_pooled.csv", threshold)
            r = against_chance(base["y_true"].to_numpy(), base["y_pred"].to_numpy(),
                               base["score"].to_numpy())
            out.setdefault(name, {})[foot] = r
            print(f"{name:14s} {foot:6s} {r['n_correct']:6d}/{r['n']:<3d} {r['accuracy']:6.2f} "
                  f"{r['binomial_p']:9.2g} {r['auc']:6.3f} {r['mannwhitney_p']:9.2g}")
    for foot in FEET:
        m = pd.read_csv(MAMBA_DIR / f"mamba_{foot}_subjects.csv")
        r = against_chance(m["y_true"].to_numpy(), m["y_pred"].to_numpy(), m["y_score"].to_numpy())
        out.setdefault("mamba", {})[foot] = r
        print(f"{'mamba':14s} {foot:6s} {r['n_correct']:6d}/{r['n']:<3d} {r['accuracy']:6.2f} "
              f"{r['binomial_p']:9.2g} {r['auc']:6.3f} {r['mannwhitney_p']:9.2g}")
    print()
    return out


def main() -> None:
    foot = "both"
    mamba = pd.read_csv(MAMBA_DIR / f"mamba_{foot}_subjects.csv").set_index("subject_id").sort_index()
    y_true = mamba["y_true"].to_numpy()
    m = {"y_pred": mamba["y_pred"].to_numpy(), "score": mamba["y_score"].to_numpy()}

    results = {"foot": foot, "n_subjects": int(len(y_true)), "n_boot": N_BOOT,
               "against_chance": chance_table(),
               "mamba": {"accuracy": round(float(np.mean(m["y_pred"] == y_true)), 4),
                         "auc": round(auc(y_true, m["score"]), 4)},
               "vs": {}}

    print(f"Mamba (both feet, {len(y_true)} subjects): "
          f"acc={results['mamba']['accuracy']:.3f} auc={results['mamba']['auc']:.3f}\n")
    print(f"{'baseline':14s} {'dAcc':>7s} {'b01/b10':>9s} {'McNemar p':>10s} "
          f"{'dAUC':>7s} {'DeLong p':>9s} {'dAUC 95% CI':>18s}")

    for name, (pred_dir, threshold) in BASELINES.items():
        base = subject_table(pred_dir / f"{name}_{foot}_pooled.csv", threshold)
        assert list(base.index) == list(mamba.index), f"{name}: subject sets differ"
        b = {"y_pred": base["y_pred"].to_numpy(), "score": base["score"].to_numpy()}

        d_acc = float(np.mean(m["y_pred"] == y_true) - np.mean(b["y_pred"] == y_true))
        b01, b10, p_mc = mcnemar_exact(y_true, m["y_pred"], b["y_pred"])
        d_auc, p_dl = delong_p(y_true, m["score"], b["score"])
        boot = paired_bootstrap(y_true, m, b)

        results["vs"][name] = {
            "baseline_accuracy": round(float(np.mean(b["y_pred"] == y_true)), 4),
            "baseline_auc": round(auc(y_true, b["score"]), 4),
            "delta_accuracy": round(d_acc, 4), "mcnemar": {"b01": b01, "b10": b10, "p": round(p_mc, 5)},
            "delta_auc": round(d_auc, 4), "delong_p": round(p_dl, 5), **boot,
        }
        print(f"{name:14s} {d_acc:+7.3f} {b01:4d}/{b10:<4d} {p_mc:10.4f} "
              f"{d_auc:+7.3f} {p_dl:9.4f}  [{boot['auc_ci'][0]:+.3f}, {boot['auc_ci'][1]:+.3f}]")

    OUT_PATH.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
