"""Score-level fusion of several Mamba runs' subject predictions.

Each run writes predictions/mamba_{foot}_subjects.csv (one row per subject:
subject_id, y_true, y_pred, y_score). Averaging y_score across runs that use
different token families/window lengths is a standard late-fusion ensemble
and is evaluated with the same metrics/bootstrap as a single run.

Usage:
  python scripts/fuse_predictions.py --out outputs/mamba_fused \
      --foot both run_dir_a run_dir_b run_dir_c
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.folds import FoldResult
from baseline.metrics import bootstrap_metrics, compute_point_metrics, confusion_counts, summarize_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--foot", default="both")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    frames = []
    for run_dir in args.run_dirs:
        path = run_dir / "predictions" / f"mamba_{args.foot}_subjects.csv"
        df = pd.read_csv(path).set_index("subject_id").sort_index()
        frames.append(df)
        print(f"loaded {path} ({len(df)} subjects)")

    base = frames[0]
    for other in frames[1:]:
        if not base.index.equals(other.index) or not (base["y_true"] == other["y_true"]).all():
            raise SystemExit("Runs disagree on subjects/labels; cannot fuse")

    mean_score = np.mean([f["y_score"].to_numpy() for f in frames], axis=0)
    y_true = base["y_true"].to_numpy()
    y_pred = (mean_score >= args.threshold).astype(int)

    point = compute_point_metrics(y_true, y_pred, mean_score)
    boot = summarize_bootstrap(bootstrap_metrics(y_true, y_pred, mean_score, n_boot=args.n_boot))
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fused_results.json").write_text(
        json.dumps(
            {
                "runs": [str(d) for d in args.run_dirs],
                "foot": args.foot,
                "threshold": args.threshold,
                "point": {k: (None if np.isnan(v) else round(float(v), 4)) for k, v in point.items()},
                "bootstrap": boot,
                "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            },
            indent=1,
        )
    )
    pd.DataFrame(
        {"subject_id": base.index, "y_true": y_true, "y_pred": y_pred, "y_score": mean_score}
    ).to_csv(args.out / f"mamba_{args.foot}_subjects_fused.csv", index=False)
    print(
        f"fused {len(frames)} runs -> acc={point['accuracy']:.3f} prec_w={point['precision_w']:.3f} "
        f"rec_w={point['recall_w']:.3f} f1_w={point['f1_w']:.3f} auc={point['auc']:.3f}"
    )


if __name__ == "__main__":
    main()
