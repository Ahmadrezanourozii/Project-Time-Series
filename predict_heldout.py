"""Final held-out evaluation: train the gait ensemble on all 100 labelled
subjects and predict PD (1) / control (0) for every anonymised recording.

Why this model and not the Mamba bundle: under the same subject-wise folds the
gait-feature ensemble reaches 0.89 accuracy / 0.95 AUC against Mamba's 0.77 /
0.85 (paired bootstrap of the accuracy difference: +0.12, 95 % CI 0.03-0.21),
and stays at 0.85 accuracy when test recordings are cut to 35 s. Details in
CLAUDE.md section 14.

Deployment choices:
  * fit on all 100 subjects -- cross-validation chose the recipe; the deployed
    model should then use every labelled subject, not 60 or 80 of them
  * each recording is processed on its own; nothing is estimated from the
    held-out cohort, so a single new patient gets the same answer as in a batch
  * threshold 0.5 on the averaged probability: the training set is balanced and
    cross-validated sensitivity/specificity at 0.5 are 0.85/0.89

Usage:
  python predict_heldout.py --input path/to/HeldOut_Eval_Anonymized \
      --template path/to/prediction_template.xlsx --out outputs/heldout/predictions.xlsx
  python predict_heldout.py ... --cv     # also re-run the cross-validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gait.evaluate import load_labels, predefined_oof, repeated_oof, summarize
from gait.features import extract_directory
from gait.models import GaitEnsemble, feature_matrix

OUT_DIR = Path("outputs/heldout")


def training_recordings(train_dir: Path | None) -> list[Path]:
    """One _01 recording per labelled subject."""
    if train_dir is not None:
        paths = sorted(train_dir.glob("*_01.txt"))
    else:
        from baseline.features import discover_unique_gait_files

        paths = discover_unique_gait_files()
    if len(paths) != 100:
        raise SystemExit(f"expected 100 training recordings, found {len(paths)}")
    return paths


def cached_features(paths: list[Path], cache: Path) -> pd.DataFrame:
    if cache.exists():
        features = pd.read_csv(cache, index_col=0)
        if sorted(features.index) == sorted(p.name for p in paths):
            return features
    features = extract_directory(paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(cache)
    return features


def write_submission(template: Path, predictions: pd.Series, out: Path) -> None:
    """Fill the template's Prediction column in place; AnonID column and row order untouched."""
    from openpyxl import load_workbook

    wb = load_workbook(template)
    ws = wb.active
    header = [c.value for c in ws[1]]
    id_col, pred_col = header.index("AnonID") + 1, header.index("Prediction") + 1
    ids = [ws.cell(row=r, column=id_col).value for r in range(2, ws.max_row + 1)]
    ids = [i for i in ids if i is not None]
    missing = set(ids) ^ set(predictions.index)
    if missing:
        raise SystemExit(f"template and recordings disagree on: {sorted(missing)}")
    for r, anon_id in enumerate(ids, start=2):
        ws.cell(row=r, column=pred_col).value = int(predictions[anon_id])
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)

    check = pd.read_excel(out)
    assert list(check.columns) == ["AnonID", "Prediction"], check.columns
    assert len(check) == len(ids) and check["AnonID"].tolist() == ids
    assert check["Prediction"].isin([0, 1]).all() and check["Prediction"].notna().all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Directory of Eval_XXX.txt recordings")
    parser.add_argument("--template", type=Path, required=True, help="prediction_template.xlsx")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "predictions.xlsx")
    parser.add_argument("--train-dir", type=Path, default=None,
                        help="Flat directory of <subject>_01.txt (default: Data/splits)")
    parser.add_argument("--seeds", type=int, default=10, help="Seeds per tree member in the final fit")
    parser.add_argument("--cv", action="store_true", help="Report subject-wise CV before predicting")
    args = parser.parse_args()

    train_paths = training_recordings(args.train_dir)
    eval_paths = sorted(args.input.glob("*.txt"))
    if not eval_paths:
        raise SystemExit(f"no .txt recordings in {args.input}")

    train = cached_features(train_paths, OUT_DIR / "gait_features_train.csv")
    train.index = [name.split("_")[0] for name in train.index]
    X, y = feature_matrix(train), load_labels(list(train.index))
    held = cached_features(eval_paths, OUT_DIR / "gait_features_heldout.csv")
    held.index = [Path(name).stem for name in held.index]
    print(f"train: {X.shape[0]} subjects ({y.sum()} PD) x {X.shape[1]} features | held-out: {len(held)} recordings")

    report: dict = {}
    if args.cv:
        report["predefined_folds"] = summarize(y, predefined_oof(GaitEnsemble, X, y))
        report["repeated_5fold_x10"] = summarize(y, repeated_oof(GaitEnsemble, X, y, repeats=10))
        for name, r in report.items():
            print(f"  CV {name}: acc={r['acc']:.3f} auc={r['auc']:.3f} bacc={r['bacc']:.3f}")

    model = GaitEnsemble(n_seeds=args.seeds).fit(X, y)
    members = model.member_proba(held[X.columns])
    score = members.mean(axis=0)
    result = pd.DataFrame(
        {"score_pd": score.round(4), "Prediction": (score >= 0.5).astype(int),
         **{f"p_{n}": m.round(4) for n, m in zip(["lr", "rf", "et", "hgb"], members)},
         "duration_s": held["duration_s"].round(1)},
        index=held.index,
    )
    result.index.name = "AnonID"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_DIR / "heldout_scores.csv")
    write_submission(args.template, result["Prediction"], args.out)
    result[["Prediction"]].to_csv(args.out.with_suffix(".csv"))

    report["heldout"] = {
        "n": int(len(result)), "n_pd": int(result["Prediction"].sum()),
        "members_agree": float(np.mean(((members >= 0.5) == (score >= 0.5)).all(axis=0))),
        "borderline_0.4_0.6": int(np.sum((score > 0.4) & (score < 0.6))),
    }
    (OUT_DIR / "heldout_report.json").write_text(json.dumps(report, indent=1))
    print(result.to_string())
    print(f"\n{report['heldout']['n_pd']} PD / {len(result) - report['heldout']['n_pd']} control -> {args.out}")


if __name__ == "__main__":
    main()
