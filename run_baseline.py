#!/usr/bin/env python3
"""Baseline classification pipeline: Random Forest & SVM-RBF for PD vs HC gait.

Pipeline
--------
1. Sanity-check PD/HC labels against the filename convention.
2. Build (or load cached) the window-level feature table (72 features/window).
3. For each model x foot-config, run all 5 folds (leakage-safe scaling +
   GridSearchCV on the provided train/validation split) and pool the
   out-of-fold test predictions (disjoint across folds -> full 100-subject
   coverage).
4. Bootstrap (1000x) Accuracy/Sensitivity/Specificity/AUC on the pooled
   predictions, write one LaTeX table per model.
5. Plot the best-performing model's confusion matrix per foot-config.
"""

from __future__ import annotations

import json

import pandas as pd

from baseline.config import (
    FIGURES_DIR,
    FOOT_NFEATURES,
    MODEL_DISPLAY,
    N_BOOTSTRAP,
    N_FOLDS,
    OUTPUT_DIR,
    PREDICTIONS_DIR,
    PRIMARY_METRIC,
    RANDOM_STATE,
    TABLES_DIR,
)
from baseline.features import discover_unique_gait_files, load_or_build_window_features, parse_subject_id, select_foot_columns
from baseline.folds import pool_fold_results, run_fold
from baseline.labels import label_from_subject_id, load_labels
from baseline.metrics import bootstrap_metrics, compute_point_metrics, confusion_counts, summarize_bootstrap
from baseline.reporting import build_results_table, plot_confusion_matrix, save_latex_table


def sanity_check_labels() -> None:
    """Check filename-derived labels against the demographics CSV, restricted to
    the subjects that actually appear in Data/splits (the demographics CSV
    covers a larger cohort than the 100 subjects used in this deliverable)."""
    labels = load_labels()
    used_subjects = sorted({parse_subject_id(p) for p in discover_unique_gait_files()})
    for subject_id in used_subjects:
        expected = label_from_subject_id(subject_id)
        actual = labels.loc[subject_id, "label"]
        if expected != actual:
            raise ValueError(f"Label mismatch for {subject_id}: filename says {expected}, CSV says {actual}")
    print(f"Label sanity check OK: {len(used_subjects)} subjects.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sanity_check_labels()

    print("Building/loading window feature table...")
    window_features = load_or_build_window_features()
    print(f"Window feature table: {window_features.shape[0]} windows, {window_features.shape[1]} columns.")

    model_names = ["random_forest", "svm_rbf"]
    foot_names = ["left", "right", "both"]

    pooled = {}
    all_best_params = {}

    for model_name in model_names:
        for foot in foot_names:
            print(f"\n=== {MODEL_DISPLAY[model_name]} | {foot} ===")
            foot_features = select_foot_columns(window_features, foot)

            fold_results = []
            for fold_idx in range(1, N_FOLDS + 1):
                result = run_fold(fold_idx, foot, model_name, foot_features)
                fold_results.append(result)
                print(f"  Fold {fold_idx}: best_params={result.best_params}")

            pooled_result = pool_fold_results(fold_results)
            pooled[(model_name, foot)] = pooled_result
            all_best_params[f"{model_name}__{foot}"] = pooled_result.best_params

            pd.DataFrame(
                {
                    "subject_id": pooled_result.subject_id,
                    "window_idx": pooled_result.window_idx,
                    "y_true": pooled_result.y_true,
                    "y_pred": pooled_result.y_pred,
                    "y_score": pooled_result.y_score,
                }
            ).to_csv(PREDICTIONS_DIR / f"{model_name}_{foot}_pooled.csv", index=False)

    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(all_best_params, f, indent=2, default=str)

    print("\nComputing bootstrap metrics and building LaTeX tables...")
    metrics_cache = {}
    for model_name in model_names:
        rows = []
        for foot in foot_names:
            pr = pooled[(model_name, foot)]
            point = compute_point_metrics(pr.y_true, pr.y_pred, pr.y_score)
            boot = bootstrap_metrics(pr.y_true, pr.y_pred, pr.y_score, n_boot=N_BOOTSTRAP, random_state=RANDOM_STATE)
            summary = summarize_bootstrap(boot)
            metrics_cache[(model_name, foot)] = {"point": point, "summary": summary, "pooled": pr}
            rows.append(
                {
                    "Data": foot.capitalize(),
                    "Features (#)": FOOT_NFEATURES[foot],
                    "Acc (%)": summary["accuracy"],
                    "Sen (%)": summary["sensitivity"],
                    "Spe (%)": summary["specificity"],
                    "AUC": summary["auc"],
                }
            )
        df = build_results_table(rows)
        print(f"\n{MODEL_DISPLAY[model_name]} results:\n{df.to_string(index=False)}")
        save_latex_table(
            df,
            TABLES_DIR / f"{MODEL_DISPLAY[model_name]}_results.tex",
            caption=f"{MODEL_DISPLAY[model_name]} performance (pooled 5-fold CV, bootstrap mean $\\pm$ std).",
            label=f"tab:{model_name}_results",
        )

    print("\nPlotting confusion matrices for the best-performing model per foot-config...")
    for foot in foot_names:
        best_model = max(model_names, key=lambda m: metrics_cache[(m, foot)]["point"][PRIMARY_METRIC])
        pr = metrics_cache[(best_model, foot)]["pooled"]
        tp, fp, tn, fn = confusion_counts(pr.y_true, pr.y_pred)
        cm = [[tn, fp], [fn, tp]]
        out_path = FIGURES_DIR / f"confusion_{foot}_{best_model}.png"
        plot_confusion_matrix(cm, title=f"{foot.capitalize()} — {MODEL_DISPLAY[best_model]}", out_path=out_path)
        print(f"  {foot}: best={MODEL_DISPLAY[best_model]} -> {out_path}")

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
