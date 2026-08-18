#!/usr/bin/env python3
"""Deep-learning classification pipeline: 1D-ResNet & GRU for PD vs HC gait,
trained directly on raw VGRF signal windows (not the 72 hand-crafted stats).

Mirrors run_baseline.py's structure exactly so results are apples-to-apples
comparable to the classical RandomForest/SVM-RBF pipeline:
1. Build (or load cached) the raw per-timestep window table (500 timesteps x
   18 channels/window).
2. For each model x foot-config, run all 5 folds: per fold, grid-search
   (dropout x lr) with validation-AUC early stopping, refit the winning
   config on train+validation, predict on test. Pool the disjoint test-fold
   predictions across folds (full 100-subject coverage).
3. Bootstrap (1000x) Accuracy/Sensitivity/Specificity/AUC on the pooled
   predictions, write one LaTeX table per model.
4. Plot the best-performing DL model's confusion matrix per foot-config.
"""

from __future__ import annotations

import json

import pandas as pd

from baseline.config import MODEL_DISPLAY, N_BOOTSTRAP, PRIMARY_METRIC, RANDOM_STATE
from baseline.folds import FoldResult, pool_fold_results
from baseline.metrics import bootstrap_metrics, compute_point_metrics, confusion_counts, summarize_bootstrap
from baseline.reporting import build_results_table, plot_confusion_matrix, save_latex_table
from dl.config import CACHE_DIR, FIGURES_DIR, FOOT_NCHANNELS, OUTPUT_DIR, PREDICTIONS_DIR, TABLES_DIR
from dl.train import run_fold
from dl.windows import load_or_build_raw_windows


def _load_pooled_from_csv(model_name: str, foot: str) -> FoldResult:
    """Reconstruct a pooled FoldResult from a previously written predictions CSV.

    Used to resume a sweep that was interrupted after some (model, foot)
    combos already completed and wrote their pooled predictions -- avoids
    redoing already-finished fold-cycles. Per-fold best_params aren't
    preserved across a resume (only known in-memory during the original
    run), so best_params carries a placeholder note instead.
    """
    df = pd.read_csv(PREDICTIONS_DIR / f"{model_name}_{foot}_pooled.csv")
    return FoldResult(
        fold_idx=-1,
        foot=foot,
        model_name=model_name,
        subject_id=df["subject_id"].to_numpy(),
        window_idx=df["window_idx"].to_numpy(),
        y_true=df["y_true"].to_numpy(),
        y_pred=df["y_pred"].to_numpy(),
        y_score=df["y_score"].to_numpy(),
        best_params={"note": "resumed from cached predictions CSV; per-fold params not preserved"},
    )


def main() -> None:
    for directory in (OUTPUT_DIR, CACHE_DIR, PREDICTIONS_DIR, TABLES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print("Building/loading raw window table...")
    table = load_or_build_raw_windows()
    print(f"Raw window table: {table.data.shape[0]} windows, {table.data.shape[1]} channels, {table.data.shape[2]} timesteps.")

    model_names = ["gru"]
    foot_names = ["left", "right", "both"]

    pooled = {}
    all_best_params = {}

    for model_name in model_names:
        for foot in foot_names:
            print(f"\n=== {MODEL_DISPLAY[model_name]} | {foot} ===")

            cache_csv = PREDICTIONS_DIR / f"{model_name}_{foot}_pooled.csv"
            if cache_csv.exists():
                print(f"  Found existing {cache_csv.name}, skipping (resuming an interrupted sweep).")
                pooled_result = _load_pooled_from_csv(model_name, foot)
                pooled[(model_name, foot)] = pooled_result
                all_best_params[f"{model_name}__{foot}"] = pooled_result.best_params
                continue

            fold_results = []
            for fold_idx in range(1, 6):
                result = run_fold(fold_idx, foot, model_name, table)
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
            ).to_csv(cache_csv, index=False)

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
                    "Features (#)": FOOT_NCHANNELS[foot],
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

    print("\nPlotting confusion matrices for the best-performing DL model per foot-config...")
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
