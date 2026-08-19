"""Empirical leakage audit: check the invariants instead of trusting the code.

Every check either passes or raises. Run it before submitting results.

  python scripts/audit_leakage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import baseline.folds as folds_mod
from baseline.config import N_FOLDS
from dl.datasets import assemble_fold_splits, fit_normalizer
from dl.windows import select_foot_channels
from run_mamba import load_window_table

NPZ = Path("outputs/dl/cache/stat_windows_30s.npz")
PASS, FAIL = "  PASS", "  FAIL"
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(name)


def audit_folds() -> None:
    print("\n1. Fold construction")
    all_test: list[str] = []
    for fold in range(1, N_FOLDS + 1):
        df = folds_mod.load_fold_assignment(fold)
        counts = df["split"].value_counts().to_dict()
        check(f"fold {fold}: 60/20/20 subjects",
              counts == {"train": 60, "validation": 20, "test": 20}, str(counts))
        check(f"fold {fold}: no subject in two splits", df["subject_id"].is_unique)
        all_test += df.loc[df["split"] == "test", "subject_id"].tolist()
    check("test sets disjoint across folds", len(all_test) == len(set(all_test)),
          f"{len(all_test)} rows, {len(set(all_test))} unique")
    check("pooled test covers 100 subjects once", len(set(all_test)) == 100)


def audit_split_isolation(table) -> None:
    print("\n2. Split isolation at window level")
    for fold in (1, 3, 5):
        splits = assemble_fold_splits(fold, "both", table)
        tr, va, te = (set(splits[k].subject_id) for k in ("train", "validation", "test"))
        check(f"fold {fold}: train ∩ test empty", not (tr & te))
        check(f"fold {fold}: validation ∩ test empty", not (va & te))
        check(f"fold {fold}: train ∩ validation empty", not (tr & va))
        n = sum(len(splits[k].y) for k in ("train", "validation", "test"))
        check(f"fold {fold}: every window assigned exactly once", n == len(table.subject_id),
              f"{n} vs {len(table.subject_id)}")


def audit_normalisation(table) -> None:
    print("\n3. Normalisation fitted on training data only")
    fold = 1
    splits = assemble_fold_splits(fold, "both", table)
    foot_data = select_foot_channels(table, "both")
    assignment = folds_mod.load_fold_assignment(fold)
    train_subjects = set(assignment.loc[assignment["split"] == "train", "subject_id"])
    train_mask = np.array([s in train_subjects for s in table.subject_id])

    mean_tr, std_tr = fit_normalizer(foot_data[train_mask])
    mean_all, std_all = fit_normalizer(foot_data)

    # Normalised training data must be centred; a normaliser fitted on everything would not be identical
    check("train split has ~zero mean after normalisation",
          abs(float(splits["train"].X.mean())) < 1e-3, f"{float(splits['train'].X.mean()):.2e}")
    check("train-only and all-data statistics differ (so all-data was NOT used)",
          not np.allclose(mean_tr, mean_all, atol=1e-6) or not np.allclose(std_tr, std_all, atol=1e-6))

    # Reproduce the test split by hand with train-only stats -> must match what the pipeline produced
    test_subjects = set(assignment.loc[assignment["split"] == "test", "subject_id"])
    test_mask = np.array([s in test_subjects for s in table.subject_id])
    manual = ((foot_data[test_mask] - mean_tr) / std_tr).astype(np.float32)
    check("test split normalised with TRAIN statistics",
          np.allclose(manual, splits["test"].X, atol=1e-5),
          f"max diff {np.abs(manual - splits['test'].X).max():.2e}")

    manual_wrong = ((foot_data[test_mask] - mean_all) / std_all).astype(np.float32)
    check("test split NOT normalised with all-data statistics",
          not np.allclose(manual_wrong, splits["test"].X, atol=1e-5))


def audit_feature_independence() -> None:
    """Features of one recording must not depend on any other recording."""
    print("\n4. Feature extraction independent per recording")
    from baseline.features import ALL_CHANNELS, drop_gait_init, load_raw_recording, parse_subject_id
    from baseline.config import SKIP_INIT_SEC, SPLITS_DIR
    from scripts.build_feature_sequences import per_second_tokens

    path = sorted(SPLITS_DIR.glob("Fold_1/test/*.txt"))[0]
    df = drop_gait_init(load_raw_recording(path), skip_sec=SKIP_INIT_SEC)
    tokens_alone = per_second_tokens(df[ALL_CHANNELS].to_numpy(dtype=np.float32))

    table = load_window_table(NPZ)
    sid = parse_subject_id(path)
    cached = table.data[table.subject_id == sid]
    rebuilt = np.stack([tokens_alone[s : s + 30] for s in range(0, len(tokens_alone) - 29, 10)])
    rebuilt = np.transpose(rebuilt, (0, 2, 1))
    check(f"{sid}: tokens computed in isolation match the batch-built cache",
          cached.shape == rebuilt.shape and np.allclose(cached, rebuilt, atol=1e-4),
          f"{cached.shape} vs {rebuilt.shape}")


def audit_threshold_source() -> None:
    """The calibrated threshold must come from validation subjects only."""
    print("\n5. Decision threshold provenance")
    src = Path("run_mamba.py").read_text()
    block = src.split("if args.calibrate_threshold")[1].split("del best_val_model")[0]
    check("threshold block reads the validation split", 'splits["validation"]' in block)
    check("threshold block never reads the test split", 'splits["test"]' not in block)
    check("threshold uses the grid-search model (trained on train only)", "best_val_model" in block)


def audit_inference_path() -> None:
    """predict.py must reuse stored statistics, never fit new ones."""
    print("\n6. Inference path")
    src = Path("predict.py").read_text()
    check("predict.py applies stored normaliser", "apply_normalizer" in src)
    check("predict.py never fits a normaliser", "fit_normalizer" not in src)
    check("predict.py reads mean/std from the bundle", 'bundle["mean"]' in src and 'bundle["std"]' in src)


def audit_model_selection() -> None:
    """Configurations were compared on pooled test accuracy; check whether the
    reported one also wins on validation, which is the leakage-free criterion."""
    print("\n7. Model selection criterion (selection bias)")
    rows = []
    for f in sorted(Path("outputs/mamba/experiment_archive").glob("*/**/results.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        args = r.get("args", {})
        res = r.get("results", {}).get("both")
        if not res or "subject_mean_prob" not in res or args.get("split_mode") == "window":
            continue
        if args.get("smoke") or res["subject_mean_prob"].get("n", 0) < 100:
            continue  # smoke runs use a handful of subjects; not comparable
        bp = res.get("best_params_per_fold", {})
        aucs = []
        for v in bp.values():
            if isinstance(v, dict):
                if "val_auc" in v:
                    aucs.append(v["val_auc"])
                else:
                    aucs += [w["val_auc"] for w in v.values() if isinstance(w, dict) and "val_auc" in w]
        if not aucs:
            continue
        rows.append({
            "run": f.parts[2] + "/" + f.parts[3],
            "npz": Path(str(r["args"]["npz"])).stem,
            "size": r["args"]["model_size"],
            "val_auc": float(np.mean(aucs)),
            "test_acc": res["subject_mean_prob"]["point"]["accuracy"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  (no archived runs to compare)")
        return
    by_val = df.sort_values("val_auc", ascending=False).head(5)
    by_test = df.sort_values("test_acc", ascending=False).head(5)
    print("  top 5 by VALIDATION AUC (leakage-free criterion):")
    for _, r in by_val.iterrows():
        print(f"    {r['npz'][:26]:26s} {r['size']:6s} val={r['val_auc']:.3f} test={r['test_acc']:.3f}")
    print("  top 5 by TEST accuracy (what was actually used to pick):")
    for _, r in by_test.iterrows():
        print(f"    {r['npz'][:26]:26s} {r['size']:6s} val={r['val_auc']:.3f} test={r['test_acc']:.3f}")

    val_winner = by_val.iloc[0]
    test_winner = by_test.iloc[0]
    check("token family agrees between the two criteria",
          val_winner["npz"].startswith("stat_windows") and test_winner["npz"].startswith("stat_windows"),
          f"validation picks {val_winner['npz']}, test picks {test_winner['npz']}")
    gap = test_winner["test_acc"] - val_winner["test_acc"]
    print(f"  selection bias: choosing on test gives {test_winner['test_acc']:.3f}, "
          f"choosing on validation gives {val_winner['test_acc']:.3f} → optimism {gap:+.3f}")
    check("selection bias is disclosed in the report",
          "selection" in Path("report/Sections/03_Experiments_and_Results.tex").read_text().lower())
    print(f"  Spearman corr(validation AUC, test accuracy) = "
          f"{df['val_auc'].corr(df['test_acc'], method='spearman'):.2f} over {len(df)} configurations")


def main() -> None:
    table = load_window_table(NPZ)
    print(f"Auditing with {NPZ.name}: {table.data.shape}, {len(set(table.subject_id))} subjects")
    audit_folds()
    audit_split_isolation(table)
    audit_normalisation(table)
    audit_feature_independence()
    audit_threshold_source()
    audit_inference_path()
    audit_model_selection()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("All leakage checks passed.")


if __name__ == "__main__":
    main()
