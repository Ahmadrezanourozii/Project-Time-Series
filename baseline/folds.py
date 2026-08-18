"""Fold-aware subject split assignment, leakage-safe scaling, and GridSearchCV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from .config import FOLD_ASSIGNMENTS_JSON, LABEL_TO_INT, PARAM_GRIDS, RANDOM_STATE, SCORING_METRIC, SPLITS_DIR
from .features import parse_subject_id


def load_fold_assignment(fold_idx: int) -> pd.DataFrame:
    """Subject -> split ('train'/'validation'/'test') for one fold.

    Reads Data/splits/Fold_{fold_idx}/{training,validation,test}/*.txt
    filenames, or -- on machines without Data/ (Kaggle) -- the committed
    metadata/fold_assignments.json mirror of the same structure. Asserts
    exactly 60/20/20 subjects and that no subject appears in more than one
    split within this fold.
    """
    rows = []
    if SPLITS_DIR.exists():
        fold_dir = SPLITS_DIR / f"Fold_{fold_idx}"
        for split_name, folder in [("train", "training"), ("validation", "validation"), ("test", "test")]:
            for path in sorted((fold_dir / folder).glob("*.txt")):
                rows.append((parse_subject_id(path), split_name))
    else:
        import json

        assignments = json.loads(FOLD_ASSIGNMENTS_JSON.read_text())[str(fold_idx)]
        for split_name in ("train", "validation", "test"):
            rows.extend((subject_id, split_name) for subject_id in assignments[split_name])

    df = pd.DataFrame(rows, columns=["subject_id", "split"])
    if not df["subject_id"].is_unique:
        raise ValueError(f"Fold {fold_idx}: a subject appears in more than one split")

    counts = df["split"].value_counts()
    expected = {"train": 60, "validation": 20, "test": 20}
    for split_name, expected_n in expected.items():
        if counts.get(split_name, 0) != expected_n:
            raise ValueError(
                f"Fold {fold_idx}: expected {expected_n} '{split_name}' subjects, got {counts.get(split_name, 0)}"
            )
    return df


@dataclass
class FoldResult:
    fold_idx: int
    foot: str
    model_name: str
    subject_id: np.ndarray
    window_idx: np.ndarray
    y_true: np.ndarray  # 0/1, 1 = PD
    y_pred: np.ndarray  # 0/1
    y_score: np.ndarray  # continuous, higher = more PD-like
    best_params: dict


def _build_estimator(model_name: str) -> Pipeline:
    """ANOVA feature selection (k tuned via PARAM_GRIDS' 'selectk__k') feeding
    into the classifier -- k is chosen jointly with the classifier's own
    hyperparameters by the same GridSearchCV/PredefinedSplit in run_fold().
    """
    if model_name == "random_forest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
    elif model_name == "svm_rbf":
        clf = SVC(kernel="rbf", random_state=RANDOM_STATE, class_weight="balanced")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return Pipeline([("selectk", SelectKBest(f_classif)), ("clf", clf)])


def _split_block(foot_features: pd.DataFrame, feat_cols: list[str], split_name: str):
    block = foot_features.loc[foot_features["split"] == split_name]
    X = block[feat_cols].to_numpy(dtype=float)
    y = block["label"].map(LABEL_TO_INT).to_numpy()
    subject_id = block["subject_id"].to_numpy()
    window_idx = block["window_idx"].to_numpy()
    return X, y, subject_id, window_idx


def run_fold(
    fold_idx: int,
    foot: str,
    model_name: str,
    foot_features: pd.DataFrame,
    scoring: str = SCORING_METRIC,
) -> FoldResult:
    """Run one fold for one foot-config and one model.

    foot_features: output of features.select_foot_columns(window_features, foot)
    (columns: subject_id, window_idx, label, <feature cols>).

    Steps: merge fold's subject->split assignment onto the window rows, fit
    RobustScaler on the TRAIN split only, transform validation/test with it,
    run GridSearchCV (over a Pipeline[SelectKBest(f_classif) -> classifier])
    using a PredefinedSplit built from train+validation (so hyperparameter
    selection -- including the ANOVA k -- uses the actual provided validation
    subjects), refit the best pipeline on train+validation, then predict on
    test.
    """
    feat_cols = [c for c in foot_features.columns if c not in ("subject_id", "window_idx", "label")]

    assignment = load_fold_assignment(fold_idx)
    merged = foot_features.merge(assignment, on="subject_id", how="inner")

    X_train, y_train, _, _ = _split_block(merged, feat_cols, "train")
    X_val, y_val, _, _ = _split_block(merged, feat_cols, "validation")
    X_test, y_test, subj_test, win_test = _split_block(merged, feat_cols, "test")

    scaler = RobustScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_trainval = np.vstack([X_train_s, X_val_s])
    y_trainval = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([np.full(len(X_train_s), -1), np.zeros(len(X_val_s), dtype=int)])
    ps = PredefinedSplit(test_fold)

    grid = GridSearchCV(
        _build_estimator(model_name),
        PARAM_GRIDS[model_name],
        scoring=scoring,
        cv=ps,
        refit=True,
        n_jobs=-1,
    )
    grid.fit(X_trainval, y_trainval)

    best = grid.best_estimator_
    y_pred = best.predict(X_test_s)
    if model_name == "svm_rbf":
        y_score = best.decision_function(X_test_s)
    else:
        y_score = best.predict_proba(X_test_s)[:, 1]

    return FoldResult(
        fold_idx=fold_idx,
        foot=foot,
        model_name=model_name,
        subject_id=subj_test,
        window_idx=win_test,
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        best_params=grid.best_params_,
    )


def pool_fold_results(fold_results: list[FoldResult]) -> FoldResult:
    """Concatenate several folds' test predictions for the same (model, foot).

    Fold test sets are disjoint, so this covers every subject's windows exactly
    once. fold_idx on the returned FoldResult is -1 to indicate 'pooled'.
    """
    if not fold_results:
        raise ValueError("fold_results is empty")
    foot = fold_results[0].foot
    model_name = fold_results[0].model_name

    return FoldResult(
        fold_idx=-1,
        foot=foot,
        model_name=model_name,
        subject_id=np.concatenate([f.subject_id for f in fold_results]),
        window_idx=np.concatenate([f.window_idx for f in fold_results]),
        y_true=np.concatenate([f.y_true for f in fold_results]),
        y_pred=np.concatenate([f.y_pred for f in fold_results]),
        y_score=np.concatenate([f.y_score for f in fold_results]),
        best_params={f.fold_idx: f.best_params for f in fold_results},
    )
