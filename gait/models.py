"""The subject-level gait classifier: a soft-voting ensemble of four model
families on the features of gait/features.py.

The four members fail differently (linear vs. axis-aligned splits vs.
randomised splits vs. boosting), so averaging their probabilities lowers the
variance that dominates with 100 training subjects. Hyperparameters are fixed
conservative defaults, not tuned on any test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Bookkeeping columns of gait.features.recording_features, never model inputs:
# body mass, recording length and stride counts say nothing about the disease.
NON_FEATURES = ["bw_proxy_N", "duration_s", "L_n_strides", "R_n_strides"]


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    return features.drop(columns=[c for c in NON_FEATURES if c in features.columns])


def _members(seed: int) -> list:
    return [
        make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=seed),
        ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=seed),
        HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10,
                                       l2_regularization=1.0, random_state=seed),
    ]


class GaitEnsemble(BaseEstimator, ClassifierMixin):
    """Mean PD probability of the four members, each trained with `n_seeds` seeds."""

    def __init__(self, n_seeds: int = 1) -> None:
        self.n_seeds = n_seeds

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GaitEnsemble":
        self.columns_ = list(X.columns)
        self.classes_ = np.array([0, 1])
        self.members_ = [[m.fit(X, y) for m in _members(seed)] for seed in range(self.n_seeds)]
        return self

    def member_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(n_members, n_samples) PD probability, averaged over seeds."""
        X = X[self.columns_]
        per_seed = [[m.predict_proba(X)[:, 1] for m in members] for members in self.members_]
        return np.mean(per_seed, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.member_proba(X).mean(axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
