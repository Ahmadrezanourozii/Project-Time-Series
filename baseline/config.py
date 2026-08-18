"""Shared paths, constants, and hyperparameter grids for the baseline pipeline."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
SPLITS_DIR = DATA_DIR / "splits"
DEMOGRAPHICS_CSV = DATA_DIR / "demographics_with_bmi.csv"

# Committed metadata mirror of Data/splits, for machines without Data/ (Kaggle)
FOLD_ASSIGNMENTS_JSON = PROJECT_ROOT / "metadata" / "fold_assignments.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"
CACHE_DIR = OUTPUT_DIR / "cache"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Sampling & windowing (100 Hz per Data/format.txt)
FS_HZ = 100.0
SKIP_INIT_SEC = 5.0
WINDOW_SEC = 5.0
STEP_SEC = 1.0
WINDOW_SAMPLES = int(WINDOW_SEC * FS_HZ)  # 500
STEP_SAMPLES = int(STEP_SEC * FS_HZ)  # 250

LEFT_CHANNELS = [f"L{i}" for i in range(1, 9)] + ["TotalL"]  # 9 channels
RIGHT_CHANNELS = [f"R{i}" for i in range(1, 9)] + ["TotalR"]  # 9 channels
FOOT_CONFIGS: dict[str, list[str]] = {
    "left": LEFT_CHANNELS,
    "right": RIGHT_CHANNELS,
    "both": LEFT_CHANNELS + RIGHT_CHANNELS,
}

STAT_NAMES = [
    "mean",
    "std",
    "skew",
    "kurt",
    "rms",
    "zcr",
    "median",
    "min",
    "max",
    "spec_energy",
    "spec_entropy",
]

FOOT_NFEATURES = {foot: len(channels) * len(STAT_NAMES) for foot, channels in FOOT_CONFIGS.items()}

RAW_COLUMNS = (
    ["time"]
    + [f"L{i}" for i in range(1, 9)]
    + [f"R{i}" for i in range(1, 9)]
    + ["TotalL", "TotalR"]
)

FILENAME_RE = re.compile(r"^([A-Za-z]{2}(?:Co|Pt)\d+)_(\d+)\.txt$")

N_FOLDS = 5
RANDOM_STATE = 42
N_BOOTSTRAP = 1000

DEGENERATE_STD_EPS = 1e-8
DEGENERATE_FILL = 0.0

LABEL_TO_INT = {"HC": 0, "PD": 1}
INT_TO_LABEL = {0: "HC", 1: "PD"}
POSITIVE_LABEL_INT = 1  # PD is the positive class

SCORING_METRIC = "roc_auc"  # GridSearchCV scoring
PRIMARY_METRIC = "accuracy"  # tie-break for "best performing model" confusion-matrix selection

CACHE_VERSION = "v3"

PARAM_GRIDS = {
    "random_forest": {
        "selectk__k": [20, 40, "all"],
        "clf__n_estimators": [500],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", 0.3, 0.5, 1.0],
    },
    "svm_rbf": {
        "selectk__k": [20, 40, "all"],
        "clf__C": (2.0 ** np.arange(-5, 16, 2)).tolist(),
        "clf__gamma": (2.0 ** np.arange(-15, 4, 2)).tolist(),
    },
}

MODEL_DISPLAY = {
    "random_forest": "RandomForest",
    "svm_rbf": "SVM_RBF",
    "resnet": "ResNet",
    "gru": "GRU",
    "mamba": "Mamba",
}

COLOR_DARK_BLUE = "#003366"
COLOR_ORANGE = "#FF9933"
