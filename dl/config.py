"""DL-only constants: grid values, training/early-stopping schedule, device, paths.

Shared constants (windowing, folds, labels, bootstrap) are imported from
baseline.config rather than redefined here, so the DL pipeline can never
drift out of sync with the classical pipeline's protocol.
"""

from __future__ import annotations

import torch

from baseline.config import (  # noqa: F401
    FOOT_CONFIGS,
    FOOT_NFEATURES,
    LABEL_TO_INT,
    MODEL_DISPLAY,
    N_BOOTSTRAP,
    N_FOLDS,
    POSITIVE_LABEL_INT,
    PRIMARY_METRIC,
    PROJECT_ROOT,
    RANDOM_STATE,
    SCORING_METRIC,
    SKIP_INIT_SEC,
    SPLITS_DIR,
    STEP_SAMPLES,
    WINDOW_SAMPLES,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dl"
CACHE_DIR = OUTPUT_DIR / "cache"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

CACHE_VERSION = "v2"

FOOT_NCHANNELS = {"left": 9, "right": 9, "both": 18}

# Hyperparameter grid: dropout values are each model's own class default plus
# one higher-regularization alternative; lr grid shared across models.
# GRU's grid is cut from 4 combos (dropout x lr) to 2 -- dropout fixed at
# its class default (0.2), only lr is searched -- to keep runtime manageable
# on CPU across all 15 fold-cycles. ResNet keeps the full 2x2 grid (it runs
# on MPS and already completed a full sweep well within budget).
LR_GRID = [1e-3, 3e-4]
DROPOUT_GRID = {
    "resnet": [0.1, 0.3],
    "gru": [0.2],
    "mamba": [0.1, 0.3],
}

BATCH_SIZE = 32
WEIGHT_DECAY = 1e-3
EARLY_STOP_PATIENCE = 3
NUM_WORKERS = 0

# Per-model epoch cap: the recurrent model converges within a handful of
# epochs even under a much higher cap, so 15 is a safe ceiling that bounds
# worst-case per-combo cost without changing early-stopping behavior for
# runs that genuinely need more epochs. ResNet keeps its original cap
# (already completed a full sweep within budget at this setting).
MAX_EPOCHS_BY_MODEL = {"resnet": 30, "gru": 15, "mamba": 30}


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = resolve_device()

# The bidirectional 2-layer LSTM (hidden_size=256) triggered MPS backend
# out-of-memory errors on this machine even in a fresh process. GRU (fewer
# gates: 3 vs 4) does NOT OOM on MPS, but a direct one-epoch timing pilot on
# the "both" foot-config (largest, 18 channels) showed CPU is still faster
# (122.3s/epoch) than MPS (198.9s/epoch) -- confirms the earlier LSTM finding
# that this recurrent architecture's sequential per-timestep recurrence
# doesn't benefit from MPS; dispatch overhead outweighs any parallelism gain.
# Conv nets (ResNet) have no such issue and benefit from MPS acceleration, so
# device selection is per-model rather than global.
DEVICE_BY_MODEL = {
    "resnet": DEVICE,
    "gru": torch.device("cpu"),
    # Mamba: CUDA on Kaggle; locally the reference block runs fine on CPU/MPS
    "mamba": torch.device("cuda") if torch.cuda.is_available() else DEVICE,
}
