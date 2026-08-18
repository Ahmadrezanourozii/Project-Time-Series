"""Window-level VGRF feature extraction: 9 channels/foot x 11 stats (time-domain:
mean, std, skew, kurt, rms, zcr, median, min, max; frequency-domain: spec_energy,
spec_entropy).

Each subject's ~121s recording is windowed (5s window, 1.0s step, first 5s
dropped) and every window becomes one row of the feature table, inheriting
its subject's PD/HC label. No tsfresh, no cross-window aggregation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .config import (
    CACHE_DIR,
    CACHE_VERSION,
    DEGENERATE_FILL,
    DEGENERATE_STD_EPS,
    FILENAME_RE,
    FOOT_CONFIGS,
    RAW_COLUMNS,
    SKIP_INIT_SEC,
    SPLITS_DIR,
    STAT_NAMES,
    STEP_SAMPLES,
    WINDOW_SAMPLES,
)
from .labels import load_labels

ALL_CHANNELS = FOOT_CONFIGS["both"]  # 18 channels: L1-8,TotalL,R1-8,TotalR


def discover_unique_gait_files() -> list[Path]:
    """Unique gait recordings across all Fold_*/{training,validation,test}/*.txt."""
    all_paths = sorted(SPLITS_DIR.glob("Fold_*/*/*.txt"))
    by_name: dict[str, Path] = {}
    for path in all_paths:
        by_name.setdefault(path.name, path)
    return [by_name[name] for name in sorted(by_name)]


def parse_subject_id(path: Path) -> str:
    """GaCo02_01.txt -> GaCo02."""
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected filename: {path.name}")
    return match.group(1)


def load_raw_recording(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=RAW_COLUMNS)


def drop_gait_init(df: pd.DataFrame, skip_sec: float = SKIP_INIT_SEC) -> pd.DataFrame:
    """Remove the first ``skip_sec`` seconds so windows use steady-state gait."""
    return df.loc[df["time"] > skip_sec].reset_index(drop=True)


def window_start_indices(n_samples: int, window_samples: int, step_samples: int) -> list[int]:
    if n_samples < window_samples:
        return []
    return list(range(0, n_samples - window_samples + 1, step_samples))


def compute_window_stats(window: np.ndarray) -> tuple[float, ...]:
    """11 stats of one channel's samples within a single window, in the exact
    order of config.STAT_NAMES: mean, std, skew, kurt, rms, zcr, median, min,
    max, spec_energy, spec_entropy.

    When the window is flat (std below DEGENERATE_STD_EPS - e.g. swing phase or
    a standstill), skew/kurtosis are mathematically undefined and filled with
    DEGENERATE_FILL rather than left as NaN (no downstream imputation step here).
    The other 7 stats are well-defined even on a flat window (rms=|mean|, zcr=0,
    spec_entropy~=0) and need no such fallback.
    """
    eps = 1e-12
    mean_v = float(np.mean(window))
    std_v = float(np.std(window, ddof=1))
    if std_v < DEGENERATE_STD_EPS:
        skew_v, kurt_v = DEGENERATE_FILL, DEGENERATE_FILL
    else:
        with np.errstate(all="ignore"):
            skew_v = float(skew(window, bias=False))
            kurt_v = float(kurtosis(window, bias=False))
        if not np.isfinite(skew_v):
            skew_v = DEGENERATE_FILL
        if not np.isfinite(kurt_v):
            kurt_v = DEGENERATE_FILL

    rms_v = float(np.sqrt(np.mean(np.square(window))))
    demeaned = window - mean_v
    zcr_v = float(np.mean(np.abs(np.diff(np.sign(demeaned))) > 0))
    median_v = float(np.median(window))
    min_v = float(np.min(window))
    max_v = float(np.max(window))

    psd = np.abs(np.fft.rfft(window)) ** 2
    spec_energy_v = float(np.sum(psd) / len(window))
    psd_norm = psd / (np.sum(psd) + eps)
    spec_entropy_v = float(-np.sum(psd_norm * np.log2(psd_norm + eps)))

    return (
        mean_v,
        std_v,
        skew_v,
        kurt_v,
        rms_v,
        zcr_v,
        median_v,
        min_v,
        max_v,
        spec_energy_v,
        spec_entropy_v,
    )


def extract_subject_window_features(df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    """Per-window feature matrix for one subject's post-init recording.

    Inputs: df (post drop_gait_init), channels (ordered list of column names).
    Output: array of shape (n_windows, len(channels) * len(STAT_NAMES)),
    channel-major column order [ch0_mean, ch0_std, ..., ch0_spec_entropy,
    ch1_mean, ...].
    """
    starts = window_start_indices(len(df), WINDOW_SAMPLES, STEP_SAMPLES)
    values = df[channels].to_numpy(dtype=float)

    rows = []
    for start in starts:
        chunk = values[start : start + WINDOW_SAMPLES]
        row: list[float] = []
        for col in range(chunk.shape[1]):
            row.extend(compute_window_stats(chunk[:, col]))
        rows.append(row)

    if not rows:
        return np.empty((0, len(channels) * len(STAT_NAMES)))
    return np.asarray(rows, dtype=float)


def build_window_feature_table(gait_files: list[Path], labels: pd.DataFrame) -> pd.DataFrame:
    """Single pass over all unique subject recordings, computing all 18 channels'
    11 stats (198 columns) per window.

    Output columns: ['subject_id', 'window_idx', 'label', <198 feature cols>]
    Feature col names: f"{channel}__{stat}", e.g. 'L1__mean', 'TotalR__spec_entropy'.
    """
    feat_cols = [f"{ch}__{stat}" for ch in ALL_CHANNELS for stat in STAT_NAMES]
    records = []
    for path in gait_files:
        subject_id = parse_subject_id(path)
        label = labels.loc[subject_id, "label"]
        df = drop_gait_init(load_raw_recording(path))
        window_mat = extract_subject_window_features(df, ALL_CHANNELS)
        for win_idx in range(window_mat.shape[0]):
            records.append([subject_id, win_idx, label] + window_mat[win_idx].tolist())

    if not records:
        raise RuntimeError("No windows generated; check window length vs recordings.")

    return pd.DataFrame(records, columns=["subject_id", "window_idx", "label"] + feat_cols)


def load_or_build_window_features(force_rebuild: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"window_features_{CACHE_VERSION}.csv"
    if cache_path.exists() and not force_rebuild:
        return pd.read_csv(cache_path)

    labels = load_labels()
    gait_files = discover_unique_gait_files()
    table = build_window_feature_table(gait_files, labels)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, index=False)
    return table


def select_foot_columns(window_features: pd.DataFrame, foot: str) -> pd.DataFrame:
    """Slice the canonical window feature table down to one foot-config.

    foot in {'left', 'right', 'both'}. Returns subject_id/window_idx/label plus
    that foot-config's feature columns only (36/36/72).
    """
    channels = FOOT_CONFIGS[foot]
    feat_cols = [f"{ch}__{stat}" for ch in channels for stat in STAT_NAMES]
    return window_features[["subject_id", "window_idx", "label"] + feat_cols].copy()
