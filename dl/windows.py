"""Raw per-timestep VGRF window extraction for the DL pipeline.

Reuses baseline.features' raw-loading/windowing primitives (unchanged) so
window boundaries/counts are byte-for-byte identical to the classical
pipeline's windows -- but stacks the raw (500, n_channels) samples per
window instead of computing the 4 hand-crafted stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.features import (
    ALL_CHANNELS,
    discover_unique_gait_files,
    drop_gait_init,
    load_raw_recording,
    parse_subject_id,
    window_start_indices,
)
from baseline.labels import load_labels

from .config import CACHE_DIR, CACHE_VERSION, SKIP_INIT_SEC, STEP_SAMPLES, WINDOW_SAMPLES


@dataclass
class RawWindowTable:
    subject_id: np.ndarray  # (N,) str
    window_idx: np.ndarray  # (N,) int
    label: np.ndarray  # (N,) str, 'HC'/'PD'
    data: np.ndarray  # (N, len(channels), WINDOW_SAMPLES) float32, channels-first
    channels: list[str]


def _extract_subject_raw_windows(df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    """(n_windows, len(channels), WINDOW_SAMPLES) float32 array, channels-first."""
    starts = window_start_indices(len(df), WINDOW_SAMPLES, STEP_SAMPLES)
    values = df[channels].to_numpy(dtype=np.float32)  # (n_samples, n_channels)

    if not starts:
        return np.empty((0, len(channels), WINDOW_SAMPLES), dtype=np.float32)

    windows = np.stack([values[start : start + WINDOW_SAMPLES] for start in starts])  # (n_win, T, C)
    return np.transpose(windows, (0, 2, 1))  # (n_win, C, T)


def build_raw_window_table(gait_files: list[Path], labels: pd.DataFrame) -> RawWindowTable:
    subject_ids: list[str] = []
    window_idxs: list[int] = []
    label_list: list[str] = []
    chunks: list[np.ndarray] = []

    for path in gait_files:
        subject_id = parse_subject_id(path)
        label = labels.loc[subject_id, "label"]
        df = drop_gait_init(load_raw_recording(path), skip_sec=SKIP_INIT_SEC)
        subject_windows = _extract_subject_raw_windows(df, ALL_CHANNELS)
        n_win = subject_windows.shape[0]
        subject_ids.extend([subject_id] * n_win)
        window_idxs.extend(range(n_win))
        label_list.extend([label] * n_win)
        chunks.append(subject_windows)

    if not chunks:
        raise RuntimeError("No windows generated; check window length vs recordings.")

    return RawWindowTable(
        subject_id=np.asarray(subject_ids),
        window_idx=np.asarray(window_idxs, dtype=int),
        label=np.asarray(label_list),
        data=np.concatenate(chunks, axis=0),
        channels=list(ALL_CHANNELS),
    )


def _cache_path() -> Path:
    return CACHE_DIR / f"raw_windows_{CACHE_VERSION}.npz"


def load_or_build_raw_windows(force_rebuild: bool = False) -> RawWindowTable:
    cache_path = _cache_path()
    if cache_path.exists() and not force_rebuild:
        npz = np.load(cache_path, allow_pickle=False)
        return RawWindowTable(
            subject_id=npz["subject_id"],
            window_idx=npz["window_idx"],
            label=npz["label"],
            data=npz["data"],
            channels=list(npz["channels"]),
        )

    labels = load_labels()
    gait_files = discover_unique_gait_files()
    table = build_raw_window_table(gait_files, labels)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        subject_id=table.subject_id,
        window_idx=table.window_idx,
        label=table.label,
        data=table.data,
        channels=np.asarray(table.channels),
    )
    return table


def select_foot_channels(table: RawWindowTable, foot: str) -> np.ndarray:
    """Slice the channel dimension down to one foot-config. Returns (N, n_chan, T)."""
    from baseline.config import FOOT_CONFIGS

    wanted = FOOT_CONFIGS[foot]
    idx = [table.channels.index(ch) for ch in wanted]
    return table.data[:, idx, :]
