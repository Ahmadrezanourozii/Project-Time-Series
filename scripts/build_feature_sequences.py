"""Build feature-token sequence npz caches (needs local Data/).

The assignment allows DL models to consume "feature vector sequences". Raw-
signal Mamba collapses on folds 1/4 while RF on engineered stats handles all
folds, so these caches feed Mamba the same 11 stats x 18 channels = 198-dim
vectors as tokens:

  stat30 : one token per 1 s of signal, 30-token windows (30 s), step 10 s
           -> data (n_win, 198, 30), same schema as raw_windows npz
  stat60 : 60-token windows (60 s), step 15 s
  subjseq: one token per 5 s window (the existing window_features_v3 vectors),
           full recording per subject, repeat-padded to the longest subject
           -> data (100, 198, T_max); repeat-padding keeps mean-pooling valid
           without masks.

Usage: python scripts/build_feature_sequences.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.config import FS_HZ, SKIP_INIT_SEC, STAT_NAMES
from baseline.features import (
    ALL_CHANNELS,
    compute_window_stats,
    discover_unique_gait_files,
    drop_gait_init,
    load_raw_recording,
    parse_subject_id,
)
from baseline.labels import load_labels

OUT_DIR = Path("outputs/dl/cache")
FEATURE_CHANNELS = [f"{ch}__{stat}" for ch in ALL_CHANNELS for stat in STAT_NAMES]  # 198
WINDOW_FEATURES_CSV = Path("outputs/baseline/cache/window_features_v3.csv")


def per_second_tokens(values: np.ndarray) -> np.ndarray:
    """(n_samples, 18) raw -> (n_blocks, 198) stats of each full 1 s block."""
    block = int(FS_HZ)
    n_blocks = len(values) // block
    tokens = np.empty((n_blocks, len(FEATURE_CHANNELS)), dtype=np.float32)
    for b in range(n_blocks):
        chunk = values[b * block : (b + 1) * block]
        col = 0
        for c in range(chunk.shape[1]):
            tokens[b, col : col + len(STAT_NAMES)] = compute_window_stats(chunk[:, c])
            col += len(STAT_NAMES)
    return tokens


def save_npz(path: Path, subject_id, window_idx, label, data) -> None:
    np.savez_compressed(
        path,
        subject_id=np.asarray(subject_id),
        window_idx=np.asarray(window_idx, dtype=int),
        label=np.asarray(label),
        data=data.astype(np.float32),
        channels=np.asarray(FEATURE_CHANNELS),
    )
    print(f"wrote {path} {data.shape} ({path.stat().st_size / 1e6:.0f} MB)")


def build_stat_windows(token_table: dict[str, np.ndarray], labels, window_tokens: int, step_tokens: int, out: Path) -> None:
    sids, widxs, lbls, chunks = [], [], [], []
    for subject_id, tokens in token_table.items():
        if len(tokens) < window_tokens:
            padded = np.zeros((window_tokens, tokens.shape[1]), dtype=np.float32)
            padded[: len(tokens)] = tokens
            windows = padded[None]
        else:
            starts = range(0, len(tokens) - window_tokens + 1, step_tokens)
            windows = np.stack([tokens[s : s + window_tokens] for s in starts])
        windows = np.transpose(windows, (0, 2, 1))  # (n, 198, T)
        sids.extend([subject_id] * len(windows))
        widxs.extend(range(len(windows)))
        lbls.extend([labels.loc[subject_id, "label"]] * len(windows))
        chunks.append(windows)
    save_npz(out, sids, widxs, lbls, np.concatenate(chunks))


def build_subject_sequences(labels) -> None:
    """One sequence per subject from the existing 5s-window feature vectors."""
    df = pd.read_csv(WINDOW_FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in ("subject_id", "window_idx", "label")]
    assert feat_cols == FEATURE_CHANNELS, "window_features_v3 column order mismatch"
    subjects = sorted(df["subject_id"].unique())
    t_max = int(df.groupby("subject_id").size().max())

    data = np.empty((len(subjects), len(feat_cols), t_max), dtype=np.float32)
    lbls = []
    for i, subject_id in enumerate(subjects):
        seq = (
            df.loc[df["subject_id"] == subject_id]
            .sort_values("window_idx")[feat_cols]
            .to_numpy(dtype=np.float32)
        )  # (T_subj, 198)
        reps = int(np.ceil(t_max / len(seq)))
        seq = np.tile(seq, (reps, 1))[:t_max]  # repeat-pad, no zeros
        data[i] = seq.T
        lbls.append(labels.loc[subject_id, "label"])
    save_npz(OUT_DIR / "subject_feature_seq.npz", subjects, [0] * len(subjects), lbls, data)


def main() -> None:
    labels = load_labels()
    print("computing per-second stat tokens for 100 recordings...")
    token_table: dict[str, np.ndarray] = {}
    for path in discover_unique_gait_files():
        subject_id = parse_subject_id(path)
        df = drop_gait_init(load_raw_recording(path), skip_sec=SKIP_INIT_SEC)
        token_table[subject_id] = per_second_tokens(df[ALL_CHANNELS].to_numpy(dtype=np.float32))
    lens = [len(t) for t in token_table.values()]
    print(f"token lengths min/median/max = {min(lens)}/{int(np.median(lens))}/{max(lens)}")

    build_stat_windows(token_table, labels, 30, 10, OUT_DIR / "stat_windows_30s.npz")
    build_stat_windows(token_table, labels, 60, 15, OUT_DIR / "stat_windows_60s.npz")
    build_subject_sequences(labels)


if __name__ == "__main__":
    main()
