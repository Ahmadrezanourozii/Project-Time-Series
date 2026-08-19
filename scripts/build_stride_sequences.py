"""Build stride-cycle token sequences (needs local Data/).

Clinically, PD gait is characterised by stride-to-stride timing variability,
reduced swing time and asymmetry -- none of which the per-second amplitude
statistics expose directly. Here each token is one *gait cycle*: foot
contacts are detected from the per-foot total VGRF, and each stride yields
timing/loading features for both feet, so a Mamba over this sequence models
stride-to-stride dynamics directly.

Token grid: one token per LEFT-foot stride; right-foot features come from the
right stride starting nearest that left contact (a common grid is required so
the left/right/both channel selection stays aligned).

Channels are named '{TotalL|TotalR}__{feature}' so dl.windows.select_foot_channels
prefix-matching maps them onto the existing left/right/both foot configs.

Outputs (channels-first (N, C, T), same schema as the other caches):
  stride_windows_20.npz   20-stride windows, step 5
  stride_windows_40.npz   40-stride windows, step 10
  stride_seq_full.npz     one repeat-padded sequence per subject
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.config import FS_HZ, SKIP_INIT_SEC
from baseline.features import (
    discover_unique_gait_files,
    drop_gait_init,
    load_raw_recording,
    parse_subject_id,
)
from baseline.labels import load_labels

OUT_DIR = Path("outputs/dl/cache")
FEATURES = [
    "stride_time",
    "stance_time",
    "swing_time",
    "stance_ratio",
    "peak_force",
    "impulse",
    "stride_time_delta",
    "double_support",
]
CHANNELS = [f"{foot}__{feat}" for foot in ("TotalL", "TotalR") for feat in FEATURES]
MIN_STRIDE_SEC, MAX_STRIDE_SEC = 0.5, 2.5


def detect_contacts(signal: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous stance intervals (start, end) from one foot's total force."""
    threshold = 0.05 * np.max(signal)
    stance = signal > threshold
    edges = np.diff(stance.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if stance[0]:
        starts.insert(0, 0)
    if stance[-1]:
        ends.append(len(stance))
    intervals = [(s, e) for s, e in zip(starts, ends) if (e - s) > 0.15 * FS_HZ]
    return intervals


def stride_features(signal: np.ndarray, contacts: list[tuple[int, int]], max_force: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-stride features and the sample index each stride starts at."""
    rows, starts = [], []
    prev_stride = np.nan
    for i in range(len(contacts) - 1):
        (s0, e0), (s1, _) = contacts[i], contacts[i + 1]
        stride_time = (s1 - s0) / FS_HZ
        if not (MIN_STRIDE_SEC <= stride_time <= MAX_STRIDE_SEC):
            continue
        stance_time = (e0 - s0) / FS_HZ
        swing_time = (s1 - e0) / FS_HZ
        segment = signal[s0:e0]
        rows.append(
            [
                stride_time,
                stance_time,
                swing_time,
                stance_time / stride_time,
                float(np.max(segment)) / max_force,
                float(np.sum(segment)) / (max_force * FS_HZ),
                0.0 if np.isnan(prev_stride) else stride_time - prev_stride,
                0.0,  # double_support filled in later (needs the other foot)
            ]
        )
        starts.append(s0)
        prev_stride = stride_time
    return np.asarray(rows, dtype=np.float32), np.asarray(starts, dtype=int)


def build_subject_tokens(df) -> np.ndarray | None:
    """(n_left_strides, 16) token matrix for one recording, or None if too short."""
    left, right = df["TotalL"].to_numpy(np.float32), df["TotalR"].to_numpy(np.float32)
    lc, rc = detect_contacts(left), detect_contacts(right)
    if len(lc) < 6 or len(rc) < 6:
        return None

    lf, ls = stride_features(left, lc, float(np.max(left)))
    rf, rs = stride_features(right, rc, float(np.max(right)))
    if len(lf) < 5 or len(rf) < 5:
        return None

    # Double support: overlap of each left stance with any right stance.
    for idx, (s0, e0) in enumerate([lc[i] for i in range(len(lc) - 1)][: len(lf)]):
        overlap = sum(max(0, min(e0, re) - max(s0, rs_)) for rs_, re in rc)
        lf[idx, FEATURES.index("double_support")] = overlap / FS_HZ

    # Align each left stride with the nearest-starting right stride.
    nearest = np.abs(rs[None, :] - ls[:, None]).argmin(axis=1)
    return np.concatenate([lf, rf[nearest]], axis=1)  # (n_left_strides, 16)


def save_npz(path: Path, subject_id, window_idx, label, data) -> None:
    np.savez_compressed(
        path,
        subject_id=np.asarray(subject_id),
        window_idx=np.asarray(window_idx, dtype=int),
        label=np.asarray(label),
        data=data.astype(np.float32),
        channels=np.asarray(CHANNELS),
    )
    print(f"wrote {path} {data.shape} ({path.stat().st_size / 1e6:.1f} MB)")


def build_windows(tokens: dict[str, np.ndarray], labels, size: int, step: int, out: Path) -> None:
    sids, widxs, lbls, chunks = [], [], [], []
    for subject_id, tok in tokens.items():
        if len(tok) < size:
            reps = int(np.ceil(size / len(tok)))
            windows = np.tile(tok, (reps, 1))[:size][None]
        else:
            windows = np.stack([tok[s : s + size] for s in range(0, len(tok) - size + 1, step)])
        windows = np.transpose(windows, (0, 2, 1))
        sids.extend([subject_id] * len(windows))
        widxs.extend(range(len(windows)))
        lbls.extend([labels.loc[subject_id, "label"]] * len(windows))
        chunks.append(windows)
    save_npz(out, sids, widxs, lbls, np.concatenate(chunks))


def main() -> None:
    labels = load_labels()
    tokens: dict[str, np.ndarray] = {}
    for path in discover_unique_gait_files():
        subject_id = parse_subject_id(path)
        df = drop_gait_init(load_raw_recording(path), skip_sec=SKIP_INIT_SEC)
        tok = build_subject_tokens(df)
        if tok is None:
            raise RuntimeError(f"{subject_id}: stride detection failed")
        tokens[subject_id] = tok

    lens = [len(t) for t in tokens.values()]
    print(f"{len(tokens)} subjects, strides min/median/max = {min(lens)}/{int(np.median(lens))}/{max(lens)}")
    if len(tokens) != 100:
        raise RuntimeError(f"expected 100 subjects, got {len(tokens)}")

    build_windows(tokens, labels, 20, 5, OUT_DIR / "stride_windows_20.npz")
    build_windows(tokens, labels, 40, 10, OUT_DIR / "stride_windows_40.npz")
    build_windows(tokens, labels, max(lens), max(lens), OUT_DIR / "stride_seq_full.npz")


if __name__ == "__main__":
    main()
