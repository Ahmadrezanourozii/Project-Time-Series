"""Fuse amplitude-statistic tokens with stride-cycle tokens (needs local Data/).

The two token families are complementary: per-second amplitude statistics
capture loading magnitude/shape, stride-cycle features capture gait timing
and its variability (the clinical PD marker). A subject-level RF check gave
AUC 0.794 (stride only), 0.827 (amplitude only), 0.859 (both) -- so they are
fused here into a single token stream on a common 1 s grid:

    token_t = [198 amplitude stats of second t] ++ [16 stride features of the
               stride active during second t]

Outputs stride-padded windows in the usual channels-first schema:
  fused_windows_30s.npz        30 tokens, step 10
  fused_windows_30s_dense.npz  30 tokens, step 2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.config import FS_HZ, SKIP_INIT_SEC
from baseline.features import (
    ALL_CHANNELS,
    discover_unique_gait_files,
    drop_gait_init,
    load_raw_recording,
    parse_subject_id,
)
from baseline.labels import load_labels
from scripts.build_feature_sequences import FEATURE_CHANNELS, per_second_tokens
from scripts.build_stride_sequences import (
    CHANNELS as STRIDE_CHANNELS,
    build_subject_tokens,
    detect_contacts,
)

OUT_DIR = Path("outputs/dl/cache")
CHANNELS = list(FEATURE_CHANNELS) + list(STRIDE_CHANNELS)  # 198 + 16 = 214


def stride_tokens_per_second(df, n_seconds: int) -> np.ndarray:
    """(n_seconds, 16): each second carries the features of its active stride."""
    stride_tok = build_subject_tokens(df)
    if stride_tok is None:
        raise RuntimeError("stride detection failed")
    contacts = detect_contacts(df["TotalL"].to_numpy(np.float32))
    starts_sec = np.array([c[0] for c in contacts[: len(stride_tok)]]) / FS_HZ

    out = np.empty((n_seconds, stride_tok.shape[1]), dtype=np.float32)
    for t in range(n_seconds):
        idx = int(np.searchsorted(starts_sec, t, side="right") - 1)
        out[t] = stride_tok[min(max(idx, 0), len(stride_tok) - 1)]
    return out


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
    data = np.concatenate(chunks)
    np.savez_compressed(
        out,
        subject_id=np.asarray(sids),
        window_idx=np.asarray(widxs, dtype=int),
        label=np.asarray(lbls),
        data=data.astype(np.float32),
        channels=np.asarray(CHANNELS),
    )
    print(f"wrote {out} {data.shape} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    labels = load_labels()
    tokens: dict[str, np.ndarray] = {}
    for path in discover_unique_gait_files():
        subject_id = parse_subject_id(path)
        df = drop_gait_init(load_raw_recording(path), skip_sec=SKIP_INIT_SEC)
        amp = per_second_tokens(df[ALL_CHANNELS].to_numpy(dtype=np.float32))
        stride = stride_tokens_per_second(df, len(amp))
        tokens[subject_id] = np.concatenate([amp, stride], axis=1)  # (n_sec, 214)

    lens = [len(t) for t in tokens.values()]
    print(f"{len(tokens)} subjects, seconds min/median/max = {min(lens)}/{int(np.median(lens))}/{max(lens)}")
    build_windows(tokens, labels, 30, 10, OUT_DIR / "fused_windows_30s.npz")
    build_windows(tokens, labels, 30, 2, OUT_DIR / "fused_windows_30s_dense.npz")


if __name__ == "__main__":
    main()
