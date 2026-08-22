"""Token sequences with duty-cycle features added (needs local Data/).

Motivation: the eleven amplitude statistics we already compute per second do
not encode how long the foot is loaded. Measured directly, the fraction of the
recording during which the right foot carries load separates PD from HC at
AUC 0.83 on its own -- better than several trained models -- yet no single
existing statistic tracks it above |r| = 0.50. So it is added explicitly.

Per second and per channel, three features join the existing eleven:
  duty      fraction of the second the channel is loaded (> 5 % of the
            recording's own 99th percentile -- an instance-level threshold, so
            no information crosses subjects)
  onsets    number of loading onsets in that second (a cadence proxy)
  bout      mean length of the loaded stretches in that second, in seconds

14 statistics x 18 channels = 252 dimensions per token.

Outputs (same schema as the other caches):
  duty_windows_30s.npz   30-token windows, step 10
  duty_windows_60s.npz   60-token windows, step 15
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
DUTY_NAMES = ["duty", "onsets", "bout"]
ALL_STATS = list(STAT_NAMES) + DUTY_NAMES
CHANNELS = [f"{ch}__{stat}" for ch in ALL_CHANNELS for stat in ALL_STATS]  # 252


def duty_features(block: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """(duty fraction, onset count, mean loaded-bout length) for one second."""
    loaded = block > threshold
    duty = float(loaded.mean())
    if not loaded.any():
        return 0.0, 0.0, 0.0
    edges = np.diff(loaded.astype(np.int8))
    onsets = float(np.count_nonzero(edges == 1) + (1 if loaded[0] else 0))
    bout = duty / onsets / FS_HZ * len(block) if onsets else duty * len(block) / FS_HZ
    return duty, onsets, float(bout)


def per_second_tokens(values: np.ndarray) -> np.ndarray:
    """(n_samples, 18) raw -> (n_seconds, 252)."""
    block_len = int(FS_HZ)
    n_blocks = len(values) // block_len
    # one threshold per channel, from the recording itself
    thresholds = 0.05 * np.percentile(values, 99, axis=0)

    tokens = np.empty((n_blocks, len(CHANNELS)), dtype=np.float32)
    for b in range(n_blocks):
        chunk = values[b * block_len : (b + 1) * block_len]
        col = 0
        for c in range(chunk.shape[1]):
            tokens[b, col : col + len(STAT_NAMES)] = compute_window_stats(chunk[:, c])
            col += len(STAT_NAMES)
            tokens[b, col : col + 3] = duty_features(chunk[:, c], thresholds[c])
            col += 3
    return tokens


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
        subject_id=np.asarray(sids), window_idx=np.asarray(widxs, dtype=int),
        label=np.asarray(lbls), data=data.astype(np.float32), channels=np.asarray(CHANNELS),
    )
    print(f"wrote {out} {data.shape} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    labels = load_labels()
    tokens: dict[str, np.ndarray] = {}
    for path in discover_unique_gait_files():
        df = drop_gait_init(load_raw_recording(path), skip_sec=SKIP_INIT_SEC)
        tokens[parse_subject_id(path)] = per_second_tokens(df[ALL_CHANNELS].to_numpy(dtype=np.float32))

    lens = [len(t) for t in tokens.values()]
    print(f"{len(tokens)} subjects, {len(CHANNELS)} dims/token, "
          f"seconds min/median/max = {min(lens)}/{int(np.median(lens))}/{max(lens)}")
    build_windows(tokens, labels, 30, 10, OUT_DIR / "duty_windows_30s.npz")
    build_windows(tokens, labels, 60, 15, OUT_DIR / "duty_windows_60s.npz")


if __name__ == "__main__":
    main()
