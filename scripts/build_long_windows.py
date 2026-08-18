"""Build long-window / full-sequence npz caches for E2 (needs local Data/).

Same schema as raw_windows_v2.npz (subject_id, window_idx, label, data,
channels) so run_mamba.py --npz can consume them unchanged. Examples:

  python scripts/build_long_windows.py --window-sec 30 --step-sec 10 --out outputs/dl/cache/windows_30s.npz
  python scripts/build_long_windows.py --window-sec 60 --step-sec 15 --out outputs/dl/cache/windows_60s.npz
  python scripts/build_long_windows.py --full --full-sec 110 --out outputs/dl/cache/sequences_full.npz
"""

from __future__ import annotations

import argparse
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
    window_start_indices,
)
from baseline.labels import load_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--step-sec", type=float, default=15.0)
    parser.add_argument("--full", action="store_true", help="One fixed-length sequence per subject")
    parser.add_argument("--full-sec", type=float, default=110.0, help="Truncate/pad length for --full")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = load_labels()
    gait_files = discover_unique_gait_files()

    subject_ids, window_idxs, label_list, chunks = [], [], [], []
    lengths = []

    for path in gait_files:
        subject_id = parse_subject_id(path)
        label = labels.loc[subject_id, "label"]
        df = drop_gait_init(load_raw_recording(path), skip_sec=SKIP_INIT_SEC)
        values = df[ALL_CHANNELS].to_numpy(dtype=np.float32)  # (n_samples, 18)
        lengths.append(len(values))

        if args.full:
            target = int(args.full_sec * FS_HZ)
            if len(values) >= target:
                seq = values[:target]
            else:
                seq = np.zeros((target, values.shape[1]), dtype=np.float32)
                seq[: len(values)] = values
            windows = seq[None]  # (1, T, C)
        else:
            window_samples = int(args.window_sec * FS_HZ)
            step_samples = int(args.step_sec * FS_HZ)
            starts = window_start_indices(len(values), window_samples, step_samples)
            if not starts:
                # Recording shorter than one window (5 Ju subjects at 60 s):
                # zero-pad to a single window so every subject stays covered.
                print(f"NOTE: {subject_id} shorter than window ({len(values)} samples), zero-padded")
                padded = np.zeros((window_samples, values.shape[1]), dtype=np.float32)
                padded[: len(values)] = values
                windows = padded[None]
            else:
                windows = np.stack([values[s : s + window_samples] for s in starts])  # (n, T, C)

        windows = np.transpose(windows, (0, 2, 1))  # (n, C, T)
        n_win = windows.shape[0]
        subject_ids.extend([subject_id] * n_win)
        window_idxs.extend(range(n_win))
        label_list.extend([label] * n_win)
        chunks.append(windows)

    data = np.concatenate(chunks, axis=0)
    n_subjects = len(set(subject_ids))
    print(
        f"recordings: {len(gait_files)}, lengths min/median/max = "
        f"{min(lengths)}/{int(np.median(lengths))}/{max(lengths)} samples"
    )
    print(f"windows: {data.shape} across {n_subjects} subjects "
          f"({len(subject_ids) / n_subjects:.1f} windows/subject)")
    if n_subjects != 100:
        raise RuntimeError(f"Expected 100 subjects, got {n_subjects}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        subject_id=np.asarray(subject_ids),
        window_idx=np.asarray(window_idxs, dtype=int),
        label=np.asarray(label_list),
        data=data,
        channels=np.asarray(ALL_CHANNELS),
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
