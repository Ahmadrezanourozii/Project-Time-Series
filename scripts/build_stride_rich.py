"""Gait-cycle token sequences with the full per-stride feature set (needs Data/).

The earlier stride tokens (build_stride_sequences.py) carried 8 timing values
per foot and scored 0.59-0.66. The subject-level gait ensemble showed that the
discriminative information sits in force shape and the centre-of-pressure path,
so here every stride token carries all per-stride quantities of
gait/features.py for both feet (31 each) plus double support and the phase of
the right heel strike inside the left stride.

Token grid: one token per LEFT stride; the right-foot values come from the
right stride starting nearest to it. Channel names use the TotalL__/TotalR__
prefixes so dl.windows.select_foot_channels keeps working.

Outputs (channels-first (N, C, T)):
  stride_rich_20.npz   20-stride windows, step 5
  stride_rich_40.npz   40-stride windows, step 10
With --input DIR the same tokens are written for unlabelled recordings
(label 'NA'), for inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gait.features import PER_STRIDE_MEDIAN, foot_strides, load_recording

OUT_DIR = Path("outputs/dl/cache")
PER_FOOT = PER_STRIDE_MEDIAN + ["double_support"]
CHANNELS = [f"{foot}__{f}" for foot in ("TotalL", "TotalR") for f in PER_FOOT] + ["TotalL__phase"]


def _double_support(own: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    spans = other[["start", "end"]].to_numpy()
    out = []
    for s0, e0, s1 in own[["start", "end", "next"]].to_numpy():
        overlap = np.clip(np.minimum(e0, spans[:, 1]) - np.maximum(s0, spans[:, 0]), 0, None).sum()
        out.append(overlap / (s1 - s0))
    return np.asarray(out)


def recording_tokens(path: Path) -> np.ndarray:
    """(n_left_strides, len(CHANNELS)) for one recording."""
    df = load_recording(path)
    bw = float(np.mean(df["TotalL"] + df["TotalR"]))
    left, _ = foot_strides(df, "L", bw)
    right, _ = foot_strides(df, "R", bw)
    if len(left) < 5 or len(right) < 5:
        raise RuntimeError(f"{path.name}: too few strides ({len(left)}/{len(right)})")
    left["double_support"] = _double_support(left, right)
    right["double_support"] = _double_support(right, left)

    nearest = np.abs(right["start"].to_numpy()[None, :] - left["start"].to_numpy()[:, None]).argmin(axis=1)
    r = right.iloc[nearest].reset_index(drop=True)
    phase = (r["start"].to_numpy() - left["start"].to_numpy()) / (left["next"] - left["start"]).to_numpy()
    tokens = np.concatenate(
        [left[PER_FOOT].to_numpy(), r[PER_FOOT].to_numpy(), np.mod(phase, 1.0)[:, None]], axis=1
    )
    return tokens.astype(np.float32)


def build_windows(tokens: dict[str, np.ndarray], labels: dict[str, str], size: int, step: int, out: Path) -> None:
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
        lbls.extend([labels[subject_id]] * len(windows))
        chunks.append(windows)
    data = np.concatenate(chunks)
    np.savez_compressed(
        out, subject_id=np.asarray(sids), window_idx=np.asarray(widxs, dtype=int), label=np.asarray(lbls),
        data=data, channels=np.asarray(CHANNELS),
    )
    print(f"wrote {out} {data.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Unlabelled recordings (inference)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.input is None:
        from baseline.features import discover_unique_gait_files, parse_subject_id
        from baseline.labels import load_labels

        label_table = load_labels()
        paths = discover_unique_gait_files()
        ids = [parse_subject_id(p) for p in paths]
        labels = {s: label_table.loc[s, "label"] for s in ids}
        prefix = "stride_rich"
    else:
        paths = sorted(args.input.glob("*.txt"))
        ids = [p.stem for p in paths]
        labels = {s: "NA" for s in ids}
        prefix = "stride_rich_heldout"

    tokens = {s: recording_tokens(p) for s, p in zip(ids, paths)}
    lens = [len(t) for t in tokens.values()]
    print(f"{len(tokens)} recordings, {len(CHANNELS)} channels, strides min/median/max = "
          f"{min(lens)}/{int(np.median(lens))}/{max(lens)}")
    build_windows(tokens, labels, 20, 5, args.out_dir / f"{prefix}_20.npz")
    build_windows(tokens, labels, 40, 10, args.out_dir / f"{prefix}_40.npz")


if __name__ == "__main__":
    main()
