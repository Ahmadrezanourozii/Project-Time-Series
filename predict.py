"""Predict PD/HC for new gait recordings with a trained Mamba bundle.

Takes raw PhysioNet-format recordings (tab-separated, 19 columns: time, L1-L8,
R1-R8, TotalL, TotalR at 100 Hz) that the model has never seen, applies exactly
the training-time pre-processing and normalisation, and writes one decision per
recording.

The bundle (produced by `run_mamba.py --export-bundle`) holds one file per
fold, each containing the weights *and* the per-channel mean/std of that fold's
training subjects. Statistics are never recomputed from the new data -- doing
so would leak the new cohort's distribution into its own predictions. Window
scores are averaged over folds, then over each recording's windows.

Usage:
  python predict.py --bundle outputs/mamba/final/bundle --input path/to/new_txt_dir
  python predict.py --bundle ... --input ... --labels labels.csv   # also score it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.config import FOOT_CONFIGS, FS_HZ, RAW_COLUMNS, SKIP_INIT_SEC
from baseline.features import compute_window_stats, drop_gait_init, window_start_indices
from dl.datasets import apply_normalizer
from mamba_model import MambaClassifier

ALL_CHANNELS = FOOT_CONFIGS["both"]
STAT_TOKENS_PER_WINDOW = 30  # 30 s windows of 1 s tokens -- matches the trained model
TOKEN_STEP = 10


def per_second_tokens(values: np.ndarray) -> np.ndarray:
    """(n_samples, 18) raw -> (n_seconds, 198) statistics per 1 s block."""
    block = int(FS_HZ)
    n_blocks = len(values) // block
    if n_blocks == 0:
        raise ValueError("recording shorter than one second after trimming")
    tokens = []
    for b in range(n_blocks):
        chunk = values[b * block : (b + 1) * block]
        tokens.append(np.concatenate([compute_window_stats(chunk[:, c]) for c in range(chunk.shape[1])]))
    return np.asarray(tokens, dtype=np.float32)


def windows_from_recording(path: Path, foot: str) -> np.ndarray:
    """Raw .txt -> (n_windows, C, T) float32, unnormalised."""
    df = pd.read_csv(path, sep="\t", header=None, names=RAW_COLUMNS)
    df = drop_gait_init(df, skip_sec=SKIP_INIT_SEC)
    tokens = per_second_tokens(df[ALL_CHANNELS].to_numpy(dtype=np.float32))  # (T_sec, 198)

    starts = window_start_indices(len(tokens), STAT_TOKENS_PER_WINDOW, TOKEN_STEP)
    if not starts:
        padded = np.zeros((STAT_TOKENS_PER_WINDOW, tokens.shape[1]), dtype=np.float32)
        padded[: len(tokens)] = tokens
        windows = padded[None]
    else:
        windows = np.stack([tokens[s : s + STAT_TOKENS_PER_WINDOW] for s in starts])
    windows = np.transpose(windows, (0, 2, 1))  # (n, 198, T)

    wanted = set(FOOT_CONFIGS[foot])
    channel_names = [f"{ch}__{stat}" for ch in ALL_CHANNELS for stat in range(11)]
    idx = [i for i, name in enumerate(channel_names) if name.split("__")[0] in wanted]
    return windows[:, idx, :]


def load_bundle(bundle_dir: Path, foot: str) -> list[dict]:
    files = sorted(bundle_dir.glob(f"mamba_{foot}_fold*.pt"))
    if not files:
        raise SystemExit(f"No bundle files for foot={foot!r} in {bundle_dir}")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


@torch.no_grad()
def score_windows(bundle: dict, X: np.ndarray, device: torch.device) -> np.ndarray:
    model = MambaClassifier(
        in_channels=bundle["in_channels"], dropout=0.0, variant=bundle["variant"],
        bidirectional=bundle["bidirectional"], pooling=bundle["pooling"], **bundle["model_kwargs"],
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    Xn = apply_normalizer(X, bundle["mean"], bundle["std"])
    scores = []
    tensor = torch.from_numpy(Xn)
    for start in range(0, len(tensor), 128):
        logits = model(tensor[start : start + 128].to(device))
        scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Directory of .txt recordings")
    parser.add_argument("--foot", default="both", choices=["left", "right", "both"])
    parser.add_argument("--out", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--threshold", type=float, default=None, help="Overrides the bundle's threshold")
    parser.add_argument("--labels", type=Path, default=None, help="Optional CSV (subject_id,label) to score against")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = load_bundle(args.bundle, args.foot)
    threshold = args.threshold if args.threshold is not None else float(np.mean([b["threshold"] for b in bundles]))
    manifest = args.bundle / "manifest.json"
    if manifest.exists():
        print("bundle:", json.loads(manifest.read_text()).get("token_kind"), f"({len(bundles)} fold models)")
    print(f"device={device}  foot={args.foot}  threshold={threshold:.3f}")

    recordings = sorted(args.input.glob("*.txt"))
    if not recordings:
        raise SystemExit(f"No .txt recordings found in {args.input}")

    rows = []
    for path in recordings:
        X = windows_from_recording(path, args.foot)
        fold_scores = [score_windows(b, X, device) for b in bundles]  # each (n_windows,)
        window_scores = np.mean(fold_scores, axis=0)
        score = float(window_scores.mean())
        rows.append(
            {
                "recording": path.name,
                "subject_id": path.stem.split("_")[0],
                "n_windows": len(X),
                "score_pd": round(score, 4),
                "prediction": "PD" if score >= threshold else "HC",
            }
        )
        print(f"  {path.name:20s} windows={len(X):3d} score={score:.3f} -> {rows[-1]['prediction']}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} recordings, "
          f"{(df.prediction == 'PD').sum()} PD / {(df.prediction == 'HC').sum()} HC)")

    if args.labels:
        truth = pd.read_csv(args.labels).set_index("subject_id")["label"]
        merged = df.join(truth, on="subject_id").dropna(subset=["label"])
        if len(merged):
            from baseline.metrics import compute_point_metrics

            y_true = (merged["label"] == "PD").astype(int).to_numpy()
            y_pred = (merged["prediction"] == "PD").astype(int).to_numpy()
            m = compute_point_metrics(y_true, y_pred, merged["score_pd"].to_numpy())
            print(f"scored {len(merged)} labelled recordings: acc={m['accuracy']:.3f} "
                  f"prec_w={m['precision_w']:.3f} rec_w={m['recall_w']:.3f} "
                  f"f1_w={m['f1_w']:.3f} auc={m['auc']:.3f}")


if __name__ == "__main__":
    main()
