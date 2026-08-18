"""Deliverable 3 entry point: train/evaluate the Mamba classifier.

Runs locally (CPU/MPS, pure-PyTorch fallback block, --smoke) and on Kaggle
GPU (CUDA, mamba_ssm kernels). Mirrors dl/train.py's protocol exactly --
grid search (dropout x lr) with early stopping on validation ROC-AUC,
refit on train+validation at the winning epoch, predict the fold's test
subjects, pool the 5 disjoint test folds -- and adds AMP, sequence-length-
aware batch sizing with an OOM halve-and-retry guard, optional gradient
checkpointing, and subject-level aggregation on top.

Examples:
  python run_mamba.py --smoke
  python run_mamba.py --npz /kaggle/input/pd-gait-vgrf-windows/raw_windows_v2.npz \
      --fold-json /kaggle/input/pd-gait-vgrf-windows/fold_assignments.json \
      --output-dir /kaggle/working/outputs --foot all --require-cuda-kernels
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

import baseline.folds as folds_mod
from baseline.folds import FoldResult, pool_fold_results
from baseline.metrics import bootstrap_metrics, compute_point_metrics, confusion_counts, summarize_bootstrap
from baseline.reporting import plot_confusion_matrix, save_latex_table
from dl.aggregate import aggregate_subjects
from dl.config import WEIGHT_DECAY
from dl.datasets import FoldSplitData, WindowDataset, assemble_fold_splits, concat_splits
from dl.seeding import fold_seed, set_all_seeds
from dl.windows import RawWindowTable
from mamba_model import HAVE_MAMBA_SSM, MODEL_SIZES, MambaClassifier

FOOT_ORDER = ["left", "right", "both"]


# ---------------------------------------------------------------------------
# Data loading


def load_window_table(npz_path: Path) -> RawWindowTable:
    npz = np.load(npz_path, allow_pickle=False)
    return RawWindowTable(
        subject_id=npz["subject_id"],
        window_idx=npz["window_idx"],
        label=npz["label"],
        data=npz["data"],
        channels=list(npz["channels"]),
    )


def restrict_for_smoke(table: RawWindowTable, fold_idx: int, per_split: dict[str, int], windows_per_subject: int) -> RawWindowTable:
    """Class-balanced subject subsample + capped windows/subject, for fast end-to-end tests."""
    assignment = folds_mod.load_fold_assignment(fold_idx)
    label_by_subject = {}
    for sid, lbl in zip(table.subject_id, table.label):
        label_by_subject[sid] = lbl

    keep_subjects: list[str] = []
    for split, n_subj in per_split.items():
        candidates = assignment.loc[assignment["split"] == split, "subject_id"].tolist()
        for wanted_label in ("HC", "PD"):
            picked = [s for s in candidates if label_by_subject.get(s) == wanted_label][: n_subj // 2]
            keep_subjects.extend(picked)

    mask = np.isin(table.subject_id, keep_subjects) & (table.window_idx < windows_per_subject)
    return RawWindowTable(
        subject_id=table.subject_id[mask],
        window_idx=table.window_idx[mask],
        label=table.label[mask],
        data=table.data[mask],
        channels=table.channels,
    )


# ---------------------------------------------------------------------------
# Training


def auto_batch_size(seq_len: int) -> int:
    if seq_len <= 1000:
        return 128
    if seq_len <= 3000:
        return 32
    if seq_len <= 6000:
        return 16
    return 8


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _augment(X: torch.Tensor, noise_std: float, channel_dropout: float) -> torch.Tensor:
    if noise_std > 0:
        X = X + noise_std * torch.randn_like(X)
    if channel_dropout > 0:
        keep = (torch.rand(X.shape[0], X.shape[1], 1, device=X.device) >= channel_dropout).float()
        X = X * keep
    return X


@torch.no_grad()
def predict_scores(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int, use_amp: bool) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(X)
    while True:
        try:
            scores = []
            for start in range(0, len(tensor), batch_size):
                batch = tensor[start : start + batch_size].to(device)
                with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                    logits = model(batch)
                scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            return np.concatenate(scores)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise
            batch_size //= 2
            print(f"    [OOM] predict batch halved to {batch_size}")


def train_config(
    build_fn,
    train_split: FoldSplitData,
    val_split: FoldSplitData | None,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int | None,
    lr: float,
    batch_size: int,
    accum_steps: int,
    use_amp: bool,
    cosine: bool,
    noise_std: float,
    channel_dropout: float,
    min_epochs: int = 0,
) -> tuple[nn.Module, int, float, int]:
    """Train one config; returns (model, best_epoch, best_val_auc, batch_size_used).

    On CUDA OOM the batch is halved (accumulation doubled, preserving the
    effective batch) and training restarts from scratch for this config.
    """
    while True:
        try:
            return _train_once(
                build_fn, train_split, val_split, device, seed, max_epochs, patience,
                lr, batch_size, accum_steps, use_amp, cosine, noise_std, channel_dropout,
                min_epochs,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 2:
                raise
            batch_size //= 2
            accum_steps *= 2
            print(f"    [OOM] train batch halved to {batch_size} (accum x{accum_steps})")


def _train_once(
    build_fn, train_split, val_split, device, seed, max_epochs, patience,
    lr, batch_size, accum_steps, use_amp, cosine, noise_std, channel_dropout,
    min_epochs=0,
):
    set_all_seeds(seed)
    generator = torch.Generator().manual_seed(seed)

    model = build_fn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if cosine:
        # 3-epoch linear warmup then cosine decay -- stabilizes the early
        # epochs where Mamba runs were observed to collapse on some folds.
        warmup = min(3, max_epochs)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(max_epochs - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # num_workers=0: data is already an in-RAM tensor; forked workers only add
    # RAM pressure and fork+CUDA instability on Kaggle batch sessions.
    n_train = len(train_split.y)
    effective_bs = min(batch_size, n_train)
    loader = DataLoader(
        WindowDataset(train_split),
        batch_size=effective_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
        # drop_last guards BatchNorm-style size-1 failures; disabled when the
        # dataset is so small (subject-level sequences) it would drop everything
        drop_last=n_train > effective_bs,
    )

    best_state = deepcopy(model.state_dict())
    best_auc = -np.inf
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        for step, (X_batch, y_batch) in enumerate(loader):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if noise_std > 0 or channel_dropout > 0:
                X_batch = _augment(X_batch, noise_std, channel_dropout)
            with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                loss = criterion(model(X_batch), y_batch) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        if val_split is None:
            continue  # refit stage: fixed epoch count, no monitoring

        val_scores = predict_scores(model, val_split.X, device, batch_size, use_amp)
        val_auc = roc_auc_score(val_split.y, val_scores)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if patience is not None and epochs_since_improve >= patience and epoch >= min_epochs:
                break

    if val_split is None:
        return model, max_epochs, float("nan"), batch_size
    model.load_state_dict(best_state)
    return model, best_epoch, best_auc, batch_size


def run_fold_mamba(fold_idx: int, foot: str, table: RawWindowTable, args, model_kwargs: dict) -> FoldResult:
    splits = assemble_fold_splits(fold_idx, foot, table)
    for split_a, split_b in [("train", "test"), ("validation", "test"), ("train", "validation")]:
        overlap = set(splits[split_a].subject_id) & set(splits[split_b].subject_id)
        if overlap:
            raise RuntimeError(f"Fold {fold_idx}: subject leakage between {split_a}/{split_b}: {overlap}")

    device = resolve_device()
    in_channels = splits["train"].X.shape[1]
    seq_len = splits["train"].X.shape[2]
    batch_size = args.batch_size if args.batch_size > 0 else auto_batch_size(seq_len)
    base_seed = fold_seed(args.seed, fold_idx)
    use_amp = args.amp and device.type == "cuda"

    def build_fn(dropout: float):
        return lambda: MambaClassifier(
            in_channels=in_channels, dropout=dropout, variant=args.variant,
            grad_checkpoint=args.grad_checkpoint, **model_kwargs,
        )

    grid = list(itertools.product(args.dropout_grid, args.lr_grid))
    best_combo, best_combo_auc, best_combo_epoch = None, -np.inf, 1
    best_seed = base_seed
    for dropout, lr in grid:
        t0 = time.time()
        _, epoch, val_auc, batch_size = train_config(
            build_fn(dropout), splits["train"], splits["validation"], device, base_seed,
            args.epochs, args.patience, lr, batch_size, args.accum_steps, use_amp,
            args.cosine, args.aug_noise, args.aug_channel_dropout, args.min_epochs,
        )
        print(
            f"  fold {fold_idx} {foot}: dropout={dropout} lr={lr:g} -> val AUC {val_auc:.4f} "
            f"(epoch {epoch}, {time.time() - t0:.0f}s)",
            flush=True,
        )
        if val_auc > best_combo_auc:
            best_combo_auc, best_combo, best_combo_epoch = val_auc, (dropout, lr), epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Collapse guard: some (fold, seed) pairs never leave AUC~0.5 while others
    # reach 0.9+ -- retrain the winning combo with fresh seeds and keep the best.
    retries = 0
    while best_combo_auc < args.collapse_auc and retries < args.collapse_retries:
        retries += 1
        retry_seed = base_seed + 1000 * retries
        dropout, lr = best_combo
        _, epoch, val_auc, batch_size = train_config(
            build_fn(dropout), splits["train"], splits["validation"], device, retry_seed,
            args.epochs, args.patience, lr, batch_size, args.accum_steps, use_amp,
            args.cosine, args.aug_noise, args.aug_channel_dropout, args.min_epochs,
        )
        print(
            f"  fold {fold_idx} {foot}: collapse retry {retries} (seed {retry_seed}) "
            f"-> val AUC {val_auc:.4f} (epoch {epoch})",
            flush=True,
        )
        if val_auc > best_combo_auc:
            best_combo_auc, best_combo_epoch, best_seed = val_auc, epoch, retry_seed
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dropout, lr = best_combo
    refit_model, _, _, batch_size = train_config(
        build_fn(dropout), concat_splits(splits["train"], splits["validation"]), None, device,
        best_seed, max(best_combo_epoch, 1), None, lr, batch_size, args.accum_steps, use_amp,
        args.cosine, args.aug_noise, args.aug_channel_dropout,
    )

    test_split = splits["test"]
    y_score = predict_scores(refit_model, test_split.X, device, batch_size, use_amp)
    y_pred = (y_score >= 0.5).astype(int)

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(refit_model.state_dict(), ckpt_dir / f"mamba_{foot}_fold{fold_idx}_seed{args.seed}.pt")
    del refit_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return FoldResult(
        fold_idx=fold_idx, foot=foot, model_name="mamba",
        subject_id=test_split.subject_id, window_idx=test_split.window_idx,
        y_true=test_split.y, y_pred=y_pred, y_score=y_score,
        best_params={
            "dropout": dropout, "lr": lr, "best_epoch": best_combo_epoch,
            "val_auc": best_combo_auc, "batch_size": batch_size, "seed": best_seed,
        },
    )


# ---------------------------------------------------------------------------
# Reporting


def metrics_block(result: FoldResult, n_boot: int) -> dict:
    point = compute_point_metrics(result.y_true, result.y_pred, result.y_score)
    boot = summarize_bootstrap(bootstrap_metrics(result.y_true, result.y_pred, result.y_score, n_boot=n_boot))
    tp, fp, tn, fn = confusion_counts(result.y_true, result.y_pred)
    return {
        "point": {k: (None if np.isnan(v) else round(float(v), 4)) for k, v in point.items()},
        "bootstrap": boot,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n": int(len(result.y_true)),
    }


def save_predictions(result: FoldResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["subject_id,window_idx,y_true,y_pred,y_score"]
    lines += [
        f"{s},{w},{t},{p},{sc:.6f}"
        for s, w, t, p, sc in zip(result.subject_id, result.window_idx, result.y_true, result.y_pred, result.y_score)
    ]
    out_path.write_text("\n".join(lines) + "\n")


def save_tables_and_figures(all_results: dict, output_dir: Path) -> None:
    import pandas as pd

    for level in ("window", "subject_mean_prob"):
        rows = []
        for foot in FOOT_ORDER:
            if foot not in all_results:
                continue
            boot = all_results[foot][level]["bootstrap"]
            rows.append(
                {
                    "Data": foot.capitalize(),
                    "Acc (%)": boot["accuracy"],
                    "Pre (%)": boot["precision_w"],
                    "Rec (%)": boot["recall_w"],
                    "F1": boot["f1_w"],
                    "Sen (%)": boot["sensitivity"],
                    "Spe (%)": boot["specificity"],
                    "AUC": boot["auc"],
                }
            )
        if not rows:
            continue
        df = pd.DataFrame(rows)
        save_latex_table(
            df,
            output_dir / "tables" / f"mamba_{level}_results.tex",
            caption=f"Mamba results ({level.replace('_', ' ')} level, weighted Precision/Recall/F1, mean $\\pm$ std over 1000 bootstrap resamples).",
            label=f"tab:mamba_{level}",
        )

    for foot, blocks in all_results.items():
        for level in ("window", "subject_mean_prob"):
            c = blocks[level]["confusion"]
            cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            title = f"Mamba – {foot} ({'window' if level == 'window' else 'subject'} level)"
            plot_confusion_matrix(cm, title, output_dir / "figures" / f"confusion_{foot}_{level}.png")


# ---------------------------------------------------------------------------
# Main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, default=Path("outputs/dl/cache/raw_windows_v2.npz"))
    parser.add_argument("--fold-json", type=Path, default=None, help="Override path to fold_assignments.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mamba"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--foot", default="all", choices=["left", "right", "both", "all"])
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--model-size", default="base", choices=list(MODEL_SIZES))
    parser.add_argument("--variant", default="mamba2", choices=["mamba1", "mamba2"])
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=8, help="No early stop before this epoch")
    parser.add_argument("--collapse-auc", type=float, default=0.70, help="Retry fold when best val AUC is below this")
    parser.add_argument("--collapse-retries", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto from sequence length")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr-grid", default="1e-3,3e-4")
    parser.add_argument("--dropout-grid", default="0.1,0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cosine", dest="cosine", action="store_false", help="Cosine+warmup is on by default")
    parser.add_argument("--aug-noise", type=float, default=0.0)
    parser.add_argument("--aug-channel-dropout", type=float, default=0.0)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true", help="Tiny balanced subset, fold 1, 2 epochs")
    parser.add_argument("--require-cuda-kernels", action="store_true")
    args = parser.parse_args()
    args.lr_grid = [float(x) for x in args.lr_grid.split(",")]
    args.dropout_grid = [float(x) for x in args.dropout_grid.split(",")]
    args.folds = [int(x) for x in args.folds.split(",")]
    return args


def main() -> None:
    args = parse_args()
    t_start = time.time()

    if args.require_cuda_kernels and not HAVE_MAMBA_SSM:
        raise SystemExit("mamba_ssm is not importable but --require-cuda-kernels was set")

    if args.fold_json is not None:
        folds_mod.FOLD_ASSIGNMENTS_JSON = args.fold_json
        folds_mod.SPLITS_DIR = Path("/nonexistent-force-json")

    if args.smoke:
        args.folds = [1]
        args.epochs = 2
        args.patience = None
        args.min_epochs = 0
        args.collapse_retries = 0
        args.model_size = "small"
        args.lr_grid = args.lr_grid[:1]
        args.dropout_grid = args.dropout_grid[:1]
        args.n_boot = 50
        if args.batch_size <= 0:
            args.batch_size = 16

    model_kwargs = dict(MODEL_SIZES[args.model_size], d_state=args.d_state)
    device = resolve_device()
    print(
        f"device={device} amp={args.amp and device.type == 'cuda'} mamba_ssm={HAVE_MAMBA_SSM} "
        f"variant={args.variant} size={args.model_size} folds={args.folds}"
    )

    table = load_window_table(args.npz)
    if args.smoke:
        table = restrict_for_smoke(table, 1, {"train": 4, "validation": 2, "test": 2}, windows_per_subject=10)
        print(f"smoke: {len(table.subject_id)} windows, {len(set(table.subject_id))} subjects")

    feet = FOOT_ORDER if args.foot == "all" else [args.foot]
    all_results: dict[str, dict] = {}

    for foot in feet:
        fold_results = [run_fold_mamba(fold_idx, foot, table, args, model_kwargs) for fold_idx in args.folds]
        pooled = pool_fold_results(fold_results)

        if len(args.folds) == 5 and not args.smoke:
            counts = {s: int(np.sum(pooled.subject_id == s)) for s in set(pooled.subject_id)}
            if len(counts) != 100:
                raise RuntimeError(f"Pooled test covers {len(counts)} subjects, expected 100")

        blocks = {
            "window": metrics_block(pooled, args.n_boot),
            "subject_mean_prob": metrics_block(aggregate_subjects(pooled, "mean_prob"), args.n_boot),
            "subject_majority": metrics_block(aggregate_subjects(pooled, "majority"), args.n_boot),
            "best_params_per_fold": {str(f.fold_idx): f.best_params for f in fold_results},
        }
        all_results[foot] = blocks

        save_predictions(pooled, args.output_dir / "predictions" / f"mamba_{foot}_pooled.csv")
        save_predictions(
            aggregate_subjects(pooled, "mean_prob"),
            args.output_dir / "predictions" / f"mamba_{foot}_subjects.csv",
        )
        fmt = lambda v: "nan" if v is None else f"{v:.3f}"
        win_acc = blocks["window"]["point"]["accuracy"]
        subj = blocks["subject_mean_prob"]["point"]
        print(
            f"[{foot}] window acc={fmt(win_acc)} | subject acc={fmt(subj['accuracy'])} "
            f"prec_w={fmt(subj['precision_w'])} rec_w={fmt(subj['recall_w'])} f1_w={fmt(subj['f1_w'])} "
            f"auc={fmt(subj['auc'])}"
        )

    save_tables_and_figures(all_results, args.output_dir)

    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mamba_ssm": HAVE_MAMBA_SSM,
    }
    if HAVE_MAMBA_SSM:
        import mamba_ssm

        env["mamba_ssm_version"] = mamba_ssm.__version__

    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {"env": env, "args": args_json, "results": all_results, "runtime_sec": round(time.time() - t_start, 1)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"done in {payload['runtime_sec']}s -> {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
