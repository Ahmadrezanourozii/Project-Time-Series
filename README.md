# Project Time Series (SS 26) — PD vs HC gait classification

Deliverable 3: classification with a Mamba state-space model, trained on Kaggle GPU.

## Headline result

Subject-level, both feet, five predefined subject-wise folds pooled
(100 subjects, mean ± std over 1000 bootstrap resamples):

| Model | Acc (%) | Pre (%) | Rec (%) | F1 | Sen (%) | Spe (%) | AUC |
|---|---|---|---|---|---|---|---|
| SVM-RBF | 68 ± 5 | 69 ± 5 | 68 ± 5 | 0.68 | 66 | 70 | 0.77 |
| 1D-ResNet | 66 ± 5 | 67 ± 5 | 66 ± 5 | 0.66 | 56 | 76 | 0.76 |
| LSTM | 63 ± 5 | 64 ± 5 | 63 ± 5 | 0.63 | 54 | 72 | 0.68 |
| Random Forest | 73 ± 4 | 74 ± 4 | 73 ± 4 | 0.73 | 74 | 72 | 0.82 |
| **Mamba (ours)** | **78 ± 4** | **79 ± 4** | **78 ± 4** | **0.78** | 82 | 74 | 0.82 |

Per foot, Mamba reaches 70 % (left), 79 % (right), 78 % (both).

Under a **window-wise** split — windows of one subject in both train and test,
the protocol implicit in much of the literature — the same pipeline reports
88 % (SVM), 98 % (RF). Those numbers measure subject recognition, not disease
detection, and are reported in the paper only as a contrast.

## Repository layout

```
baseline/        classical pipeline (features, folds, metrics, reporting)
dl/              shared DL pipeline (datasets, models, training, aggregation)
mamba_model.py   MambaClassifier (mamba_ssm on CUDA, pure-PyTorch fallback locally)
run_mamba.py     entry point — both protocols, both evaluation levels
scripts/         cache builders, baseline recomputation, figures, deck
kaggle/          notebook + kernel metadata for the GPU runs
metadata/        committed fold assignments + labels (no raw data needed on Kaggle)
report/          LaTeX short report (Template.pdf = 4 pages + references)
```

Data (489 MB) and results (`outputs/`) are deliberately not tracked by git.

## Reproducing a run on Kaggle

1. Push code changes: `git push`
2. In the notebook <https://www.kaggle.com/code/ah22reza/pd-mamba-train>,
   run cell 1 (pulls the latest commit) and the training cell.
   A single configuration takes about 2–10 minutes on a T4.

Batch runs from this machine:

```bash
export KAGGLE_API_TOKEN=...            # never commit this
python -m kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4
python -m kaggle kernels status ah22reza/pd-mamba-train
python -m kaggle kernels output ah22reza/pd-mamba-train -p /tmp/run
```

Data lives in the private Kaggle dataset `ah22reza/pd-gait-vgrf-windows`;
rebuild and re-upload the caches with:

```bash
python scripts/export_fold_assignments.py     # metadata/
python scripts/build_feature_sequences.py     # amplitude-statistic tokens
python scripts/build_stride_sequences.py      # stride-cycle tokens
python scripts/build_fused_sequences.py       # fused tokens
python -m kaggle datasets version -p <staging dir> -m "message"
```

## Running locally

```bash
python run_mamba.py --smoke --foot both      # 2-minute end-to-end check (CPU/MPS)
python run_mamba.py --foot all               # full subject-wise run
python run_mamba.py --foot both --split-mode window   # leaky reference protocol
python scripts/baseline_subject_metrics.py   # baselines under both protocols
```

Locally `mamba_ssm` is unavailable, so a pure-PyTorch reference block is used;
it is correct but slow and intended for smoke tests only. Pass
`--require-cuda-kernels` on Kaggle to fail loudly if the fast kernels are missing.

## Deliverables

- `report/Template.pdf` — short report (4 content pages + references).
  The author footnote still needs your home address and date of birth.
- `Mamba Results.AhmadrezaNourozi.pptx` — 14-slide deck for the 15-minute talk.
- `outputs/mamba/final/` — predictions, metrics, tables and figures behind both.
