# Project Time Series (SS 26) — PD vs HC gait classification

Deliverable 3: classification with a Mamba state-space model, trained on Kaggle GPU.

## Final report and presentation

- [Final report (PDF)](final_report/Final_Report_Ahmadreza_Nourozi.pdf) — *Parkinson's disease detection from gait force signals with a state-space (Mamba) model: results and the role of the evaluation protocol* (4 pages + references, IEEE format)
- [Final presentation (PDF)](final_report/Final_Presentation_Ahmadreza_Nourozi.pdf) — 15-minute talk, 17 slides

## Headline result

Subject-level, both feet, five predefined subject-wise folds pooled
(100 subjects, mean ± SD over 1000 bootstrap resamples of subjects).
The Mamba configuration was selected on mean validation ROC-AUC, never on test data.

| Model | Acc (%) | Pre (%) | F1 | Sen (%) | AUC |
|---|---|---|---|---|---|
| SVM-RBF | 68 ± 5 | 69 ± 5 | 0.68 | 66 | 0.77 |
| 1D-ResNet | 66 ± 5 | 67 ± 5 | 0.66 | 56 | 0.76 |
| LSTM | 63 ± 5 | 64 ± 5 | 0.63 | 54 | 0.67 |
| Random Forest | 73 ± 4 | 74 ± 4 | 0.73 | 74 | 0.81 |
| **Mamba (ours)** | **75 ± 4** | **77 ± 4** | **0.75** | **86** | **0.85** |

Per foot, Mamba reaches 74 % (left), 71 % (right) and 75 % (both).
DeLong's test puts Mamba clearly ahead of SVM-RBF, 1D-ResNet and LSTM on AUC,
and statistically level with the tuned Random Forest (p = 0.15). The main gain
comes from feeding per-second channel statistics instead of raw samples, which
also removed a training collapse seen on two folds with raw input.

Under a **window-wise** split — windows of one subject in both train and test,
the protocol implicit in much of the literature — the same baselines report
88 % (SVM-RBF) and 98 % (Random Forest). Those numbers measure subject
recognition, not disease detection, and are reported only as a contrast.

## Repository layout

```
baseline/        classical pipeline (features, folds, metrics, reporting)
dl/              shared DL pipeline (datasets, models, training, aggregation)
mamba_model.py   MambaClassifier (mamba_ssm on CUDA, pure-PyTorch fallback locally)
run_mamba.py     entry point — both protocols, both evaluation levels
scripts/         cache builders, baseline recomputation, figures, deck
kaggle/          notebook + kernel metadata for the GPU runs
metadata/        committed fold assignments + labels (no raw data needed on Kaggle)
report/          LaTeX source of the short report
final_report/    submitted final report and presentation (PDF)
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

1. Data analysis (May 2026)
2. Baseline models — TSFresh features with Random Forest and SVM (linear / RBF), 5-fold CV (July 2026)
3. Mamba classifier, final report and presentation (September 2026) — see [`final_report/`](final_report/)

Predictions, metrics, tables and figures behind the report are written to
`outputs/mamba/final/` by `run_mamba.py` (not tracked by git).
