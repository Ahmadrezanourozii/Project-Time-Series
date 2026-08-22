# CLAUDE.md — Project knowledge base

**Project**: Time Series (SS 26), FAU Erlangen-Nürnberg — Parkinson's disease (PD)
vs healthy control (HC) classification from gait vertical ground reaction force
(VGRF) signals.
**Owner**: Ahmadreza Nourozi (matriculation 23726011, M.Sc. AI).
**Supervisor**: Tomás.
**Last updated**: 2026-08-22. **The project is finished and submitted-ready.**

This file is the single place to re-read before touching the project again.
It records what was tried, what the numbers were, what broke, and what is still
open — so no experiment needs to be repeated.

---

## 1. Current status (read this first)

| Item | State |
|---|---|
| Deliverable 1 (data analysis) | Done, submitted (`DataAnalysis.AhmadrezaNourozi.pptx`) |
| Deliverable 2 (baselines) | Done, submitted (`Baseline Models.AhmadrezaNourozi.pptx`) |
| Deliverable 3 (Mamba) | **Done.** ~20 GPU sessions, 48 configurations. Reported model: 60 s amplitude tokens, base encoder, 3 seeds → **75 % accuracy, 0.85 AUC** |
| Short report | **Done and compliant** — `report/Template.pdf`, 4 content pages + 1 reference page |
| Presentation | **Done** — `Mamba Results.AhmadrezaNourozi.pptx`, 15 slides for 15 min |
| Inference on unseen data | **Done and tested on Kaggle** — `predict.py`, bundle dataset, standalone notebook |
| Leakage audit | **Done** — `scripts/audit_leakage.py`, 34 checks, all pass |
| Improvement attempt | **Done, negative** — see §11. The ceiling is reached; do not re-run |

**Nothing is open.** The only remaining action is external: rotate the Kaggle
API token (§6).

**Supervisor feedback (2026-08-19)**: *"80% sounds good. I will later ask everyone
to try inference in a new set."* → the subject-wise result is accepted, and the
inference path is a real requirement. It is built and verified (§7). Note the
number later became 75 % after the selection-bias fix (§2); the drop is a
correction, not a regression, and the reasoning is in the report.

---

## 2. The result

Subject-level, both feet, five predefined subject-wise folds pooled over 100
subjects; mean ± std over 1000 bootstrap resamples. All models evaluated with
identical code and the same aggregation rule.

| Model | Acc (%) | Pre (%) | Rec (%) | F1 | Sen (%) | Spe (%) | AUC |
|---|---|---|---|---|---|---|---|
| SVM-RBF | 68 ± 5 | 69 ± 5 | 68 ± 5 | 0.68 | 66 | 70 | 0.77 |
| 1D-ResNet | 66 ± 5 | 67 ± 5 | 66 ± 5 | 0.66 | 56 | 76 | 0.76 |
| LSTM | 63 ± 5 | 64 ± 5 | 63 ± 5 | 0.63 | 54 | 72 | 0.68 |
| Random Forest | 73 ± 4 | 74 ± 4 | 73 ± 4 | 0.73 | 74 | 72 | 0.82 |
| **Mamba (ours)** | **75 ± 4** | **77 ± 4** | **75 ± 4** | **0.75** | 86 | 64 | **0.85** |

Per foot (Mamba, 3 seeds): left 74, right 71, both 75. Random Forest is better
on the right foot (78) and level on the left (74), so the win is on both feet
and on AUC, not across the board.
Confusion (both feet, subject level): TP 43, FN 7, TN 32, FP 18.

**Which configuration is "the" model, and why it changed.** The reported model
is 60 s amplitude-statistic tokens with the base encoder, 3 seeds
(`outputs/mamba/final/primary_model/`). It is *not* the configuration with the
best test accuracy — that was 30 s tokens with the large encoder at 0.78–0.80.
The leakage audit (§5, problem 12) showed that picking among 45 configurations
by pooled test accuracy leaks the test set into the model definition. Ranking
the same 45 by mean validation AUC picks the 60 s/base model instead, and its
test accuracy of 0.75 is the number that estimates performance on new subjects.
The ~3-point difference is the optimism a test-based search would have added.
Both criteria agree on the part that matters: amplitude-statistic tokens win.

### The two-protocol story (the project's main methodological finding)

| Model | Subject-wise (leakage-free) | Window-wise (leaky) |
|---|---|---|
| SVM-RBF | 68 | 88 |
| Random Forest | 73 | 98 |
| Mamba, raw 5 s windows | 66–69 | **99.7** |

Under the window-wise split, windows of one subject land in both train and
test. The model then recognises the *person*, not the disease. This explains
why published accuracies on this database span 70–99 %. A fully random window
split with RF on the cached feature matrix gives **100.0 %** — total leakage.

Note: the originally requested target (Acc 88.6 / Pre 89.7 / Rec 88.6 /
F1 0.885) is almost exactly the **SVM-RBF leaky number (88 %)**. Those targets
therefore very likely came from a window-level evaluation, not a
subject-disjoint one. This was checked, not assumed.

---

## 3. Data

- **Source**: PhysioNet *Gait in Parkinson's Disease* (studies Ga, Ju, Si).
- **Subjects**: 100 (50 PD / 50 HC), only the normal-walk `_01` recordings.
  Dual-task `_10` recordings exist in the wild but are **not** in `Data/`.
- **Signals**: 8 sensors per foot + 2 per-foot totals = 18 channels, 100 Hz,
  ~2 min per subject (3 918–25 865 samples after trimming; median 11 618).
- **Folds**: given by the course, `Data/splits/Fold_{1..5}/{training,validation,test}`,
  60/20/20 subjects, class-balanced, test sets disjoint → pooling the five test
  folds covers each subject exactly once.
- `Data/` is 489 MB and **not** in git. `metadata/fold_assignments.json` +
  `metadata/labels.csv` mirror the split structure so Kaggle needs no raw data.

### Cached representations (all in `outputs/dl/cache/`, all channels-first `(N, C, T)`)

| File | Shape | What it is |
|---|---|---|
| `raw_windows_v2.npz` | (10407, 18, 500) | raw 5 s windows, 1 s step — the Deliverable-2 DL input |
| `windows_30s.npz` / `windows_60s.npz` | (827, 18, 3000) / (363, 18, 6000) | raw long windows |
| `sequences_full.npz` | (100, 18, 11000) | one 110 s raw sequence per subject |
| **`stat_windows_30s.npz`** | **(827, 198, 30)** | **amplitude-statistic tokens — the winning input** |
| `stat_windows_60s.npz` | (363, 198, 60) | same, 60 s windows |
| `stat_windows_{15,30,60}s_dense.npz` | (9407/3996/1309, 198, T) | dense steps (more training windows) |
| `subject_feature_seq.npz` | (100, 198, 254) | one repeat-padded sequence per subject |
| `stride_windows_{20,40}.npz` | (1576/612, 16, T) | stride-cycle tokens |
| `stride_seq_full.npz` | (100, 16, 166) | one stride sequence per subject |
| `fused_windows_30s{,_dense}.npz` | (827/3996, 214, 30) | amplitude ++ stride tokens |

**Amplitude-statistic token** = one token per second of signal, holding 11
statistics (mean, std, skew, kurtosis, RMS, zero-crossing rate, median, min,
max, spectral energy, spectral entropy) for each of the 18 channels → 198 dims.
Built by `scripts/build_feature_sequences.py` (~5 min, recomputes per-second
tokens for all 100 recordings).

**Stride-cycle token** = one token per gait cycle: stride/stance/swing time,
stance ratio, normalised peak force, impulse, stride-time increment,
double-support time — for both feet (16 dims). Foot contacts detected by
thresholding per-foot total force at 5 % of max; strides outside 0.5–2.5 s
rejected. 34–166 strides per subject (median 101). Built by
`scripts/build_stride_sequences.py`.

---

## 4. Complete experiment log

All raw results and kernel logs are archived under
`outputs/mamba/experiment_archive/kaggle_run{6..16}/`. 46 real configurations
were run (plus smoke tests). Subject-level accuracy on **both feet** unless noted.

### 4.1 Input representation — the single biggest factor

| Input | Subject acc | Worst-fold val AUC | Run |
|---|---|---|---|
| Raw samples, 5 s | 0.66 → 0.69 | **0.53** (collapse) | run6, run7 |
| Raw samples, 30 s | 0.69 | 0.63 | run8 |
| Raw samples, 60 s | 0.65 | 0.59 | run8 |
| Stride-cycle tokens, 20/40 strides | 0.59 / 0.66 | 0.71 | run13 |
| Subject-level feature sequence | 0.70 | 0.72 | run9 |
| Fused (amplitude ++ stride) | 0.72–0.79 | 0.71 | run13, run14 |
| **Amplitude statistics, 30 s** | **0.76 → 0.80** | **0.75–0.77** | run9, run10, run12 |
| Amplitude statistics, 60 s | 0.75–0.77 | 0.73 | run9, run10 |
| Dense amplitude (15 s / 30 s) | 0.74–0.78 | — | run11 |

**Finding**: raw signals make training collapse on folds 1 and 4 (validation
AUC ≈ 0.53 while other folds exceed 0.92). Neither longer windows, nor seed
restarts, nor a bigger encoder repaired it. Per-second statistics remove the
collapse entirely and add ~7 accuracy points. Denser window steps (5× more
training windows) did **not** help further — the bottleneck is the number of
subjects, not the number of windows.

### 4.2 Architecture

| Factor | Result |
|---|---|
| Encoder size base (d=128, L=4) → large (d=256, L=6) | 0.72 → **0.80** (the real gain) |
| Bidirectional vs causal, **matched size (base)** | 0.72 vs 0.72; causal AUC **0.866** vs 0.822 |
| Model `small` (d=64, L=2) | 0.73 |
| Mamba2 (Triton kernels) | used throughout; Mamba1 not benchmarked |

**Important correction**: an earlier note claimed bidirectionality was worth
+8 points. That comparison was invalid — it compared a *base causal* model with
a *large bidirectional* one, confounding size with direction. At matched size
there is no benefit, and the causal variant ranks subjects slightly better.
The reported model still uses the bidirectional block (it was the default when
the final runs were made), but the report presents this as a negative result.

### 4.3 Training recipe

| Setting | Value | Why |
|---|---|---|
| Optimiser | AdamW, weight decay 1e-3 | — |
| LR grid | 3e-4, 1e-4 | 1e-3 won only on collapsed folds |
| Dropout grid | 0.1, 0.3 | 0.5 did not help (run14) |
| Schedule | 3 warm-up epochs + cosine | added after collapses |
| Epochs | ≤ 30, patience 6, min 8 | best epoch is usually **1–4**: the model overfits almost immediately |
| Batch | 128 (auto-scaled by sequence length, halved on OOM) | |
| Seeds | 42, 43, 44 for the reported model | single seeds range 0.72–0.80 |
| Collapse guard | retrain fold with a fresh seed if val AUC < 0.70 (≤3 attempts) | |
| AMP | on (CUDA) | |

**Finding**: heavier regularisation did not beat the plain recipe; ensembling
5 seeds was not better than 3; late fusion across token families (run13, run14)
scored 0.72–0.77, i.e. **below** the best single model — diversity did not help
here.

### 4.4 Decision rule

- Subject score = mean of its window scores; decision threshold calibrated per
  fold on that fold's **validation** subjects (maximising balanced accuracy)
  using the model trained on training data only. Test data never involved.
- Calibrated vs fixed 0.5 threshold: differs by ≤ 2 points either way.
- Majority vote over window predictions ≈ mean-probability rule (±1 point).

### 4.5 What was tried at subject level with classical models (sanity checks, local)

| Features | Subject acc | AUC |
|---|---|---|
| Stride-cycle summary (mean/std/CV) | 0.69–0.72 | 0.79–0.80 |
| Amplitude window features (mean/std per subject) | 0.72 | 0.83 |
| Both combined | 0.74 | **0.86** |
| Canonical PD markers added (DFA, autocorrelation, RMSSD, asymmetry) | 0.74 | 0.85 |

**Finding**: even the clinically canonical stride-variability markers cap around
AUC 0.85 under subject-disjoint evaluation. The Mamba result (AUC 0.82,
accuracy 0.78) sits at that ceiling. ~0.80 is the realistic limit of this
dataset with 100 subjects and this protocol — not a tuning failure.

---

## 5. Problems hit and how they were fixed

| # | Problem | Diagnosis | Fix |
|---|---|---|---|
| 1 | `pip`/`kaggle` shebangs pointed at an old venv path | venv was copied from `Deliverable2Baselinemodels/` | always call `.venv/bin/python -m pip`, never `.venv/bin/pip` |
| 2 | `kaggle datasets create --private` rejected | new CLI: private is the default, flag removed | drop the flag |
| 3 | Kernel ran on **P100**, unsupported by torch/mamba kernels (sm_60) | `enable_gpu` alone does not pin the accelerator | `"machine_shape": "NvidiaTeslaT4"` in kernel metadata **and** `--accelerator NvidiaTeslaT4` on push; assert `get_device_capability() >= (7,5)` in cell 1 |
| 4 | Dataset not mounted at `/kaggle/input/<slug>` | Kaggle now nests under `/kaggle/input/datasets/...` | walk `/kaggle/input` to find `raw_windows_v2.npz` |
| 5 | `mamba-ssm==2.2.2` had no wheel → source build | pinned version too old for torch 2.10/cu128 | install **latest** first: gets `mamba-ssm 2.3.2.post1` + `causal-conv1d 1.6.2.post1` prebuilt |
| 6 | Batch session died mid-run, outputs lost | too much work in one session; `num_workers=2` forked workers | `num_workers=0`, bound each run's scope, `PYTHONUNBUFFERED=1` |
| 7 | Training collapse on folds 1 & 4 (val AUC ≈ 0.53) | raw signal + few subjects → degenerate optimum | feature tokens (real fix) + warm-up/cosine + min-epochs + seed-retry guard |
| 8 | Multi-seed runs crashed in leaky mode: *"Seed runs produced different row orders"* | the random window split was seeded with the **model** seed | split RNG seeded by fold only — the split is protocol, not model |
| 9 | Long windows dropped 5 short recordings (95 subjects) | `Ju` recordings are as short as 39 s | zero-pad recordings shorter than one window |
| 10 | LaTeX missing (`pdflatex not found`); BasicTeX needs sudo | — | TinyTeX in `~/Library/TinyTeX` (no sudo), then `tlmgr install multirow booktabs psnfss courier times helvetic` |
| 11 | A fabricated number (97 %) sat in the report's protocol table | written before that run finished | replaced with `---`; all surrounding claims rewritten to cite only measured baselines |
| 12 | **Configurations were compared on pooled test accuracy** | each run kept train/val/test apart, but the *choice between runs* used test | rank all 45 by mean validation AUC instead; retrain the winner. Costs ~3 accuracy points |
| 13 | `predict.py` built 30-token windows for a 60-token model | Mamba accepts any T, so nothing errored | window length travels in `manifest.json`; inference refuses to guess |
| 14 | Bundle would not load without `mamba_ssm` | fallback block has a different parameter layout | explicit one-line error pointing at the Kaggle notebook |

### The audit itself

`scripts/audit_leakage.py` checks the invariants empirically rather than by
reading code — it reproduces the test split by hand from training statistics and
compares against what the pipeline produced, confirms the same split is *not*
reproducible from all-data statistics, recomputes one recording's tokens in
isolation, and greps the threshold block for any reference to the test split.
Run it before submitting anything: `python scripts/audit_leakage.py`.

---

## 6. Infrastructure (how to run anything)

### GitHub → Kaggle loop
- Repo: <https://github.com/Ahmadrezanourozii/Project-Time-Series> (public).
  Auth via `gh auth login` + `gh auth setup-git` (the SSH key on this machine is
  *not* registered with GitHub).
- Kaggle dataset (private): `ah22reza/pd-gait-vgrf-windows` — all npz caches +
  `fold_assignments.json` + `labels.csv` + demographics.
- Kaggle notebook: `ah22reza/pd-mamba-train`.
- Kaggle account is **ah22reza** (not the GitHub name).
- Token: use `export KAGGLE_API_TOKEN=KGAT_...` in the shell only. **The token
  used during development was pasted into a chat and should be rotated.**

```bash
# push code, then trigger a batch run
git push
python -m kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4
python -m kaggle kernels status ah22reza/pd-mamba-train
python -m kaggle kernels output ah22reza/pd-mamba-train -p /tmp/run
# add/refresh data
python -m kaggle datasets version -p <staging dir> -m "message"
```

Interactive editing in the Kaggle UI is much faster for iteration: after the
first run the session stays warm, so `git pull` + one training cell takes
seconds instead of a 3–6 min cold start.

### Timings on a Kaggle T4
- amplitude tokens, 5 folds, 1 foot, 2×2 grid: **~2 min**
- same with 3 seeds, 3 feet: ~20 min
- raw 5 s windows, 5 folds, 1 foot: **~100 min** (this is why raw input is expensive)
- raw 5 s, leaky split, 3 feet: 5.4 h

### Local commands
```bash
python run_mamba.py --smoke --foot both              # 2-min end-to-end check (CPU/MPS)
python run_mamba.py --foot all                        # full subject-wise run
python run_mamba.py --foot both --split-mode window   # leaky reference protocol
python scripts/baseline_subject_metrics.py            # baselines, both protocols
python scripts/make_report_figures.py                 # report figures (PDF)
python scripts/make_presentation.py                   # rebuild the deck
```
Locally `mamba_ssm` is unavailable, so `mamba_minimal.py` (pure-PyTorch
reference scan) is used — correct but slow, smoke tests only. Pass
`--require-cuda-kernels` on Kaggle to fail loudly if the fast kernels are absent.

---

## 7. Inference on unseen data (the supervisor's next request)

`run_mamba.py --export-bundle DIR` writes one `.pt` per fold containing the
weights **plus the per-channel mean/std of that fold's training subjects**, the
model config and the calibrated threshold, and a `manifest.json`.

```bash
python predict.py --bundle outputs/mamba/final/bundle --input <dir of new .txt> \
                  --foot both --out predictions.csv
python predict.py ... --labels metadata/labels.csv   # also scores it, if labels exist
```

`predict.py` reads raw PhysioNet-format `.txt` (19 columns, 100 Hz), drops the
first 5 s, builds per-second statistic tokens, cuts 30-token windows, applies
each fold's stored normalisation, averages window scores across folds, then
averages per recording.

**Verified**: the windows `predict.py` rebuilds from a raw file are
*byte-identical* (max abs diff 0.0) to the cached training windows for the same
subject, for both `both` and `left` channel selections. This is the check that
matters — if tokenisation drifted, inference would be silently wrong. The full
path was then run on Kaggle over 20 raw recordings and produced sensible,
well-separated scores (controls 0.03–0.23, patients 0.79–0.97).

**Read this before quoting any number from `predict.py`.** The bundle holds one
model per (fold, seed). For a genuinely new subject that is exactly right: no
fold model has seen them, so averaging all five is a legitimate ensemble. But
the 20 recordings used for the smoke test were fold 1's test subjects, which
means folds 2–5 had them in *training*. The resulting 100 % accuracy is
therefore meaningless as a performance estimate — it only shows the path runs
end to end. The honest estimate for unseen subjects stays the 75 % of
Table~2. The same effect explains the mean score gap of 0.20 against
`run_mamba.py`'s predictions, which use only the fold that held each subject
out.

**Never** recompute normalisation statistics from the new cohort: that leaks the
test distribution into its own predictions.

Two defects that testing caught here, both of which would have failed silently:
- `predict.py` hardcoded 30-token windows while the deployed model uses 60. A
  Mamba encoder accepts any sequence length, so this produced plausible but
  wrong scores. The window length now travels in `manifest.json` and inference
  refuses to guess it.
- Loading a bundle without `mamba_ssm` failed with a wall of shape mismatches,
  because the pure-PyTorch fallback block has a different parameter layout.
  Inference now says so in one line. Run it where the CUDA kernels exist.

---

## 8. Repository map

```
baseline/       classical pipeline: config, labels, features, folds, metrics, reporting
dl/             shared DL pipeline: config, windows, datasets, models, train, aggregate, seeding
mamba_model.py  MambaClassifier (+ BiMambaLayer); mamba_minimal.py = CPU fallback
run_mamba.py    main entry point: both protocols, both evaluation levels, bundle export
predict.py      inference on unseen recordings
scripts/        export_fold_assignments, build_{feature,stride,fused,long}_sequences,
                baseline_subject_metrics, fuse_predictions, make_report_figures, make_presentation
kaggle/         mamba-train.ipynb + kernel-metadata.json
metadata/       fold_assignments.json, labels.csv  (committed; Kaggle needs no raw data)
report/         LaTeX short report → Template.pdf
outputs/        NOT in git: caches, predictions, figures, experiment_archive/
```

Key implementation details worth remembering:
- `baseline/folds.py::load_fold_assignment` falls back to
  `metadata/fold_assignments.json` when `Data/splits` is absent (this is how
  Kaggle runs without the 489 MB dataset).
- `baseline/metrics.py` returns PD-positive *and* class-weighted precision /
  recall / F1. Weighted recall equals accuracy by construction — that identity
  is asserted, and it is why the requested target had Recall == Accuracy.
- `dl/windows.py::select_foot_channels` matches feature channels by prefix
  (`L1__mean` → `L1`), so left/right/both work for raw and token tables alike.
- Metrics are bootstrapped 1000× (subjects for subject-level, windows for
  window-level).

---

## 9. Deliverables and what still needs your input

- `report/Template.pdf` — 4 content pages + 1 reference page, ICASSP template.
  **The author footnote still needs your home address and date of birth.**
- `Mamba Results.AhmadrezaNourozi.pptx` — 14 slides, FAU template, ~1 min/slide.
- `outputs/mamba/final/` — predictions, metrics, tables, report figures.
- `outputs/mamba/experiment_archive/` — every run's `results.json` and kernel log.

### Open items
1. The last Kaggle run (kernel v17) produces: the deployable **inference
   bundle**, the clean one-factor **ablations** (uni vs bi at matched size,
   mean vs mean+max pooling), and the **leaky number for the reported model**
   (the `---` cell in report Table 3). Fetch with
   `kaggle kernels output ah22reza/pd-mamba-train -p /tmp/run17`.
2. Rotate the Kaggle API token.
3. If more accuracy is ever needed, the honest lever is **more subjects**
   (dual-task recordings, other VGRF cohorts) — not more tuning. Everything
   tried here plateaus at 0.78–0.80 because the ceiling is the sample size.

---

## 10. Things not to redo

- Do **not** feed raw 100 Hz samples to Mamba expecting better results — tried at
  5 s, 30 s, 60 s and full 110 s sequences; all worse and two folds collapse.
- Do **not** chase denser window steps — 5× more windows changed nothing.
- Do **not** expect late fusion / more seeds to break 0.80 — both tried, both flat.
- Do **not** expect stride-cycle tokens alone to work — 0.59–0.66, though they
  are complementary to amplitude features in a *subject-level* RF (AUC 0.86).
- Do **not** compare our numbers with papers that split by window — reproduce
  their protocol first (`--split-mode window`) and the numbers jump to 88–100 %.

---

## 11. The improvement attempt (2026-08-22) — negative, do not repeat

After the report was written we asked whether 75 % could be beaten. The answer
is no, and the evidence is strong enough that further search is not worth GPU
time.

**What was tried**, all judged by mean validation AUC, never by test:

| Candidate | Validation AUC | Verdict |
|---|---|---|
| incumbent (60 s amplitude tokens, base) | 0.873 | reference |
| larger encoder | 0.885 | ΔAcc −0.030, CI contains 0 |
| subject-balanced sampling (`--balance-subjects`) | 0.879 | ΔAcc −0.010, CI contains 0 |
| duty-cycle tokens, 60 s | 0.838 | worse |
| duty-cycle tokens, 30 s | 0.841 | worse |

`duty60` scored 0.81 on **test** — tempting, and exactly the trap the selection
audit exists to prevent. Its validation AUC is in the bottom tier, so it was
rejected.

**Three measurements that explain the ceiling:**

1. **A single scalar matches the deep model.** The fraction of the recording
   during which the right foot carries load — one number, no learning, logistic
   regression per fold — reaches **0.77 accuracy / 0.830 AUC**. Mamba reaches
   0.77 / 0.854 at threshold 0.5, Random Forest 0.73 / 0.815. A paired bootstrap
   over the same 100 subjects puts *every* pairwise difference inside a 95 %
   interval containing zero. Four very different model classes land in a 0.04
   AUC band: the signal is saturated.
2. **The prescribed folds carry a study confound.** The three PhysioNet
   sub-studies have different PD rates (Ga 0.58, Ju 0.50, Si 0.43) and the folds
   are not stratified by study. A classifier using only "which study" has signal
   in training but scores **AUC 0.46 at test** — below chance. Capacity spent on
   study identity is actively subtracted from test performance, which explains
   the fold-1/4 collapses and the seed variance better than any optimiser story.
3. **n = 100 is the binding constraint.** The 95 % CI on 0.75 accuracy is
   [0.67, 0.83]. Every configuration ever run, from 0.65 to 0.80, falls inside
   one interval. Detecting a true five-point gain at 80 % power would need
   roughly 1200 subjects.

**Two claims that did not survive verification.** An expert-review agent
suggested (a) fusing Mamba with the stance scalar reaches AUC 0.898 and (b) duty
features are the missing ingredient. Reproducing (a) gave AUC 0.856 against
0.854 — ΔAUC +0.002, nothing. For (b), a cheap subject-level probe *before*
spending GPU time showed 0.849 with the duty features versus 0.850 without.
Both were rejected on our own measurements. **Always reproduce a suggested
number before acting on it.**

**Honest stopping criterion, now met:** stop when the paired-bootstrap CI of
(candidate − incumbent) contains zero for every candidate in the last rounds.
It does, for all 48.

---

## 12. Quality-control findings (2026-08-22)

A sentence-by-sentence pass over the report caught real errors. Re-run these
checks if the report is ever edited again.

| Found | Fix |
|---|---|
| Methods said "the reported model is the largest" encoder | It is the middle one (d=128, L=4). Corrected |
| Methods said all three representations "share the above windowing" (5 s) | False: statistic tokens are grouped into 30 s / 60 s windows. Rewritten, and the pipeline figure updated |
| Figures printed a literal `\%` in axis labels and subtitles | matplotlib does not interpret LaTeX escapes; use a plain `%` |
| Experiment 1 quoted 80 % without qualification | It is the best configuration of that representation, not the reported model (75 %). Now stated |
| Run counts stale ("fifteen sessions", "forty-five configurations") | Now twenty and forty-eight |
| The report mixed conventions: baselines from bootstrap means, Mamba from point estimates (LSTM read 0.68 vs a true 0.673) | `scripts/fill_report_tables.py` now writes **every** cell of the results table from JSON with one convention |
| Deck and report disagreed on AUC (0.86 vs 0.85) | Deck now reads the same files with the same rounding; a check confirms cell-by-cell agreement |
| Window-level bootstrap resampled rows independently | Windows of one subject are near-duplicates; `bootstrap_metrics(..., groups=...)` now resamples whole subjects. Intervals were 2× too narrow |

### Compliance state of `report/Template.pdf`

Verified with `pdffonts`, `pdfimages`, `pdfinfo`, not by assumption:

- 4 content pages + 1 page holding only references
- 16/16 fonts embedded **and** subsetted (the IEEE spec's main requirement)
- zero raster images — all figures are vector, so the 300 dpi rule cannot be violated
- not encrypted; abstract 145 words; five keywords
- every figure and table referenced; no undefined references; no overfull boxes
- author footnote complete (name, email, matriculation, programme, address, date of birth)

### Commands to re-verify everything

```bash
python scripts/audit_leakage.py                 # 34 leakage checks
python scripts/fill_report_tables.py            # regenerate the results table from JSON
python scripts/make_report_figures.py           # regenerate all figures (vector)
python scripts/make_presentation.py             # rebuild the deck from the same JSON
cd report && pdflatex Template && bibtex Template && pdflatex Template && pdflatex Template
pdffonts Template.pdf ; pdfimages -list Template.pdf ; pdfinfo Template.pdf
```

---

## 13. If you come back to this project

**Start with `RESUME.md`** — it holds the prompt to paste into a new session and
lists what is not stored in the repo (Kaggle token, `Data/`, `outputs/`).

1. Read §1, §2, §10 and §11 first. They tell you the result, the ceiling and
   what not to retry.
2. The deliverables are `report/Template.pdf` and
   `Mamba Results.AhmadrezaNourozi.pptx`. Both are generated from
   `outputs/mamba/final/` — never edit numbers by hand; run the scripts in §12.
3. If a new test set arrives, use the standalone Kaggle notebook
   `ah22reza/pd-mamba-predict` with the bundle dataset `ah22reza/pd-mamba-bundle`
   attached. Expect roughly 75 % on genuinely unseen subjects.
4. If the goal is a better number, the lever is **more subjects**, not more
   tuning — §11 explains why in measured terms.
5. Rotate the Kaggle token before doing anything else if it has not been done.
