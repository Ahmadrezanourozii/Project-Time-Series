# How to restart work on this project

Paste the block below as your **first message** in a new session. Everything
else lives in `CLAUDE.md`; this file only exists so you do not have to remember
what to say.

---

## The starter prompt (copy this)

```
This is my Parkinson's-disease gait classification project (Time Series, SS26, FAU).
It is finished and submitted-ready. Before doing anything, read CLAUDE.md in this
folder end to end — it is the knowledge base with every experiment, result,
failure and decision from the previous sessions. Also read RESUME.md.

Key facts so you do not re-derive them:
- Final model: Mamba on 60 s amplitude-statistic tokens, base encoder, 3 seeds.
  75% subject-level accuracy, 0.85 AUC, no data leakage.
- The ceiling is reached. CLAUDE.md section 11 proves it with measurements.
  Do not restart the architecture/hyper-parameter search.
- Deliverables: report/Template.pdf and Mamba Results.AhmadrezaNourozi.pptx.
  Both are generated from outputs/mamba/final/ — never edit numbers by hand.

What I want to do now: <SAY WHAT YOU WANT HERE>

For Kaggle you will need this token (I generate a fresh one each time):
KAGGLE_API_TOKEN=<PASTE A FRESH TOKEN FROM kaggle.com > Settings > API>
```

Replace the last two lines with whatever you actually need. If the task does not
touch Kaggle, leave the token out entirely.

---

## What is NOT stored in the repo (you must supply it again)

| Thing | Where to get it |
|---|---|
| **Kaggle API token** | kaggle.com → Settings → API → Create New Token. Never commit it; export it in the shell only. |
| **GitHub auth** | Should still work via `gh auth status`. If not: `gh auth login` then `gh auth setup-git`. |
| **`Data/` (489 MB)** | Not in git by design. It is still on this machine at `Data/`. If lost, it is the PhysioNet *Gait in Parkinson's Disease* database plus the course-provided `splits/` folders. |
| **`outputs/` (caches, models, results)** | Not in git either. Still on this machine. If lost, rebuild with the scripts listed in CLAUDE.md §12 — but the raw `Data/` is required. |

---

## Things that live outside this machine

| Resource | Address |
|---|---|
| Code | <https://github.com/Ahmadrezanourozii/Project-Time-Series> (public) |
| Data (private Kaggle dataset) | `ah22reza/pd-gait-vgrf-windows` |
| Trained model bundle | `ah22reza/pd-mamba-bundle` |
| Training notebook | `ah22reza/pd-mamba-train` |
| Inference notebook | `ah22reza/pd-mamba-predict` |

Kaggle account is **ah22reza**; the GitHub account is **Ahmadrezanourozii**.

---

## The three most likely reasons you are back

**A new test set arrived.** Open `ah22reza/pd-mamba-predict` on Kaggle with the
bundle dataset attached, upload the new `.txt` recordings as a dataset, set
`NEW_DATA` in cell 2, Run All. No retraining. Expect roughly 75 % on genuinely
unseen subjects; sensitivity runs higher than specificity.

**The supervisor asked for a change to the report or slides.** Edit the LaTeX in
`report/Sections/`, then rebuild. If any *number* changes, regenerate rather
than retype: `python scripts/fill_report_tables.py`, then
`python scripts/make_report_figures.py`, then `python scripts/make_presentation.py`.
Re-check compliance with the commands in CLAUDE.md §12.

**You want a better accuracy.** Read CLAUDE.md §11 first. A single scalar
feature already matches the deep model, the prescribed folds carry a study
confound that hurts at test, and with 100 subjects every configuration we ran
falls inside one 95 % confidence interval. The lever is more subjects, not more
tuning.
