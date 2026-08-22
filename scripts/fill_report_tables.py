"""Generate the whole results table of the report from the saved run outputs.

Every cell -- baselines and Mamba alike -- is written from JSON by this script,
so the report cannot drift from the runs and cannot mix conventions between
rows. The presentation reads the same files with the same rounding, so the two
documents agree cell by cell.

Convention (one, applied everywhere):
  Acc, Pre   mean +/- std over 1000 bootstrap resamples of the subjects
  F1, Sen, Spe, AUC   point estimate on the pooled predictions

  python scripts/fill_report_tables.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FINAL = Path("outputs/mamba/final")
TEX = Path("report/Sections/03_Experiments_and_Results.tex")
FEET = ("Left", "Right", "Both")
BASELINES = [("SVM-RBF", "svm_rbf"), ("1D-ResNet", "resnet"), ("LSTM", "lstm"),
             ("Random Forest", "random_forest")]


def cells(block: dict, bold: bool = False) -> str:
    point, boot = block["point"], block["bootstrap"]
    values = [
        boot["accuracy"].replace("±", "$\\pm$"),
        boot["precision_w"].replace("±", "$\\pm$"),
        f"{point['f1_w']:.2f}",
        f"{point['sensitivity'] * 100:.0f}",
        f"{point['auc']:.2f}",
    ]
    if bold:
        values = [f"\\textbf{{{v}}}" for v in values]
    return " & ".join(values)


def main() -> None:
    src = FINAL / "primary_model" / "results.json"
    mamba = json.loads(src.read_text())["results"]
    base = json.loads((FINAL / "baseline_comparison.json").read_text())["subject_wise"]

    lines: list[str] = []
    for i, (label, key) in enumerate(BASELINES):
        for j, foot in enumerate(FEET):
            head = f"\\multirow{{1}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(f"    {head} & {foot:5s} & {cells(base[key][foot.lower()]['subject'])} \\\\")
        lines.append("    \\midrule")

    for j, foot in enumerate(FEET):
        head = "\\multirow{1}{*}{\\textbf{Mamba (ours)}}" if j == 0 else ""
        block = mamba[foot.lower()]["subject_mean_prob"]
        lines.append(f"    {head} & {foot:5s} & {cells(block, bold=(foot == 'Both'))} \\\\")

    body = "\n".join(lines)
    tex = TEX.read_text()
    start = tex.index("Model & Data & Acc (\\%)")
    start = tex.index("\\midrule", start) + len("\\midrule\n")
    end = tex.index("    \\bottomrule", start)
    TEX.write_text(tex[:start] + body + "\n" + tex[end:])

    print(f"rewrote Table 2 from {src} and baseline_comparison.json")
    for foot in FEET:
        p = mamba[foot.lower()]["subject_mean_prob"]["point"]
        print(f"  Mamba {foot:5s} acc={p['accuracy']:.3f} f1={p['f1_w']:.3f} auc={p['auc']:.3f}")


if __name__ == "__main__":
    main()
