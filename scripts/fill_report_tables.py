"""Generate the whole results table of the report from the saved run outputs.

Every cell -- baselines and Mamba alike -- is written from JSON by this script,
so the report cannot drift from the runs and cannot mix conventions between
rows. The presentation reads the same files with the same rounding, so the two
documents agree cell by cell.

Convention (one, applied everywhere):
  Acc, Pre            mean +/- std over 1000 bootstrap resamples of the subjects
  F1, Sen, AUC        the same bootstrap's std, attached to the point estimate
                      of the mean (see MEAN_SOURCE below)
  p                   one-sided exact binomial test of the accuracy against the
                      50 % chance level, from scripts/significance_tests.py

The spread on F1, Sen and AUC comes from the identical resampling that already
produced the Acc and Pre cells -- the same 1000 draws of whole subjects, the
same seed -- so no new statistic is computed here, only a field that was
already in the JSON is read.

MEAN_SOURCE: for F1, Sen and AUC the *mean* printed is the point estimate on
the pooled predictions, not the bootstrap mean. The two agree in 40 of the 45
cells; where they disagree the bootstrap mean is high by one unit in the last
digit (resampling bias on a bounded statistic). Keeping the point estimate
leaves every published number unchanged. Run with --audit to list the cells
where the two differ.

  python scripts/fill_report_tables.py [--audit]
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


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def boot_sd(text: str) -> str:
    """'0.85 +/- 0.04' -> '0.04'. The spread half of a bootstrap summary."""
    return text.split("±")[1].strip()


def metrics(block: dict) -> list[tuple[str, float]]:
    """The five metric cells of one row, as (printed text, numeric value).

    Acc and Pre keep the bootstrap summary string they always had. F1, Sen and
    AUC gain the same bootstrap's spread, attached to the point mean so that
    no previously published mean moves (see MEAN_SOURCE in the module docstring).
    """
    point, boot = block["point"], block["bootstrap"]
    acc, pre = boot["accuracy"], boot["precision_w"]
    f1 = f"{point['f1_w']:.2f} ± {boot_sd(boot['f1_w'])}"
    sen = f"{point['sensitivity'] * 100:.0f} ± {boot_sd(boot['sensitivity'])}"
    auc = f"{point['auc']:.2f} ± {boot_sd(boot['auc'])}"
    return [
        (acc, float(acc.split("±")[0])),
        (pre, float(pre.split("±")[0])),
        (f1, round(point["f1_w"], 2)),
        (sen, round(point["sensitivity"] * 100)),
        (auc, round(point["auc"], 2)),
    ]


def render(cells: list[tuple[str, float]], bests: list[float], p_value: float) -> str:
    """Bold a cell only when it is the best in its column; ties bold together."""
    out = []
    for (text, value), best in zip(cells, bests):
        text = text.replace("±", "$\\pm$")
        out.append(f"\\textbf{{{text}}}" if value == best else text)
    out.append(fmt_p(p_value))  # no winner in the p column
    return " & ".join(out)


def audit(rows: list[tuple[str, str, dict]]) -> None:
    """List cells where the bootstrap mean rounds differently from the point."""
    print("\ncells where the bootstrap mean differs from the printed point mean:")
    found = False
    for label, foot, block in rows:
        point, boot = block["point"], block["bootstrap"]
        for name, shown, booted in (
            ("F1 ", f"{point['f1_w']:.2f}", boot["f1_w"]),
            ("Sen", f"{point['sensitivity'] * 100:.0f}", boot["sensitivity"]),
            ("AUC", f"{point['auc']:.2f}", boot["auc"]),
        ):
            bmean = booted.split("±")[0].strip()
            if bmean != shown:
                found = True
                print(f"  {label:14s} {foot:5s} {name}  printed {shown}  bootstrap mean {bmean}")
    if not found:
        print("  none")


def main() -> None:
    src = FINAL / "primary_model" / "results.json"
    mamba = json.loads(src.read_text())["results"]
    base = json.loads((FINAL / "baseline_comparison.json").read_text())["subject_wise"]
    chance = json.loads((FINAL / "significance.json").read_text())["against_chance"]

    # Collect every row first: the bold mark is a property of the whole column,
    # so no row can be rendered until all fifteen are known.
    rows: list[tuple[str, str, dict, float]] = []
    for label, key in BASELINES:
        for foot in FEET:
            rows.append((label, foot, base[key][foot.lower()]["subject"],
                         chance[key][foot.lower()]["binomial_p"]))
    for foot in FEET:
        rows.append(("\\textbf{Mamba (ours)}", foot,
                     mamba[foot.lower()]["subject_mean_prob"],
                     chance["mamba"][foot.lower()]["binomial_p"]))

    table = [metrics(block) for _, _, block, _ in rows]
    bests = [max(row[col][1] for row in table) for col in range(5)]

    lines: list[str] = []
    for i, ((label, foot, _, p_value), cells) in enumerate(zip(rows, table)):
        head = f"\\multirow{{1}}{{*}}{{{label}}}" if i % len(FEET) == 0 else ""
        lines.append(f"    {head} & {foot:5s} & {render(cells, bests, p_value)} \\\\")
        if i % len(FEET) == len(FEET) - 1 and i != len(rows) - 1:
            lines.append("    \\addlinespace[2pt]")

    body = "\n".join(lines)
    tex = TEX.read_text()
    start = tex.index("Model & Data & Acc (\\%)")
    start = tex.index("\\midrule", start) + len("\\midrule\n")
    end = tex.index("    \\bottomrule", start)
    TEX.write_text(tex[:start] + body + "\n" + tex[end:])

    print(f"rewrote Table 3 from {src} and baseline_comparison.json")
    names = ["Acc", "Pre", "F1 ", "Sen", "AUC"]
    print("  best per column (bold): " +
          ", ".join(f"{n} {b:g}" for n, b in zip(names, bests)))
    if "--audit" in sys.argv:
        audit([(l, f, b) for l, f, b, _ in rows])


if __name__ == "__main__":
    main()
