"""Write the Mamba rows of the report's main table from results.json.

The other rows are baselines and stay as they are; only the placeholders
MAMBA_L / MAMBA_R / MAMBA_B are replaced, so the numbers in the report can
never drift from the numbers in the run that produced them.

  python scripts/fill_report_tables.py [results.json]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FINAL = Path("outputs/mamba/final")
TEX = Path("report/Sections/03_Experiments_and_Results.tex")


def row(bootstrap: dict, point: dict) -> str:
    """One LaTeX row body: Acc, Pre, F1, Sen, AUC."""
    acc = bootstrap["accuracy"].replace("±", "$\\pm$")
    pre = bootstrap["precision_w"].replace("±", "$\\pm$")
    f1 = f"{point['f1_w']:.2f}"
    sen = f"{point['sensitivity'] * 100:.0f}"
    auc = f"{point['auc']:.2f}"
    return f"{acc} & {pre} & {f1} & {sen} & {auc}"


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else FINAL / "primary_model" / "results.json"
    if not src.exists():
        src = FINAL / "primary_amplitude30_3seed" / "results.json"
    results = json.loads(src.read_text())["results"]

    tex = TEX.read_text()
    for foot, key in (("left", "MAMBA_L"), ("right", "MAMBA_R"), ("both", "MAMBA_B")):
        block = results[foot]["subject_mean_prob"]
        body = row(block["bootstrap"], block["point"])
        if foot == "both":  # bold the headline row
            body = " & ".join(f"\\textbf{{{c.strip()}}}" for c in body.split("&"))
        # re.sub would interpret LaTeX backslashes in the replacement, so substitute by hand
        # the placeholders are written MAMBA\_L in LaTeX (escaped underscore)
        pattern = re.compile(re.escape(key.replace("_", "\\_")) + r"\s*&(\s*&)*")
        match = pattern.search(tex)
        if not match:
            raise SystemExit(f"placeholder {key} not found in {TEX}")
        tex = tex[: match.start()] + body + " " + tex[match.end() :]
    TEX.write_text(tex)

    print(f"filled Mamba rows from {src}")
    for foot in ("left", "right", "both"):
        p = results[foot]["subject_mean_prob"]["point"]
        print(f"  {foot:6s} acc={p['accuracy']:.3f} f1_w={p['f1_w']:.3f} auc={p['auc']:.3f} "
              f"sen={p['sensitivity']:.3f} spe={p['specificity']:.3f}")


if __name__ == "__main__":
    main()
