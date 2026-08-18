"""LaTeX results tables and themed confusion-matrix plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .config import COLOR_DARK_BLUE

TABLE_COLUMNS = ["Data", "Features (#)", "Acc (%)", "Sen (%)", "Spe (%)", "AUC"]

BLUES_CMAP = LinearSegmentedColormap.from_list("dark_blue_scale", ["#FFFFFF", COLOR_DARK_BLUE])


def build_results_table(rows: list[dict]) -> pd.DataFrame:
    """rows: list of dicts (Left/Right/Both order) with keys matching TABLE_COLUMNS."""
    return pd.DataFrame(rows)[TABLE_COLUMNS]


def save_latex_table(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, escape=False, caption=caption, label=label)
    out_path.write_text(latex)


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    """Plot a 2x2 confusion matrix (rows=true, cols=predicted, order HC/PD).

    Style: whitegrid base, custom white->Dark Blue colormap, large bold
    annotations colored for contrast against each cell, no gridlines.
    """
    cm = np.asarray(cm)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap=BLUES_CMAP)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HC", "PD"], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["HC", "PD"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="semibold")
    ax.set_ylabel("True", fontsize=12, fontweight="semibold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(False)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else COLOR_DARK_BLUE
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color=color,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
