"""Vector (PDF) figures for the short report.

Produces, in outputs/mamba/final/figures_report/:
  fig1_pipeline.pdf   end-to-end pipeline, raw VGRF -> Mamba -> PD/HC decision
  fig2_protocols.pdf  the two evaluation protocols side by side
  fig3_results.pdf    subject-wise accuracy per model/foot vs the leaky protocol
  fig4_confusion.pdf  confusion matrices of the final Mamba model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DARK_BLUE = "#003366"
ORANGE = "#FF9933"
GREY = "#8C8C8C"
OUT = Path("outputs/mamba/final/figures_report")
FINAL = Path("outputs/mamba/final")


def _box(ax, x, y, w, h, text, color, fontsize=7.5, text_color="white"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=0, facecolor=color)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.9, color=GREY))


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y, h = 0.52, 0.34
    boxes = [
        (0.005, 0.145, "Raw VGRF\n16 sensors + 2 totals\n100 Hz, 100 subjects", DARK_BLUE),
        (0.170, 0.150, "Pre-processing\ndrop first 5 s\n5 s windows, 1 s step", DARK_BLUE),
        (0.340, 0.165, "Tokenisation\n11 stats $\\times$ 18 ch per 1 s\n(+ stride-cycle features)", ORANGE),
        (0.525, 0.150, "Bidirectional\nMamba (SSM)\n$d=256$, 6 layers", DARK_BLUE),
        (0.695, 0.140, "Mean\\,$\\|$\\,max pool\n+ linear head\nwindow score", DARK_BLUE),
        (0.855, 0.140, "Subject decision\nmean score over\nwindows $\\rightarrow$ PD / HC", ORANGE),
    ]
    for x, w, text, color in boxes:
        _box(ax, x, y, w, h, text, color)
        if x > 0.01:
            _arrow(ax, x - 0.022, y + h / 2, x - 0.003, y + h / 2)

    ax.text(0.5, 0.30, "5 predefined subject-wise folds (60 / 20 / 20 subjects) — "
                       "no subject appears in more than one split",
            ha="center", va="center", fontsize=7.5, color=DARK_BLUE)
    ax.text(0.5, 0.16, "per-channel z-score fitted on training windows only  •  "
                       "hyper-parameters chosen on validation  •  test touched once",
            ha="center", va="center", fontsize=7, color=GREY)
    fig.savefig(OUT / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_protocols() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    rng = np.random.RandomState(0)
    n_subj, n_win = 10, 12

    for ax, leaky in zip(axes, (False, True)):
        grid = np.zeros((n_subj, n_win))
        if leaky:
            grid = rng.rand(n_subj, n_win) < 0.2  # test windows scattered everywhere
        else:
            grid[np.array([2, 7])] = 1  # whole subjects held out
        ax.imshow(grid, cmap=matplotlib.colors.ListedColormap([DARK_BLUE, ORANGE]),
                  aspect="auto", vmin=0, vmax=1)
        ax.set_xlabel("windows of one recording", fontsize=8)
        ax.set_ylabel("subjects", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        title = ("(b) Window-wise split — LEAKY\nsame subject in train and test"
                 if leaky else
                 "(a) Subject-wise split — used here\nwhole subjects held out")
        ax.set_title(title, fontsize=8.5, fontweight="bold", color=DARK_BLUE)

    handles = [plt.Line2D([], [], marker="s", linestyle="", color=DARK_BLUE, label="train"),
               plt.Line2D([], [], marker="s", linestyle="", color=ORANGE, label="test")]
    fig.legend(handles=handles, ncol=2, loc="lower center", frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(OUT / "fig2_protocols.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_results(mamba: dict, base: dict, leaky_mamba: dict | None) -> None:
    models = ["svm_rbf", "resnet", "lstm", "random_forest", "mamba"]
    display = {"svm_rbf": "SVM-RBF", "resnet": "1D-ResNet", "lstm": "LSTM",
               "random_forest": "Random Forest", "mamba": "Mamba (ours)"}
    feet = ["left", "right", "both"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    width, xs = 0.26, np.arange(len(models))
    for i, foot in enumerate(feet):
        vals = []
        for m in models:
            if m == "mamba":
                vals.append(mamba["results"][foot]["subject_mean_prob"]["point"]["accuracy"] * 100)
            else:
                vals.append(base["subject_wise"][m][foot]["subject"]["point"]["accuracy"] * 100)
        color = [DARK_BLUE, GREY, ORANGE][i]
        bars = ax1.bar(xs + (i - 1) * width, vals, width, label=foot.capitalize(), color=color)
        ax1.bar_label(bars, fmt="%.0f", fontsize=6, padding=1)

    ax1.set_xticks(xs); ax1.set_xticklabels([display[m] for m in models], fontsize=7.5, rotation=12)
    ax1.set_ylabel("Subject-level accuracy (\\%)", fontsize=8)
    ax1.set_ylim(50, 90); ax1.tick_params(labelsize=7.5)
    ax1.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax1.set_title("(a) Subject-wise protocol (headline)", fontsize=8.5, fontweight="bold", color=DARK_BLUE)
    ax1.grid(axis="y", alpha=0.25); ax1.set_axisbelow(True)

    pairs = [
        ("Random\nForest", base["subject_wise"]["random_forest"]["both"]["subject"]["point"]["accuracy"] * 100,
         base["leaky_window"]["random_forest"]["both"]["point"]["accuracy"] * 100),
        ("SVM-RBF", base["subject_wise"]["svm_rbf"]["both"]["subject"]["point"]["accuracy"] * 100,
         base["leaky_window"]["svm_rbf"]["both"]["point"]["accuracy"] * 100),
    ]
    if leaky_mamba is not None:
        pairs.append(("Mamba", mamba["results"]["both"]["subject_mean_prob"]["point"]["accuracy"] * 100,
                      leaky_mamba["results"]["both"]["window"]["point"]["accuracy"] * 100))

    xs2 = np.arange(len(pairs))
    ax2.bar(xs2 - 0.2, [p[1] for p in pairs], 0.4, label="subject-wise", color=DARK_BLUE)
    ax2.bar(xs2 + 0.2, [p[2] for p in pairs], 0.4, label="leaky window", color=ORANGE)
    ax2.set_xticks(xs2); ax2.set_xticklabels([p[0] for p in pairs], fontsize=7.5)
    ax2.set_ylim(50, 105); ax2.tick_params(labelsize=7.5)
    ax2.set_ylabel("Accuracy (\\%)", fontsize=8)
    ax2.legend(fontsize=7, frameon=False, loc="lower right")
    ax2.set_title("(b) Protocol effect", fontsize=8.5, fontweight="bold", color=DARK_BLUE)
    ax2.grid(axis="y", alpha=0.25); ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_results.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_confusion(mamba: dict) -> None:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#FFFFFF", DARK_BLUE])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for ax, foot in zip(axes, ("left", "right", "both")):
        c = mamba["results"][foot]["subject_mean_prob"]["confusion"]
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=15,
                        fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else DARK_BLUE)
        acc = mamba["results"][foot]["subject_mean_prob"]["point"]["accuracy"] * 100
        ax.set_title(f"{foot.capitalize()} — {acc:.0f}\\% acc.", fontsize=8.5,
                     fontweight="bold", color=DARK_BLUE)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["HC", "PD"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["HC", "PD"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.grid(False)
    axes[0].set_ylabel("True", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_confusion.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "text.usetex": False, "axes.edgecolor": GREY})
    OUT.mkdir(parents=True, exist_ok=True)

    mamba = json.loads((FINAL / "primary_amplitude30_3seed" / "results.json").read_text())
    base = json.loads((FINAL / "baseline_comparison.json").read_text())
    leaky_path = FINAL / "leaky_amp30" / "results.json"
    leaky = json.loads(leaky_path.read_text()) if leaky_path.exists() else None

    fig_pipeline()
    fig_protocols()
    fig_results(mamba, base, leaky)
    fig_confusion(mamba)
    print(f"wrote {len(list(OUT.glob('*.pdf')))} figures to {OUT}")


if __name__ == "__main__":
    main()
