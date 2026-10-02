"""Vector (PDF) figures for the short report.

Colour choices follow the categorical/sequential split: foot configurations and
protocols are identities (categorical hues, fixed order, never cycled), while
confusion counts are magnitudes (a single hue, light to dark). The categorical
slots below pass the colour-vision-deficiency and lightness checks of
`validate_palette.js`; do not substitute them by eye.

Writes to outputs/mamba/final/figures_report/:
  fig1_pipeline.pdf   raw VGRF -> tokens -> Mamba -> per-subject decision
  fig2_protocols.pdf  the two evaluation protocols side by side
  fig3_results.pdf    the protocol effect (single column, drawn at 1:1)
  fig4_confusion.pdf  confusion matrices of the final model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Categorical slots (validated): blue, orange, aqua. Fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Ink, never a series colour
INK, INK_SOFT, RULE = "#1c1c1a", "#5a5a55", "#d5d5cf"
STAGE_FILL, TOKEN_FILL = "#22405f", "#c2531f"

OUT = Path("outputs/mamba/final/figures_report")
FINAL = Path("outputs/mamba/final")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.6,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "text.color": INK,
    "axes.labelcolor": INK,
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- fig 1


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 0.70))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.44, 0.94)
    ax.axis("off")

    stages = [
        ("Raw VGRF", "18 channels, 100 Hz\n100 subjects", STAGE_FILL),
        ("Pre-processing", "drop first 5 s\nof gait initiation", STAGE_FILL),
        ("Feature extraction", "11 statistics/channel\nper 1 s -> 60 s windows", TOKEN_FILL),
        ("Mamba encoder", "state-space blocks\nmean + max pooling", STAGE_FILL),
        ("Subject decision", "mean window score\nPD / HC", TOKEN_FILL),
    ]
    y, h, gap = 0.46, 0.46, 0.026
    w = (1.0 - gap * (len(stages) - 1)) / len(stages)
    for i, (title, body, fill) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.016",
                           linewidth=0, facecolor=fill)
        )
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
                fontsize=7.4, color="white", fontweight="bold")
        ax.text(x + w / 2, y + h * 0.33, body, ha="center", va="center",
                fontsize=6.3, color="white", linespacing=1.45, alpha=0.92)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.003, y + h / 2), (x - 0.003, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=7,
                                         linewidth=0.8, color=INK_SOFT))

    fig.savefig(OUT / "fig1_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2


def fig_protocols() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.55))
    rng = np.random.RandomState(0)
    n_subj, n_win = 12, 16
    cmap = ListedColormap([BLUE, ORANGE])

    for ax, leaky in zip(axes, (False, True)):
        grid = np.zeros((n_subj, n_win))
        if leaky:
            grid = (rng.rand(n_subj, n_win) < 0.2).astype(float)
        else:
            grid[[3, 8]] = 1.0
        # Drawn as rectangles rather than imshow: an image would be rasterised
        # into the PDF at whatever dpi matplotlib picks, and IEEE requires
        # >= 300 dpi for raster art. Rectangles stay vector at any zoom.
        for i in range(n_subj):
            for j in range(n_win):
                ax.add_patch(plt.Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, linewidth=0,
                                           facecolor=ORANGE if grid[i, j] else BLUE))
        ax.set_xlim(0, n_win); ax.set_ylim(n_subj, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("windows of one recording  →", fontsize=7, color=INK_SOFT)

    axes[0].set_ylabel("subjects", fontsize=7, color=INK_SOFT)
    axes[0].set_title("(a)  Subject-wise split — used here", fontsize=8.2,
                      fontweight="bold", color=INK, pad=7, loc="left")
    axes[1].set_title("(b)  Window-wise split — leaky", fontsize=8.2,
                      fontweight="bold", color=INK, pad=7, loc="left")
    axes[0].text(0.5, -0.30, "whole subjects held out", transform=axes[0].transAxes,
                 ha="center", fontsize=6.8, color=INK_SOFT)
    axes[1].text(0.5, -0.30, "the same subject appears in train and test",
                 transform=axes[1].transAxes, ha="center", fontsize=6.8, color=INK_SOFT)

    handles = [plt.Line2D([], [], marker="s", markersize=5, linestyle="", color=BLUE, label="training"),
               plt.Line2D([], [], marker="s", markersize=5, linestyle="", color=ORANGE, label="test")]
    fig.legend(handles=handles, ncol=2, loc="lower center", frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.08), handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_protocols_tmp.pdf", bbox_inches="tight", pad_inches=0.02)
    (OUT / "fig3_protocols_tmp.pdf").rename(OUT / "fig2_protocols.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3


def fig_results(base: dict) -> None:
    r"""Effect of the split unit, single panel.

    Drawn at the final print size (\columnwidth = 244.7 pt = 3.39 in) so that
    \includegraphics[width=\columnwidth] applies no scaling: the point sizes
    below are the point sizes on the page, next to an 8 pt caption. The earlier
    two-panel version was authored 7.0 in wide and squeezed to 0.48x, which is
    what made its labels unreadable.
    """
    pairs = [
        ("Random\nForest",
         base["subject_wise"]["random_forest"]["both"]["subject"]["point"]["accuracy"] * 100,
         base["leaky_window"]["random_forest"]["both"]["point"]["accuracy"] * 100),
        ("SVM-RBF",
         base["subject_wise"]["svm_rbf"]["both"]["subject"]["point"]["accuracy"] * 100,
         base["leaky_window"]["svm_rbf"]["both"]["point"]["accuracy"] * 100),
    ]

    # Width chosen so that the tight bounding box comes out at exactly
    # \columnwidth (244.7 pt); see the docstring.
    fig, ax = plt.subplots(figsize=(3.68, 2.30))

    xs = np.arange(len(pairs))
    b1 = ax.bar(xs - 0.19, [p[1] for p in pairs], 0.34, label="subject-wise", color=BLUE, linewidth=0)
    b2 = ax.bar(xs + 0.19, [p[2] for p in pairs], 0.34, label="window-wise", color=ORANGE, linewidth=0)
    ax.bar_label(b1, fmt="%.0f", fontsize=7.4, padding=2, color=INK_SOFT)
    ax.bar_label(b2, fmt="%.0f", fontsize=7.4, padding=2, color=INK_SOFT)
    # The gap itself is the message, so state it once above each pair.
    for i, p in enumerate(pairs):
        top = max(p[1], p[2]) + 9
        ax.plot([i - 0.19, i - 0.19, i + 0.19, i + 0.19],
                [top - 2.5, top, top, top - 2.5], color=INK_SOFT, linewidth=0.6)
        ax.text(i, top + 1.5, f"+{p[2] - p[1]:.0f} pts", fontsize=8.0,
                color=INK, ha="center", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8.4, color=INK)
    ax.set_xlim(-0.55, len(pairs) - 0.45)
    ax.set_ylim(45, 114)
    ax.set_yticks([50, 75, 100])
    ax.tick_params(labelsize=8.0, length=0)
    ax.set_ylabel("Accuracy (%)", fontsize=8.6)
    ax.legend(fontsize=8.0, frameon=False, loc="lower center", handlelength=1.1,
              handletextpad=0.45, ncol=2, columnspacing=1.2, bbox_to_anchor=(0.5, -0.44))
    ax.set_title("Effect of the split unit", fontsize=9.0, fontweight="bold",
                 color=INK, loc="left", pad=6)
    ax.grid(axis="y", alpha=0.35, linewidth=0.5, color=RULE)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_results.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- fig 4


def fig_confusion(mamba: dict) -> None:
    """Counts are a magnitude -> one hue, light to dark (never categorical)."""
    ramp = LinearSegmentedColormap.from_list("blues", ["#f2f6fc", BLUE])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.78))

    for ax, foot in zip(axes, ("left", "right", "both")):
        block = mamba["results"][foot]["subject_mean_prob"]
        c = block["confusion"]
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        # Cells as rectangles, not imshow, so the figure stays fully vector.
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1.4,
                                           edgecolor="white", facecolor=ramp(cm[i, j] / 50)))
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14,
                        fontweight="bold", color="white" if cm[i, j] > 27 else INK)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
        acc = block["point"]["accuracy"] * 100
        sen = block["point"]["sensitivity"] * 100
        spe = block["point"]["specificity"] * 100
        # Title sits above the sub-line, not on top of it: the sub-line is
        # anchored just clear of the axes and the title padded past its cap
        # height, so the two never collide at any figure size.
        ax.set_title(f"{foot.capitalize()}", fontsize=8.2, fontweight="bold", color=INK, pad=14)
        ax.text(0.5, 1.025, f"acc {acc:.0f}%  ·  sen {sen:.0f}%  ·  spe {spe:.0f}%",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.4, color=INK_SOFT)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["HC", "PD"], fontsize=7.4)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["HC", "PD"], fontsize=7.4)
        ax.set_xlabel("Predicted", fontsize=7.2)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

    axes[0].set_ylabel("True", fontsize=7.2)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / "fig4_confusion.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    primary = FINAL / "primary_model" / "results.json"
    if not primary.exists():
        primary = FINAL / "primary_amplitude30_3seed" / "results.json"
    mamba = json.loads(primary.read_text())
    base = json.loads((FINAL / "baseline_comparison.json").read_text())

    fig_pipeline()
    fig_protocols()
    fig_results(base)
    fig_confusion(mamba)
    print(f"wrote {len(list(OUT.glob('*.pdf')))} figures to {OUT} (source: {primary.parent.name})")


if __name__ == "__main__":
    main()
