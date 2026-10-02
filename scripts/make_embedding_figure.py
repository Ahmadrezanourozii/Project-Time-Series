"""Unsupervised PCA / t-SNE embedding of the 100 subjects (vector PDF).

One point per subject. The features are exactly the ones the reported model is
fed -- the 11 amplitude statistics of each of the 18 channels, computed on every
full second of the recording after the first five seconds are dropped (198
dimensions) -- averaged over the subject's recording and then z-scored per
feature across the 100 subjects.

The per-second statistics are NOT recomputed here: ``per_second_tokens`` is
imported from ``scripts/build_feature_sequences.py``, which is the same function
that built the training caches, so the embedding cannot drift from the model's
input. This figure is a visualisation only -- it is never fitted, never split
and never used for any reported number.

Writes to outputs/mamba/final/figures_report/:
  fig5_embedding.pdf          the two-panel figure
  subject_mean_features.npz   the (100, 198) matrix, so re-runs are instant

  python scripts/make_embedding_figure.py [--rebuild]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from baseline.config import SKIP_INIT_SEC
from baseline.features import (
    ALL_CHANNELS,
    discover_unique_gait_files,
    drop_gait_init,
    load_raw_recording,
    parse_subject_id,
)
from baseline.labels import load_labels

# Importing this module also installs the shared rcParams (serif/Times, type-42
# fonts, ink colours), so this figure matches the other four exactly.
from scripts.build_feature_sequences import per_second_tokens
from scripts.make_report_figures import BLUE, INK, INK_SOFT, OUT, RULE

SEED = 42  # fixed, so the t-SNE layout is reproducible
PERPLEXITY = 30
PD_RED = "#d1332e"  # PD
HC_BLUE = BLUE  # HC -- the report's categorical blue
CACHE = OUT / "subject_mean_features.npz"


def subject_mean_features(rebuild: bool = False):
    """(100, 198) mean per-second feature vector per subject, plus ids/labels."""
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        return z["subject_id"], z["label"], z["data"]

    labels = load_labels()
    sids, lbls, rows = [], [], []
    paths = discover_unique_gait_files()
    print(f"computing per-second features for {len(paths)} recordings...")
    for path in paths:
        subject_id = parse_subject_id(path)
        df = drop_gait_init(load_raw_recording(path), skip_sec=SKIP_INIT_SEC)
        tokens = per_second_tokens(df[ALL_CHANNELS].to_numpy(dtype=np.float32))
        rows.append(tokens.mean(axis=0))  # average the subject's seconds
        sids.append(subject_id)
        lbls.append(labels.loc[subject_id, "label"])

    data = np.stack(rows).astype(np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, subject_id=np.asarray(sids),
                        label=np.asarray(lbls), data=data)
    print(f"wrote {CACHE} {data.shape}")
    return np.asarray(sids), np.asarray(lbls), data


def panel(ax, xy, is_pd, title, xlabel, ylabel) -> None:
    for mask, color, name in ((~is_pd, HC_BLUE, "HC"), (is_pd, PD_RED, "PD")):
        ax.scatter(xy[mask, 0], xy[mask, 1], s=15, c=color, label=name,
                   linewidths=0.4, edgecolors="white", zorder=3)
    ax.set_title(title, fontsize=8.6, fontweight="bold", color=INK, loc="left", pad=5)
    ax.set_xlabel(xlabel, fontsize=8.0)
    ax.set_ylabel(ylabel, fontsize=8.0)
    ax.tick_params(labelsize=7.4, length=0)
    ax.grid(alpha=0.35, linewidth=0.5, color=RULE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Autoscale only, with a margin: no point may be dropped or clipped.
    ax.margins(0.08)


def main() -> None:
    sids, lbls, data = subject_mean_features("--rebuild" in sys.argv)
    is_pd = lbls == "PD"
    print(f"{len(sids)} subjects: {int(is_pd.sum())} PD, {int((~is_pd).sum())} HC")

    # z-score per feature over all 100 subjects (unsupervised: no fold, no split)
    z = (data - data.mean(axis=0)) / np.where(data.std(axis=0) < 1e-8, 1.0, data.std(axis=0))

    pca = PCA(n_components=2, random_state=SEED)
    xy_pca = pca.fit_transform(z)
    ev = pca.explained_variance_ratio_
    print(f"explained variance: PC1 {ev[0] * 100:.1f}%, PC2 {ev[1] * 100:.1f}% "
          f"(PC1+PC2 {ev[:2].sum() * 100:.1f}%)")

    xy_tsne = TSNE(n_components=2, perplexity=PERPLEXITY, init="pca",
                   learning_rate="auto", random_state=SEED).fit_transform(z)

    # Width tuned so the tight bounding box lands on \textwidth (506.5 pt),
    # i.e. the figure prints at 1:1 and the point sizes above are the point
    # sizes on the page.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.30, 2.20))
    panel(ax1, xy_pca, is_pd, "(a) PCA", "PC1", "PC2")
    panel(ax2, xy_tsne, is_pd, "(b) t-SNE", "t-SNE 1", "t-SNE 2")

    handles, labels_ = ax1.get_legend_handles_labels()
    ax2.legend(handles, labels_, title="Diagnosis", fontsize=7.6,
               title_fontsize=7.6, frameon=False, loc="center left",
               bbox_to_anchor=(1.01, 0.5), handletextpad=0.3, borderpad=0.2)

    fig.tight_layout(w_pad=2.2)
    out = OUT / "fig5_embedding.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
