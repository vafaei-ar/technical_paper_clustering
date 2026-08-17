from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from generate_author_revision_materials import _candidate_table, _column, _heatmap_matrix, _profile_table


STROKE_CLUSTER_LABELS = [
    "Preserved\n(n=5,720)",
    "Renal-anaemic\n(n=2,647)",
    "Hyperglycaemic\n(n=1,468)",
]
SEPSIS_CLUSTER_LABELS = [
    "Neutrophil-predominant\n(n=11,454)",
    "IG-high dysfunction\n(n=1,362)",
    "Eosinophil-lymphocyte\n(n=3,026)",
]


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=9)


def make_model_selection_figure(stroke_run: Path, sepsis_run: Path, output_dir: Path) -> list[Path]:
    """Create Figure 2 without refitting any clustering model."""
    datasets = [("A", "Stroke", _candidate_table(stroke_run)), ("B", "Sepsis", _candidate_table(sepsis_run))]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)

    x_values: list[float] = []
    y_values: list[float] = []
    prepared = []
    for panel, title, table in datasets:
        sil_col = _column(table, ["silhouette", "silhouette_mean"])
        stability_col = _column(table, ["subsample_ari_mean", "mean_subsample_ari", "subsample_ari"])
        reduction_col = _column(table, ["reduction"])
        clusterer_col = _column(table, ["clusterer"])
        k_col = _column(table, ["k"])
        plot = table.copy()
        plot[sil_col] = pd.to_numeric(plot[sil_col], errors="coerce")
        plot[stability_col] = pd.to_numeric(plot[stability_col], errors="coerce")
        plot = plot.dropna(subset=[sil_col, stability_col])
        x_values.extend(plot[sil_col].tolist())
        y_values.extend(plot[stability_col].tolist())
        prepared.append((panel, title, plot, sil_col, stability_col, reduction_col, clusterer_col, k_col))

    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    xpad = max(0.01, 0.06 * (xmax - xmin))
    ypad = max(0.015, 0.06 * (ymax - ymin))

    for ax, (panel, title, plot, sil_col, stability_col, reduction_col, clusterer_col, k_col) in zip(axes, prepared):
        ax.scatter(
            plot[sil_col],
            plot[stability_col],
            s=28,
            alpha=0.48,
            color="0.45",
            edgecolors="none",
            rasterized=True,
        )
        selected = plot[
            (plot[reduction_col].astype(str).str.lower() == "pca")
            & (plot[clusterer_col].astype(str).str.lower() == "kmeans")
            & (pd.to_numeric(plot[k_col], errors="coerce") == 3)
        ]
        if not selected.empty:
            row = selected.sort_values(stability_col, ascending=False).iloc[0]
            ax.scatter(
                [row[sil_col]],
                [row[stability_col]],
                s=170,
                marker="*",
                color="black",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                f"Selected: PCA + k-means, k=3\nSilhouette {row[sil_col]:.3f}; ARI {row[stability_col]:.3f}",
                xy=(row[sil_col], row[stability_col]),
                xytext=(12, -12),
                textcoords="offset points",
                fontsize=8.5,
                va="top",
                ha="left",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "lw": 0.6},
            )
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, min(1.01, ymax + ypad))
        ax.set_xlabel("Silhouette score", fontsize=10)
        ax.set_title(f"{panel}  {title}", loc="left", fontsize=11, fontweight="bold")
        ax.grid(True, linewidth=0.5, alpha=0.18)
        _style_axes(ax)
        ax.text(
            0.98,
            0.04,
            "Higher separation →\nHigher reproducibility ↑",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

    axes[0].set_ylabel("Repeated-subsample ARI", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.93, wspace=0.16)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure2_model_selection_publication.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _draw_heatmap_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    panel: str,
    cluster_labels: list[str],
    limit: float,
):
    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_title(f"{panel}  {title}", loc="left", fontsize=10.5, fontweight="bold", pad=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cluster_labels, fontsize=8.0)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def make_phenotype_heatmap_figure(stroke_run: Path, sepsis_run: Path, output_dir: Path) -> list[Path]:
    """Create Figure 3 from existing profile tables. No clustering or preprocessing is rerun."""
    specs = [
        ("A", "Stroke: continuous features", _profile_table(stroke_run, "continuous"), 12, "continuous", STROKE_CLUSTER_LABELS),
        ("B", "Stroke: binary features", _profile_table(stroke_run, "binary"), 12, "binary", STROKE_CLUSTER_LABELS),
        ("C", "Sepsis: continuous features", _profile_table(sepsis_run, "continuous"), 16, "continuous", SEPSIS_CLUSTER_LABELS),
        ("D", "Sepsis: binary features", _profile_table(sepsis_run, "binary"), 10, "binary", SEPSIS_CLUSTER_LABELS),
    ]

    prepared = []
    domain_limit = {"continuous": 0.5, "binary": 0.5}
    for panel, title, profile, top_n, domain, labels in specs:
        matrix, _ = _heatmap_matrix(profile, top_n)
        domain_limit[domain] = max(domain_limit[domain], float(np.nanmax(np.abs(matrix.to_numpy()))))
        prepared.append((panel, title, matrix, domain, labels))

    # Heatmaps occupy the upper two rows. Horizontal shared color bars live below
    # each column, leaving the centre gap entirely free for long right-panel labels.
    fig = plt.figure(figsize=(12.8, 10.0))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.055],
        width_ratios=[1.0, 1.0],
        hspace=0.38,
        wspace=0.78,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    cax_cont = fig.add_subplot(gs[2, 0])
    cax_binary = fig.add_subplot(gs[2, 1])

    images: dict[str, object] = {}
    for ax, (panel, title, matrix, domain, labels) in zip(axes, prepared):
        images[domain] = _draw_heatmap_panel(ax, matrix, title, panel, labels, domain_limit[domain])

    cb_cont = fig.colorbar(images["continuous"], cax=cax_cont, orientation="horizontal")
    cb_cont.ax.tick_params(labelsize=8, length=2)
    cb_cont.set_label("Median difference divided by cohort IQR", fontsize=8.5, labelpad=4)

    cb_binary = fig.colorbar(images["binary"], cax=cax_binary, orientation="horizontal")
    cb_binary.ax.tick_params(labelsize=8, length=2)
    cb_binary.set_label("Standardized prevalence difference", fontsize=8.5, labelpad=4)

    fig.subplots_adjust(left=0.14, right=0.985, top=0.975, bottom=0.08)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure3_phenotype_heatmaps_publication.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate publication-ready Figures 2 and 3 from existing manuscript-output tables without refitting models."
    )
    parser.add_argument("--stroke-run", required=True)
    parser.add_argument("--sepsis-run", required=True)
    parser.add_argument("--output-dir", default="results/publication_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    outputs = []
    outputs.extend(make_model_selection_figure(Path(args.stroke_run), Path(args.sepsis_run), output_dir))
    outputs.extend(make_phenotype_heatmap_figure(Path(args.stroke_run), Path(args.sepsis_run), output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
