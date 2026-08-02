from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from tpcluster.display import display_name
from tpcluster.figure_style import add_panel_label, apply_publication_style, save_figure


COMPACT_CLUSTER_LABELS = {
    "stroke": {
        0: "Preserved",
        1: "Renal–anaemic",
        2: "Hyperglycaemic",
    },
    "sepsis": {
        0: "Neutrophil",
        1: "IG-high",
        2: "Eosinophil–\nlymphocyte",
    },
}


def _cluster_sizes(run_dir: Path) -> dict[int, int]:
    table = pd.read_csv(
        run_dir / "manuscript_outputs" / "tables" / "table_cluster_sizes.csv"
    )
    return {
        int(row.primary_cluster): int(row.n)
        for row in table.itertuples(index=False)
    }


def _heatmap_matrix(profile: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, str]:
    value_col = (
        "median_difference_iqr"
        if "median_difference_iqr" in profile.columns
        else "standardised_difference"
    )
    profile = profile.copy()
    label_col = "feature_label" if "feature_label" in profile.columns else "feature"
    profile["feature_display"] = profile[label_col].map(display_name)
    profile["absolute"] = pd.to_numeric(profile[value_col], errors="coerce").abs()
    top = profile.groupby("feature")["absolute"].max().nlargest(top_n).index
    matrix = profile[profile.feature.isin(top)].pivot_table(
        index="feature_display",
        columns="cluster",
        values=value_col,
        aggfunc="first",
    )
    matrix = matrix.reindex(columns=[0, 1, 2])
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values().index]
    return matrix, value_col


def _draw_heatmap(
    ax: plt.Axes,
    cax: plt.Axes,
    profile_path: Path,
    cohort: str,
    sizes: dict[int, int],
    top_n: int,
    title: str,
) -> None:
    profile = pd.read_csv(profile_path)
    matrix, value_col = _heatmap_matrix(profile, top_n)
    limit = max(0.5, float(np.nanmax(np.abs(matrix.to_numpy()))))

    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        [
            f"{COMPACT_CLUSTER_LABELS[cohort][cluster]}\nn={sizes[cluster]:,}"
            for cluster in range(3)
        ],
        rotation=0,
        ha="center",
        fontsize=9.5,
    )
    ax.set_title(title, loc="left", pad=6)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    label = (
        "Median difference / IQR"
        if value_col == "median_difference_iqr"
        else "Standardized prevalence difference"
    )
    colorbar = ax.figure.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=8.5)
    colorbar.set_label(label, fontsize=9, labelpad=8)


def make_combined_heatmap_figure(
    stroke_run_dir: str | Path,
    sepsis_run_dir: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    """Generate the collision-free four-panel phenotype heatmap figure."""
    apply_publication_style()
    stroke_run = Path(stroke_run_dir)
    sepsis_run = Path(sepsis_run_dir)

    fig = plt.figure(figsize=(16.8, 11.6))
    grid = GridSpec(
        2,
        5,
        figure=fig,
        width_ratios=(1.0, 0.045, 0.34, 1.0, 0.045),
        height_ratios=(1.0, 1.15),
        hspace=0.40,
        wspace=0.26,
    )

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 3]),
    ]
    caxes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 4]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 4]),
    ]
    for row in (0, 1):
        spacer = fig.add_subplot(grid[row, 2])
        spacer.axis("off")

    specifications = [
        (
            axes[0],
            caxes[0],
            stroke_run / "manuscript_outputs" / "tables" / "table_continuous_cluster_profiles_processed_scale.csv",
            "stroke",
            _cluster_sizes(stroke_run),
            12,
            "Stroke: continuous features",
        ),
        (
            axes[1],
            caxes[1],
            stroke_run / "manuscript_outputs" / "tables" / "table_binary_cluster_profiles.csv",
            "stroke",
            _cluster_sizes(stroke_run),
            12,
            "Stroke: binary features",
        ),
        (
            axes[2],
            caxes[2],
            sepsis_run / "manuscript_outputs" / "tables" / "table_continuous_cluster_profiles_processed_scale.csv",
            "sepsis",
            _cluster_sizes(sepsis_run),
            16,
            "Sepsis: continuous features",
        ),
        (
            axes[3],
            caxes[3],
            sepsis_run / "manuscript_outputs" / "tables" / "table_binary_cluster_profiles.csv",
            "sepsis",
            _cluster_sizes(sepsis_run),
            10,
            "Sepsis: binary features",
        ),
    ]

    for panel, specification in zip("ABCD", specifications):
        _draw_heatmap(*specification)
        add_panel_label(specification[0], panel)

    fig.text(
        0.5,
        0.015,
        "Zero denotes the cohort-wide reference; positive values indicate relative enrichment.",
        ha="center",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.085)
    return save_figure(fig, Path(output_dir) / "figure3_phenotype_heatmaps", formats)
