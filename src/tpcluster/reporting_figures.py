from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from tpcluster.core import prepare_matrix
from tpcluster.display import display_name
from tpcluster.figure_style import (
    SHORT_LABELS,
    add_panel_label,
    apply_publication_style,
    save_figure,
)


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


def _load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manuscript_outputs" / "analysis_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _cluster_sizes(run_dir: Path) -> dict[int, int]:
    table = pd.read_csv(
        run_dir / "manuscript_outputs" / "tables" / "table_cluster_sizes.csv"
    )
    return {
        int(row.primary_cluster): int(row.n)
        for row in table.itertuples(index=False)
    }


def make_workflow_figure(
    output_dir: str | Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(11.0, 2.2))
    ax.axis("off")
    labels = [
        "Cohort\nconstruction",
        "Feature\ngovernance",
        "Preprocessing",
        "Candidate\nmodels",
        "Quality\ngates",
        "Canonical\nphenotypes",
        "Post hoc clinical\ninterpretation",
    ]
    xs = np.linspace(0.07, 0.93, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x,
            0.58,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10.5,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.10",
                "facecolor": "white",
                "edgecolor": "0.35",
                "linewidth": 1.0,
            },
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1] - 0.055, 0.58),
                xytext=(x + 0.055, 0.58),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "0.35"},
            )
    ax.text(
        0.21,
        0.21,
        "Clustering variables only",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    ax.text(
        0.84,
        0.21,
        "Demographics, SDoH, and outcomes excluded from clustering",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(pad=0.2)
    return save_figure(fig, Path(output_dir) / "figure1_workflow", formats)


def _stratified_display_sample(
    plot: pd.DataFrame,
    max_points: int,
    seed: int,
) -> pd.DataFrame:
    """Downsample for visualization while preserving cluster membership."""
    if len(plot) <= max_points:
        return plot.reset_index(drop=True)

    sampled: list[pd.DataFrame] = []
    total = len(plot)
    for _, group in plot.groupby("cluster", sort=True):
        target = max(100, int(round(max_points * len(group) / total)))
        sampled.append(group.sample(n=min(len(group), target), random_state=seed))
    return pd.concat(sampled, ignore_index=True)


def _pca_panel(
    ax: plt.Axes,
    config_path: str | Path,
    run_dir: Path,
    cohort: str,
    seed: int = 11,
    max_points: int = 5000,
) -> None:
    config = _load_config(config_path)
    frame = pd.read_parquet(config["input_path"])
    assignments = pd.read_parquet(
        run_dir / "manuscript_outputs" / "final_cluster_assignments.parquet"
    )
    merged = frame.merge(assignments, on=config["id_column"], validate="one_to_one")
    _, _, scaled, *_ = prepare_matrix(
        frame,
        config["features"]["primary"],
        config.get("preprocessing", {}),
    )
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(scaled)
    plot = pd.DataFrame(
        {
            "PC1": coordinates[:, 0],
            "PC2": coordinates[:, 1],
            "cluster": merged["primary_cluster"].to_numpy(),
        }
    )
    plot = _stratified_display_sample(plot, max_points=max_points, seed=seed)

    sizes = _cluster_sizes(run_dir)
    for cluster in sorted(plot["cluster"].unique()):
        subset = plot[plot["cluster"] == cluster]
        ax.scatter(
            subset.PC1,
            subset.PC2,
            s=8,
            alpha=0.28,
            linewidths=0,
            label=f"{SHORT_LABELS[cohort][int(cluster)]} (n={sizes[int(cluster)]:,})",
            rasterized=True,
        )
    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title(cohort.capitalize(), loc="left", pad=5)
    ax.legend(frameon=False, loc="best", handletextpad=0.4, borderaxespad=0.3)


def make_combined_pca_figure(
    stroke_config: str | Path,
    sepsis_config: str | Path,
    stroke_run_dir: str | Path,
    sepsis_run_dir: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    _pca_panel(axes[0], stroke_config, Path(stroke_run_dir), "stroke")
    _pca_panel(axes[1], sepsis_config, Path(sepsis_run_dir), "sepsis")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.text(
        0.5,
        0.01,
        "Two-dimensional visualization; clustering used the full retained PCA representation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.2)
    return save_figure(fig, Path(output_dir) / "figure2_pca_projection", formats)


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
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        [
            f"{COMPACT_CLUSTER_LABELS[cohort][cluster]}\nn={sizes[cluster]:,}"
            for cluster in range(3)
        ],
        rotation=0,
        ha="center",
        fontsize=9,
    )
    ax.set_title(title, loc="left", pad=5)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    label = (
        "Median difference / IQR"
        if value_col == "median_difference_iqr"
        else "Standardized prevalence difference"
    )
    colorbar = ax.figure.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label(label, fontsize=8.5, labelpad=7)


def make_combined_heatmap_figure(
    stroke_run_dir: str | Path,
    sepsis_run_dir: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    apply_publication_style()
    stroke_run = Path(stroke_run_dir)
    sepsis_run = Path(sepsis_run_dir)
    fig = plt.figure(figsize=(12.6, 11.2))
    grid = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=(1.0, 0.045, 1.0, 0.045),
        height_ratios=(1.0, 1.15),
        hspace=0.38,
        wspace=0.28,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 2]),
    ]
    caxes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 3]),
    ]
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
        0.01,
        "Zero denotes the cohort-wide reference; positive values indicate relative enrichment.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.08)
    return save_figure(fig, Path(output_dir) / "figure3_phenotype_heatmaps", formats)


def _prepare_effect_size_table(path: Path, metric: str, top_n: int) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table[table["effect_size_name"] == metric].copy()
    label_col = "variable_label" if "variable_label" in table.columns else "variable"
    table["label"] = table[label_col].map(display_name)
    return table.sort_values("effect_size", ascending=False).head(top_n).sort_values("effect_size")


def _effect_size_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    axis_label: str,
    x_max: float,
) -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No variables available", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    else:
        y = np.arange(len(table))
        ax.hlines(y, 0, table.effect_size, color="0.82", linewidth=1.0, zorder=1)
        ax.scatter(table.effect_size, y, s=42, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(table.label, fontsize=9)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(axis_label)
    ax.set_title(title, loc="left", pad=5)
    ax.grid(axis="x", color="0.92", linewidth=0.7)


def make_effect_size_figure(
    stroke_run_dir: str | Path,
    sepsis_run_dir: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    apply_publication_style()
    stroke_path = (
        Path(stroke_run_dir)
        / "manuscript_outputs"
        / "tables"
        / "table_effect_size_focused_results.csv"
    )
    sepsis_path = (
        Path(sepsis_run_dir)
        / "manuscript_outputs"
        / "tables"
        / "table_effect_size_focused_results.csv"
    )
    stroke_cat = _prepare_effect_size_table(stroke_path, "Cramers_V", 8)
    stroke_cont = _prepare_effect_size_table(stroke_path, "epsilon_squared", 8)
    sepsis_cat = _prepare_effect_size_table(sepsis_path, "Cramers_V", 8)
    sepsis_cont = _prepare_effect_size_table(sepsis_path, "epsilon_squared", 8)

    cat_max = max(
        0.10,
        1.15
        * max(
            stroke_cat.effect_size.max() if not stroke_cat.empty else 0,
            sepsis_cat.effect_size.max() if not sepsis_cat.empty else 0,
        ),
    )
    cont_max = max(
        0.02,
        1.15
        * max(
            stroke_cont.effect_size.max() if not stroke_cont.empty else 0,
            sepsis_cont.effect_size.max() if not sepsis_cont.empty else 0,
        ),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
    specifications = [
        (axes[0, 0], stroke_cat, "Stroke: categorical variables", "Cramér's V", cat_max),
        (axes[0, 1], stroke_cont, "Stroke: continuous variables", "Epsilon-squared", cont_max),
        (axes[1, 0], sepsis_cat, "Sepsis: categorical variables", "Cramér's V", cat_max),
        (axes[1, 1], sepsis_cont, "Sepsis: continuous variables", "Epsilon-squared", cont_max),
    ]
    for panel, specification in zip("ABCD", specifications):
        _effect_size_panel(*specification)
        add_panel_label(specification[0], panel)

    fig.text(
        0.5,
        0.01,
        "Post hoc contextual comparisons. Metrics are displayed separately because Cramér's V and epsilon-squared are not directly comparable.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2.0, w_pad=2.0)
    return save_figure(fig, Path(output_dir) / "figure4_posthoc_effect_sizes", formats)


def write_figure_manifest(
    output_dir: str | Path,
    stroke_run_dir: str | Path,
    sepsis_run_dir: str | Path,
    generated_files: list[Path],
) -> Path:
    output_dir = Path(output_dir)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stroke_run_dir": str(Path(stroke_run_dir)),
        "sepsis_run_dir": str(Path(sepsis_run_dir)),
        "stroke_analysis_manifest": _load_manifest(Path(stroke_run_dir)),
        "sepsis_analysis_manifest": _load_manifest(Path(sepsis_run_dir)),
        "generated_files": [str(path) for path in generated_files],
        "pca_display_seed": 11,
        "pca_max_display_points_per_cohort": 5000,
        "stability_selection_figure_included": False,
        "stability_note": "Detailed stability-selection visualization reserved for future work; current paper reports existing stability summaries in tables.",
        "figure_standard": "Scientific Figure Master Standard v8",
        "effect_size_metrics_separated": True,
        "heatmap_colorbars_dedicated": True,
    }
    path = manifest_dir / "paper_figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
