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
    LONG_LABELS,
    SHORT_LABELS,
    add_panel_label,
    apply_publication_style,
    save_figure,
)


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
        "Candidate\nclustering",
        "Quality\ngates",
        "Canonical\nphenotypes",
        "Post hoc clinical\ninterpretation",
    ]
    xs = np.linspace(0.07, 0.93, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x,
            0.55,
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
                xy=(xs[index + 1] - 0.055, 0.55),
                xytext=(x + 0.055, 0.55),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "0.35"},
            )
    ax.text(
        0.21,
        0.16,
        "Clustering variables only",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    ax.text(
        0.84,
        0.16,
        "Demographics, SDoH, and outcomes excluded from clustering",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(pad=0.2)
    return save_figure(fig, Path(output_dir) / "figure1_workflow", formats)


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
    if len(plot) > max_points:
        plot = (
            plot.groupby("cluster", group_keys=False)
            .apply(
                lambda group: group.sample(
                    min(len(group), max(100, int(max_points * len(group) / len(plot)))),
                    random_state=seed,
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
    sizes = _cluster_sizes(run_dir)
    for cluster in sorted(plot.cluster.unique()):
        subset = plot[plot.cluster == cluster]
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
        "Two-dimensional visual projections; clustering used the full retained PCA representation.",
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
    profile["absolute"] = profile[value_col].abs()
    top = profile.groupby("feature")["absolute"].max().nlargest(top_n).index
    matrix = profile[profile.feature.isin(top)].pivot_table(
        index="feature_label" if "feature_label" in profile else "feature",
        columns="cluster",
        values=value_col,
        aggfunc="first",
    )
    matrix = matrix.reindex(columns=[0, 1, 2])
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values().index]
    return matrix, value_col


def _draw_heatmap(
    ax: plt.Axes,
    profile_path: Path,
    cohort: str,
    sizes: dict[int, int],
    top_n: int,
    title: str,
) -> plt.cm.ScalarMappable:
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
    ax.set_yticklabels(matrix.index)
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        [f"{SHORT_LABELS[cohort][cluster]}\nn={sizes[cluster]:,}" for cluster in range(3)],
        rotation=0,
        ha="center",
    )
    ax.set_title(title, loc="left", pad=5)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    image._tpcluster_value_col = value_col
    return image


def make_combined_heatmap_figure(
    stroke_run_dir: str | Path,
    sepsis_run_dir: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    apply_publication_style()
    stroke_run = Path(stroke_run_dir)
    sepsis_run = Path(sepsis_run_dir)
    fig = plt.figure(figsize=(11.2, 11.0))
    grid = GridSpec(2, 2, figure=fig, height_ratios=(1.0, 1.2), hspace=0.34, wspace=0.50)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    specifications = [
        (
            axes[0],
            stroke_run / "manuscript_outputs" / "tables" / "table_continuous_cluster_profiles_processed_scale.csv",
            "stroke",
            _cluster_sizes(stroke_run),
            14,
            "Stroke: continuous features",
        ),
        (
            axes[1],
            stroke_run / "manuscript_outputs" / "tables" / "table_binary_cluster_profiles.csv",
            "stroke",
            _cluster_sizes(stroke_run),
            12,
            "Stroke: binary features",
        ),
        (
            axes[2],
            sepsis_run / "manuscript_outputs" / "tables" / "table_continuous_cluster_profiles_processed_scale.csv",
            "sepsis",
            _cluster_sizes(sepsis_run),
            16,
            "Sepsis: continuous features",
        ),
        (
            axes[3],
            sepsis_run / "manuscript_outputs" / "tables" / "table_binary_cluster_profiles.csv",
            "sepsis",
            _cluster_sizes(sepsis_run),
            12,
            "Sepsis: binary features",
        ),
    ]
    images = []
    for panel, specification in zip("ABCD", specifications):
        image = _draw_heatmap(*specification)
        add_panel_label(specification[0], panel)
        images.append(image)
    for ax, image in zip(axes, images):
        value_col = image._tpcluster_value_col
        label = (
            "Median difference / IQR"
            if value_col == "median_difference_iqr"
            else "Standardized prevalence difference"
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.025)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label(label, fontsize=8.5)
    fig.text(
        0.5,
        0.01,
        "Zero denotes the cohort-wide reference; positive values indicate relative enrichment.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, Path(output_dir) / "figure3_phenotype_heatmaps", formats)


def _effect_size_panel(ax: plt.Axes, path: Path, cohort: str, top_n: int = 10) -> None:
    table = pd.read_csv(path)
    table = table.sort_values("effect_size", ascending=False).head(top_n).copy()
    label_col = "variable_label" if "variable_label" in table else "variable"
    table["label"] = table[label_col].map(display_name)
    table = table.sort_values("effect_size")
    y = np.arange(len(table))
    ax.scatter(table.effect_size, y, s=38, zorder=3)
    ax.hlines(y, 0, table.effect_size, color="0.78", linewidth=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table.label)
    ax.axvline(0.05, color="0.72", linestyle="--", linewidth=0.9)
    ax.axvline(0.15, color="0.72", linestyle="--", linewidth=0.9)
    ax.set_xlim(left=0)
    ax.set_xlabel("Effect size")
    ax.set_title(cohort.capitalize(), loc="left", pad=5)
    ax.grid(axis="x", color="0.92", linewidth=0.7)


def make_effect_size_figure(
    stroke_run_dir: str | Path,
    sepsis_run_dir: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> list[Path]:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    _effect_size_panel(
        axes[0],
        Path(stroke_run_dir)
        / "manuscript_outputs"
        / "tables"
        / "table_effect_size_focused_results.csv",
        "stroke",
    )
    _effect_size_panel(
        axes[1],
        Path(sepsis_run_dir)
        / "manuscript_outputs"
        / "tables"
        / "table_effect_size_focused_results.csv",
        "sepsis",
    )
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.text(
        0.5,
        0.01,
        "Post hoc comparisons. Effect sizes are Cramér's V for categorical variables and epsilon-squared for continuous variables.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.3)
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
    }
    path = manifest_dir / "paper_figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
