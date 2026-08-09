from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _profile_table(run_dir: Path, kind: str) -> pd.DataFrame:
    table_dir = run_dir / "manuscript_outputs" / "tables"
    path = (
        table_dir / "table_continuous_cluster_profiles_processed_scale.csv"
        if kind == "continuous"
        else table_dir / "table_binary_cluster_profiles.csv"
    )
    return pd.read_csv(path)


def _heatmap_matrix(profile: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, str]:
    value_col = (
        "median_difference_iqr"
        if "median_difference_iqr" in profile.columns
        else "standardised_difference"
    )
    label_col = "feature_label" if "feature_label" in profile.columns else "feature"
    profile = profile.copy()
    profile["label"] = profile[label_col].astype(str).str.replace("_", " ", regex=False)
    profile["abs"] = pd.to_numeric(profile[value_col], errors="coerce").abs()
    top = profile.groupby("feature")["abs"].max().nlargest(top_n).index
    matrix = profile[profile["feature"].isin(top)].pivot_table(
        index="label", columns="cluster", values=value_col, aggfunc="first"
    )
    matrix = matrix.reindex(columns=[0, 1, 2])
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values().index]
    return matrix, value_col


def _draw_panel(ax, title: str, profile: pd.DataFrame, top_n: int) -> None:
    matrix, value_col = _heatmap_matrix(profile, top_n)
    limit = max(0.5, float(np.nanmax(np.abs(matrix.to_numpy()))))
    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Cluster 0", "Cluster 1", "Cluster 2"], fontsize=8.5)
    ax.tick_params(length=0, pad=3)

    # Attach the colorbar tightly to its own panel. Using an axes divider prevents
    # the bar from colliding with labels in the neighboring panel.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.035)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7.5, pad=2)
    cbar.set_label(
        "Median difference / IQR"
        if value_col == "median_difference_iqr"
        else "Standardized prevalence difference",
        fontsize=7.5,
        labelpad=4,
    )


def make_figure(stroke_run: Path, sepsis_run: Path, output_dir: Path) -> list[Path]:
    specs = [
        ("Stroke: continuous features", _profile_table(stroke_run, "continuous"), 12),
        ("Stroke: binary features", _profile_table(stroke_run, "binary"), 12),
        ("Sepsis: continuous features", _profile_table(sepsis_run, "continuous"), 16),
        ("Sepsis: binary features", _profile_table(sepsis_run, "binary"), 10),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))
    for ax, (title, profile, top_n) in zip(axes.ravel(), specs):
        _draw_panel(ax, title, profile, top_n)

    # Extra horizontal space is reserved for the long clinical labels, while the
    # colorbars themselves remain close to each heatmap frame.
    fig.subplots_adjust(left=0.11, right=0.985, top=0.965, bottom=0.07, wspace=0.43, hspace=0.34)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure3_phenotype_heatmaps_compact_v2.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate refined manuscript phenotype heatmaps.")
    parser.add_argument("--stroke-run", required=True)
    parser.add_argument("--sepsis-run", required=True)
    parser.add_argument("--output-dir", default="results/author_revision_materials")
    args = parser.parse_args()

    outputs = make_figure(Path(args.stroke_run), Path(args.sepsis_run), Path(args.output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
