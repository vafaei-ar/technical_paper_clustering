from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from tpcluster.display import display_name


def latest_full_run(output_dir: str | Path) -> Path:
    runs = [
        path
        for path in Path(output_dir).iterdir()
        if path.is_dir() and (path / "manuscript_outputs").exists()
    ]
    if not runs:
        raise FileNotFoundError(f"No manuscript-output run found under {output_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def add_display_columns(table: pd.DataFrame) -> pd.DataFrame:
    polished = table.copy()
    if "feature" in polished.columns:
        polished["feature_label"] = polished["feature"].map(display_name)
    if "variable" in polished.columns:
        polished["variable_label"] = polished["variable"].map(display_name)
    return polished


def render_heatmap(
    profile: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    value_col = (
        "median_difference_iqr"
        if "median_difference_iqr" in profile.columns
        else "standardised_difference"
    )
    plotted = profile.copy()
    plotted["absolute"] = plotted[value_col].abs()
    top_features = plotted.groupby("feature")["absolute"].max().nlargest(18).index
    matrix = plotted[plotted["feature"].isin(top_features)].pivot_table(
        index="feature",
        columns="cluster",
        values=value_col,
        aggfunc="first",
    )
    matrix = matrix.reindex(columns=sorted(matrix.columns))
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values().index]
    display_index = [display_name(feature) for feature in matrix.index]

    limit = max(0.5, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(8.8, max(5.8, 0.34 * len(matrix))))
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(display_index, fontsize=8.5)
    ax.set_xticks(np.arange(len(matrix.columns)))

    if "cluster_label" in plotted.columns:
        cluster_labels = (
            plotted[["cluster", "cluster_label"]]
            .drop_duplicates()
            .set_index("cluster")["cluster_label"]
            .to_dict()
        )
        xlabels = [cluster_labels.get(int(cluster), f"Cluster {cluster}") for cluster in matrix.columns]
    else:
        xlabels = [f"Cluster {cluster}" for cluster in matrix.columns]

    ax.set_xticklabels(xlabels, rotation=18, ha="right", fontsize=9)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Canonical clinical phenotype")
    ax.set_ylabel("Feature")
    fig.colorbar(image, ax=ax, label=value_col.replace("_", " "))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def polish_run(config_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    cohort = config["dataset_name"]
    run_dir = latest_full_run(config["output_dir"])
    manuscript_dir = run_dir / "manuscript_outputs"
    table_dir = manuscript_dir / "tables"
    figure_dir = manuscript_dir / "figures"

    for csv_path in sorted(table_dir.glob("*.csv")):
        table = pd.read_csv(csv_path)
        polished = add_display_columns(table)
        polished.to_csv(csv_path, index=False)

    continuous_path = table_dir / "table_continuous_cluster_profiles_processed_scale.csv"
    binary_path = table_dir / "table_binary_cluster_profiles.csv"

    if continuous_path.exists():
        render_heatmap(
            pd.read_csv(continuous_path),
            figure_dir / "figure_continuous_heatmap.png",
            f"{cohort.capitalize()} continuous-feature effects",
        )
    if binary_path.exists():
        render_heatmap(
            pd.read_csv(binary_path),
            figure_dir / "figure_binary_heatmap.png",
            f"{cohort.capitalize()} binary-feature effects",
        )

    print(f"Applied manuscript display labels: {manuscript_dir}")
    return manuscript_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    polish_run(args.config)


if __name__ == "__main__":
    main()
