from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_full_run(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_dir()
        and (path / "internal_metrics.csv").exists()
        and (path / "seed_stability.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed clustering run found under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def model_key(reduction: str, clusterer: str, k: int) -> tuple[str, str, int]:
    return str(reduction), str(clusterer), int(k)


def model_registry(dataset: str, path: str | Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    cfg = load_yaml(path)
    if dataset not in cfg:
        raise KeyError(f"Dataset {dataset!r} not found in {path}")

    registry: dict[tuple[str, str, int], dict[str, Any]] = {}
    dataset_cfg = cfg[dataset]
    primary = dataset_cfg.get("primary")
    if primary:
        registry[model_key(primary["reduction"], primary["clusterer"], primary["k"])] = {
            **primary,
            "selection_status": "primary",
        }
    for status in ("sensitivity", "profiled"):
        for item in dataset_cfg.get(status, []):
            registry[model_key(item["reduction"], item["clusterer"], item["k"])] = {
                **item,
                "selection_status": status,
            }
    return registry


def summarize_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "reduction", "clusterer", "k"]
    summary = metrics.groupby(group_cols, as_index=False).agg(
        n_seeds=("seed", "nunique"),
        n_components_min=("n_components", "min"),
        n_components_max=("n_components", "max"),
        silhouette_mean=("silhouette", "mean"),
        silhouette_sd=("silhouette", "std"),
        silhouette_min=("silhouette", "min"),
        silhouette_max=("silhouette", "max"),
        calinski_harabasz_mean=("calinski_harabasz", "mean"),
        calinski_harabasz_sd=("calinski_harabasz", "std"),
        davies_bouldin_mean=("davies_bouldin", "mean"),
        davies_bouldin_sd=("davies_bouldin", "std"),
        minimum_cluster_n_min=("minimum_cluster_n", "min"),
        minimum_cluster_fraction_min=("minimum_cluster_fraction", "min"),
        cluster_size_gate_pass_rate=("passes_cluster_size_gate", "mean"),
        all_seeds_pass_cluster_size_gate=("passes_cluster_size_gate", "all"),
    )
    return summary


def summarize_seed_stability(seed_stability: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "reduction", "clusterer", "k"]
    return seed_stability.groupby(group_cols, as_index=False).agg(
        seed_pair_count=("adjusted_rand_index", "size"),
        seed_ari_mean=("adjusted_rand_index", "mean"),
        seed_ari_median=("adjusted_rand_index", "median"),
        seed_ari_sd=("adjusted_rand_index", "std"),
        seed_ari_min=("adjusted_rand_index", "min"),
        seed_ari_max=("adjusted_rand_index", "max"),
    )


def summarize_reference_seed(metrics: pd.DataFrame, default_seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in metrics.groupby(["dataset", "reduction", "clusterer", "k"]):
        reference = group[group["seed"] == default_seed]
        if reference.empty:
            reference = group.sort_values("seed").head(1)
        row = reference.iloc[0]
        rows.append(
            {
                "dataset": key[0],
                "reduction": key[1],
                "clusterer": key[2],
                "k": int(key[3]),
                "reference_seed": int(row["seed"]),
                "reference_n_components": int(row["n_components"]),
                "reference_silhouette": float(row["silhouette"]),
                "reference_calinski_harabasz": float(row["calinski_harabasz"]),
                "reference_davies_bouldin": float(row["davies_bouldin"]),
                "reference_minimum_cluster_n": int(row["minimum_cluster_n"]),
                "reference_minimum_cluster_fraction": float(row["minimum_cluster_fraction"]),
                "reference_passes_cluster_size_gate": bool(row["passes_cluster_size_gate"]),
            }
        )
    return pd.DataFrame(rows)


def load_subsample_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "subsample_stability_summary.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["dataset", "reduction", "clusterer", "k"]
        )
    table = pd.read_csv(path)
    keep = [
        "dataset",
        "candidate",
        "reduction",
        "clusterer",
        "k",
        "mean_subsample_ari",
        "median_subsample_ari",
        "sd_subsample_ari",
        "minimum_subsample_ari",
        "p10_subsample_ari",
        "p25_subsample_ari",
        "minimum_cluster_fraction",
        "mean_minimum_cluster_fraction",
    ]
    keep = [column for column in keep if column in table.columns]
    table = table[keep].copy()
    if "minimum_cluster_fraction" in table.columns:
        table = table.rename(
            columns={
                "minimum_cluster_fraction": "subsample_minimum_cluster_fraction",
                "mean_minimum_cluster_fraction": "subsample_mean_minimum_cluster_fraction",
            }
        )
    return table


def add_ranks(table: pd.DataFrame) -> pd.DataFrame:
    ranked = table.copy()
    eligible = ranked["all_seeds_pass_cluster_size_gate"].fillna(False)
    rank_specs = {
        "silhouette_rank": ("silhouette_mean", False),
        "calinski_harabasz_rank": ("calinski_harabasz_mean", False),
        "davies_bouldin_rank": ("davies_bouldin_mean", True),
        "seed_stability_rank": ("seed_ari_mean", False),
        "subsample_stability_rank": ("mean_subsample_ari", False),
    }
    for output, (column, ascending) in rank_specs.items():
        ranked[output] = np.nan
        if column not in ranked.columns:
            continue
        values = ranked.loc[eligible, column]
        ranked.loc[eligible, output] = values.rank(
            method="min", ascending=ascending, na_option="bottom"
        )
    return ranked


def build_summary(
    config_path: str | Path,
    models_path: str | Path,
    run_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    cfg = load_yaml(config_path)
    dataset = str(cfg["dataset_name"])
    run_dir = Path(run_dir) if run_dir else latest_full_run(cfg["output_dir"])

    metrics = pd.read_csv(run_dir / "internal_metrics.csv")
    stability = pd.read_csv(run_dir / "seed_stability.csv")
    default_seed = int(cfg.get("seeds", [11])[0])

    summary = summarize_seed_metrics(metrics)
    summary = summary.merge(
        summarize_seed_stability(stability),
        on=["dataset", "reduction", "clusterer", "k"],
        how="left",
        validate="one_to_one",
    )
    summary = summary.merge(
        summarize_reference_seed(metrics, default_seed),
        on=["dataset", "reduction", "clusterer", "k"],
        how="left",
        validate="one_to_one",
    )

    subsample = load_subsample_summary(run_dir)
    if not subsample.empty:
        summary = summary.merge(
            subsample,
            on=["dataset", "reduction", "clusterer", "k"],
            how="left",
            validate="one_to_one",
        )

    registry = model_registry(dataset, models_path)
    names: list[str] = []
    statuses: list[str] = []
    rationales: list[str] = []
    for row in summary.itertuples(index=False):
        key = model_key(row.reduction, row.clusterer, row.k)
        item = registry.get(key, {})
        names.append(
            str(item.get("name", f"{dataset}_{row.reduction}_{row.clusterer}_k{row.k}"))
        )
        statuses.append(str(item.get("selection_status", "not_shortlisted")))
        rationales.append(str(item.get("rationale", "")))
    summary.insert(1, "candidate", names)
    summary.insert(2, "selection_status", statuses)
    summary.insert(3, "selection_rationale", rationales)
    summary["selected_primary"] = summary["selection_status"].eq("primary")
    summary = add_ranks(summary)

    status_order = pd.CategoricalDtype(
        ["primary", "sensitivity", "profiled", "not_shortlisted"], ordered=True
    )
    summary["selection_status"] = summary["selection_status"].astype(status_order)
    summary = summary.sort_values(
        ["selection_status", "reduction", "clusterer", "k"]
    ).reset_index(drop=True)
    summary["selection_status"] = summary["selection_status"].astype(str)

    output_dir = run_dir / "manuscript_outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tableS_candidate_model_selection.csv"
    summary.to_csv(output_path, index=False)
    return summary, output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate the complete clustering grid, seed agreement, reference-seed "
            "metrics, and available repeated-subsample results into a supplementary "
            "candidate-model selection table. This does not refit clustering models."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--models", default="configs/manuscript_models.yaml"
    )
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    summary, output_path = build_summary(args.config, args.models, args.run_dir)
    print(f"Rows: {len(summary)}")
    print(f"Output: {output_path}")
    print(
        summary.loc[
            summary["selection_status"].ne("not_shortlisted"),
            [
                "candidate",
                "selection_status",
                "silhouette_mean",
                "davies_bouldin_mean",
                "seed_ari_mean",
                "mean_subsample_ari",
                "minimum_cluster_fraction_min",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
