from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .core import fit_clusterer, prepare_matrix, reduce_matrix


def _minimum_cluster_gate(labels: np.ndarray, cfg: dict) -> tuple[bool, int, float]:
    counts = pd.Series(labels).value_counts()
    minimum_n = int(counts.min())
    minimum_fraction = float(minimum_n / len(labels))
    required_n = max(
        int(cfg.get("minimum_cluster_n", 30)),
        int(np.ceil(len(labels) * float(cfg.get("minimum_cluster_fraction", 0.01)))),
    )
    return minimum_n >= required_n, minimum_n, minimum_fraction


def run(config_path: str | Path) -> Path:
    with open(config_path, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    frame = pd.read_parquet(cfg["input_path"])
    id_column = cfg["id_column"]
    if id_column not in frame.columns:
        raise KeyError(f"ID column not found: {id_column}")

    raw, processed, x_scaled, removed, plausibility, log_report, clipping = prepare_matrix(
        frame,
        cfg["features"]["primary"],
        cfg.get("preprocessing", {}),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output_dir"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    raw.to_parquet(output_dir / "analysis_features_raw.parquet", index=False)
    processed.to_parquet(output_dir / "analysis_features_processed.parquet", index=False)
    pd.DataFrame({"removed_feature": removed}).to_csv(output_dir / "removed_zero_variance_features.csv", index=False)
    plausibility.to_csv(output_dir / "plausibility_replacements.csv", index=False)
    log_report.to_csv(output_dir / "log1p_transform_report.csv", index=False)
    clipping.to_csv(output_dir / "clipping_report.csv", index=False)

    metric_rows: list[dict] = []
    assignment_rows: list[pd.DataFrame] = []
    seed_rows: list[dict] = []

    for reduction in cfg["reductions"]:
        for seed in cfg["seeds"]:
            z, reduction_info = reduce_matrix(x_scaled, reduction, int(seed))
            for clusterer in cfg["clusterers"]:
                for k in cfg["k_values"]:
                    labels = fit_clusterer(z, clusterer, int(k), int(seed))
                    passed, minimum_n, minimum_fraction = _minimum_cluster_gate(labels, cfg.get("quality_gates", {}))
                    metric_rows.append({
                        "dataset": cfg["dataset_name"],
                        "reduction": reduction,
                        "clusterer": clusterer,
                        "k": int(k),
                        "seed": int(seed),
                        "n_components": reduction_info["n_components"],
                        "silhouette": float(silhouette_score(z, labels)),
                        "calinski_harabasz": float(calinski_harabasz_score(z, labels)),
                        "davies_bouldin": float(davies_bouldin_score(z, labels)),
                        "minimum_cluster_n": minimum_n,
                        "minimum_cluster_fraction": minimum_fraction,
                        "passes_cluster_size_gate": passed,
                    })
                    assignment_rows.append(pd.DataFrame({
                        id_column: frame[id_column].to_numpy(),
                        "dataset": cfg["dataset_name"],
                        "reduction": reduction,
                        "clusterer": clusterer,
                        "k": int(k),
                        "seed": int(seed),
                        "cluster": labels,
                    }))

    metrics = pd.DataFrame(metric_rows)
    assignments = pd.concat(assignment_rows, ignore_index=True)
    metrics.to_csv(output_dir / "internal_metrics.csv", index=False)
    assignments.to_parquet(output_dir / "cluster_assignments.parquet", index=False)

    for (reduction, clusterer, k), group in assignments.groupby(["reduction", "clusterer", "k"]):
        by_seed = {int(seed): rows.sort_values(id_column)["cluster"].to_numpy() for seed, rows in group.groupby("seed")}
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            seed_rows.append({
                "dataset": cfg["dataset_name"],
                "reduction": reduction,
                "clusterer": clusterer,
                "k": int(k),
                "seed_a": seed_a,
                "seed_b": seed_b,
                "adjusted_rand_index": float(adjusted_rand_score(by_seed[seed_a], by_seed[seed_b])),
            })
    pd.DataFrame(seed_rows).to_csv(output_dir / "seed_stability.csv", index=False)

    print(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full clustering grid.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
