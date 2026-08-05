from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from generate_method_audit import pca_retention_table, stability_method_audit


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "results" / "stroke" / "20260101_000000"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "dataset": ["stroke"],
            "reduction": ["pca"],
            "clusterer": ["kmeans"],
            "k": [3],
            "seed": [11],
            "n_components": [7],
            "silhouette": [0.2],
            "calinski_harabasz": [100.0],
            "davies_bouldin": [1.8],
            "minimum_cluster_n": [50],
            "minimum_cluster_fraction": [0.1],
            "passes_cluster_size_gate": [True],
        }
    ).to_csv(run_dir / "internal_metrics.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["stroke", "stroke"],
            "candidate": ["stroke_pca_kmeans_k3", "stroke_pca_kmeans_k3"],
            "reduction": ["pca", "pca"],
            "clusterer": ["kmeans", "kmeans"],
            "k": [3, 3],
            "reference_seed": [11, 11],
            "subsample_seed": [1001, 1002],
            "sample_fraction": [0.8, 0.8],
            "sample_n": [80, 80],
            "ari_vs_full": [0.95, 0.96],
            "minimum_cluster_size": [10, 11],
            "maximum_cluster_size": [50, 49],
            "minimum_cluster_fraction": [0.125, 0.1375],
            "full_minimum_cluster_size": [15, 15],
            "full_minimum_cluster_fraction": [0.15, 0.15],
        }
    ).to_csv(run_dir / "subsample_stability_detailed.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["stroke"],
            "candidate": ["stroke_pca_kmeans_k3"],
            "reduction": ["pca"],
            "clusterer": ["kmeans"],
            "k": [3],
            "mean_subsample_ari": [0.955],
        }
    ).to_csv(run_dir / "subsample_stability_summary.csv", index=False)

    config_path = tmp_path / "stroke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "stroke",
                "output_dir": str(tmp_path / "results" / "stroke"),
                "features": {"primary": ["a", "b", "c"]},
                "seeds": [11],
            }
        ),
        encoding="utf-8",
    )
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "stroke": {
                    "primary": {
                        "name": "stroke_pca_kmeans_k3",
                        "reduction": "pca",
                        "clusterer": "kmeans",
                        "k": 3,
                        "seed": 11,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path, models_path, run_dir


def test_pca_retention_table(tmp_path: Path):
    config_path, models_path, run_dir = _write_fixture(tmp_path)
    table, output_path = pca_retention_table(config_path, models_path, run_dir)
    assert output_path.exists()
    assert table.loc[0, "retained_component_count"] == 7
    assert table.loc[0, "input_feature_count"] == 3


def test_stability_method_audit(tmp_path: Path):
    config_path, _, run_dir = _write_fixture(tmp_path)
    audit, output_path = stability_method_audit(config_path, run_dir)
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["pca_refit_within_each_subsample"] is True
    assert loaded["preprocessing_refit_within_each_subsample"] is False
    assert audit["n_repeats"] == 2
