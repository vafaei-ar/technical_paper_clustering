from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from generate_candidate_model_summary import build_summary


def test_build_summary_writes_complete_grid(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "stroke" / "run"
    run_dir.mkdir(parents=True)

    metric_rows = []
    for reduction, clusterer, k in [
        ("pca", "kmeans", 3),
        ("none", "agglomerative", 2),
    ]:
        for seed, offset in [(11, 0.00), (23, 0.01)]:
            metric_rows.append(
                {
                    "dataset": "stroke",
                    "reduction": reduction,
                    "clusterer": clusterer,
                    "k": k,
                    "seed": seed,
                    "n_components": 8 if reduction == "pca" else 28,
                    "silhouette": 0.30 + offset,
                    "calinski_harabasz": 100.0 + seed,
                    "davies_bouldin": 1.1 - offset,
                    "minimum_cluster_n": 100,
                    "minimum_cluster_fraction": 0.10,
                    "passes_cluster_size_gate": True,
                }
            )
    pd.DataFrame(metric_rows).to_csv(run_dir / "internal_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "dataset": "stroke",
                "reduction": reduction,
                "clusterer": clusterer,
                "k": k,
                "seed_a": 11,
                "seed_b": 23,
                "adjusted_rand_index": ari,
            }
            for reduction, clusterer, k, ari in [
                ("pca", "kmeans", 3, 0.98),
                ("none", "agglomerative", 2, 1.00),
            ]
        ]
    ).to_csv(run_dir / "seed_stability.csv", index=False)

    pd.DataFrame(
        [
            {
                "dataset": "stroke",
                "candidate": "stroke_pca_kmeans_k3",
                "reduction": "pca",
                "clusterer": "kmeans",
                "k": 3,
                "mean_subsample_ari": 0.97,
                "median_subsample_ari": 0.98,
                "sd_subsample_ari": 0.01,
                "minimum_subsample_ari": 0.94,
                "p10_subsample_ari": 0.95,
                "p25_subsample_ari": 0.96,
                "minimum_cluster_fraction": 0.09,
                "mean_minimum_cluster_fraction": 0.10,
            }
        ]
    ).to_csv(run_dir / "subsample_stability_summary.csv", index=False)

    config_path = tmp_path / "stroke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "stroke",
                "output_dir": str(tmp_path / "results" / "stroke"),
                "seeds": [11, 23],
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
                        "reference_seed": 11,
                        "rationale": "Primary test model.",
                    },
                    "profiled": [
                        {
                            "name": "stroke_raw_agg_k2",
                            "reduction": "none",
                            "clusterer": "agglomerative",
                            "k": 2,
                            "reference_seed": 11,
                            "rationale": "Profiled comparison.",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    summary, output_path = build_summary(config_path, models_path, run_dir)

    assert output_path.exists()
    assert len(summary) == 2
    primary = summary.loc[summary["selected_primary"]].iloc[0]
    assert primary["candidate"] == "stroke_pca_kmeans_k3"
    assert primary["selection_status"] == "primary"
    assert primary["mean_subsample_ari"] == 0.97
    assert primary["seed_ari_mean"] == 0.98
    assert summary["all_seeds_pass_cluster_size_gate"].all()
