from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from examples.verify_synthetic_recovery import verify_recovery


def test_verify_recovery_writes_reports(tmp_path: Path) -> None:
    cohort = pd.DataFrame(
        {
            "PATID": ["A", "B", "C", "D", "E", "F"],
            "feature_a": [0, 0, 1, 1, 2, 2],
            "cluster_truth": [0, 0, 1, 1, 2, 2],
        }
    )
    input_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(input_path, index=False)

    run_dir = tmp_path / "results" / "20260804_000000"
    run_dir.mkdir(parents=True)
    assignments = pd.DataFrame(
        {
            "PATID": cohort["PATID"],
            "dataset": "synthetic",
            "reduction": "pca",
            "clusterer": "kmeans",
            "k": 3,
            "seed": 11,
            "cluster": [2, 2, 0, 0, 1, 1],
        }
    )
    assignments.to_parquet(run_dir / "cluster_assignments.parquet", index=False)

    config = {
        "dataset_name": "synthetic",
        "input_path": str(input_path),
        "id_column": "PATID",
        "output_dir": str(tmp_path / "results"),
        "features": {"primary": ["feature_a"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = verify_recovery(config_path, run_dir=run_dir, minimum_ari=0.80)

    assert report["adjusted_rand_index"] == 1.0
    assert report["passes_verification"] is True
    assert report["truth_excluded_from_clustering_features"] is True

    verification_dir = run_dir / "verification"
    assert (verification_dir / "synthetic_recovery_verification.csv").exists()
    json_report = json.loads(
        (verification_dir / "synthetic_recovery_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert json_report["passes_verification"] is True
