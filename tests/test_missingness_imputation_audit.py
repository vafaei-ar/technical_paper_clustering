from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from generate_missingness_imputation_audit import build_manifest, build_missingness_table, generate


def test_build_missingness_table_counts_repository_stage_missingness(tmp_path: Path) -> None:
    limits = tmp_path / "limits.yaml"
    limits.write_text("lab:\n  min: 0\n  max: 10\n", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "lab": [1.0, np.nan, 99.0, 4.0],
            "skewed": [0.0, -2.0, 3.0, np.nan],
        }
    )
    cfg = {
        "dataset_name": "demo",
        "input_path": "data/demo_imputed.parquet",
        "features": {"primary": ["lab", "skewed"]},
        "preprocessing": {
            "plausibility_limits_path": str(limits),
            "log1p_features": ["skewed"],
        },
    }

    table = build_missingness_table(frame, cfg).set_index("feature")
    assert table.loc["lab", "missing_at_repository_input_n"] == 1
    assert table.loc["lab", "plausibility_values_replaced_n"] == 1
    assert (
        table.loc["lab", "missing_immediately_before_repository_median_imputation_n"]
        == 2
    )
    assert table.loc["skewed", "missing_at_repository_input_n"] == 1
    assert table.loc["skewed", "negative_values_to_missing_before_log1p_n"] == 1
    assert (
        table.loc[
            "skewed", "missing_immediately_before_repository_median_imputation_n"
        ]
        == 2
    )
    assert bool(table.loc["lab", "source_filename_suggests_prior_imputation"])


def test_manifest_does_not_claim_upstream_method_is_known(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    cfg = {
        "dataset_name": "demo",
        "input_path": "data/demo_imputed.parquet",
        "features": {"primary": ["x"]},
        "preprocessing": {},
    }
    table = build_missingness_table(frame, cfg)
    manifest = build_manifest(cfg, table)
    assert manifest["upstream_imputation_status"]["known_from_this_repository"] is False
    assert manifest["repository_stage_imputation"]["method"] == "feature-wise median imputation"
    assert "fully observed raw measurements" in manifest["clinical_scale_interpretation"]


def test_generate_writes_table_and_manifest(tmp_path: Path) -> None:
    data_path = tmp_path / "demo.parquet"
    pd.DataFrame({"id": [1, 2, 3], "x": [1.0, np.nan, 3.0]}).to_parquet(data_path)
    output_root = tmp_path / "results"
    run_dir = output_root / "run1"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"silhouette": [0.1]}).to_csv(run_dir / "internal_metrics.csv", index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "demo",
                "input_path": str(data_path),
                "id_column": "id",
                "output_dir": str(output_root),
                "features": {"primary": ["x"]},
                "preprocessing": {},
            }
        ),
        encoding="utf-8",
    )

    table, table_path, manifest_path = generate(config_path, run_dir)
    assert len(table) == 1
    assert table_path.exists()
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "demo"
