from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from generate_outcome_definition_audit import build_audit, infer_binary_threshold


def test_infer_binary_threshold_exact() -> None:
    duration = pd.Series([1, 2, 3, 4, 5, 6])
    outcome = pd.Series([0, 0, 0, 1, 1, 1])
    result = infer_binary_threshold(duration, outcome)
    assert result["threshold_identified"] is True
    assert result["agreement"] == 1.0


def test_build_outcome_audit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "results" / "run1"
    (run_dir / "manuscript_outputs").mkdir(parents=True)
    data_dir.mkdir()

    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "enc_duration": [2.0, 5.0, 8.0, 10.0],
            "prolonged_los": [0, 0, 1, 1],
            "DISCHARGE_STATUS": ["HO", "SN", "EX", None],
        }
    )
    input_path = data_dir / "cohort.parquet"
    frame.to_parquet(input_path, index=False)

    config = {
        "dataset_name": "demo",
        "input_path": str(input_path),
        "id_column": "ID",
        "output_dir": str(tmp_path / "results"),
        "features": {"primary": ["feature_a"]},
    }
    profile = {
        "demo": {
            "continuous_outcomes": ["enc_duration"],
            "categorical_outcomes": ["DISCHARGE_STATUS"],
            "binary_outcomes": ["prolonged_los"],
        }
    }
    config_path = tmp_path / "config.yaml"
    profile_path = tmp_path / "profiles.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    table, audit, table_path, manifest_path = build_audit(
        config_path, profile_path, run_dir
    )
    assert len(table) == 3
    assert table["excluded_from_clustering_features"].all()
    assert audit["outcomes_excluded_from_clustering"] is True
    assert audit["prolonged_los_threshold_inference"]["agreement"] == 1.0
    assert table_path.exists()
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["duration_definition"]["derivation_available_in_repository"] is False
