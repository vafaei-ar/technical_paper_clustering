from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from generate_unit_of_analysis_audit import build_audit


def write_config(tmp_path: Path, input_path: Path, output_dir: Path) -> Path:
    config = {
        "dataset_name": "demo",
        "input_path": str(input_path),
        "id_column": "PATID",
        "output_dir": str(output_dir),
        "features": {"primary": ["feature_a"]},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def make_run(output_dir: Path) -> Path:
    run_dir = output_dir / "20260804_120000"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_csv(run_dir / "internal_metrics.csv", index=False)
    return run_dir


def test_unique_configured_identifier(tmp_path: Path) -> None:
    input_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "PATID": [1, 2, 3],
            "ENCOUNTERID": [10, 20, 30],
            "ADMIT_DATE": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "feature_a": [0.1, 0.2, 0.3],
        }
    ).to_parquet(input_path, index=False)
    output_dir = tmp_path / "results"
    run_dir = make_run(output_dir)
    config_path = write_config(tmp_path, input_path, output_dir)

    table, manifest, table_path, manifest_path = build_audit(config_path, run_dir)

    patid = table.loc[table.column.eq("PATID")].iloc[0]
    assert bool(patid.is_unique_among_nonmissing)
    assert patid.duplicate_rows_beyond_first_n == 0
    assert manifest["inferred_analytic_unit_from_frozen_file"] == "one analytic row per configured identifier"
    assert manifest["encounter_identifier_candidates_present"] == ["ENCOUNTERID"]
    assert "ADMIT_DATE" in manifest["date_or_time_columns_present"]
    assert table_path.exists()
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["clustering_features_include_configured_id"] is False


def test_repeated_configured_identifier_is_flagged(tmp_path: Path) -> None:
    input_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "PATID": [1, 1, 2, None],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_parquet(input_path, index=False)
    output_dir = tmp_path / "results"
    run_dir = make_run(output_dir)
    config_path = write_config(tmp_path, input_path, output_dir)

    table, manifest, _, _ = build_audit(config_path, run_dir)

    row = table.iloc[0]
    assert row.values_repeated_n == 1
    assert row.duplicate_rows_beyond_first_n == 1
    assert row.maximum_rows_per_value == 2
    assert not bool(row.is_unique_among_nonmissing)
    assert manifest["inferred_analytic_unit_from_frozen_file"] == "multiple analytic rows per configured identifier"
    assert "requires review" in manifest["duplicate_patient_or_encounter_risk"]
