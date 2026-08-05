from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PATIENT_ALIASES = [
    "PATID",
    "PATIENT_ID",
    "PATIENTID",
    "PERSON_ID",
    "PERSONID",
    "MRN",
]
ENCOUNTER_ALIASES = [
    "ENCOUNTERID",
    "ENCOUNTER_ID",
    "ENCOUNTERID_x",
    "VISIT_ID",
    "VISITID",
    "INDEX_ENCOUNTER_ID",
]
DATE_HINTS = (
    "date",
    "admit",
    "discharge",
    "encounter_start",
    "encounter_end",
    "index_time",
    "index_datetime",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_completed_run(output_dir: str | Path) -> Path:
    candidates = [
        path
        for path in Path(output_dir).iterdir()
        if path.is_dir() and (path / "internal_metrics.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run found under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def matching_columns(columns: list[str], aliases: list[str]) -> list[str]:
    lookup = {str(column).lower(): str(column) for column in columns}
    return [lookup[alias.lower()] for alias in aliases if alias.lower() in lookup]


def duplicated_value_summary(series: pd.Series) -> dict[str, Any]:
    nonmissing = series.dropna()
    counts = nonmissing.value_counts(dropna=False)
    repeated = counts[counts > 1]
    duplicate_rows = int(counts.sub(1).clip(lower=0).sum())
    return {
        "nonmissing_n": int(nonmissing.shape[0]),
        "missing_n": int(series.isna().sum()),
        "unique_n": int(nonmissing.nunique(dropna=True)),
        "values_repeated_n": int(repeated.shape[0]),
        "duplicate_rows_beyond_first_n": duplicate_rows,
        "maximum_rows_per_value": int(counts.max()) if not counts.empty else 0,
        "is_unique_among_nonmissing": bool(repeated.empty),
    }


def build_audit(config_path: str | Path, run_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    cfg = load_yaml(config_path)
    dataset = str(cfg["dataset_name"])
    input_path = Path(cfg["input_path"])
    frame = pd.read_parquet(input_path)
    configured_id = str(cfg["id_column"])
    if configured_id not in frame.columns:
        raise KeyError(f"Configured ID column not found: {configured_id}")

    run_dir = Path(run_dir) if run_dir else latest_completed_run(cfg["output_dir"])
    output_root = run_dir / "manuscript_outputs"
    table_dir = output_root / "tables"
    manifest_dir = output_root / "manifests"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    columns = [str(column) for column in frame.columns]
    patient_columns = matching_columns(columns, PATIENT_ALIASES)
    encounter_columns = matching_columns(columns, ENCOUNTER_ALIASES)
    date_columns = [
        column for column in columns if any(hint in column.lower() for hint in DATE_HINTS)
    ]

    candidate_columns: list[tuple[str, str]] = [(configured_id, "configured_id")]
    for column in patient_columns:
        if column != configured_id:
            candidate_columns.append((column, "patient_identifier_candidate"))
    for column in encounter_columns:
        if column != configured_id:
            candidate_columns.append((column, "encounter_identifier_candidate"))

    rows: list[dict[str, Any]] = []
    for column, role in candidate_columns:
        summary = duplicated_value_summary(frame[column])
        rows.append(
            {
                "dataset": dataset,
                "column": column,
                "role": role,
                "total_rows": int(len(frame)),
                **summary,
            }
        )

    table = pd.DataFrame(rows)
    configured_summary = duplicated_value_summary(frame[configured_id])
    unique_configured_id = configured_summary["is_unique_among_nonmissing"] and configured_summary["missing_n"] == 0

    if unique_configured_id:
        inferred_unit = "one analytic row per configured identifier"
        duplicate_risk = "none detected for configured identifier"
    else:
        inferred_unit = "multiple analytic rows per configured identifier"
        duplicate_risk = "repeated configured identifiers detected; patient-level independence requires review"

    manifest: dict[str, Any] = {
        "dataset": dataset,
        "input_path": str(input_path),
        "total_rows": int(len(frame)),
        "configured_id_column": configured_id,
        "configured_id_summary": configured_summary,
        "patient_identifier_candidates_present": patient_columns,
        "encounter_identifier_candidates_present": encounter_columns,
        "date_or_time_columns_present": date_columns,
        "inferred_analytic_unit_from_frozen_file": inferred_unit,
        "duplicate_patient_or_encounter_risk": duplicate_risk,
        "interpretation_limits": [
            "Uniqueness of the configured identifier establishes one row per configured identifier, not necessarily one row per biologic patient unless the identifier is confirmed to be patient-level.",
            "Absence of a separate encounter identifier prevents independent confirmation that the configured identifier represents a patient rather than an index encounter.",
            "Cohort-construction and index-event rules must be confirmed from the upstream source pipeline or data dictionary.",
        ],
        "clustering_assignment_merge_contract": "The manuscript-output generator merges assignments to the frozen cohort using validate='one_to_one'.",
        "clustering_features_include_configured_id": configured_id in cfg.get("features", {}).get("primary", []),
    }

    table_path = table_dir / "tableS_unit_of_analysis_duplicate_audit.csv"
    manifest_path = manifest_dir / "unit_of_analysis_duplicate_audit.json"
    table.to_csv(table_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return table, manifest, table_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen cohort unit of analysis and repeated patient/encounter identifiers without rerunning clustering."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    table, manifest, table_path, manifest_path = build_audit(args.config, args.run_dir)
    print(table.to_string(index=False))
    print(f"Table: {table_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Inferred unit: {manifest['inferred_analytic_unit_from_frozen_file']}")
    print(f"Duplicate risk: {manifest['duplicate_patient_or_encounter_risk']}")


if __name__ == "__main__":
    main()
