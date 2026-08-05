from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tpcluster.core import apply_plausibility_limits


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_full_run(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    runs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and (path / "internal_metrics.csv").exists()
    ]
    if not runs:
        raise FileNotFoundError(f"No completed clustering run found under {output_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def build_missingness_table(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    features = list(cfg["features"]["primary"])
    preprocessing = cfg.get("preprocessing", {})
    missing_features = [feature for feature in features if feature not in frame.columns]
    if missing_features:
        raise KeyError(f"Missing configured features: {missing_features}")

    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    n_rows = len(numeric)
    original_missing = numeric.isna().sum()

    plausibility_adjusted, plausibility_report = apply_plausibility_limits(
        numeric,
        preprocessing.get("plausibility_limits_path"),
    )
    plausibility_lookup = (
        plausibility_report.set_index("feature")["replaced_n"].to_dict()
        if not plausibility_report.empty
        else {}
    )

    transformed = plausibility_adjusted.copy()
    negative_to_missing: dict[str, int] = {}
    for feature in preprocessing.get("log1p_features", []):
        if feature not in transformed.columns:
            continue
        values = pd.to_numeric(transformed[feature], errors="coerce")
        invalid = values < 0
        negative_to_missing[feature] = int(invalid.sum())
        transformed[feature] = np.log1p(values.mask(invalid))

    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    missing_before_median = transformed.isna().sum()
    medians = transformed.median(numeric_only=True)

    rows: list[dict[str, Any]] = []
    for feature in features:
        missing_n = int(original_missing[feature])
        pre_imputation_n = int(missing_before_median[feature])
        rows.append(
            {
                "dataset": cfg["dataset_name"],
                "feature": feature,
                "n_rows": n_rows,
                "missing_at_repository_input_n": missing_n,
                "missing_at_repository_input_pct": missing_n / n_rows if n_rows else np.nan,
                "plausibility_values_replaced_n": int(plausibility_lookup.get(feature, 0)),
                "negative_values_to_missing_before_log1p_n": int(negative_to_missing.get(feature, 0)),
                "missing_immediately_before_repository_median_imputation_n": pre_imputation_n,
                "missing_immediately_before_repository_median_imputation_pct": pre_imputation_n / n_rows if n_rows else np.nan,
                "repository_imputation_value_on_processed_scale": float(medians.get(feature, np.nan)),
                "repository_imputation_method": "single median imputation by feature after plausibility filtering and configured transformations",
                "source_input_filename": Path(cfg["input_path"]).name,
                "source_filename_suggests_prior_imputation": "imput" in Path(cfg["input_path"]).name.lower(),
            }
        )
    return pd.DataFrame(rows)


def build_manifest(cfg: dict[str, Any], table: pd.DataFrame) -> dict[str, Any]:
    n_rows = int(table["n_rows"].iloc[0]) if not table.empty else 0
    total_input_missing = int(table["missing_at_repository_input_n"].sum())
    total_pre_imputation_missing = int(
        table["missing_immediately_before_repository_median_imputation_n"].sum()
    )
    return {
        "dataset": cfg["dataset_name"],
        "input_path": cfg["input_path"],
        "n_rows": n_rows,
        "n_configured_features": int(len(table)),
        "repository_stage_imputation": {
            "method": "feature-wise median imputation",
            "timing": "after numeric coercion, plausibility filtering, configured log1p transformations, and replacement of infinite values; before zero-variance removal, clipping, scaling, PCA, and clustering",
            "fit_scope": "full analytic cohort",
            "multiple_imputation": False,
            "missingness_indicators_added": False,
            "total_missing_cells_at_repository_input": total_input_missing,
            "total_missing_cells_immediately_before_median_imputation": total_pre_imputation_missing,
        },
        "upstream_imputation_status": {
            "known_from_this_repository": False,
            "input_filename_suggests_prior_imputation": "imput" in Path(cfg["input_path"]).name.lower(),
            "interpretation": (
                "The repository cannot establish the upstream imputation algorithm or diagnostics from the parquet file alone. "
                "Those details must be obtained from the cohort-construction pipeline."
            ),
        },
        "clinical_scale_interpretation": (
            "Values described as original clinical scale are values from the repository input parquet before clustering transformations. "
            "They may include upstream-imputed values and therefore should not be described as fully observed raw measurements."
        ),
    }


def generate(config_path: str | Path, run_dir: str | Path | None = None) -> tuple[pd.DataFrame, Path, Path]:
    cfg = load_yaml(config_path)
    frame = pd.read_parquet(cfg["input_path"])
    run_dir = Path(run_dir) if run_dir else latest_full_run(cfg["output_dir"])
    table = build_missingness_table(frame, cfg)

    table_dir = run_dir / "manuscript_outputs" / "tables"
    manifest_dir = run_dir / "manuscript_outputs" / "manifests"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    table_path = table_dir / "tableS_missingness_and_repository_imputation.csv"
    manifest_path = manifest_dir / "missingness_imputation_audit.json"
    table.to_csv(table_path, index=False)
    manifest_path.write_text(
        json.dumps(build_manifest(cfg, table), indent=2), encoding="utf-8"
    )
    return table, table_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit feature-level missingness and the median-imputation step performed by this repository. "
            "This does not rerun clustering."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    table, table_path, manifest_path = generate(args.config, args.run_dir)
    print(f"Dataset: {table['dataset'].iloc[0]}")
    print(f"Rows: {int(table['n_rows'].iloc[0])}")
    print(f"Features: {len(table)}")
    print(f"Missing cells at repository input: {int(table['missing_at_repository_input_n'].sum())}")
    print(
        "Missing cells immediately before repository median imputation: "
        f"{int(table['missing_immediately_before_repository_median_imputation_n'].sum())}"
    )
    print(f"Table: {table_path}")
    print(f"Audit: {manifest_path}")


if __name__ == "__main__":
    main()
