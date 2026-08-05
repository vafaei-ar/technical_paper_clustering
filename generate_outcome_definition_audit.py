from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DISCHARGE_GROUPS = {
    "HO": "Home/self-care",
    "RH": "Inpatient rehabilitation",
    "SN": "Skilled nursing facility",
    "EX": "Expired",
    "HS": "Hospice",
    "HH": "Home health",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_full_run(output_dir: str | Path) -> Path:
    candidates = [
        p for p in Path(output_dir).iterdir()
        if p.is_dir() and (p / "manuscript_outputs").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run found under {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def infer_binary_threshold(duration: pd.Series, outcome: pd.Series) -> dict[str, Any]:
    d = pd.to_numeric(duration, errors="coerce")
    y = pd.to_numeric(outcome, errors="coerce")
    valid = d.notna() & y.isin([0, 1])
    d = d[valid]
    y = y[valid].astype(int)
    result: dict[str, Any] = {
        "evaluated_n": int(valid.sum()),
        "threshold_identified": False,
        "operator": None,
        "threshold": None,
        "agreement": None,
    }
    if d.empty or y.nunique() < 2:
        return result

    positives = d[y == 1]
    negatives = d[y == 0]
    candidates = sorted(set([
        float(positives.min()),
        float(negatives.max()),
        float(np.nextafter(negatives.max(), np.inf)),
    ]))
    best: tuple[float, str, float] | None = None
    for threshold in candidates:
        for operator, prediction in (
            (">=", d >= threshold),
            (">", d > threshold),
        ):
            agreement = float((prediction.astype(int) == y).mean())
            candidate = (agreement, operator, threshold)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is not None:
        result.update({
            "threshold_identified": bool(best[0] == 1.0),
            "operator": best[1],
            "threshold": best[2],
            "agreement": best[0],
            "minimum_positive_duration": float(positives.min()),
            "maximum_negative_duration": float(negatives.max()),
        })
    return result


def build_audit(
    config_path: str | Path,
    profile_config_path: str | Path = "configs/profile_columns.yaml",
    run_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    cfg = load_yaml(config_path)
    profiles = load_yaml(profile_config_path)
    cohort = str(cfg["dataset_name"])
    profile_cfg = profiles[cohort]
    frame = pd.read_parquet(cfg["input_path"])
    run_dir = Path(run_dir) if run_dir else latest_full_run(cfg["output_dir"])

    primary = set(cfg["features"]["primary"])
    categories = {
        "continuous_outcome": profile_cfg.get("continuous_outcomes", []),
        "categorical_outcome": profile_cfg.get("categorical_outcomes", []),
        "binary_outcome": profile_cfg.get("binary_outcomes", []),
    }
    rows: list[dict[str, Any]] = []
    for role, variables in categories.items():
        for variable in variables:
            present = variable in frame.columns
            series = frame[variable] if present else pd.Series(dtype=float)
            numeric = pd.to_numeric(series, errors="coerce")
            row: dict[str, Any] = {
                "dataset": cohort,
                "variable": variable,
                "role": role,
                "present_in_input": present,
                "excluded_from_clustering_features": variable not in primary,
                "n_total": int(len(frame)),
                "n_missing": int(series.isna().sum()) if present else None,
                "missing_fraction": float(series.isna().mean()) if present else None,
                "n_unique_nonmissing": int(series.nunique(dropna=True)) if present else None,
                "minimum": float(numeric.min()) if present and numeric.notna().any() else None,
                "median": float(numeric.median()) if present and numeric.notna().any() else None,
                "maximum": float(numeric.max()) if present and numeric.notna().any() else None,
                "definition_source": "upstream analytic input; derivation code not present in this repository",
            }
            if role == "binary_outcome" and present:
                row["prevalence"] = float(numeric.mean())
            rows.append(row)

    table = pd.DataFrame(rows)
    audit: dict[str, Any] = {
        "dataset": cohort,
        "input_path": cfg["input_path"],
        "outcomes_excluded_from_clustering": bool(table["excluded_from_clustering_features"].all()),
        "duration_definition": {
            "variable": "enc_duration" if "enc_duration" in frame.columns else None,
            "derivation_available_in_repository": False,
            "unit_available_in_repository": False,
            "required_upstream_confirmation": [
                "start timestamp",
                "end timestamp",
                "time unit",
                "ED-to-inpatient merge handling",
                "negative or implausible duration handling",
            ],
        },
        "discharge_mapping": {
            "applied_in_manuscript_generator": cohort == "stroke" and "DISCHARGE_STATUS" in frame.columns,
            "mapping": DISCHARGE_GROUPS,
            "all_other_or missing codes": "Other institutional/other",
        },
    }

    if "DISCHARGE_STATUS" in frame.columns:
        counts = frame["DISCHARGE_STATUS"].astype("string").fillna("Missing").value_counts(dropna=False)
        audit["discharge_code_counts"] = {str(k): int(v) for k, v in counts.items()}
        audit["discharge_codes_not_explicitly_mapped"] = sorted(
            set(str(x) for x in counts.index) - set(DISCHARGE_GROUPS)
        )

    if "enc_duration" in frame.columns and "prolonged_los" in frame.columns:
        audit["prolonged_los_threshold_inference"] = infer_binary_threshold(
            frame["enc_duration"], frame["prolonged_los"]
        )
        audit["prolonged_los_definition_available_in_repository"] = False
        audit["prolonged_los_note"] = (
            "The threshold is inferred empirically from the frozen analytic variables; "
            "the upstream clinical definition and duration unit still require confirmation."
        )

    output_dir = run_dir / "manuscript_outputs"
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "tableS_outcome_definition_audit.csv"
    manifest_path = manifest_dir / "outcome_definition_audit.json"
    table.to_csv(table_path, index=False)
    manifest_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return table, audit, table_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit outcome variables without refitting clustering models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-config", default="configs/profile_columns.yaml")
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    table, audit, table_path, manifest_path = build_audit(
        args.config, args.profile_config, args.run_dir
    )
    print(table.to_string(index=False))
    print(f"Outcome table: {table_path}")
    print(f"Outcome audit: {manifest_path}")
    print(f"All outcomes excluded from clustering: {audit['outcomes_excluded_from_clustering']}")
    if "prolonged_los_threshold_inference" in audit:
        print("Prolonged LOS inference:", audit["prolonged_los_threshold_inference"])


if __name__ == "__main__":
    main()
