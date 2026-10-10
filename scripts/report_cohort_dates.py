"""Report aggregate index-date coverage of frozen stroke and sepsis cohorts.

Reads only IDs and verified index dates, never exports patient-level data.

Examples:
    python scripts/report_cohort_dates.py
    python scripts/report_cohort_dates.py --sepsis-index-file PATH/first_sepsis_encounters.parquet

The optional index file is joined by PATID to the exact frozen analysis cohort.
Unmatched patients are reported, not silently excluded from provenance checks.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_FILES = {
    "stroke": Path("data/stroke/stroke_cohort_imputed.parquet"),
    "sepsis": Path("data/sepsis/sepsis_cohort_imputed_safe.parquet"),
}
DEFAULT_INDEX_COLUMNS = {"stroke": "DX_DATE_stroke", "sepsis": "DX_DATE_sepsis"}
EXPECTED_COHORT_N = {"stroke": 9835, "sepsis": 15842}


def _column_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pq.ParquetFile(path).schema_arrow.names
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    raise ValueError(f"Unsupported file format for {path}; use Parquet or CSV.")


def _read_columns(path: Path, columns: list[str], id_column: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, usecols=columns, dtype={id_column: "string"})
    else:
        data = pd.read_parquet(path, columns=columns)
    # Normalize ID dtype for joins, but do not remove leading zeros or alter values.
    data[id_column] = data[id_column].astype("string")
    return data


def summarize_cohort(
    path: str | Path,
    *,
    cohort: str,
    date_column: str,
    id_column: str = "PATID",
    expected_n: int | None = None,
    index_file: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize the date range, verifying IDs if dates come from another file."""
    path = Path(path)
    available = _column_names(path)
    if id_column not in available:
        raise ValueError(f"{cohort}: ID column {id_column!r} is missing from {path}.")

    has_internal_dates = date_column in available
    if not has_internal_dates and index_file is None:
        candidates = [
            name for name in available
            if "DATE" in name.upper() or "TIME" in name.upper()
        ]
        raise ValueError(
            f"{cohort}: missing [{date_column!r}] in {path}. "
            f"Available date-like columns: {candidates}. "
            f"To use verified dates from the original extraction, pass "
            f"--{cohort}-index-file PATH/first_{cohort}_encounters.parquet. "
            f"Alternatively, use --{cohort}-column for a verified index date."
        )

    source = Path(index_file) if index_file is not None else path
    if index_file is not None:
        source_columns = _column_names(source)
        missing_source = [
            col for col in [id_column, date_column] if col not in source_columns
        ]
        if missing_source:
            raise ValueError(
                f"{cohort}: source {source} is missing {missing_source}. "
                f"Available date-like columns: "
                f"{[x for x in source_columns if 'DATE' in x.upper()]}"
            )

    use_source_join = source != path
    frozen = _read_columns(
        path, [id_column] if use_source_join else [id_column, date_column], id_column
    )

    joined_count = len(frozen)
    unmatched = 0
    source_extra = 0
    if use_source_join:
        dates = _read_columns(source, [id_column, date_column], id_column)
        source_missing_ids = int(dates[id_column].isna().sum())
        # pandas.merge matches null keys. Exclude null source identifiers instead.
        dates = dates.dropna(subset=[id_column])
        repeated_ids = int(dates[id_column].duplicated().sum())
        if repeated_ids:
            raise ValueError(
                f"{cohort}: date source has {repeated_ids} duplicate patient IDs. "
                "Resolve the first-encounter record before joining."
            )
        valid_frozen_ids = frozen[id_column].dropna().unique()
        source_extra = int((~dates[id_column].isin(valid_frozen_ids)).sum())
        frozen = frozen.merge(
            dates, on=id_column, how="left", validate="many_to_one", indicator=True
        )
        unmatched = int((frozen["_merge"] == "left_only").sum())
        frozen = frozen.drop(columns=["_merge"])
    else:
        source_missing_ids = 0

    values = frozen[date_column]
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    missing_dates = int(values.isna().sum())
    unparseable = int(parsed.isna().sum()) - missing_dates
    valid = parsed.dropna()
    old_cutoff = pd.Timestamp("1900-01-01", tz="UTC")
    tomorrow = pd.Timestamp(date.today(), tz="UTC") + pd.Timedelta(days=1)

    return {
        "cohort": cohort,
        "file": str(path),
        "index_date_source_file": str(source),
        "index_date_source_joined_by_patient_id": use_source_join,
        "index_date_column": date_column,
        "patient_id_column": id_column,
        "row_count": int(joined_count),
        "unique_patients": int(frozen[id_column].nunique(dropna=True)),
        "missing_patient_ids": int(frozen[id_column].isna().sum()),
        "duplicate_patient_ids": int(frozen[id_column].dropna().duplicated().sum()),
        "expected_row_count": expected_n,
        "matches_expected_count": expected_n is None or joined_count == expected_n,
        "unmatched_patients_in_date_source": unmatched,
        "source_patients_not_in_frozen_cohort": source_extra,
        "date_source_missing_patient_ids": source_missing_ids,
        "all_frozen_patients_matched": unmatched == 0,
        "valid_index_dates": int(len(valid)),
        "missing_index_dates": missing_dates,
        "unparseable_index_dates": unparseable,
        "first_index_date": valid.min().date().isoformat() if len(valid) else None,
        "last_index_date": valid.max().date().isoformat() if len(valid) else None,
        "index_dates_before_1900": int((valid < old_cutoff).sum()),
        "index_dates_after_today": int((valid > tomorrow).sum()),
        "range_complete_for_frozen_cohort": (
            unmatched == 0 and not missing_dates and not unparseable
            and bool(len(valid))
        ),
        "note": (
            "Date-source and analysis cohort releases must also be verified "
            "as the same extraction; matching IDs alone does not prove provenance."
        ) if use_source_join else "Index date is taken from the frozen analysis file.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stroke", type=Path, default=DEFAULT_FILES["stroke"])
    parser.add_argument("--sepsis", type=Path, default=DEFAULT_FILES["sepsis"])
    parser.add_argument("--stroke-column", default=DEFAULT_INDEX_COLUMNS["stroke"])
    parser.add_argument("--sepsis-column", default=DEFAULT_INDEX_COLUMNS["sepsis"])
    parser.add_argument(
        "--stroke-index-file", type=Path, default=None,
        help="Explicit original first-stroke-encounter file, if needed.",
    )
    parser.add_argument(
        "--sepsis-index-file", type=Path, default=None,
        help="Original first-sepsis-encounter Parquet/CSV containing PATID and DX_DATE_sepsis.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/provenance/cohort_index_date_summary.json"),
        help="Aggregate-only JSON output (under gitignored results/ by default).",
    )
    args = parser.parse_args(argv)

    results = []
    errors = []
    for cohort in ("stroke", "sepsis"):
        try:
            results.append(
                summarize_cohort(
                    getattr(args, cohort),
                    cohort=cohort,
                    date_column=getattr(args, f"{cohort}_column"),
                    expected_n=EXPECTED_COHORT_N[cohort],
                    index_file=getattr(args, f"{cohort}_index_file"),
                )
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            errors.append({"cohort": cohort, "error": str(exc)})

    report = {
        "description": "Frozen cohort index-date provenance (aggregate only)",
        "cohorts": results,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved aggregate report to {args.output}")

    if any(
        not r["matches_expected_count"] or not r["range_complete_for_frozen_cohort"]
        or r["duplicate_patient_ids"]
        or r["index_dates_before_1900"] or r["index_dates_after_today"]
        for r in results
    ):
        print("WARNING: Review cohort counts, date coverage, and date quality before publication.")
    if errors:
        print("INCOMPLETE: Fix the reported errors before using both study periods.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
