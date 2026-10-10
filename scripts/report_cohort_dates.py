"""Report aggregate index-date coverage of frozen stroke/sepsis Parquet files.

This read-only check loads only the patient ID and index-date column. It never
exports patient-level records.

From the repository root:

    python scripts/report_cohort_dates.py

To use alternate input locations, pass --stroke PATH and --sepsis PATH.
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


def summarize_cohort(
    path: str | Path,
    *,
    cohort: str,
    date_column: str,
    id_column: str = "PATID",
    expected_n: int | None = None,
) -> dict[str, Any]:
    """Summarize index dates without loading any other patient-level columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{cohort}: cannot find {path}. Pass --{cohort} with the actual "
            "location of the frozen cohort Parquet file."
        )

    available = pq.ParquetFile(path).schema_arrow.names
    missing = [col for col in (id_column, date_column) if col not in available]
    if missing:
        date_candidates = [
            name for name in available
            if "DATE" in name.upper() or "TIME" in name.upper()
        ]
        raise ValueError(
            f"{cohort}: missing {missing} in {path}. "
            f"Available date-like columns: {date_candidates}. "
            f"Use --{cohort}-column for a verified index date. "
            "Do not silently substitute admission date."
        )

    frame = pd.read_parquet(path, columns=[id_column, date_column])
    original = frame[date_column]
    parsed = pd.to_datetime(original, errors="coerce", utc=True)
    missing_dates = int(original.isna().sum())
    unparseable_dates = int(parsed.isna().sum()) - missing_dates
    valid = parsed.dropna()
    old_cutoff = pd.Timestamp("1900-01-01", tz="UTC")
    tomorrow = pd.Timestamp(date.today(), tz="UTC") + pd.Timedelta(days=1)

    return {
        "cohort": cohort,
        "file": str(path),
        "index_date_column": date_column,
        "patient_id_column": id_column,
        "row_count": int(len(frame)),
        "unique_patients": int(frame[id_column].nunique(dropna=True)),
        "missing_patient_ids": int(frame[id_column].isna().sum()),
        "duplicate_patient_ids": int(frame[id_column].dropna().duplicated().sum()),
        "expected_row_count": expected_n,
        "matches_expected_count": expected_n is None or len(frame) == expected_n,
        "valid_index_dates": int(len(valid)),
        "missing_index_dates": missing_dates,
        "unparseable_index_dates": unparseable_dates,
        "first_index_date": valid.min().date().isoformat() if len(valid) else None,
        "last_index_date": valid.max().date().isoformat() if len(valid) else None,
        "index_dates_before_1900": int((valid < old_cutoff).sum()),
        "index_dates_after_today": int((valid > tomorrow).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stroke", type=Path, default=DEFAULT_FILES["stroke"])
    parser.add_argument("--sepsis", type=Path, default=DEFAULT_FILES["sepsis"])
    parser.add_argument("--stroke-column", default=DEFAULT_INDEX_COLUMNS["stroke"])
    parser.add_argument("--sepsis-column", default=DEFAULT_INDEX_COLUMNS["sepsis"])
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/provenance/cohort_index_date_summary.json"),
        help="Where to write aggregate-only JSON (default is gitignored).",
    )
    args = parser.parse_args(argv)

    results = [
        summarize_cohort(
            getattr(args, cohort),
            cohort=cohort,
            date_column=getattr(args, f"{cohort}_column"),
            expected_n=EXPECTED_COHORT_N[cohort],
        )
        for cohort in ("stroke", "sepsis")
    ]
    report = {
        "description": "Frozen cohort index-date provenance (aggregate only)",
        "cohorts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved aggregate report to {args.output}")

    if any(not row["matches_expected_count"] for row in results):
        print("WARNING: Cohort size differs from the manuscript sample size.")
    if any(
        row["missing_index_dates"] or row["unparseable_index_dates"]
        or row["index_dates_before_1900"] or row["index_dates_after_today"]
        for row in results
    ):
        print("WARNING: Review the date-quality flags before reporting the study period.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
