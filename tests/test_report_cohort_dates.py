"""Aggregate provenance checks tested on synthetic records, never on PHI."""

import pandas as pd
import pytest

from scripts.report_cohort_dates import summarize_cohort


def test_index_date_range_and_missingness(tmp_path):
    path = tmp_path / "stroke.parquet"
    pd.DataFrame({
        "PATID": ["a", "b", "c", "d"],
        "DX_DATE_stroke": ["2014-01-03", "2020-02-29", None, "bad-date"],
        "unused_private_field": ["patient-private"] * 4,
    }).to_parquet(path, index=False)

    summary = summarize_cohort(
        path, cohort="stroke", date_column="DX_DATE_stroke", expected_n=4
    )
    assert summary["first_index_date"] == "2014-01-03"
    assert summary["last_index_date"] == "2020-02-29"
    assert summary["row_count"] == 4
    assert summary["unique_patients"] == 4
    assert summary["valid_index_dates"] == 2
    assert summary["missing_index_dates"] == 1
    assert summary["unparseable_index_dates"] == 1
    assert summary["matches_expected_count"] is True
    assert "patient-private" not in str(summary)


def test_index_date_does_not_silently_fallback(tmp_path):
    path = tmp_path / "sepsis.parquet"
    pd.DataFrame({
        "PATID": ["p1"],
        "ADMIT_DATE": ["2018-04-10"],
    }).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Available date-like columns"):
        summarize_cohort(path, cohort="sepsis", date_column="DX_DATE_sepsis")


def test_duplicates_are_reported(tmp_path):
    path = tmp_path / "stroke.parquet"
    pd.DataFrame({
        "PATID": [1, 1, 2],
        "DX_DATE_stroke": pd.to_datetime(
            ["2018-01-01", "2018-02-01", "2019-01-01"]
        ),
    }).to_parquet(path, index=False)

    summary = summarize_cohort(
        path, cohort="stroke", date_column="DX_DATE_stroke"
    )
    assert summary["row_count"] == 3
    assert summary["unique_patients"] == 2
    assert summary["duplicate_patient_ids"] == 1
