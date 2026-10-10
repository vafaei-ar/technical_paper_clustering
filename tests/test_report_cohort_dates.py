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


def test_join_to_original_first_sepsis_encounters(tmp_path):
    frozen_path = tmp_path / "sepsis_cohort_imputed_safe.parquet"
    dates_path = tmp_path / "first_sepsis_encounters.parquet"
    pd.DataFrame({
        "PATID": ["001", "002", "003"],
        "Glucose": [5.0, 6.0, 7.0],
    }).to_parquet(frozen_path, index=False)
    pd.DataFrame({
        "PATID": ["001", "002", "003", "other"],
        "DX_DATE_sepsis": ["2015-06-12", "2018-03-04", "2020-07-19", "1980-01-01"],
    }).to_parquet(dates_path, index=False)

    summary = summarize_cohort(
        frozen_path, cohort="sepsis", date_column="DX_DATE_sepsis",
        expected_n=3, index_file=dates_path,
    )
    assert summary["first_index_date"] == "2015-06-12"
    assert summary["last_index_date"] == "2020-07-19"
    assert summary["valid_index_dates"] == 3
    assert summary["source_patients_not_in_frozen_cohort"] == 1
    assert summary["unmatched_patients_in_date_source"] == 0
    assert summary["range_complete_for_frozen_cohort"] is True


def test_partial_join_is_marked_incomplete(tmp_path):
    frozen_path = tmp_path / "sepsis.parquet"
    dates_path = tmp_path / "first_sepsis_encounters.parquet"
    pd.DataFrame({"PATID": ["1", "2", "3"]}).to_parquet(frozen_path)
    pd.DataFrame({
        "PATID": ["1", "2"],
        "DX_DATE_sepsis": ["2018-01-01", "2019-01-01"],
    }).to_parquet(dates_path)
    summary = summarize_cohort(
        frozen_path, cohort="sepsis", date_column="DX_DATE_sepsis",
        index_file=dates_path,
    )
    assert summary["unmatched_patients_in_date_source"] == 1
    assert summary["valid_index_dates"] == 2
    assert summary["missing_index_dates"] == 1
    assert summary["range_complete_for_frozen_cohort"] is False


def test_duplicate_source_ids_fail_safely(tmp_path):
    frozen_path = tmp_path / "sepsis.parquet"
    dates_path = tmp_path / "first_sepsis_encounters.parquet"
    pd.DataFrame({"PATID": ["1", "2"]}).to_parquet(frozen_path)
    pd.DataFrame({
        "PATID": ["1", "1"],
        "DX_DATE_sepsis": ["2017-01-01", "2020-01-01"],
    }).to_parquet(dates_path)
    with pytest.raises(ValueError, match="duplicate patient IDs"):
        summarize_cohort(
            frozen_path, cohort="sepsis", date_column="DX_DATE_sepsis",
            index_file=dates_path,
        )


def test_csv_source_can_be_used(tmp_path):
    frozen_path = tmp_path / "sepsis.parquet"
    dates_path = tmp_path / "first_sepsis_encounters.csv"
    pd.DataFrame({"PATID": ["001", "002"]}).to_parquet(frozen_path)
    pd.DataFrame({
        "PATID": ["001", "002"],
        "DX_DATE_sepsis": ["2017-01-01", "2020-01-01"],
    }).to_csv(dates_path, index=False)
    summary = summarize_cohort(
        frozen_path, cohort="sepsis", date_column="DX_DATE_sepsis",
        index_file=dates_path,
    )
    assert summary["unmatched_patients_in_date_source"] == 0
    assert summary["first_index_date"] == "2017-01-01"


def test_cli_keeps_stroke_summary_when_sepsis_dates_missing(tmp_path):
    from scripts.report_cohort_dates import main
    stroke_path = tmp_path / "stroke.parquet"
    sepsis_path = tmp_path / "sepsis.parquet"
    output_path = tmp_path / "report.json"
    pd.DataFrame({
        "PATID": ["s1"],
        "DX_DATE_stroke": ["2015-04-06"],
    }).to_parquet(stroke_path)
    pd.DataFrame({"PATID": ["p1"]}).to_parquet(sepsis_path)
    exit_code = main([
        "--stroke", str(stroke_path),
        "--sepsis", str(sepsis_path),
        "--output", str(output_path),
    ])
    import json
    report = json.loads(output_path.read_text())
    assert exit_code == 2
    assert report["cohorts"][0]["first_index_date"] == "2015-04-06"
    assert report["errors"][0]["cohort"] == "sepsis"

