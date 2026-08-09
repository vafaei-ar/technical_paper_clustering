from __future__ import annotations

import pandas as pd

import generate_author_revision_materials as arm


def test_continuous_summary_uses_one_decimal_iqr():
    row = arm._continuous_summary(pd.Series([1.0, 2.0, 3.0, 4.0]), "Test", "Age")
    assert row["summary"] == "2.5 (1.8-3.2)"
    assert row["n"] == 4


def test_categorical_summary_reports_counts_and_percentages():
    rows = arm._categorical_summary(pd.Series(["A", "A", "B", None]), "Test", "Sex")
    by_level = {row["level"]: row for row in rows}
    assert by_level["A"]["summary"] == "2 (50.0%)"
    assert by_level["B"]["summary"] == "1 (25.0%)"
    assert by_level["Missing"]["summary"] == "1 (25.0%)"
