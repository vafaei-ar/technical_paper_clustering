"""Data-free regression test for the aggregate patient-overlap CLI.

All identifiers below are synthetic test strings, never research records.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_stroke_overlap_and_label_join_use_patient_ids(tmp_path: Path):
    technical = tmp_path / "technical.csv"
    clinical = tmp_path / "clinical.csv"
    tech_labels = tmp_path / "technical_labels.csv"
    clin_labels = tmp_path / "clinical_labels.csv"

    pd.DataFrame({
        "PATID": ["test-A", "test-B", "test-C", "test-D"],
        "Ischemic_Stroke": [1, 1, 1, 0],
        "Hemorrhagic_Stroke": [0, 0, 1, 1],
    }).to_csv(technical, index=False)
    # Deliberately reversed patient order: row-order matching is invalid.
    pd.DataFrame({"PATID": ["test-B", "test-A"]}).to_csv(clinical, index=False)
    pd.DataFrame({
        "technical_PATID": ["test-D", "test-C", "test-A", "test-B"],
        "primary_cluster": ["Other", "Dual", "T0", "T1"],
    }).to_csv(tech_labels, index=False)
    pd.DataFrame({
        "clinical_PATID": ["test-B", "test-A"],
        "clinical_cluster": ["C1", "C0"],
    }).to_csv(clin_labels, index=False)

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_stroke_cohort_overlap.py"
    result = subprocess.run([
        sys.executable, str(script),
        "--technical-cohort", str(technical),
        "--clinical-cohort", str(clinical),
        "--outdir", str(tmp_path / "output"),
        "--expected-technical", "4",
        "--expected-clinical", "2",
        "--technical-labels", str(tech_labels),
        "--clinical-labels", str(clin_labels),
        "--technical-label-id-column", "technical_PATID",
        "--clinical-label-id-column", "clinical_PATID",
        "--technical-label-column", "primary_cluster",
        "--clinical-label-column", "clinical_cluster",
    ], check=True, capture_output=True, text=True)
    summary_path = tmp_path / "output" / "stroke_cohort_overlap_aggregate.json"
    summary = json.loads(summary_path.read_text())
    assert summary["clinical_exactly_equals_technical_ischemic_only"]
    assert summary["clinical_subset_of_technical"]
    assert summary["shared_n"] == 2
    assert summary["technical_indicator_counts"]["dual_coded"] == 1
    assert summary["technical_indicator_counts"]["hemorrhagic_only"] == 1
    assert summary["cluster_ari_shared_patients"] == 1.0
    assert "test-A" not in summary_path.read_text()
    assert "test-B" not in summary_path.read_text()
    assert "Aggregate results only" in result.stdout
