from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from generate_anchor_nonanchor_summary import build_summary


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "results" / "stroke" / "20260101_000000"
    table_dir = run_dir / "manuscript_outputs" / "tables"
    table_dir.mkdir(parents=True)

    continuous_rows = []
    raw_rows = []
    for cluster in [0, 1, 2]:
        for feature, effect in [
            ("Glucose", 1.5 if cluster == 2 else -0.2),
            ("Hematocrit", -1.2 if cluster == 1 else 0.3),
            ("Creatinine", 0.9 if cluster == 1 else -0.1),
            ("Sodium", 0.7 - 0.1 * cluster),
            ("Platelets", -0.6 + 0.1 * cluster),
            ("Potassium", 0.4 + 0.1 * cluster),
            ("Leukocytes", 0.3 + 0.1 * cluster),
            ("Hemoglobin", -0.2 + 0.1 * cluster),
        ]:
            continuous_rows.append(
                {
                    "cluster": cluster,
                    "feature": feature,
                    "median_difference_iqr": effect,
                }
            )
            raw_rows.append(
                {
                    "cluster": cluster,
                    "feature": feature,
                    "median": 10 + cluster,
                    "q1": 9 + cluster,
                    "q3": 11 + cluster,
                    "n_nonmissing": 100,
                }
            )
    pd.DataFrame(continuous_rows).to_csv(
        table_dir / "table_continuous_cluster_profiles_processed_scale.csv",
        index=False,
    )
    pd.DataFrame(raw_rows).to_csv(
        table_dir / "table_continuous_cluster_profiles_raw_scale.csv",
        index=False,
    )

    binary_rows = []
    for cluster in [0, 1, 2]:
        for feature, effect in [
            ("Diabetes_Mellitus", 0.8 - 0.1 * cluster),
            ("Heart_failure", 0.6 - 0.1 * cluster),
            ("Hypertension", 0.5 - 0.1 * cluster),
        ]:
            binary_rows.append(
                {
                    "cluster": cluster,
                    "feature": feature,
                    "standardised_difference": effect,
                    "prevalence": 0.4,
                    "overall_prevalence": 0.3,
                }
            )
    pd.DataFrame(binary_rows).to_csv(
        table_dir / "table_binary_cluster_profiles.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "primary_cluster": [0, 1, 2],
            "primary_cluster_label": ["Preserved", "Renal-anaemic", "Hyperglycaemic"],
            "n": [120, 80, 60],
        }
    ).to_csv(table_dir / "table_cluster_sizes.csv", index=False)

    config = {
        "dataset_name": "stroke",
        "output_dir": str(tmp_path / "results" / "stroke"),
    }
    config_path = tmp_path / "stroke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    anchors = {
        "stroke": {
            "anchor_features": [
                {"feature": "Glucose", "role": "glucose role"},
                {"feature": "Hematocrit", "role": "hematocrit role"},
                {"feature": "Creatinine", "role": "creatinine role"},
            ],
            "note": "anchors are naming variables",
        }
    }
    anchors_path = tmp_path / "anchors.yaml"
    anchors_path.write_text(yaml.safe_dump(anchors), encoding="utf-8")
    return config_path, anchors_path, run_dir


def test_build_summary_separates_anchor_and_nonanchor_features(tmp_path: Path):
    config_path, anchors_path, run_dir = write_fixture(tmp_path)
    summary, output_path, audit_path = build_summary(
        config_path,
        anchors_path=anchors_path,
        run_dir=run_dir,
        top_n=2,
    )

    assert output_path.exists()
    assert audit_path.exists()
    anchors = summary[summary["evidence_role"] == "canonical_label_anchor"]
    assert set(anchors["feature"]) == {"Glucose", "Hematocrit", "Creatinine"}
    nonanchors = summary[summary["evidence_role"] == "non_anchor_evidence"]
    assert not set(nonanchors["feature"]).intersection(set(anchors["feature"]))
    assert nonanchors.groupby(["cluster", "domain"]).size().max() <= 2


def test_anchor_audit_marks_anchors_as_not_independent_validation(tmp_path: Path):
    config_path, anchors_path, run_dir = write_fixture(tmp_path)
    _, _, audit_path = build_summary(
        config_path,
        anchors_path=anchors_path,
        run_dir=run_dir,
        top_n=2,
    )
    audit = pd.read_csv(audit_path)
    assert len(audit) == 3
    assert audit["anchor_is_independent_validation"].eq(False).all()
