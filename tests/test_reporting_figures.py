from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tpcluster.reporting_figures import (
    make_combined_heatmap_figure,
    make_effect_size_figure,
    make_workflow_figure,
    write_figure_manifest,
)


def _write_run(root: Path, cohort: str) -> Path:
    run = root / cohort / "run"
    tables = run / "manuscript_outputs" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        {
            "primary_cluster": [0, 1, 2],
            "primary_cluster_label": ["A", "B", "C"],
            "primary_cluster_short_label": ["A", "B", "C"],
            "n": [50, 30, 20],
            "fraction": [0.5, 0.3, 0.2],
        }
    ).to_csv(tables / "table_cluster_sizes.csv", index=False)

    continuous = []
    binary = []
    for feature_index in range(6):
        for cluster in range(3):
            continuous.append(
                {
                    "feature": f"feature_{feature_index}",
                    "feature_label": f"Feature {feature_index}",
                    "cluster": cluster,
                    "median_difference_iqr": (cluster - 1) * (feature_index + 1) / 5,
                }
            )
            binary.append(
                {
                    "feature": f"binary_{feature_index}",
                    "feature_label": f"Binary {feature_index}",
                    "cluster": cluster,
                    "standardised_difference": (cluster - 1) * (feature_index + 1) / 10,
                }
            )
    pd.DataFrame(continuous).to_csv(
        tables / "table_continuous_cluster_profiles_processed_scale.csv",
        index=False,
    )
    pd.DataFrame(binary).to_csv(
        tables / "table_binary_cluster_profiles.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "variable": ["outcome_a", "outcome_b", "race", "duration"],
            "variable_label": ["Outcome A", "Outcome B", "Race", "Duration"],
            "effect_size": [0.18, 0.08, 0.02, 0.04],
            "effect_size_name": [
                "Cramers_V",
                "epsilon_squared",
                "Cramers_V",
                "epsilon_squared",
            ],
        }
    ).to_csv(tables / "table_effect_size_focused_results.csv", index=False)
    (run / "manuscript_outputs" / "analysis_manifest.json").write_text(
        json.dumps({"dataset": cohort}), encoding="utf-8"
    )
    return run


def test_workflow_figure_writes_png_and_pdf(tmp_path: Path):
    outputs = make_workflow_figure(tmp_path, formats=("png", "pdf"))
    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert all(path.exists() for path in outputs)


def test_combined_heatmap_and_effect_size_figures(tmp_path: Path):
    stroke = _write_run(tmp_path, "stroke")
    sepsis = _write_run(tmp_path, "sepsis")
    heatmaps = make_combined_heatmap_figure(
        stroke, sepsis, tmp_path / "figures", formats=("png",)
    )
    effects = make_effect_size_figure(
        stroke, sepsis, tmp_path / "figures", formats=("png",)
    )
    assert heatmaps[0].exists()
    assert effects[0].exists()
    assert heatmaps[0].stat().st_size > 0
    assert effects[0].stat().st_size > 0


def test_figure_manifest_records_scope_and_layout(tmp_path: Path):
    stroke = _write_run(tmp_path, "stroke")
    sepsis = _write_run(tmp_path, "sepsis")
    generated = [tmp_path / "figure.png"]
    generated[0].write_bytes(b"test")
    path = write_figure_manifest(tmp_path, stroke, sepsis, generated)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["stability_selection_figure_included"] is False
    assert manifest["pca_display_seed"] == 11
    assert manifest["effect_size_metrics_separated"] is True
    assert manifest["heatmap_colorbars_dedicated"] is True
    assert manifest["figure_standard"] == "Scientific Figure Master Standard v8"
