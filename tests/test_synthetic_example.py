from __future__ import annotations

from pathlib import Path

import yaml

from examples.generate_synthetic_clinical_example import generate_synthetic_cohort


def test_synthetic_cohort_is_reproducible_and_unique() -> None:
    first = generate_synthetic_cohort(n_per_cluster=12, seed=17)
    second = generate_synthetic_cohort(n_per_cluster=12, seed=17)

    assert first.equals(second)
    assert len(first) == 36
    assert first["PATID"].is_unique
    assert set(first["cluster_truth"]) == {0, 1, 2}
    assert first.isna().sum().sum() == 0


def test_synthetic_truth_is_excluded_from_clustering_features() -> None:
    config_path = Path("examples/synthetic_config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["id_column"] == "PATID"
    assert "cluster_truth" not in config["features"]["primary"]
    assert set(config["k_values"]) == {2, 3, 4}
