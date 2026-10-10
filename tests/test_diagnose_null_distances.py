"""Synthetic-data tests for aggregate-only sepsis-null distance diagnostics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from scripts.diagnose_null_distances import (
    fit_adaptive_and_fixed,
    run,
    sampled_distance_summary,
)


def test_distance_decomposition_matches_sklearn():
    rng = np.random.default_rng(20)
    z = np.concatenate([
        rng.normal(loc=mu, scale=.8, size=(15, 3))
        for mu in (-1.2, 0.0, 1.2)
    ])
    labels = np.repeat([9, 2, 5], 15)
    sample = np.sort(rng.choice(len(z), 33, replace=False))
    overall, groups = sampled_distance_summary(z, labels, sample)
    expected = silhouette_samples(z[sample], labels[sample])
    assert overall["silhouette_mean"] == pytest.approx(float(expected.mean()))
    assert overall["negative_silhouette_fraction"] == pytest.approx(
        float(np.mean(expected < 0))
    )
    assert overall["sample_size"] == 33
    assert sum(group["sample_cluster_n"] for group in groups) == 33
    assert [g["cluster_size_rank"] for g in groups] == [1, 2, 3]
    assert all(g["mean_within_distance_a"] > 0 for g in groups)
    assert all(g["mean_nearest_other_distance_b"] > 0 for g in groups)


def test_summary_is_invariant_to_relabeling():
    rng = np.random.default_rng(55)
    z = rng.normal(size=(90, 5))
    labels = np.repeat([0, 1, 2], 30)
    ix = np.arange(90)
    a, _ = sampled_distance_summary(z, labels, ix)
    b, _ = sampled_distance_summary(z, np.array([4, 7, 10])[labels], ix)
    assert a == b


def test_fixed_pca_and_reference_aggregation_are_separate():
    rng = np.random.default_rng(7)
    n = 240
    membership = np.repeat(np.arange(3), n // 3)
    features = [f"f{i}" for i in range(7)]
    data = rng.normal(size=(n, len(features)))
    data[:, 0] += membership * 3.0
    data[:, 1] += membership * .8
    frame = pd.DataFrame(data, columns=features)
    frame.insert(0, "PATID", [f"synthetic-{i}" for i in range(n)])
    cfg = {
        "id_column": "PATID",
        "features": {"primary": features},
        "preprocessing": {"scaler": "standard", "remove_zero_variance": True},
    }
    fits = fit_adaptive_and_fixed(frame, cfg, seed=11, fixed_components=4)
    assert fits["fixed_4"][1].shape == (n, 4)
    assert fits["adaptive_90pct"][1].shape[0] == n
    assert 0 <= adjusted_rand_score(
        fits["fixed_4"][0], fits["adaptive_90pct"][0]
    ) <= 1
    report, rows, groups = run(
        frame, cfg, repeats=2, sample_size=150, fixed_components=4
    )
    assert report["reference_repeats"] == 2
    assert set(rows["mode"]) == {"adaptive_90pct", "fixed_4"}
    assert len(rows) == 4
    assert len(groups) == 2 * 3 * 3
    assert "PATID" not in rows.columns
    assert "PATID" not in groups.columns
    assert report["observed"]["fixed_4"]["pca_components"] == 4
    assert report["null"]["fixed_4"]["pca_components_mean"] == 4
    assert 0 <= report["observed_adaptive_vs_fixed_ari"] <= 1
