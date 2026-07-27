import numpy as np
import pandas as pd

from tpcluster.core import fit_clusterer, prepare_matrix, reduce_matrix
from tpcluster.phenotypes import (
    apply_canonical_mapping,
    canonical_cluster_mapping,
    remap_profile_clusters,
)


def test_prepare_matrix_removes_constant_and_scales():
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 5, 5, 5]})
    raw, processed, scaled, removed, *_ = prepare_matrix(
        frame,
        ["a", "b"],
        {
            "remove_zero_variance": True,
            "scaler": "robust",
            "clip_quantiles": [0.0, 1.0],
        },
    )
    assert list(raw.columns) == ["a", "b"]
    assert list(processed.columns) == ["a"]
    assert removed == ["b"]
    assert scaled.shape == (4, 1)


def test_prepare_matrix_applies_log1p():
    frame = pd.DataFrame({"x": [0.0, 1.0, 3.0]})
    _, processed, _, _, _, report, _ = prepare_matrix(
        frame,
        ["x"],
        {
            "remove_zero_variance": False,
            "scaler": "none",
            "log1p_features": ["x"],
        },
    )
    assert np.allclose(processed["x"], np.log1p(frame["x"]))
    assert report.loc[0, "feature"] == "x"


def test_reduction_and_clusterers_return_expected_shapes():
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(50, 6))
    reduced, info = reduce_matrix(matrix, "pca", 11)
    assert reduced.shape[0] == 50
    assert info["n_components"] == reduced.shape[1]
    for method in ["kmeans", "gmm", "agglomerative"]:
        labels = fit_clusterer(reduced, method, 3, 11)
        assert labels.shape == (50,)
        assert len(np.unique(labels)) == 3


def test_stroke_canonical_mapping_is_permutation_invariant():
    frame = pd.DataFrame(
        {
            "Glucose": [105, 110, 225, 240, 112, 118],
            "Hematocrit": [42, 43, 40, 41, 30, 31],
            "Creatinine": [0.8, 0.9, 1.0, 1.1, 1.8, 1.7],
        }
    )
    labels = pd.Series([7, 7, 3, 3, 9, 9])
    mapping = canonical_cluster_mapping("stroke", frame, labels)
    assert mapping == {7: 0, 9: 1, 3: 2}
    assert apply_canonical_mapping(labels, mapping).tolist() == [0, 0, 2, 2, 1, 1]


def test_sepsis_canonical_mapping_is_permutation_invariant():
    frame = pd.DataFrame(
        {
            "IG #": [0.10, 0.12, 0.90, 0.80, 0.08, 0.09],
            "Eosinophils %": [0.5, 0.6, 0.3, 0.4, 3.5, 3.2],
            "Lymphocytes %": [12, 13, 8, 9, 25, 24],
        }
    )
    labels = pd.Series([8, 8, 4, 4, 6, 6])
    mapping = canonical_cluster_mapping("sepsis", frame, labels)
    assert mapping == {8: 0, 4: 1, 6: 2}
    assert apply_canonical_mapping(labels, mapping).tolist() == [0, 0, 1, 1, 2, 2]


def test_profile_tables_are_remapped_to_canonical_ids():
    profile = pd.DataFrame(
        {
            "cluster": [7, 3, 9],
            "feature": ["a", "a", "a"],
            "median_difference_iqr": [0.1, 0.2, 0.3],
        }
    )
    remapped = remap_profile_clusters(profile, {7: 0, 9: 1, 3: 2})
    assert remapped["cluster"].tolist() == [0, 1, 2]
    assert remapped["raw_cluster"].tolist() == [7, 9, 3]
