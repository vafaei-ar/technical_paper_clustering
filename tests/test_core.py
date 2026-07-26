import numpy as np
import pandas as pd

from tpcluster.core import fit_clusterer, prepare_matrix, reduce_matrix


def test_prepare_matrix_removes_constant_and_scales():
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 5, 5, 5]})
    raw, processed, scaled, removed, *_ = prepare_matrix(
        frame,
        ["a", "b"],
        {"remove_zero_variance": True, "scaler": "robust", "clip_quantiles": [0.0, 1.0]},
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
        {"remove_zero_variance": False, "scaler": "none", "log1p_features": ["x"]},
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
