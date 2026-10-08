from __future__ import annotations

import numpy as np
import pandas as pd

from methods_revision_analyses import (
    balanced_block_matrix,
    gaussian_copula_null,
    matched_agreement,
)


def test_matched_agreement_is_label_invariant() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2])
    alternative = np.array([2, 2, 0, 0, 1, 1])
    agreement, mapping = matched_agreement(reference, alternative)
    assert agreement == 1.0
    assert set(mapping.values()) == {0, 1, 2}


def test_balanced_block_equalizes_total_block_energy() -> None:
    processed = pd.DataFrame(
        {
            "continuous_a": [0.0, 1.0, 2.0, 3.0],
            "continuous_b": [1.0, 2.0, 3.0, 4.0],
            "binary_a": [0, 1, 0, 1],
        }
    )
    scaled = processed.to_numpy(dtype=float)
    balanced, info = balanced_block_matrix(processed, scaled)
    binary_mask = np.array([False, False, True])
    continuous_mask = ~binary_mask
    continuous_energy = np.mean(
        np.sum(balanced[:, continuous_mask] ** 2, axis=1)
    )
    binary_energy = np.mean(
        np.sum(balanced[:, binary_mask] ** 2, axis=1)
    )
    assert np.isclose(continuous_energy, 1.0)
    assert np.isclose(binary_energy, 1.0)
    assert info["continuous_feature_count"] == 2
    assert info["binary_feature_count"] == 1


def test_gaussian_copula_null_preserves_discrete_marginal_counts() -> None:
    n = 200
    frame = pd.DataFrame(
        {
            "binary": np.r_[np.zeros(150), np.ones(50)],
            "continuous": np.linspace(0.0, 1.0, n),
        }
    )
    draw = gaussian_copula_null(
        frame,
        ["binary", "continuous"],
        np.random.default_rng(123),
    )
    assert set(draw["binary"].unique()).issubset({0.0, 1.0})
    assert abs(draw["binary"].mean() - 0.25) < 0.08
    assert draw["continuous"].between(0.0, 1.0).all()
