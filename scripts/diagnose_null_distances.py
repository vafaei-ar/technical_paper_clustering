"""Compare sampled sepsis cluster-distance geometry with Gaussian-copula references.

Run from the repository root:
    python scripts/diagnose_null_distances.py

Fit each clustering model to the full data, then calculate patient-level
silhouette components inside one fixed row sample shared by all comparisons.
Only aggregate metrics are saved. Existing analysis outputs are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from methods_revision_analyses import gaussian_copula_null, load_yaml
from tpcluster.core import fit_clusterer, prepare_matrix, reduce_matrix


DEFAULT_CONFIG = Path("configs/sepsis.yaml")
EXPECTED_N = 15842
EXPECTED_FEATURES = 52
REFERENCE_SEED = 11
BASE_NULL_SEED = 1073001
STATS = (
    "silhouette_mean", "silhouette_p05", "silhouette_p50",
    "negative_silhouette_fraction", "mean_within_distance_a",
    "mean_nearest_other_distance_b", "mean_separation_margin_b_minus_a",
    "between_to_total_ss_fraction", "smallest_cluster_fraction",
)


def sampled_distance_summary(z: np.ndarray, labels: np.ndarray,
                             sampled_indices: np.ndarray) -> tuple[dict, list[dict]]:
    """Compute exact Euclidean a(i), b(i), and silhouette inside a fixed sample.

    a(i): mean distance to other patients in the same sampled cluster.
    b(i): minimum mean distance to patients in any other sampled cluster.
    Original cluster assignments are always fitted on the full cohort.
    """
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels)
    ix = np.asarray(sampled_indices)
    sample = z[ix]
    sample_labels = labels[ix]
    groups, sample_sizes = np.unique(sample_labels, return_counts=True)
    if len(groups) < 2 or np.any(sample_sizes < 2):
        raise ValueError("Sample must contain at least two members of every cluster")
    distances = cdist(sample, sample, metric="euclidean")
    a = np.zeros(len(ix), dtype=float)
    b = np.zeros(len(ix), dtype=float)
    for group in groups:
        same = sample_labels == group
        count = int(same.sum())
        a[same] = distances[np.ix_(same, same)].sum(axis=1) / (count - 1)
        other_means = [
            distances[np.ix_(same, sample_labels == competitor)].mean(axis=1)
            for competitor in groups if competitor != group
        ]
        b[same] = np.min(np.vstack(other_means), axis=0)
    denominator = np.maximum(a, b)
    silhouette = np.divide(b - a, denominator,
                           out=np.zeros_like(a), where=denominator > 0)

    center = z.mean(axis=0)
    total_ss = float(np.square(z - center).sum())
    within_ss = sum(
        float(np.square(z[labels == g] - z[labels == g].mean(axis=0)).sum())
        for g in np.unique(labels)
    )
    full_sizes = np.unique(labels, return_counts=True)[1]
    overall = {
        "silhouette_mean": float(silhouette.mean()),
        "silhouette_p05": float(np.quantile(silhouette, .05)),
        "silhouette_p50": float(np.median(silhouette)),
        "negative_silhouette_fraction": float(np.mean(silhouette < 0)),
        "mean_within_distance_a": float(a.mean()),
        "mean_nearest_other_distance_b": float(b.mean()),
        "mean_separation_margin_b_minus_a": float((b - a).mean()),
        "between_to_total_ss_fraction": float(1 - within_ss / total_ss),
        "smallest_cluster_fraction": float(full_sizes.min() / len(labels)),
        "pca_components": int(z.shape[1]),
        "sample_size": int(len(ix)),
    }
    # Compare clusters by full-cohort size, not arbitrary model label.
    groups_by_size = sorted(np.unique(labels),
                            key=lambda g: -int((labels == g).sum()))
    per_cluster = []
    for size_rank, group in enumerate(groups_by_size, start=1):
        mask = sample_labels == group
        in_full = labels == group
        centroid = z[in_full].mean(axis=0)
        per_cluster.append({
            "cluster_size_rank": size_rank,
            "full_cluster_fraction": float(in_full.mean()),
            "sample_cluster_n": int(mask.sum()),
            "silhouette_mean": float(silhouette[mask].mean()),
            "negative_silhouette_fraction": float(np.mean(silhouette[mask] < 0)),
            "mean_within_distance_a": float(a[mask].mean()),
            "mean_nearest_other_distance_b": float(b[mask].mean()),
            "mean_separation_margin_b_minus_a": float((b[mask] - a[mask]).mean()),
            "full_cluster_rms_radius": float(np.sqrt(np.mean(
                np.square(z[in_full] - centroid).sum(axis=1)
            ))),
        })
    return overall, per_cluster


def fit_adaptive_and_fixed(frame: pd.DataFrame, cfg: dict, *, seed: int,
                           fixed_components: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    features = cfg["features"]["primary"]
    _, _, scaled, *_ = prepare_matrix(frame, features, cfg.get("preprocessing", {}))
    if not 1 <= fixed_components <= min(scaled.shape):
        raise ValueError("Fixed PCA component count must fit the processed matrix")
    adaptive, _ = reduce_matrix(scaled, "pca", seed)
    fixed = PCA(n_components=fixed_components, svd_solver="full",
                random_state=seed).fit_transform(scaled)
    return {
        "adaptive_90pct": (fit_clusterer(adaptive, "kmeans", 3, seed), adaptive),
        f"fixed_{fixed_components}": (
            fit_clusterer(fixed, "kmeans", 3, seed), fixed
        ),
    }


def run(frame: pd.DataFrame, cfg: dict, *, repeats: int, sample_size: int,
        fixed_components: int, base_seed: int = BASE_NULL_SEED) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if repeats < 1 or sample_size < 50 or sample_size > len(frame):
        raise ValueError("Invalid reference count or sample size")
    ix = np.sort(np.random.default_rng(2026).choice(
        len(frame), size=sample_size, replace=False
    ))
    observed_fits = fit_adaptive_and_fixed(
        frame, cfg, seed=REFERENCE_SEED, fixed_components=fixed_components
    )
    mode_names = list(observed_fits)
    observed_by_mode = {}
    observed_clusters = []
    for mode, (labels, z) in observed_fits.items():
        summary, groups = sampled_distance_summary(z, labels, ix)
        observed_by_mode[mode] = summary
        observed_clusters.extend(
            {"source": "observed", "mode": mode, **g} for g in groups
        )
    obs_ari = float(adjusted_rand_score(
        observed_fits[mode_names[0]][0], observed_fits[mode_names[1]][0]
    ))
    null_rows = []
    null_group_rows = []
    for i in range(repeats):
        seed = base_seed + i
        null_features = gaussian_copula_null(
            frame, cfg["features"]["primary"], np.random.default_rng(seed)
        )
        null_frame = pd.concat([
            frame[[cfg["id_column"]]].reset_index(drop=True),
            null_features.reset_index(drop=True),
        ], axis=1)
        fitted = fit_adaptive_and_fixed(
            null_frame, cfg, seed=seed, fixed_components=fixed_components
        )
        for mode, (labels, z) in fitted.items():
            summary, groups = sampled_distance_summary(z, labels, ix)
            null_rows.append({"reference": i, "mode": mode, **summary})
            null_group_rows.extend(
                {"reference": i, "mode": mode, **g} for g in groups
            )
        print(f"Distance reference {i + 1}/{repeats} completed", flush=True)

    null_df = pd.DataFrame(null_rows)
    group_df = pd.DataFrame(null_group_rows)
    null_by_mode = {}
    for mode in mode_names:
        subset = null_df[null_df["mode"] == mode]
        null_by_mode[mode] = {
            name: {
                "mean": float(subset[name].mean()),
                "p05": float(subset[name].quantile(.05)),
                "p95": float(subset[name].quantile(.95)),
            }
            for name in STATS
        }
        null_by_mode[mode]["pca_components_mean"] = float(
            subset["pca_components"].mean()
        )
    groups_aggregate = pd.concat([
        pd.DataFrame(observed_clusters).assign(reference=-1),
        group_df.assign(source="null"),
    ], ignore_index=True)
    report = {
        "n_full_cohort": int(len(frame)),
        "sample_size": int(sample_size),
        "reference_repeats": int(repeats),
        "fixed_pca_components": int(fixed_components),
        "observed_adaptive_vs_fixed_ari": obs_ari,
        "observed": observed_by_mode,
        "null": null_by_mode,
        "interpretation_note": (
            "Labels fit the full cohort. Sampled silhouettes use distances "
            "only among the SAME sampled rows in every run. These exploratory "
            "values do not replace full-cohort manuscript silhouettes. "
            "Reference percentiles reflect only the small exploratory null "
            "sample and are not formal significance tests. Ranked clusters "
            "are ordered by size and are not phenotype labels."
        ),
    }
    return report, null_df, groups_aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-repeats", type=int, default=12)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--fixed-components", type=int, default=25)
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/null_distance_diagnostics"))
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    if cfg.get("dataset_name") != "sepsis":
        parser.error("This focused diagnostic is specified for the sepsis cohort")
    frame = pd.read_parquet(cfg["input_path"])
    if len(frame) != EXPECTED_N or len(cfg["features"]["primary"]) != EXPECTED_FEATURES:
        parser.error("Sepsis input must have 15,842 rows and 52 configured features")
    report, reference_table, groups = run(
        frame, cfg, repeats=args.reference_repeats,
        sample_size=args.sample_size, fixed_components=args.fixed_components
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "sepsis_distance_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    reference_table.to_csv(
        args.outdir / "sepsis_reference_distance_summary.csv", index=False
    )
    groups.to_csv(
        args.outdir / "sepsis_cluster_distance_aggregates.csv", index=False
    )
    print(json.dumps(report, indent=2))
    print(f"Aggregate outputs saved to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
