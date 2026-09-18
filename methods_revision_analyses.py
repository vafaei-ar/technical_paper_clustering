from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, rankdata
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from tpcluster.core import fit_clusterer, prepare_matrix, reduce_matrix


PRIMARY_REDUCTION = "pca"
PRIMARY_CLUSTERER = "kmeans"
PRIMARY_K = 3
REFERENCE_SEED = 11

WBC_COUNT_FEATURES = [
    "Basophils #",
    "Eosinophils #",
    "IG #",
    "Lymphocytes #",
    "Monocytes #",
    "Neutrophils #",
]
WBC_PERCENT_FEATURES = [
    "Basophils %",
    "Eosinophils %",
    "IG %",
    "Lymphocytes %",
    "Monocytes %",
    "Neutrophils %",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_full_run(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    runs = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and (p / "internal_metrics.csv").exists()
        and (p / "seed_stability.csv").exists()
    ]
    if not runs:
        raise FileNotFoundError(f"No completed clustering run under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def fit_primary(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    features: list[str] | None = None,
    seed: int = REFERENCE_SEED,
    reduction: str = PRIMARY_REDUCTION,
    k: int = PRIMARY_K,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    use_features = list(features or cfg["features"]["primary"])
    _, processed, scaled, *_ = prepare_matrix(
        frame,
        use_features,
        cfg.get("preprocessing", {}),
    )
    z, _ = reduce_matrix(scaled, reduction, seed)
    labels = fit_clusterer(z, PRIMARY_CLUSTERER, k, seed)
    return labels, z, processed, scaled


def matched_agreement(
    reference: np.ndarray, alternative: np.ndarray
) -> tuple[float, dict[int, int]]:
    ref_values = np.unique(reference)
    alt_values = np.unique(alternative)
    contingency = np.zeros((len(alt_values), len(ref_values)), dtype=int)
    for i, alt in enumerate(alt_values):
        for j, ref in enumerate(ref_values):
            contingency[i, j] = int(
                np.sum((alternative == alt) & (reference == ref))
            )
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {
        int(alt_values[i]): int(ref_values[j])
        for i, j in zip(row_ind, col_ind)
    }
    mapped = np.array(
        [mapping.get(int(x), -999999) for x in alternative], dtype=int
    )
    return float(np.mean(mapped == reference)), mapping


def is_binary_column(series: pd.Series) -> bool:
    values = set(
        pd.to_numeric(series, errors="coerce").dropna().unique().tolist()
    )
    return bool(values) and values.issubset({0, 1})


def balanced_block_matrix(
    processed: pd.DataFrame,
    scaled: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    columns = list(processed.columns)
    binary_mask = np.array(
        [is_binary_column(processed[c]) for c in columns], dtype=bool
    )
    continuous_mask = ~binary_mask
    if not binary_mask.any() or not continuous_mask.any():
        raise ValueError(
            "Balanced-block sensitivity requires both binary and continuous features"
        )

    balanced = np.array(scaled, dtype=float, copy=True)
    block_info: dict[str, Any] = {}
    for name, mask in (
        ("continuous", continuous_mask),
        ("binary", binary_mask),
    ):
        block = balanced[:, mask]
        energy = float(np.mean(np.sum(np.square(block), axis=1)))
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError(f"Non-positive {name} block energy: {energy}")
        weight = 1.0 / np.sqrt(energy)
        balanced[:, mask] *= weight
        block_info[f"{name}_feature_count"] = int(mask.sum())
        block_info[
            f"{name}_preweight_mean_squared_distance_contribution"
        ] = energy
        block_info[f"{name}_weight"] = float(weight)

    block_info["definition"] = (
        "Each preprocessed feature block was multiplied by the reciprocal "
        "square root of its mean per-patient squared Euclidean norm, giving "
        "the continuous and binary blocks equal total squared-distance "
        "contribution before PCA and k-means."
    )
    return balanced, block_info


def centroid_profile_correlations(
    scaled: np.ndarray,
    reference: np.ndarray,
    alternative: np.ndarray,
    mapping: dict[int, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for alt_cluster, ref_cluster in sorted(
        mapping.items(), key=lambda x: x[1]
    ):
        ref_centroid = scaled[reference == ref_cluster].mean(axis=0)
        alt_centroid = scaled[alternative == alt_cluster].mean(axis=0)
        if np.std(ref_centroid) == 0 or np.std(alt_centroid) == 0:
            corr = np.nan
        else:
            corr = float(
                np.corrcoef(ref_centroid, alt_centroid)[0, 1]
            )
        rows.append(
            {
                "reference_cluster": int(ref_cluster),
                "alternative_cluster": int(alt_cluster),
                "centroid_profile_correlation": corr,
            }
        )
    return pd.DataFrame(rows)


def run_balanced_block(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    reference_labels, reference_z, processed, scaled = fit_primary(
        frame, cfg
    )
    balanced, weights = balanced_block_matrix(processed, scaled)
    balanced_z, pca_info = reduce_matrix(
        balanced, PRIMARY_REDUCTION, REFERENCE_SEED
    )
    balanced_labels = fit_clusterer(
        balanced_z, PRIMARY_CLUSTERER, PRIMARY_K, REFERENCE_SEED
    )
    ari = float(adjusted_rand_score(reference_labels, balanced_labels))
    agreement, mapping = matched_agreement(
        reference_labels, balanced_labels
    )
    profile_corr = centroid_profile_correlations(
        scaled, reference_labels, balanced_labels, mapping
    )
    profile_corr.to_csv(
        output_dir
        / f"balanced_block_profile_correlations_{cfg['dataset_name']}.csv",
        index=False,
    )
    pd.DataFrame(
        {
            cfg["id_column"]: frame[cfg["id_column"]].to_numpy(),
            "reference_cluster": reference_labels,
            "balanced_block_cluster": balanced_labels,
        }
    ).to_parquet(
        output_dir
        / f"balanced_block_assignments_{cfg['dataset_name']}.parquet",
        index=False,
    )
    return {
        "dataset": cfg["dataset_name"],
        "analysis": "balanced_block",
        "reference_reduction": PRIMARY_REDUCTION,
        "reference_clusterer": PRIMARY_CLUSTERER,
        "reference_k": PRIMARY_K,
        "ari_vs_reference": ari,
        "matched_patient_agreement": agreement,
        "reference_silhouette": float(
            silhouette_score(reference_z, reference_labels)
        ),
        "balanced_silhouette": float(
            silhouette_score(balanced_z, balanced_labels)
        ),
        "balanced_pca_components": int(pca_info["n_components"]),
        "mean_centroid_profile_correlation": float(
            profile_corr["centroid_profile_correlation"].mean()
        ),
        "minimum_centroid_profile_correlation": float(
            profile_corr["centroid_profile_correlation"].min()
        ),
        **weights,
    }


def run_representation_and_k_sensitivity(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    _, _, scaled, *_ = prepare_matrix(
        frame,
        cfg["features"]["primary"],
        cfg.get("preprocessing", {}),
    )
    labels: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for reduction in ("none", "pca"):
        for k in (2, 3, 4):
            z, info = reduce_matrix(scaled, reduction, REFERENCE_SEED)
            current = fit_clusterer(
                z, PRIMARY_CLUSTERER, k, REFERENCE_SEED
            )
            labels[f"{reduction}_k{k}"] = current
            counts = pd.Series(current).value_counts()
            rows.append(
                {
                    "dataset": cfg["dataset_name"],
                    "reduction": reduction,
                    "clusterer": PRIMARY_CLUSTERER,
                    "k": k,
                    "n_components": int(info["n_components"]),
                    "silhouette": float(
                        silhouette_score(z, current)
                    ),
                    "calinski_harabasz": float(
                        calinski_harabasz_score(z, current)
                    ),
                    "davies_bouldin": float(
                        davies_bouldin_score(z, current)
                    ),
                    "minimum_cluster_fraction": float(
                        counts.min() / len(current)
                    ),
                }
            )
    primary = labels["pca_k3"]
    for key, current in labels.items():
        agreement, _ = matched_agreement(primary, current)
        for row in rows:
            if f"{row['reduction']}_k{row['k']}" == key:
                row["ari_vs_pca_k3"] = float(
                    adjusted_rand_score(primary, current)
                )
                row["matched_agreement_vs_pca_k3"] = agreement
                break
    return pd.DataFrame(rows)


def run_sepsis_redundancy(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    reference_labels, _, _, _ = fit_primary(frame, cfg)
    primary_features = list(cfg["features"]["primary"])
    variants = {
        "drop_wbc_percentages_keep_counts": [
            f
            for f in primary_features
            if f not in WBC_PERCENT_FEATURES
        ],
        "drop_wbc_counts_keep_percentages": [
            f
            for f in primary_features
            if f not in WBC_COUNT_FEATURES
        ],
    }
    rows: list[dict[str, Any]] = []
    for name, features in variants.items():
        labels, z, _, _ = fit_primary(
            frame, cfg, features=features
        )
        agreement, _ = matched_agreement(
            reference_labels, labels
        )
        counts = pd.Series(labels).value_counts()
        rows.append(
            {
                "dataset": cfg["dataset_name"],
                "variant": name,
                "n_features": len(features),
                "dropped_features": "; ".join(
                    sorted(set(primary_features) - set(features))
                ),
                "ari_vs_reference": float(
                    adjusted_rand_score(reference_labels, labels)
                ),
                "matched_patient_agreement": agreement,
                "silhouette": float(silhouette_score(z, labels)),
                "minimum_cluster_fraction": float(
                    counts.min() / len(labels)
                ),
            }
        )
    return pd.DataFrame(rows)


def gaussian_copula_null(
    frame: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    numeric = (
        frame[features]
        .apply(pd.to_numeric, errors="coerce")
        .copy()
    )
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    n, p = numeric.shape
    latent = np.empty((n, p), dtype=float)
    sorted_values: list[np.ndarray] = []
    for j, column in enumerate(numeric.columns):
        values = numeric[column].to_numpy(dtype=float)
        ranks = rankdata(values, method="average")
        u = np.clip((ranks - 0.5) / n, 1e-6, 1 - 1e-6)
        latent[:, j] = norm.ppf(u)
        sorted_values.append(np.sort(values))

    corr = np.corrcoef(latent, rowvar=False)
    corr = np.nan_to_num(
        corr, nan=0.0, posinf=0.0, neginf=0.0
    )
    np.fill_diagonal(corr, 1.0)
    eigval, eigvec = np.linalg.eigh(corr)
    eigval = np.clip(eigval, 1e-6, None)
    corr_psd = (eigvec * eigval) @ eigvec.T
    scale = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(scale, scale)
    np.fill_diagonal(corr_psd, 1.0)

    draw = rng.multivariate_normal(
        np.zeros(p), corr_psd, size=n
    )
    u_draw = norm.cdf(draw)
    out = pd.DataFrame(index=frame.index)
    for j, column in enumerate(numeric.columns):
        idx = np.floor(u_draw[:, j] * n).astype(int)
        idx = np.clip(idx, 0, n - 1)
        out[column] = sorted_values[j][idx]
    return out


def full_refit_subsample_ari(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    full_labels: np.ndarray,
    n_repeats: int,
    sample_fraction: float,
    base_seed: int,
) -> np.ndarray:
    n = len(frame)
    sample_n = int(np.floor(n * sample_fraction))
    values: list[float] = []
    for repeat in range(n_repeats):
        seed = base_seed + repeat
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(n, size=sample_n, replace=False)
        )
        subset = frame.iloc[selected].reset_index(drop=True)
        labels, _, _, _ = fit_primary(
            subset, cfg, seed=seed
        )
        values.append(
            float(
                adjusted_rand_score(
                    full_labels[selected], labels
                )
            )
        )
    return np.asarray(values, dtype=float)


def run_null_reference(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
    null_repeats: int,
    null_subsamples: int,
    sample_fraction: float,
    base_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    real_labels, real_z, _, _ = fit_primary(frame, cfg)
    real_sub = full_refit_subsample_ari(
        frame,
        cfg,
        real_labels,
        n_repeats=null_subsamples,
        sample_fraction=sample_fraction,
        base_seed=base_seed + 50000,
    )
    real_silhouette = float(
        silhouette_score(real_z, real_labels)
    )
    rows: list[dict[str, Any]] = []
    for repeat in range(null_repeats):
        seed = base_seed + repeat
        rng = np.random.default_rng(seed)
        null_features = gaussian_copula_null(
            frame, cfg["features"]["primary"], rng
        )
        null_frame = (
            frame[[cfg["id_column"]]]
            .reset_index(drop=True)
            .copy()
        )
        for feature in cfg["features"]["primary"]:
            null_frame[feature] = (
                null_features[feature].to_numpy()
            )
        null_labels, null_z, _, _ = fit_primary(
            null_frame, cfg, seed=seed
        )
        null_sub = full_refit_subsample_ari(
            null_frame,
            cfg,
            null_labels,
            n_repeats=null_subsamples,
            sample_fraction=sample_fraction,
            base_seed=(
                base_seed + 100000 + repeat * 1000
            ),
        )
        rows.append(
            {
                "dataset": cfg["dataset_name"],
                "null_repeat": repeat,
                "seed": seed,
                "silhouette": float(
                    silhouette_score(null_z, null_labels)
                ),
                "mean_full_refit_subsample_ari": float(
                    null_sub.mean()
                ),
                "minimum_full_refit_subsample_ari": float(
                    null_sub.min()
                ),
            }
        )
    null_table = pd.DataFrame(rows)
    null_table.to_csv(
        output_dir
        / f"null_reference_replicates_{cfg['dataset_name']}.csv",
        index=False,
    )
    summary = {
        "dataset": cfg["dataset_name"],
        "null_model": (
            "Single Gaussian-copula reference preserving empirical "
            "univariate marginals and approximate rank-correlation "
            "structure, with no explicit latent mixture groups."
        ),
        "null_repeats": null_repeats,
        "null_subsamples_per_repeat": null_subsamples,
        "sample_fraction": sample_fraction,
        "real_silhouette": real_silhouette,
        "real_mean_full_refit_subsample_ari": float(
            real_sub.mean()
        ),
        "real_minimum_full_refit_subsample_ari": float(
            real_sub.min()
        ),
        "null_silhouette_mean": float(
            null_table["silhouette"].mean()
        ),
        "null_silhouette_p95": float(
            null_table["silhouette"].quantile(0.95)
        ),
        "null_subsample_ari_mean": float(
            null_table[
                "mean_full_refit_subsample_ari"
            ].mean()
        ),
        "null_subsample_ari_p95": float(
            null_table[
                "mean_full_refit_subsample_ari"
            ].quantile(0.95)
        ),
        "empirical_p_silhouette": float(
            (
                1
                + (
                    null_table["silhouette"]
                    >= real_silhouette
                ).sum()
            )
            / (len(null_table) + 1)
        ),
        "empirical_p_subsample_ari": float(
            (
                1
                + (
                    null_table[
                        "mean_full_refit_subsample_ari"
                    ]
                    >= float(real_sub.mean())
                ).sum()
            )
            / (len(null_table) + 1)
        ),
    }
    return null_table, summary


def candidate_audit(
    run_dir: Path, dataset: str
) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "internal_metrics.csv")
    seed = pd.read_csv(run_dir / "seed_stability.csv")
    key = ["dataset", "reduction", "clusterer", "k"]
    audit = metrics.groupby(key, as_index=False).agg(
        n_seeds=("seed", "nunique"),
        silhouette_mean=("silhouette", "mean"),
        silhouette_sd=("silhouette", "std"),
        calinski_harabasz_mean=(
            "calinski_harabasz",
            "mean",
        ),
        calinski_harabasz_sd=(
            "calinski_harabasz",
            "std",
        ),
        davies_bouldin_mean=("davies_bouldin", "mean"),
        davies_bouldin_sd=("davies_bouldin", "std"),
        minimum_cluster_n=("minimum_cluster_n", "min"),
        minimum_cluster_fraction=(
            "minimum_cluster_fraction",
            "min",
        ),
        cluster_size_gate_pass_rate=(
            "passes_cluster_size_gate",
            "mean",
        ),
    )
    stable = seed.groupby(key, as_index=False).agg(
        seed_ari_mean=("adjusted_rand_index", "mean"),
        seed_ari_sd=("adjusted_rand_index", "std"),
        seed_ari_min=("adjusted_rand_index", "min"),
    )
    audit = audit.merge(
        stable, on=key, how="left", validate="one_to_one"
    )
    full_refit = (
        run_dir
        / "subsample_stability_full_refit_summary.csv"
    )
    if full_refit.exists():
        sub = pd.read_csv(full_refit)
        keep = [
            c
            for c in [
                "dataset",
                "reduction",
                "clusterer",
                "k",
                "mean_subsample_ari",
                "median_subsample_ari",
                "minimum_subsample_ari",
                "p10_subsample_ari",
                "minimum_cluster_fraction",
            ]
            if c in sub.columns
        ]
        sub = sub[keep].rename(
            columns={
                "minimum_cluster_fraction": (
                    "subsample_minimum_cluster_fraction"
                )
            }
        )
        audit = audit.merge(
            sub, on=key, how="left", validate="one_to_one"
        )
    audit["seed_stability_note"] = np.where(
        audit["clusterer"]
        .astype(str)
        .str.lower()
        .eq("agglomerative"),
        (
            "Deterministic estimator; across-seed ARI=1 is "
            "expected and is not evidence of initialization "
            "robustness."
        ),
        "Across-seed ARI reflects initialization robustness.",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reviewer-driven robustness analyses for the "
            "Clustro technical paper."
        )
    )
    parser.add_argument(
        "--stroke-config",
        default="configs/stroke_methods_revision.yaml",
    )
    parser.add_argument(
        "--sepsis-config",
        default="configs/sepsis.yaml",
    )
    parser.add_argument("--stroke-run")
    parser.add_argument("--sepsis-run")
    parser.add_argument(
        "--output-dir",
        default="results/methods_revision",
    )
    parser.add_argument(
        "--null-repeats", type=int, default=20
    )
    parser.add_argument(
        "--null-subsamples", type=int, default=10
    )
    parser.add_argument(
        "--sample-fraction", type=float, default=0.80
    )
    parser.add_argument(
        "--base-seed", type=int, default=73001
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "stroke": load_yaml(args.stroke_config),
        "sepsis": load_yaml(args.sepsis_config),
    }
    run_dirs = {
        "stroke": (
            Path(args.stroke_run)
            if args.stroke_run
            else latest_full_run(
                configs["stroke"]["output_dir"]
            )
        ),
        "sepsis": (
            Path(args.sepsis_run)
            if args.sepsis_run
            else latest_full_run(
                configs["sepsis"]["output_dir"]
            )
        ),
    }

    summary_rows: list[dict[str, Any]] = []
    null_summaries: list[dict[str, Any]] = []
    for cohort, cfg in configs.items():
        frame = pd.read_parquet(cfg["input_path"])

        audit = candidate_audit(
            run_dirs[cohort], cohort
        )
        audit.to_csv(
            output_dir / f"candidate_audit_{cohort}.csv",
            index=False,
        )

        selection = run_representation_and_k_sensitivity(
            frame, cfg
        )
        selection.to_csv(
            output_dir
            / f"representation_k_sensitivity_{cohort}.csv",
            index=False,
        )

        balanced = run_balanced_block(
            frame, cfg, output_dir
        )
        summary_rows.append(balanced)

        if cohort == "sepsis":
            redundancy = run_sepsis_redundancy(
                frame, cfg
            )
            redundancy.to_csv(
                output_dir
                / "sepsis_wbc_redundancy_sensitivity.csv",
                index=False,
            )

        _, null_summary = run_null_reference(
            frame,
            cfg,
            output_dir,
            null_repeats=args.null_repeats,
            null_subsamples=args.null_subsamples,
            sample_fraction=args.sample_fraction,
            base_seed=(
                args.base_seed
                + (0 if cohort == "stroke" else 1000000)
            ),
        )
        null_summaries.append(null_summary)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "balanced_block_summary.csv",
        index=False,
    )
    pd.DataFrame(null_summaries).to_csv(
        output_dir / "null_reference_summary.csv",
        index=False,
    )
    manifest = {
        "stroke_config": args.stroke_config,
        "sepsis_config": args.sepsis_config,
        "stroke_run": str(run_dirs["stroke"]),
        "sepsis_run": str(run_dirs["sepsis"]),
        "null_repeats": args.null_repeats,
        "null_subsamples": args.null_subsamples,
        "sample_fraction": args.sample_fraction,
        "primary_model_for_sensitivity": {
            "reduction": PRIMARY_REDUCTION,
            "clusterer": PRIMARY_CLUSTERER,
            "k": PRIMARY_K,
            "seed": REFERENCE_SEED,
        },
        "note": (
            "These analyses are reviewer-driven sensitivities. "
            "Results should be inspected before any manuscript "
            "narrative or phenotype naming is finalized."
        ),
    }
    (
        output_dir / "methods_revision_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        "Wrote reviewer-driven analysis outputs to "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
