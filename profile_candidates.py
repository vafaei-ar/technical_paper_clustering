from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

from tpcluster.core import fit_clusterer, prepare_matrix, reduce_matrix


def latest_full_run(output_dir: str | Path) -> Path:
    runs = [p for p in Path(output_dir).iterdir() if p.is_dir() and (p / "internal_metrics.csv").exists() and len(pd.read_csv(p / "internal_metrics.csv")) >= 270]
    if not runs:
        raise FileNotFoundError(f"No full run found under {output_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def is_binary(series: pd.Series) -> bool:
    values = set(pd.to_numeric(series, errors="coerce").dropna().unique().tolist())
    return values.issubset({0, 1})


def continuous_profile(frame: pd.DataFrame, features: list[str], cluster_col: str, candidate: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        overall = pd.to_numeric(frame[feature], errors="coerce")
        overall_median = float(overall.median())
        overall_iqr = float(overall.quantile(0.75) - overall.quantile(0.25))
        for cluster, group in frame.groupby(cluster_col):
            values = pd.to_numeric(group[feature], errors="coerce")
            median = float(values.median())
            effect = (median - overall_median) / overall_iqr if overall_iqr > 0 else np.nan
            rows.append({
                "candidate": candidate,
                "cluster": int(cluster),
                "cluster_n": int(len(group)),
                "feature": feature,
                "mean": float(values.mean()),
                "sd": float(values.std()),
                "median": median,
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "overall_median": overall_median,
                "overall_iqr": overall_iqr,
                "median_difference_iqr": effect,
                "absolute_median_difference_iqr": abs(effect) if np.isfinite(effect) else np.nan,
            })
    return pd.DataFrame(rows)


def binary_profile(frame: pd.DataFrame, features: list[str], cluster_col: str, candidate: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        overall = pd.to_numeric(frame[feature], errors="coerce")
        overall_prevalence = float(overall.mean())
        for cluster, group in frame.groupby(cluster_col):
            values = pd.to_numeric(group[feature], errors="coerce")
            prevalence = float(values.mean())
            pooled = (prevalence + overall_prevalence) / 2
            denominator = np.sqrt(pooled * (1 - pooled))
            effect = (prevalence - overall_prevalence) / denominator if denominator > 0 else np.nan
            rows.append({
                "candidate": candidate,
                "cluster": int(cluster),
                "cluster_n": int(len(group)),
                "feature": feature,
                "prevalence": prevalence,
                "overall_prevalence": overall_prevalence,
                "prevalence_difference": prevalence - overall_prevalence,
                "standardised_difference": effect,
                "absolute_standardised_difference": abs(effect) if np.isfinite(effect) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", default="configs/candidates.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    with open(args.candidates, encoding="utf-8") as file:
        candidates = yaml.safe_load(file)[cfg["dataset_name"]]

    frame = pd.read_parquet(cfg["input_path"])
    raw, processed, x_scaled, *_ = prepare_matrix(frame, cfg["features"]["primary"], cfg.get("preprocessing", {}))
    run_dir = latest_full_run(cfg["output_dir"])
    output_dir = run_dir / "candidate_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = pd.DataFrame({cfg["id_column"]: frame[cfg["id_column"]].to_numpy()})
    agreement_rows = []
    labels_by_name = {}
    for candidate in candidates:
        z, _ = reduce_matrix(x_scaled, candidate["reduction"], int(candidate.get("seed", 11)))
        labels = fit_clusterer(z, candidate["clusterer"], int(candidate["k"]), int(candidate.get("seed", 11)))
        name = candidate["name"]
        labels_by_name[name] = labels
        assignments[name] = labels

        candidate_dir = output_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        profile_frame = processed.copy()
        profile_frame["cluster"] = labels
        binary_features = [f for f in processed.columns if is_binary(processed[f])]
        continuous_features = [f for f in processed.columns if f not in binary_features]
        cont = continuous_profile(profile_frame, continuous_features, "cluster", name)
        binary = binary_profile(profile_frame, binary_features, "cluster", name)
        cont.to_csv(candidate_dir / "continuous_feature_profiles.csv", index=False)
        binary.to_csv(candidate_dir / "binary_feature_profiles.csv", index=False)
        cont.sort_values(["cluster", "absolute_median_difference_iqr"], ascending=[True, False]).groupby("cluster", group_keys=False).head(15).to_csv(candidate_dir / "top_continuous_features.csv", index=False)
        binary.sort_values(["cluster", "absolute_standardised_difference"], ascending=[True, False]).groupby("cluster", group_keys=False).head(15).to_csv(candidate_dir / "top_binary_features.csv", index=False)
        counts = pd.Series(labels).value_counts().sort_index().rename_axis("cluster").reset_index(name="cluster_n")
        counts["candidate"] = name
        counts["cluster_fraction"] = counts["cluster_n"] / len(labels)
        counts.to_csv(candidate_dir / "cluster_sizes.csv", index=False)

    assignments.to_parquet(output_dir / "candidate_assignments.parquet", index=False)
    names = list(labels_by_name)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            agreement_rows.append({"candidate_a": name_a, "candidate_b": name_b, "adjusted_rand_index": adjusted_rand_score(labels_by_name[name_a], labels_by_name[name_b])})
    pd.DataFrame(agreement_rows).to_csv(output_dir / "candidate_pairwise_ari.csv", index=False)
    print(output_dir)


if __name__ == "__main__":
    main()
