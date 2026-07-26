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


def fit_solution(matrix: np.ndarray, candidate: dict, seed: int) -> np.ndarray:
    z, _ = reduce_matrix(matrix, candidate["reduction"], seed)
    return fit_clusterer(z, candidate["clusterer"], int(candidate["k"]), seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", default="configs/candidates.yaml")
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--sample-fraction", type=float, default=0.80)
    parser.add_argument("--base-seed", type=int, default=1001)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    with open(args.candidates, encoding="utf-8") as file:
        candidates = yaml.safe_load(file)[cfg["dataset_name"]]

    frame = pd.read_parquet(cfg["input_path"])
    _, _, x_scaled, *_ = prepare_matrix(frame, cfg["features"]["primary"], cfg.get("preprocessing", {}))
    run_dir = latest_full_run(cfg["output_dir"])
    n = len(frame)
    sample_n = int(np.floor(n * args.sample_fraction))
    rows = []

    for candidate in candidates:
        reference_seed = int(candidate.get("seed", 11))
        full_labels = fit_solution(x_scaled, candidate, reference_seed)
        full_counts = pd.Series(full_labels).value_counts()
        for repeat in range(args.n_repeats):
            seed = args.base_seed + repeat
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(n, size=sample_n, replace=False))
            labels = fit_solution(x_scaled[selected], candidate, seed)
            counts = pd.Series(labels).value_counts()
            rows.append({
                "dataset": cfg["dataset_name"],
                "candidate": candidate["name"],
                "reduction": candidate["reduction"],
                "clusterer": candidate["clusterer"],
                "k": int(candidate["k"]),
                "reference_seed": reference_seed,
                "subsample_seed": seed,
                "sample_fraction": args.sample_fraction,
                "sample_n": sample_n,
                "ari_vs_full": adjusted_rand_score(full_labels[selected], labels),
                "minimum_cluster_size": int(counts.min()),
                "maximum_cluster_size": int(counts.max()),
                "minimum_cluster_fraction": float(counts.min() / sample_n),
                "full_minimum_cluster_size": int(full_counts.min()),
                "full_minimum_cluster_fraction": float(full_counts.min() / n),
            })

    detailed = pd.DataFrame(rows)
    detailed.to_csv(run_dir / "subsample_stability_detailed.csv", index=False)
    summary = detailed.groupby(["dataset", "candidate", "reduction", "clusterer", "k"], as_index=False).agg(
        mean_subsample_ari=("ari_vs_full", "mean"),
        median_subsample_ari=("ari_vs_full", "median"),
        sd_subsample_ari=("ari_vs_full", "std"),
        minimum_subsample_ari=("ari_vs_full", "min"),
        p10_subsample_ari=("ari_vs_full", lambda x: x.quantile(0.10)),
        p25_subsample_ari=("ari_vs_full", lambda x: x.quantile(0.25)),
        minimum_cluster_fraction=("minimum_cluster_fraction", "min"),
        mean_minimum_cluster_fraction=("minimum_cluster_fraction", "mean"),
    )
    summary.to_csv(run_dir / "subsample_stability_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
