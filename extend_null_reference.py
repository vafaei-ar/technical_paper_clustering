from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from methods_revision_analyses import (
    fit_primary,
    gaussian_copula_null,
    load_yaml,
)


def run_one(config_path: str, repeats: int, base_seed: int) -> tuple[pd.DataFrame, dict]:
    cfg = load_yaml(config_path)
    frame = pd.read_parquet(cfg["input_path"])
    real_labels, real_z, _, _ = fit_primary(frame, cfg)
    real_silhouette = float(silhouette_score(real_z, real_labels))

    rows = []
    for repeat in range(repeats):
        seed = base_seed + repeat
        rng = np.random.default_rng(seed)
        null_features = gaussian_copula_null(
            frame, cfg["features"]["primary"], rng
        )
        null_frame = frame[[cfg["id_column"]]].reset_index(drop=True).copy()
        for feature in cfg["features"]["primary"]:
            null_frame[feature] = null_features[feature].to_numpy()
        labels, z, _, _ = fit_primary(null_frame, cfg, seed=seed)
        rows.append(
            {
                "dataset": cfg["dataset_name"],
                "null_repeat": repeat,
                "seed": seed,
                "silhouette": float(silhouette_score(z, labels)),
            }
        )

    table = pd.DataFrame(rows)
    summary = {
        "dataset": cfg["dataset_name"],
        "null_repeats": repeats,
        "real_silhouette": real_silhouette,
        "null_silhouette_mean": float(table["silhouette"].mean()),
        "null_silhouette_sd": float(table["silhouette"].std()),
        "null_silhouette_p05": float(table["silhouette"].quantile(0.05)),
        "null_silhouette_p50": float(table["silhouette"].quantile(0.50)),
        "null_silhouette_p95": float(table["silhouette"].quantile(0.95)),
        "null_silhouette_max": float(table["silhouette"].max()),
        "n_null_ge_real": int((table["silhouette"] >= real_silhouette).sum()),
        "empirical_p_silhouette": float(
            (1 + (table["silhouette"] >= real_silhouette).sum())
            / (repeats + 1)
        ),
        "null_model": (
            "Single Gaussian-copula reference preserving empirical univariate "
            "marginals and approximate rank-correlation structure, with no "
            "explicit latent mixture groups."
        ),
    }
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend the Gaussian-copula null comparison for silhouette only. "
            "This is intentionally cheaper than rerunning nested subsampling."
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
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=900001)
    parser.add_argument(
        "--output-dir",
        default="results/methods_revision",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i, config in enumerate(
        [args.stroke_config, args.sepsis_config]
    ):
        table, summary = run_one(
            config,
            args.repeats,
            args.base_seed + i * 1000000,
        )
        dataset = summary["dataset"]
        table.to_csv(
            output_dir
            / f"null_reference_silhouette_extended_{dataset}.csv",
            index=False,
        )
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(
        output_dir / "null_reference_silhouette_extended_summary.csv",
        index=False,
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
