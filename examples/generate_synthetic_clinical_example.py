from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_cohort(n_per_cluster: int = 120, seed: int = 20260804) -> pd.DataFrame:
    """Create a small non-identifiable cohort with three known latent groups."""
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    specifications = [
        {
            "cluster_truth": 0,
            "age_mean": 58,
            "creatinine_mean": 0.9,
            "glucose_mean": 105,
            "hemoglobin_mean": 14.2,
            "diabetes_p": 0.15,
            "kidney_disease_p": 0.08,
            "heart_failure_p": 0.06,
        },
        {
            "cluster_truth": 1,
            "age_mean": 72,
            "creatinine_mean": 2.1,
            "glucose_mean": 120,
            "hemoglobin_mean": 10.4,
            "diabetes_p": 0.35,
            "kidney_disease_p": 0.70,
            "heart_failure_p": 0.45,
        },
        {
            "cluster_truth": 2,
            "age_mean": 64,
            "creatinine_mean": 1.1,
            "glucose_mean": 245,
            "hemoglobin_mean": 13.4,
            "diabetes_p": 0.85,
            "kidney_disease_p": 0.20,
            "heart_failure_p": 0.20,
        },
    ]

    for specification in specifications:
        n = int(n_per_cluster)
        rows.append(
            pd.DataFrame(
                {
                    "PATID": [
                        f"SYN-{specification['cluster_truth']}-{index:04d}"
                        for index in range(n)
                    ],
                    "AGE_AT_EVENT": rng.normal(specification["age_mean"], 8, n),
                    "Creatinine": np.clip(
                        rng.normal(specification["creatinine_mean"], 0.30, n),
                        0.2,
                        None,
                    ),
                    "Glucose": np.clip(
                        rng.normal(specification["glucose_mean"], 28, n),
                        45,
                        None,
                    ),
                    "Hemoglobin": np.clip(
                        rng.normal(specification["hemoglobin_mean"], 1.1, n),
                        5,
                        None,
                    ),
                    "Diabetes": rng.binomial(1, specification["diabetes_p"], n),
                    "Chronic_kidney_disease": rng.binomial(
                        1, specification["kidney_disease_p"], n
                    ),
                    "Heart_failure": rng.binomial(
                        1, specification["heart_failure_p"], n
                    ),
                    "cluster_truth": specification["cluster_truth"],
                }
            )
        )

    cohort = pd.concat(rows, ignore_index=True)
    cohort["AGE_AT_EVENT"] = cohort["AGE_AT_EVENT"].round(1)
    cohort["Creatinine"] = cohort["Creatinine"].round(2)
    cohort["Glucose"] = cohort["Glucose"].round(1)
    cohort["Hemoglobin"] = cohort["Hemoglobin"].round(1)
    return cohort.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/data/synthetic_clinical_cohort.parquet")
    parser.add_argument("--n-per-cluster", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort = generate_synthetic_cohort(args.n_per_cluster, args.seed)
    cohort.to_parquet(output_path, index=False)
    cohort.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Rows: {len(cohort)}")
    print(f"Parquet: {output_path}")
    print(f"CSV: {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
