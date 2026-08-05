# Executable synthetic example

This example demonstrates the repository workflow without protected health information. The generated data are entirely synthetic and contain three known latent groups designed to resemble preserved, renal-anaemic, and hyperglycaemic clinical patterns.

## Run

From the repository root with the project virtual environment activated:

```bash
python examples/generate_synthetic_clinical_example.py
python -m tpcluster.cli --config examples/synthetic_config.yaml
python examples/verify_synthetic_recovery.py --config examples/synthetic_config.yaml
```

The generator writes both CSV and Parquet files under `examples/data/`. The clustering run writes its outputs under `results/synthetic/`. The verification script automatically uses the most recent synthetic run unless `--run-dir` is supplied.

## Expected checks

A successful run should produce:

- `internal_metrics.csv`
- `cluster_assignments.parquet`
- `seed_stability.csv`
- preprocessing audit files
- `verification/synthetic_recovery_verification.csv`
- `verification/synthetic_recovery_verification.json`

The true synthetic group is retained only in the generated input as `cluster_truth`; it is deliberately excluded from the clustering feature list. After model fitting, the verification script compares `cluster_truth` with the prespecified PCA + k-means, k=3, seed=11 solution using adjusted Rand index (ARI). The default verification threshold is ARI >= 0.80 and can be changed with `--minimum-ari`.

Because ARI is invariant to cluster-label permutations, predicted cluster numbers do not need to match the numeric truth labels directly.

## Purpose and limitations

The example verifies installation, configuration parsing, preprocessing, candidate-model fitting, quality gates, output schemas, deterministic reruns, and recovery of known synthetic latent structure. It is not intended to simulate the full complexity, missingness, coding practices, or causal structure of real EHR data.
