# Executable synthetic example

This example demonstrates the repository workflow without protected health information. The generated data are entirely synthetic and contain three known latent groups designed to resemble preserved, renal-anaemic, and hyperglycaemic clinical patterns.

## Run

From the repository root with the project virtual environment activated:

```bash
python examples/generate_synthetic_clinical_example.py
python -m tpcluster.cli --config examples/synthetic_config.yaml
```

The generator writes both CSV and Parquet files under `examples/data/`. The clustering run writes its outputs under `results/synthetic/`.

## Expected checks

A successful run should produce:

- `internal_metrics.csv`
- `cluster_assignments.parquet`
- `seed_stability.csv`
- preprocessing audit files

The true synthetic group is retained only in the generated input as `cluster_truth`; it is deliberately excluded from the clustering feature list. It can be used afterward to calculate external agreement for software verification, but it must not influence model fitting.

## Purpose and limitations

The example verifies installation, configuration parsing, preprocessing, candidate-model fitting, quality gates, output schemas, and deterministic reruns. It is not intended to simulate the full complexity, missingness, coding practices, or causal structure of real EHR data.
