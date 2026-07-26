# Technical Paper Clustering

Reproducible clustering analysis for the frozen stroke and sepsis cohorts used in the technical paper.

## Repository policy

This repository contains code and configuration only. Patient-level data, intermediate matrices, cluster assignments, logs, figures, tables, archives, and all generated results are excluded by `.gitignore`.

## Expected private data locations

Place the frozen input files at:

```text
data/stroke/stroke_cohort_imputed.parquet
data/sepsis/sepsis_cohort_imputed_safe.parquet
```

These files are intentionally not tracked by Git.

## Environment

```bash
conda create -n eda python=3.11 -y
conda activate eda
python -m pip install -e .
pytest -q
```

## Full reproduction

```bash
bash reproduce_all.sh
```

The workflow performs, for each cohort:

1. the complete clustering grid;
2. finalist candidate profiling;
3. 50 repeated 80% subsampling stability analyses;
4. final manuscript tables and figures.

Generated material is written under `results/`, which is ignored by Git.

## Corrected manuscript outputs

The final generator deliberately separates two scales:

- heatmaps use processed clustering-space effect sizes;
- clinical tables use the original imputed clinical scale.

Additional corrections include:

- short non-overlapping cluster labels for figures;
- simplified stroke discharge groups;
- effect-size magnitude labels alongside FDR-adjusted p-values;
- a clear PCA caption stating that the two-dimensional plot is a visual projection rather than the complete clustering space;
- an analysis manifest recording the Git commit and configuration used.

## Primary models

- Stroke: PCA plus K-means, `k=3`; `k=2` retained as sensitivity analysis.
- Sepsis: PCA plus K-means, `k=3`; `k=2` and raw-space K-means retained as sensitivity analyses.

## Data integrity

The frozen datasets used during development had the following SHA-256 hashes:

```text
stroke: ce56d3cd81c578796592f21a707d1cd97fdf68068a393606aee286b1d54844ca
sepsis: ed4fed84e5c3d533b2fc84177d471fc881e3b5a7e24c404faf10f75556963763
```

Verify local inputs before reproducing the paper outputs.
