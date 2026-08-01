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
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest
pytest -q
```

## Full reproduction

```bash
bash reproduce_all.sh
```

The workflow performs, for each cohort:

1. the complete clustering grid;
2. finalist candidate profiling;
3. repeated 80% subsampling stability analyses;
4. final cohort-level manuscript tables and figures.

Generated material is written under `results/`, which is ignored by Git.

## Corrected manuscript outputs

The final generator deliberately separates two scales:

- heatmaps use processed clustering-space effect sizes;
- clinical tables use the original imputed clinical scale.

Additional corrections include:

- permutation-invariant canonical phenotype labels;
- short non-overlapping cluster labels for figures;
- simplified stroke discharge groups;
- effect-size magnitude labels alongside FDR-adjusted p-values;
- a clear PCA caption stating that the two-dimensional plot is a visual projection rather than the complete clustering space;
- an analysis manifest recording the Git commit and configuration used.

## Paper-level publication figures

After both cohort runs and manuscript outputs exist, generate the compact paper-level figures with:

```bash
python generate_paper_figures.py \
  --stroke-run results/stroke/<stroke_run_id> \
  --sepsis-run results/sepsis/<sepsis_run_id>
```

The default outputs are written to:

```text
results/paper_figures/latest/
  figures/
    figure1_workflow.png
    figure1_workflow.pdf
    figure2_pca_projection.png
    figure2_pca_projection.pdf
    figure3_phenotype_heatmaps.png
    figure3_phenotype_heatmaps.pdf
    figure4_posthoc_effect_sizes.png
    figure4_posthoc_effect_sizes.pdf
  manifests/
    paper_figure_manifest.json
```

These figures follow the project figure standard by using compact canvases, large readable text, short cluster labels, subordinate color bars, non-rotated heatmap labels, explicit sample sizes, deterministic PCA display sampling, vector PDF export, and separated footer notes.

A dedicated stability-selection figure is intentionally not included in this paper-level set. Existing stability summaries remain available in the manuscript tables, while deeper stability-selection analyses are reserved for future work.

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
