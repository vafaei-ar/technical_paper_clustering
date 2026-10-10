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

## Candidate-model selection table

The supplementary candidate-model table consolidates the complete clustering grid rather than only the shortlisted models. It reports internal validity across seeds, cluster-size gates, pairwise seed agreement, the reference-seed solution, and available repeated-subsample agreement.

The model roles and concise selection rationales are recorded in `configs/manuscript_models.yaml`.

Generate the tables without refitting any clustering models:

```bash
python generate_candidate_model_summary.py --config configs/stroke.yaml
python generate_candidate_model_summary.py --config configs/sepsis.yaml
```

The outputs are written to each completed run:

```text
results/<cohort>/<run_id>/manuscript_outputs/tables/
  tableS_candidate_model_selection.csv
```

## Earlier paper-figure generator (legacy)

The prior four-figure workflow is retained in `generate_paper_figures.py`
for reproducibility of older manuscript versions. It is **not** the current
five-figure manuscript visualization workflow. Its outputs go to
`results/paper_figures/latest/`. Do not delete or alter this generator while
older manuscript results may still need to be reproduced.

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


## Frozen cohort index-date provenance

To verify the study periods directly from the **private, frozen Parquet files**,
run this from the repository root:

```bash
python scripts/report_cohort_dates.py
```

The script reads only `PATID` and the original index-date columns
(`DX_DATE_stroke`, `DX_DATE_sepsis`). It reports the earliest/latest dates,
row counts, distinct patients, missing/unparseable index dates, and obvious
date-range anomalies. It compares row counts with the manuscript cohorts
(stroke 9,835; sepsis 15,842). Only an **aggregate JSON** is created at
`results/provenance/cohort_index_date_summary.json`, which is excluded from Git.

If the Parquet files are not in the default `data/` locations, pass
`--stroke /absolute/path/to/stroke.parquet` and
`--sepsis /absolute/path/to/sepsis.parquet`.
If the expected date column is missing, the script lists candidate fields and
stops rather than silently substituting an admission date.

The **index-date range is not the source-database coverage period or data
extraction/freeze date**. Record those separately from your data-release
manifest or request them from the data custodian.

## Current manuscript figure set (Figures 1-5)

The authoritative, editable Python figure scripts are in `paper_figures/`.
They use completed reviewer-driven analysis outputs and **do not refit
clustering models**. First generate the methods-revision reporting outputs
using `regenerate_methods_revision_reporting.sh` when necessary, and then:

```bash
python paper_figures/make_figures.py \
  --outdir results/final_paper_figures \
  --pdf --dpi 600
```

Five figures are produced: workflow, model selection, null reference,
phenotype heatmaps, and robustness/sensitivity analysis. Private cohort data
and all generated results remain outside version control.

For layout editing, change the `TEXT_SIZE`, `TEXT_POS`, `SHOW`, or
`STYLE` dictionaries at the start of the relevant `fig*.py` script.
See `paper_figures/README.md` for data dependencies, figure-by-figure
commands, and settings.

## Branch and quality-check policy

- `main` is the integration branch for the reproducible manuscript workflow.
- Develop changes in short-lived topic branches; merge reviewed changes into
  `main` instead of maintaining parallel long-running versions.
- GitHub Actions runs syntax compilation and the existing synthetic/unit tests.
  It cannot reproduce private-patient-data analyses in CI.
- Historical `generate_author_revision_materials_v*.py` and other reporting
  scripts are deliberately retained until their outputs are archived and
  provenance dependencies verified.

