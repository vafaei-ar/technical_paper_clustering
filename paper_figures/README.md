# Clustro paper figures

This directory now uses the **same modular visualization code and layout system as the supplied aligned figure package**, with the mock/synthetic values replaced by the repository's real reviewer-driven analysis outputs.

## Files

- `clustro_style.py`: shared layout system, palette, typography, vector icons
- `clustro_data.py`: loads the real analysis outputs from `results/`
- `fig1_workflow.py`: workflow and decision logic
- `fig2_model_selection.py`: model-selection landscape
- `fig3_null_reference.py`: null-reference benchmarking
- `fig4_phenotypes.py`: real phenotype profile heatmaps and summary cards
- `fig5_robustness.py`: specification-sensitivity analyses
- `make_figures.py`: builds all five figures

The geometry, panel structure, typography, icon system, and card layouts follow the supplied visualization code rather than the earlier simplified reimplementation.

## Generate all figures

From the repository root:

```bash
source .venv/bin/activate

python paper_figures/make_figures.py \
  --outdir results/final_paper_figures_v4 \
  --pdf \
  --dpi 600
```

No local server is needed. Open or download the PNG/PDF files directly from:

```text
results/final_paper_figures_v4/
```

## Data sources

The figure code reads the existing analysis outputs rather than synthetic placeholders:

- candidate audit and full-refit ARI: `results/methods_revision_reporting/`
- 100 Gaussian-copula null replicates: `results/methods_revision/`
- phenotype profile tables: the stroke and sepsis run directories referenced by `results/methods_revision_reporting/reporting_manifest.json`
- balanced-block, representation, and WBC-redundancy sensitivity: `results/methods_revision_reporting/`

Figure 3 smooths the **actual 100 null replicate silhouettes** with a simple Gaussian-kernel density for display. Figure 4 reads the actual continuous and binary cluster-profile effect tables.
