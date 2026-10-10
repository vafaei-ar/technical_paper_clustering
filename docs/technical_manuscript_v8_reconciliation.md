# Technical manuscript v8: analysis source of truth

**Status:** author-review manuscript and supplement, not a submission-ready release. This is a non-PHI aggregate evidence index. Do not substitute the older 28-feature stroke narrative from the v6 revision pack for the 26-feature results below.

## Frozen primary reporting analyses

| Cohort | Clustering inputs | Primary model | Reporting source run |
| --- | ---: | --- | --- |
| Stroke | 26; stroke-type flags excluded | PCA k-means, k=3 | `results/stroke_methods_revision/20260918_100745` |
| Sepsis | 52 | PCA k-means, k=3 | `results/sepsis/20260918_114409` |

Run paths come from `results/methods_revision_reporting/reporting_manifest.json`; they are not remotely accessible reproducibility archives.

## Result crosswalk

| Analysis | Stroke | Sepsis |
| --- | ---: | ---: |
| Full-refit 80% subsample ARI, 50 replicates | 0.956543 | 0.975087 |
| Observed silhouette | 0.158480 | 0.169597 |
| Gaussian-copula null silhouette mean (100) | 0.119911 | 0.197282 |
| Null draws with silhouette >= observed | 0/100 | 100/100 |
| Upper-tail empirical reference p | 0.009901 | 1.000000 |
| Balanced-block ARI vs primary | 0.543240 | 0.140962 |
| Raw vs PCA k=3 ARI | 0.994420 | 0.996803 |
| Balanced-block mean matched centroid-profile correlation | 0.978553 | 0.679719 |
| Balanced-block minimum matched centroid-profile correlation | 0.958495 | 0.378163 |

Sepsis WBC redundancy: dropping **percentages**, retaining counts gives ARI **0.149053**; dropping **counts**, retaining percentages gives ARI **0.366188**. These variants must be labeled distinctly.

Principal sources, untracked by design: `tableS_candidate_model_audit_stroke.csv`, `tableS_candidate_model_audit_sepsis.csv`, `null_reference_silhouette_extended_summary.csv`, `balanced_block_summary.csv`, `sepsis_wbc_redundancy_sensitivity.csv`, `representation_k_sensitivity_stroke.csv`, `representation_k_sensitivity_sepsis.csv`, and full-refit stability summaries in the respective cohort run directories.

The null comparison is a diagnostic comparison against a selected single-component Gaussian-copula reference. It is **not** evidence of absence of biological sepsis subgroups or proof of discrete stroke subtypes.

## Manuscript and supplement alignment

The author-review manuscript v8 is based on v7 accepted preview. The five-figure sequence follows the maintained `paper_figures` package: workflow; model selection; null reference; phenotype heatmaps; robustness. The companion author-review supplement v4 adds a stage-1 versus stages-1+2 flag to its S2 candidate table, corrects the WBC labels in S8, orders S11-S14 tables with their headings, and adds S15 primary run identifiers.

Do not commit patient-level data or locally produced result directories to the repository.

## Submission blockers

- Confirm cohort source institutions, periods, extraction dates, inclusion/exclusion definitions, diagnostic code lists and exact `PATID` source interpretation.
- Obtain IRB determinations, protocol numbers, consent waivers, and governance text.
- Verify upstream sepsis missingness and imputation process.
- Provide all full per-feature phenotype profiles and re-check frozen input hashes.
- Archive the exact analysis environment and create tagged releases/DOIs.
- The **Clustro package** at `vafaei-ar/clustro` has an MIT license; this **paper-specific repository** still needs an explicit approved LICENSE file. Do not assume the parent repo's license automatically covers it.
- Before moving hardcoded choices to YAML, record the values used here (`PCA(n_components=0.90)`, `KMeans(n_init=20)`, `GaussianMixture(n_init=5)`). Changing defaults without a new run must not silently change the reported results.

The manuscript and supplement are maintained as author-review documents outside source control until provenance and release policies have been approved.
