#!/usr/bin/env bash
set -euo pipefail

# Short follow-up after the first methods-review run.
# 1) Adds full-pipeline stability for k=4, needed because k=4 has the
#    strongest internal separation metrics in sepsis.
# 2) Extends the Gaussian-copula silhouette-only reference to 100 null
#    datasets without repeating the expensive nested stability analysis.

python bootstrap_stability.py   --config configs/stroke_methods_revision.yaml   --candidates configs/methods_revision_candidates.yaml   --n-repeats 50   --sample-fraction 0.80   --refit-preprocessing

python bootstrap_stability.py   --config configs/sepsis.yaml   --candidates configs/methods_revision_candidates.yaml   --n-repeats 50   --sample-fraction 0.80   --refit-preprocessing

python extend_null_reference.py   --stroke-config configs/stroke_methods_revision.yaml   --sepsis-config configs/sepsis.yaml   --repeats "${NULL_SILHOUETTE_REPEATS:-100}"   --output-dir results/methods_revision

echo
echo "Follow-up complete."
echo "Upload:"
echo "  results/stroke_methods_revision/<latest>/subsample_stability_full_refit_summary.csv"
echo "  results/sepsis/<latest>/subsample_stability_full_refit_summary.csv"
echo "  results/methods_revision/null_reference_silhouette_extended_summary.csv"
