#!/usr/bin/env bash
set -euo pipefail

# Short follow-up after the first methods-review run.
# This adds full-pipeline stability for k=4, which is needed because k=4
# has the best internal separation metrics in sepsis.

python bootstrap_stability.py   --config configs/stroke_methods_revision.yaml   --candidates configs/methods_revision_candidates.yaml   --n-repeats 50   --sample-fraction 0.80   --refit-preprocessing

python bootstrap_stability.py   --config configs/sepsis.yaml   --candidates configs/methods_revision_candidates.yaml   --n-repeats 50   --sample-fraction 0.80   --refit-preprocessing

echo
echo "Updated full-refit stability summaries:"
echo "  results/stroke_methods_revision/<latest>/subsample_stability_full_refit_summary.csv"
echo "  results/sepsis/<latest>/subsample_stability_full_refit_summary.csv"
