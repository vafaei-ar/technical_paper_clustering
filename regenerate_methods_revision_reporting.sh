#!/usr/bin/env bash
set -euo pipefail

# Reporting-only finalization after reviewer-driven analyses.
# This does not rerun the clustering grid, null analysis, or robustness analyses.
# It regenerates profiles/tables for the finalized stroke feature set and builds
# publication figures/tables from the completed analysis outputs.

STROKE_CONFIG="configs/stroke_methods_revision.yaml"
SEPSIS_CONFIG="configs/sepsis.yaml"
CANDIDATES="configs/methods_revision_candidates.yaml"

python profile_candidates.py   --config "$STROKE_CONFIG"   --candidates "$CANDIDATES"

python generate_manuscript_outputs.py   --config "$STROKE_CONFIG"

python profile_candidates.py   --config "$SEPSIS_CONFIG"   --candidates "$CANDIDATES"

python regenerate_manuscript_outputs.py   --config "$SEPSIS_CONFIG"

python generate_methods_revision_reporting.py   --stroke-config "$STROKE_CONFIG"   --sepsis-config "$SEPSIS_CONFIG"   --methods-dir results/methods_revision   --output-dir results/methods_revision_reporting

echo
echo "Reporting package complete."
echo "Zip and upload:"
echo "  results/methods_revision_reporting"
