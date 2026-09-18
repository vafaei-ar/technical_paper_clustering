#!/usr/bin/env bash
set -euo pipefail

# Reviewer-driven revision analyses for the technical manuscript.
# Run from the repository root after activating the project environment.

STROKE_CONFIG="configs/stroke_methods_revision.yaml"
SEPSIS_CONFIG="configs/sepsis.yaml"
CANDIDATES="configs/methods_revision_candidates.yaml"
OUTPUT_DIR="results/methods_revision"

# 1. Refit the complete stroke grid without Ischemic_Stroke/Hemorrhagic_Stroke
#    as clustering inputs.
tpcluster --config "$STROKE_CONFIG"

# 2. Reuse the latest completed sepsis full-grid run unless SEPSIS_RERUN=1.
if [[ "${SEPSIS_RERUN:-0}" == "1" ]]; then
  tpcluster --config "$SEPSIS_CONFIG"
fi

# 3. Full-pipeline repeated subsampling for the explicit shortlist.
python bootstrap_stability.py   --config "$STROKE_CONFIG"   --candidates "$CANDIDATES"   --n-repeats 50   --sample-fraction 0.80   --refit-preprocessing

python bootstrap_stability.py   --config "$SEPSIS_CONFIG"   --candidates "$CANDIDATES"   --n-repeats 50   --sample-fraction 0.80   --refit-preprocessing

# 4. Balanced-block, representation/k, sepsis redundancy, null-reference,
#    and complete candidate-audit outputs.
python methods_revision_analyses.py   --stroke-config "$STROKE_CONFIG"   --sepsis-config "$SEPSIS_CONFIG"   --output-dir "$OUTPUT_DIR"   --null-repeats "${NULL_REPEATS:-20}"   --null-subsamples "${NULL_SUBSAMPLES:-10}"   --sample-fraction 0.80

printf '\nRevision outputs written to %s\n' "$OUTPUT_DIR"
printf 'Zip that directory and return it for scientific review before manuscript edits.\n'
