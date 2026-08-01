#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .
pytest -q

for cohort in stroke sepsis; do
  config="configs/${cohort}.yaml"
  tpcluster --config "$config"
  python profile_candidates.py --config "$config"
  python bootstrap_stability.py --config "$config" --n-repeats 50 --sample-fraction 0.80
  python generate_manuscript_outputs.py --config "$config"
  python polish_manuscript_labels.py --config "$config"
done

python generate_paper_figures.py

echo "All reproducible outputs were written under results/."
