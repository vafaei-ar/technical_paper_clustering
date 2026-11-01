#!/usr/bin/env python3
"""Validate private aggregate outputs, then regenerate Clustro Figures 4 and 5.

Reads existing study outputs and frozen file hashes. Never synthesizes clinical data,
refits clustering, or writes patient-level values. The manifest includes only
aggregate-input/output hashes and selected-run numerical checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "stroke": {"n": 9835, "silhouette": 0.158480, "sizes": [5697, 2660, 1478]},
    "sepsis": {"n": 15842, "silhouette": 0.169597, "sizes": [11454, 1362, 3026]},
}
FROZEN_HASHES = {
    "data/stroke/stroke_cohort_imputed.parquet": "ce56d3cd81c578796592f21a707d1cd97fdf68068a393606aee286b1d54844ca",
    "data/sepsis/sepsis_cohort_imputed_safe.parquet": "ed4fed84e5c3d533b2fc84177d471fc881e3b5a7e24c404faf10f75556963763",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing original analysis output: {path}")
    return path

def check(report_dir: Path, check_frozen_hashes: bool) -> dict:
    manifest = json.loads(required(report_dir / 'reporting_manifest.json').read_text())
    source_files = []
    checked = {}
    for cohort, target in EXPECTED.items():
        size_path = required(report_dir / f'table_cluster_sizes_{cohort}.csv')
        sensitivity_path = required(report_dir / f'representation_k_sensitivity_{cohort}.csv')
        audit_path = required(report_dir / f'tableS_candidate_model_audit_{cohort}.csv')
        source_files.extend([size_path, sensitivity_path, audit_path])
        sizes = pd.read_csv(size_path).sort_values('primary_cluster')
        actual_sizes = sizes['n'].astype(int).tolist()
        if actual_sizes != target['sizes'] or sum(actual_sizes) != target['n']:
            raise ValueError(f"{cohort} frozen-cluster sizes disagree with manuscript: {actual_sizes}")
        sensitivity = pd.read_csv(sensitivity_path)
        selected = sensitivity.loc[
            sensitivity['reduction'].astype(str).str.lower().eq('pca')
            & sensitivity['clusterer'].astype(str).str.lower().eq('kmeans')
            & pd.to_numeric(sensitivity['k'], errors='coerce').eq(3)
        ]
        if len(selected) != 1:
            raise ValueError(f"{cohort} must have one PCA k-means k=3 selected run, found {len(selected)}")
        score = float(selected.iloc[0]['silhouette'])
        if abs(score - target['silhouette']) > 0.0005:
            raise ValueError(f"{cohort} selected-run silhouette {score:.6f} != verified {target['silhouette']:.6f}")
        selected_run = Path(manifest[f'{cohort}_run'])
        if not selected_run.is_absolute():
            selected_run = ROOT / selected_run
        profile_dir = selected_run / 'manuscript_outputs' / 'tables'
        for name in ('table_continuous_cluster_profiles_processed_scale.csv',
                     'table_binary_cluster_profiles.csv'):
            source_files.append(required(profile_dir / name))
        checked[cohort] = {'n': target['n'], 'cluster_sizes': actual_sizes,
                           'selected_run_silhouette': score,
                           'source_run': str(selected_run.relative_to(ROOT)) if selected_run.is_relative_to(ROOT) else str(selected_run)}
    for name in ('null_reference_silhouette_extended_summary.csv',
                 'figure4_sensitivity_values.csv'):
        source_files.append(required(report_dir / name))
    frozen_checks = {}
    if check_frozen_hashes:
        for name, expected in FROZEN_HASHES.items():
            found = sha256(required(ROOT / name))
            if found.lower() != expected:
                raise ValueError(f"FROZEN HASH MISMATCH for {name}: {found}")
            frozen_checks[name] = {'sha256': found, 'matched': True}
    return {'cohorts': checked, 'aggregate_input_sha256': {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p)
        for p in sorted(set(source_files))
    }, 'frozen_file_checks': frozen_checks}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reporting-dir', default=str(ROOT / 'results/methods_revision_reporting'))
    parser.add_argument('--outdir', default=str(ROOT / 'results/final_paper_figures_v867'))
    parser.add_argument('--check-frozen-hashes', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    report_dir = Path(args.reporting_dir).resolve()
    outdir = Path(args.outdir).resolve()
    try:
        result = check(report_dir, args.check_frozen_hashes)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f'PRE-FLIGHT FAILED: {error}', file=sys.stderr)
        return 2
    if not args.verify_only:
        outdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CLUSTRO_REPORTING_DIR=str(report_dir))
        subprocess.run([sys.executable, str(ROOT / 'paper_figures/make_figures.py'),
                        '--only', '4', '5', '--outdir', str(outdir), '--pdf', '--dpi', '600'],
                       cwd=ROOT, env=env, check=True)
        paths = [required(outdir / name) for name in (
            'fig4_phenotypes.png','fig4_phenotypes.pdf',
            'fig5_robustness.png','fig5_robustness.pdf')]
        result['figure_sha256'] = {p.name: sha256(p) for p in paths}
        (outdir / 'figure_source_manifest_v867.json').write_text(
            json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['cohorts'], indent=2))
    print('Verified source analysis outputs.' if args.verify_only else f'Regenerated verified figures: {outdir}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
