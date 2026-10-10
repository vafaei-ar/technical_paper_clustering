"""Local-only Gaussian-copula geometry and null-stability diagnostics.

Usage (from repository root):
    python scripts/run_null_reference_extension.py --cohort sepsis --mode geometry
    python scripts/run_null_reference_extension.py --cohort sepsis --mode stability --null-repeats 100

Reads frozen patient data locally; writes only aggregate statistics.
Stability mode can be computationally expensive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from methods_revision_analyses import (
    fit_primary, gaussian_copula_null, is_binary_column,
    load_yaml, run_null_reference,
)

EXPECTED = {"stroke": (9835, 26), "sepsis": (15842, 52)}
SEEDS = {"stroke": 73001, "sepsis": 1073001}


def geometry(z, labels, n):
    z = np.asarray(z)
    center = z.mean(axis=0)
    total = np.square(z - center).sum()
    within = sum(
        np.square(z[labels == group] - z[labels == group].mean(axis=0)).sum()
        for group in np.unique(labels)
    )
    sizes = np.unique(labels, return_counts=True)[1]
    return {
        "pca_components": int(z.shape[1]),
        "between_to_total_variance_fraction": float(1 - within / total),
        "smallest_cluster_fraction": float(sizes.min() / len(labels)),
        "sampled_silhouette": float(silhouette_score(
            z, labels, sample_size=min(n, len(z)), random_state=2026
        )),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["stroke", "sepsis"], default="sepsis")
    ap.add_argument("--mode", choices=["geometry", "stability"], default="geometry")
    ap.add_argument("--geometry-repeats", type=int, default=12)
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--null-repeats", type=int, default=100)
    ap.add_argument("--null-subsamples", type=int, default=10)
    ap.add_argument("--outdir", type=Path, default=Path("results/null_reference_extension"))
    args = ap.parse_args()
    if min(args.geometry_repeats, args.null_repeats, args.null_subsamples) < 1:
        ap.error("Repeat/subsample counts must be positive")
    if args.sample_size < 50:
        ap.error("Sample size must be at least 50")

    cfg_path = Path("configs/stroke_methods_revision.yaml" if args.cohort == "stroke"
                    else "configs/sepsis.yaml")
    cfg = load_yaml(cfg_path)
    frame = pd.read_parquet(cfg["input_path"])
    features = cfg["features"]["primary"]
    n_expected, p_expected = EXPECTED[args.cohort]
    if len(frame) != n_expected or len(features) != p_expected:
        raise ValueError("Frozen cohort row/feature counts differ from manuscript; stop")

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {"cohort": args.cohort, "mode": args.mode, "input_rows": len(frame),
              "features": len(features), "configuration": str(cfg_path)}
    if args.mode == "geometry":
        labels, z, _, _ = fit_primary(frame, cfg)
        reference = frame[features].apply(pd.to_numeric, errors="coerce")
        observed = geometry(z, labels, args.sample_size)
        binary = [f for f in features if is_binary_column(reference[f])]
        old_corr = reference.corr(method="spearman").to_numpy()
        upper = np.triu_indices(len(features), 1)
        rows = []
        for i in range(args.geometry_repeats):
            seed = SEEDS[args.cohort] + i
            null = gaussian_copula_null(frame, features, np.random.default_rng(seed))
            surrogate = pd.concat(
                [frame[[cfg["id_column"]]].reset_index(drop=True),
                 null.reset_index(drop=True)], axis=1
            )
            lab, z_ref, _, _ = fit_primary(surrogate, cfg, seed=seed)
            metrics = geometry(z_ref, lab, args.sample_size)
            new_corr = null.corr(method="spearman").to_numpy()
            metrics["spearman_mae"] = float(
                np.nanmean(np.abs(old_corr[upper] - new_corr[upper]))
            )
            metrics["binary_prevalence_mae"] = (
                float(np.mean([abs(reference[f].mean() - null[f].mean())
                               for f in binary])) if binary else None
            )
            metrics["repeat"] = i
            rows.append(metrics)
            print(f"Geometry reference {i + 1}/{args.geometry_repeats}", flush=True)
        table = pd.DataFrame(rows)
        table.to_csv(args.outdir / f"geometry_references_{args.cohort}.csv", index=False)
        report["observed_geometry"] = observed
        report["reference_geometry_means"] = {
            c: float(table[c].mean()) for c in table.columns if c != "repeat"
        }
        report["reference_sampled_silhouette_p05"] = float(
            table["sampled_silhouette"].quantile(0.05)
        )
        report["sample_size"] = min(args.sample_size, len(frame))
        report["interpretation"] = (
            "Exploratory sampled silhouettes are not comparable to the full-cohort "
            "silhouettes used in the manuscript."
        )
    else:
        _, summary = run_null_reference(
            frame, cfg, args.outdir, args.null_repeats,
            args.null_subsamples, .80, SEEDS[args.cohort]
        )
        report["stability"] = summary

    dest = args.outdir / f"null_extension_summary_{args.cohort}.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
