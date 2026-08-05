from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_full_run(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_dir()
        and (path / "internal_metrics.csv").exists()
        and (path / "subsample_stability_summary.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed clustering run found under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def primary_model_for_dataset(dataset: str, models_path: str | Path) -> dict[str, Any]:
    models = load_yaml(models_path)
    if dataset not in models or "primary" not in models[dataset]:
        raise KeyError(f"Primary model not found for {dataset!r} in {models_path}")
    return models[dataset]["primary"]


def pca_retention_table(
    config_path: str | Path,
    models_path: str | Path,
    run_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    cfg = load_yaml(config_path)
    dataset = str(cfg["dataset_name"])
    run_dir = Path(run_dir) if run_dir else latest_full_run(cfg["output_dir"])
    primary = primary_model_for_dataset(dataset, models_path)

    metrics = pd.read_csv(run_dir / "internal_metrics.csv")
    selected = metrics[
        (metrics["reduction"] == primary["reduction"])
        & (metrics["clusterer"] == primary["clusterer"])
        & (metrics["k"] == int(primary["k"]))
    ].copy()
    if selected.empty:
        raise ValueError(f"Primary model metrics not found in {run_dir}")

    reference_seed = int(primary.get("seed", cfg.get("seeds", [11])[0]))
    reference = selected[selected["seed"] == reference_seed]
    if reference.empty:
        reference = selected.sort_values("seed").head(1)
    row = reference.iloc[0]

    result = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "primary_model": primary["name"],
                "reduction": primary["reduction"],
                "reference_seed": int(row["seed"]),
                "input_feature_count": len(cfg["features"]["primary"]),
                "retained_component_count": int(row["n_components"]),
                "retention_rule": "PCA retained the minimum number of components explaining at least 90% of variance",
                "target_cumulative_explained_variance": 0.90,
                "pca_solver": "full",
                "preprocessing_before_pca": "plausibility filtering where configured; log1p transformation where configured; median imputation; zero-variance removal; percentile clipping; robust scaling",
            }
        ]
    )

    output_dir = run_dir / "manuscript_outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tableS_pca_retention.csv"
    result.to_csv(output_path, index=False)
    return result, output_path


def stability_method_audit(
    config_path: str | Path,
    run_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    cfg = load_yaml(config_path)
    dataset = str(cfg["dataset_name"])
    run_dir = Path(run_dir) if run_dir else latest_full_run(cfg["output_dir"])

    detailed_path = run_dir / "subsample_stability_detailed.csv"
    if not detailed_path.exists():
        raise FileNotFoundError(f"Missing {detailed_path}")
    detailed = pd.read_csv(detailed_path)

    audit = {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "n_repeats": int(detailed["subsample_seed"].nunique()),
        "sample_fraction": float(detailed["sample_fraction"].dropna().iloc[0]),
        "sampling_without_replacement": True,
        "reference_solution": "Each candidate was fit once on the full cohort using its configured reference seed.",
        "subsample_solution": "The candidate was refit independently in each subsample using the subsample seed.",
        "ari_comparison_set": "ARI compared full-solution labels restricted to sampled observations against labels from the refit subsample solution.",
        "ari_label_handling": "Raw estimator labels were used; ARI is permutation-invariant, so canonical relabeling is unnecessary.",
        "pca_refit_within_each_subsample": True,
        "clusterer_refit_within_each_subsample": True,
        "preprocessing_refit_within_each_subsample": False,
        "preprocessing_scope": "Median imputation, zero-variance removal, clipping, and scaling were fit once on the full cohort before subsampling.",
        "important_limitation": "Because preprocessing was estimated on the full cohort before subsampling, the current resampling analysis evaluates clustering and PCA perturbation conditional on fixed full-cohort preprocessing rather than a fully nested resampling pipeline.",
        "recommended_manuscript_wording": "In repeated 80% subsamples, PCA and clustering were refit within each subsample, while preprocessing parameters were inherited from the full analytic cohort. Agreement was quantified on shared observations using the permutation-invariant adjusted Rand index.",
    }

    output_dir = run_dir / "manuscript_outputs" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stability_method_audit.json"
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit, output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PCA-retention details and audit the existing repeated-subsample implementation without refitting clustering models."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="configs/manuscript_models.yaml")
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    pca_table, pca_path = pca_retention_table(args.config, args.models, args.run_dir)
    audit, audit_path = stability_method_audit(args.config, args.run_dir)

    print(pca_table.to_string(index=False))
    print(f"PCA table: {pca_path}")
    print(f"Stability audit: {audit_path}")
    print(f"Preprocessing refit within each subsample: {audit['preprocessing_refit_within_each_subsample']}")


if __name__ == "__main__":
    main()
