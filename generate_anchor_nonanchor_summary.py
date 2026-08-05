from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tpcluster.display import display_name


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_manuscript_run(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_dir()
        and (path / "manuscript_outputs" / "tables" / "table_cluster_sizes.csv").exists()
        and (
            path
            / "manuscript_outputs"
            / "tables"
            / "table_continuous_cluster_profiles_processed_scale.csv"
        ).exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed manuscript-output run found under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_feature(available: list[str], requested: str) -> str:
    lookup = {normalise(feature): feature for feature in available}
    match = lookup.get(normalise(requested))
    if match is None:
        raise KeyError(
            f"Anchor feature {requested!r} was not found. Available features include {available[:20]}"
        )
    return match


def anchor_registry(dataset: str, anchors_path: str | Path) -> tuple[dict[str, str], str]:
    cfg = load_yaml(anchors_path)
    if dataset not in cfg:
        raise KeyError(f"Dataset {dataset!r} not found in {anchors_path}")
    entries = cfg[dataset].get("anchor_features", [])
    registry = {str(item["feature"]): str(item["role"]) for item in entries}
    return registry, str(cfg[dataset].get("note", ""))


def prepare_continuous(
    table: pd.DataFrame,
    raw_table: pd.DataFrame,
    anchor_features: set[str],
    top_n: int,
) -> pd.DataFrame:
    table = table.copy()
    table["effect_value"] = pd.to_numeric(table["median_difference_iqr"], errors="coerce")
    table["absolute_effect"] = table["effect_value"].abs()
    table["evidence_role"] = np.where(
        table["feature"].isin(anchor_features), "canonical_label_anchor", "non_anchor_evidence"
    )
    table["domain"] = "continuous"
    table["effect_metric"] = "median_difference_iqr"

    raw_columns = [
        column
        for column in ["cluster", "feature", "median", "q1", "q3", "n_nonmissing"]
        if column in raw_table.columns
    ]
    if raw_columns:
        table = table.merge(
            raw_table[raw_columns],
            on=["cluster", "feature"],
            how="left",
            validate="one_to_one",
        )

    anchors = table[table["evidence_role"] == "canonical_label_anchor"].copy()
    nonanchors = (
        table[table["evidence_role"] == "non_anchor_evidence"]
        .sort_values(["cluster", "absolute_effect"], ascending=[True, False])
        .groupby("cluster", group_keys=False)
        .head(top_n)
        .copy()
    )
    return pd.concat([anchors, nonanchors], ignore_index=True)


def prepare_binary(
    table: pd.DataFrame,
    anchor_features: set[str],
    top_n: int,
) -> pd.DataFrame:
    table = table.copy()
    table["effect_value"] = pd.to_numeric(table["standardised_difference"], errors="coerce")
    table["absolute_effect"] = table["effect_value"].abs()
    table["evidence_role"] = np.where(
        table["feature"].isin(anchor_features), "canonical_label_anchor", "non_anchor_evidence"
    )
    table["domain"] = "binary"
    table["effect_metric"] = "standardised_prevalence_difference"
    nonanchors = (
        table[table["evidence_role"] == "non_anchor_evidence"]
        .sort_values(["cluster", "absolute_effect"], ascending=[True, False])
        .groupby("cluster", group_keys=False)
        .head(top_n)
        .copy()
    )
    return nonanchors


def build_summary(
    config_path: str | Path,
    anchors_path: str | Path = "configs/canonical_anchors.yaml",
    run_dir: str | Path | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, Path, Path]:
    cfg = load_yaml(config_path)
    dataset = str(cfg["dataset_name"])
    run_dir = Path(run_dir) if run_dir else latest_manuscript_run(cfg["output_dir"])
    table_dir = run_dir / "manuscript_outputs" / "tables"
    manifest_dir = run_dir / "manuscript_outputs" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    continuous_path = table_dir / "table_continuous_cluster_profiles_processed_scale.csv"
    binary_path = table_dir / "table_binary_cluster_profiles.csv"
    raw_path = table_dir / "table_continuous_cluster_profiles_raw_scale.csv"
    sizes_path = table_dir / "table_cluster_sizes.csv"

    continuous = pd.read_csv(continuous_path)
    binary = pd.read_csv(binary_path)
    raw = pd.read_csv(raw_path)
    sizes = pd.read_csv(sizes_path)

    requested_anchors, note = anchor_registry(dataset, anchors_path)
    available = sorted(set(continuous["feature"].astype(str)))
    resolved = {
        resolve_feature(available, requested): role
        for requested, role in requested_anchors.items()
    }
    anchor_features = set(resolved)

    continuous_summary = prepare_continuous(continuous, raw, anchor_features, top_n)
    binary_summary = prepare_binary(binary, anchor_features, top_n)
    summary = pd.concat([continuous_summary, binary_summary], ignore_index=True, sort=False)

    cluster_labels = sizes.set_index("primary_cluster")["primary_cluster_label"].to_dict()
    cluster_sizes = sizes.set_index("primary_cluster")["n"].to_dict()
    summary["dataset"] = dataset
    summary["cluster_label"] = summary["cluster"].map(cluster_labels)
    summary["cluster_n"] = summary["cluster"].map(cluster_sizes)
    summary["feature_label"] = summary["feature"].map(display_name)
    summary["anchor_role"] = summary["feature"].map(resolved).fillna("")
    summary["interpretation_note"] = note

    summary["rank_within_cluster_domain"] = np.nan
    mask = summary["evidence_role"].eq("non_anchor_evidence")
    summary.loc[mask, "rank_within_cluster_domain"] = (
        summary.loc[mask]
        .groupby(["cluster", "domain"])["absolute_effect"]
        .rank(method="first", ascending=False)
    )

    preferred_columns = [
        "dataset",
        "cluster",
        "cluster_label",
        "cluster_n",
        "evidence_role",
        "anchor_role",
        "domain",
        "feature",
        "feature_label",
        "effect_metric",
        "effect_value",
        "absolute_effect",
        "rank_within_cluster_domain",
        "median",
        "q1",
        "q3",
        "n_nonmissing",
        "prevalence",
        "overall_prevalence",
        "interpretation_note",
    ]
    output_columns = [column for column in preferred_columns if column in summary.columns]
    summary = summary[output_columns].sort_values(
        ["cluster", "evidence_role", "domain", "rank_within_cluster_domain", "feature"],
        na_position="last",
    )

    output_path = table_dir / "tableS_anchor_nonanchor_feature_summary.csv"
    summary.to_csv(output_path, index=False)

    audit = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "anchor_feature": feature,
                "anchor_feature_label": display_name(feature),
                "canonicalisation_role": role,
                "present_in_processed_profile": feature in available,
                "anchor_is_independent_validation": False,
            }
            for feature, role in resolved.items()
        ]
    )
    audit_path = table_dir / "tableS_canonical_anchor_audit.csv"
    audit.to_csv(audit_path, index=False)
    return summary, output_path, audit_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export canonical phenotype anchor variables separately from the strongest "
            "non-anchor continuous and binary phenotype features. Existing clustering "
            "assignments are not refit."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchors", default="configs/canonical_anchors.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    summary, output_path, audit_path = build_summary(
        args.config,
        anchors_path=args.anchors,
        run_dir=args.run_dir,
        top_n=args.top_n,
    )
    print(f"Rows: {len(summary)}")
    print(f"Summary: {output_path}")
    print(f"Anchor audit: {audit_path}")
    print(
        summary.groupby(["cluster", "evidence_role", "domain"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
