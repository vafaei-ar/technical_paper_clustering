from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, kruskal
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from tpcluster.core import prepare_matrix

MODELS = {
    "stroke": {"primary": "stroke_pca_kmeans_k3", "sensitivity": "stroke_pca_kmeans_k2"},
    "sepsis": {"primary": "sepsis_pca_kmeans_k3", "sensitivity": "sepsis_pca_kmeans_k2", "representation": "sepsis_raw_kmeans_k3"},
}
LONG_LABELS = {
    "stroke": {0: "Preserved-haematologic, lower-comorbidity", 1: "Renal-anaemic multimorbidity", 2: "Hyperglycaemic diabetes"},
    "sepsis": {0: "Neutrophil-predominant, lower-acuity", 1: "Immature-granulocyte-high organ dysfunction", 2: "Eosinophil-lymphocyte-enriched"},
}
SHORT_LABELS = {
    "stroke": {0: "C0 Preserved haematologic", 1: "C1 Renal-anaemic", 2: "C2 Hyperglycaemic"},
    "sepsis": {0: "C0 Neutrophil-predominant", 1: "C1 IG-high organ dysfunction", 2: "C2 Eosinophil-lymphocyte"},
}
DISCHARGE_GROUPS = {"HO": "Home/self-care", "RH": "Inpatient rehabilitation", "SN": "Skilled nursing facility", "EX": "Expired", "HS": "Hospice", "HH": "Home health"}


def latest_full_run(output_dir: str | Path) -> Path:
    runs = [p for p in Path(output_dir).iterdir() if p.is_dir() and (p / "candidate_profiles" / "candidate_assignments.parquet").exists() and (p / "subsample_stability_summary.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"No completed run found under {output_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def bh(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return pd.Series(out, index=values.index)
    order = valid[np.argsort(p[valid])]
    corrected = p[order] * len(order) / np.arange(1, len(order) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    out[order] = np.minimum(corrected, 1)
    return pd.Series(out, index=values.index)


def magnitude(kind: str, value: float) -> str:
    value = abs(float(value))
    if kind == "epsilon_squared":
        return "negligible" if value < 0.01 else "small" if value < 0.06 else "moderate" if value < 0.14 else "large"
    return "negligible" if value < 0.05 else "small" if value < 0.15 else "moderate" if value < 0.30 else "large"


def raw_profiles(frame: pd.DataFrame, cluster_col: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        for cluster, index in frame.groupby(cluster_col).groups.items():
            series = values.loc[index].dropna()
            rows.append({"feature": feature, "cluster": int(cluster), "n_nonmissing": int(series.count()), "median": float(series.median()), "q1": float(series.quantile(.25)), "q3": float(series.quantile(.75)), "mean": float(series.mean()), "sd": float(series.std()), "scale": "original_imputed_clinical_scale"})
    return pd.DataFrame(rows)


def continuous_tests(frame: pd.DataFrame, cluster_col: str, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, tests = [], []
    for variable in variables:
        if variable not in frame:
            continue
        values = pd.to_numeric(frame[variable], errors="coerce")
        groups = []
        for cluster, index in frame.groupby(cluster_col).groups.items():
            series = values.loc[index].dropna()
            summaries.append({"variable": variable, "cluster": int(cluster), "n": int(series.count()), "mean": float(series.mean()), "sd": float(series.std()), "median": float(series.median()), "q1": float(series.quantile(.25)), "q3": float(series.quantile(.75))})
            groups.append(series.to_numpy())
        statistic, p_value = kruskal(*groups)
        effect = max(0, (statistic - len(groups) + 1) / (values.notna().sum() - len(groups)))
        tests.append({"variable": variable, "test": "Kruskal-Wallis", "statistic": statistic, "p_value": p_value, "effect_size": effect, "effect_size_name": "epsilon_squared", "effect_magnitude": magnitude("epsilon_squared", effect)})
    test_table = pd.DataFrame(tests)
    if not test_table.empty:
        test_table["p_fdr"] = bh(test_table["p_value"])
    return pd.DataFrame(summaries), test_table


def categorical_tests(frame: pd.DataFrame, cluster_col: str, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, tests = [], []
    for variable in variables:
        if variable not in frame:
            continue
        table = pd.crosstab(frame[cluster_col], frame[variable].astype("string").fillna("Missing"))
        for cluster in table.index:
            total = table.loc[cluster].sum()
            for level in table.columns:
                count = int(table.loc[cluster, level])
                summaries.append({"variable": variable, "cluster": int(cluster), "level": str(level), "n": count, "proportion": count / total})
        statistic, p_value, _, _ = chi2_contingency(table)
        denominator = min(table.shape[0] - 1, table.shape[1] - 1)
        effect = np.sqrt(statistic / (table.to_numpy().sum() * denominator))
        tests.append({"variable": variable, "test": "Chi-square", "statistic": statistic, "p_value": p_value, "effect_size": effect, "effect_size_name": "Cramers_V", "effect_magnitude": magnitude("Cramers_V", effect)})
    test_table = pd.DataFrame(tests)
    if not test_table.empty:
        test_table["p_fdr"] = bh(test_table["p_value"])
    return pd.DataFrame(summaries), test_table


def heatmap(profile_path: Path, output_path: Path, labels: dict[int, str], title: str) -> None:
    profile = pd.read_csv(profile_path)
    value_col = "median_difference_iqr" if "median_difference_iqr" in profile else "standardised_difference"
    profile["absolute"] = profile[value_col].abs()
    top = profile.groupby("feature")["absolute"].max().nlargest(18).index
    matrix = profile[profile["feature"].isin(top)].pivot_table(index="feature", columns="cluster", values=value_col, aggfunc="first")
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values().index]
    limit = max(.5, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(8.5, max(5.5, .32 * len(matrix))))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(matrix))); ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xticks(np.arange(len(matrix.columns))); ax.set_xticklabels([labels[int(c)] for c in matrix.columns], rotation=20, ha="right", fontsize=9)
    ax.set_title(title, pad=12); ax.set_xlabel("Cluster"); ax.set_ylabel("Feature")
    fig.colorbar(image, ax=ax, label=value_col.replace("_", " "))
    fig.tight_layout(); fig.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(fig)


def pca_plot(matrix: np.ndarray, labels: np.ndarray, short_labels: dict[int, str], output_path: Path, title: str) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=11)
    coordinates = pca.fit_transform(matrix)
    display = pd.DataFrame({"PC1": coordinates[:, 0], "PC2": coordinates[:, 1], "cluster": labels})
    if len(display) > 8000:
        display = display.sample(8000, random_state=11)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for cluster in sorted(display.cluster.unique()):
        subset = display[display.cluster == cluster]
        ax.scatter(subset.PC1, subset.PC2, s=8, alpha=.30, label=short_labels[int(cluster)])
    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title(title); ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_fraction": pca.explained_variance_ratio_, "interpretation": ["Two-dimensional visual projection; clustering used the full retained PCA representation."] * 2})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-config", default="configs/profile_columns.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    with open(args.profile_config, encoding="utf-8") as file:
        profile_cfg = yaml.safe_load(file)[cfg["dataset_name"]]

    cohort = cfg["dataset_name"]; model = MODELS[cohort]; long_labels = LONG_LABELS[cohort]; short_labels = SHORT_LABELS[cohort]
    run_dir = latest_full_run(cfg["output_dir"]); profile_dir = run_dir / "candidate_profiles"; output_dir = run_dir / "manuscript_outputs"; table_dir = output_dir / "tables"; figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(cfg["input_path"])
    assignments = pd.read_parquet(profile_dir / "candidate_assignments.parquet")
    merged = frame.merge(assignments, on=cfg["id_column"], validate="one_to_one")
    primary = model["primary"]; merged["primary_cluster"] = merged[primary].astype(int); merged["primary_cluster_label"] = merged.primary_cluster.map(long_labels); merged["primary_cluster_short_label"] = merged.primary_cluster.map(short_labels)
    if cohort == "stroke" and "DISCHARGE_STATUS" in merged:
        merged["DISCHARGE_GROUP"] = merged.DISCHARGE_STATUS.astype("string").map(DISCHARGE_GROUPS).fillna("Other institutional/other")

    columns = [cfg["id_column"], primary, model["sensitivity"], "primary_cluster", "primary_cluster_label", "primary_cluster_short_label"]
    if model.get("representation"): columns.append(model["representation"])
    merged[columns].to_parquet(output_dir / "final_cluster_assignments.parquet", index=False)
    sizes = merged.groupby(["primary_cluster", "primary_cluster_label", "primary_cluster_short_label"], as_index=False).size().rename(columns={"size": "n"}); sizes["fraction"] = sizes.n / len(merged); sizes.to_csv(table_dir / "table_cluster_sizes.csv", index=False)

    primary_dir = profile_dir / primary
    processed = pd.read_csv(primary_dir / "continuous_feature_profiles.csv"); binary = pd.read_csv(primary_dir / "binary_feature_profiles.csv")
    processed.to_csv(table_dir / "table_continuous_cluster_profiles_processed_scale.csv", index=False); binary.to_csv(table_dir / "table_binary_cluster_profiles.csv", index=False)
    raw = raw_profiles(merged, "primary_cluster", cfg["features"]["primary"]); raw["cluster_label"] = raw.cluster.map(long_labels); raw.to_csv(table_dir / "table_continuous_cluster_profiles_raw_scale.csv", index=False)
    top = processed.assign(abs_effect=processed.median_difference_iqr.abs()).sort_values(["cluster", "abs_effect"], ascending=[True, False]).groupby("cluster", group_keys=False).head(10)
    raw.merge(top[["cluster", "feature", "median_difference_iqr"]], on=["cluster", "feature"]).to_csv(table_dir / "table_top_continuous_features_raw_clinical_scale.csv", index=False)

    continuous_vars = profile_cfg.get("continuous_descriptive", []) + profile_cfg.get("continuous_outcomes", [])
    categorical_vars = profile_cfg.get("categorical_descriptive", []) + profile_cfg.get("categorical_outcomes", []) + profile_cfg.get("binary_outcomes", [])
    if cohort == "stroke" and "DISCHARGE_GROUP" in merged:
        categorical_vars = [v for v in categorical_vars if v != "DISCHARGE_STATUS"] + ["DISCHARGE_GROUP"]
    cont_summary, cont_tests = continuous_tests(merged, "primary_cluster", continuous_vars); cat_summary, cat_tests = categorical_tests(merged, "primary_cluster", categorical_vars)
    cont_summary.to_csv(table_dir / "table_continuous_outcomes_demographics.csv", index=False); cat_summary.to_csv(table_dir / "table_categorical_outcomes_demographics.csv", index=False)
    cont_tests.to_csv(table_dir / "table_continuous_statistical_tests.csv", index=False); cat_tests.to_csv(table_dir / "table_categorical_statistical_tests.csv", index=False)
    pd.concat([cont_tests.assign(domain="continuous"), cat_tests.assign(domain="categorical")], ignore_index=True).sort_values(["effect_size", "p_fdr"], ascending=[False, True]).to_csv(table_dir / "table_effect_size_focused_results.csv", index=False)

    comparisons = [{"comparison": "primary_vs_k2_sensitivity", "candidate_a": primary, "candidate_b": model["sensitivity"], "adjusted_rand_index": adjusted_rand_score(merged[primary], merged[model["sensitivity"]])}]
    if model.get("representation"):
        comparisons.append({"comparison": "pca_vs_raw_representation", "candidate_a": primary, "candidate_b": model["representation"], "adjusted_rand_index": adjusted_rand_score(merged[primary], merged[model["representation"]])})
    pd.DataFrame(comparisons).to_csv(table_dir / "table_sensitivity_agreement.csv", index=False)
    pd.read_csv(run_dir / "subsample_stability_summary.csv").to_csv(table_dir / "table_subsample_stability.csv", index=False)

    _, _, scaled, *_ = prepare_matrix(frame, cfg["features"]["primary"], cfg.get("preprocessing", {}))
    pca_plot(scaled, merged.primary_cluster.to_numpy(), short_labels, figure_dir / "figure_pca_clusters.png", f"{cohort.capitalize()} clusters: two-dimensional PCA projection").to_csv(table_dir / "pca_explained_variance.csv", index=False)
    heatmap(primary_dir / "continuous_feature_profiles.csv", figure_dir / "figure_continuous_heatmap.png", short_labels, f"{cohort.capitalize()} continuous-feature effects")
    heatmap(primary_dir / "binary_feature_profiles.csv", figure_dir / "figure_binary_heatmap.png", short_labels, f"{cohort.capitalize()} binary-feature effects")

    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    with open(output_dir / "analysis_manifest.json", "w", encoding="utf-8") as file:
        json.dump({"dataset": cohort, "primary_model": primary, "clinical_tables_scale": "original imputed values", "heatmap_scale": "processed clustering-space effect sizes", "pca_figure_note": "Two-dimensional visual projection; clustering was not limited to PC1 and PC2.", "cluster_labels": long_labels, "git_commit": commit, "config": args.config}, file, indent=2)
    print(f"Completed corrected manuscript outputs for {cohort}: {output_dir}")


if __name__ == "__main__":
    main()
