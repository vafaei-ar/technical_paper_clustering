from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from generate_author_revision_materials import _heatmap_matrix, _profile_table
from methods_revision_analyses import candidate_audit


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_reporting_run(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    runs = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and (p / "internal_metrics.csv").exists()
        and (p / "candidate_profiles" / "candidate_assignments.parquet").exists()
        and (p / "manuscript_outputs" / "tables" / "table_cluster_sizes.csv").exists()
        and (
            (p / "subsample_stability_full_refit_summary.csv").exists()
            or (p / "subsample_stability_summary.csv").exists()
        )
    ]
    if not runs:
        raise FileNotFoundError(f"No reporting-ready run found under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def full_refit_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "subsample_stability_full_refit_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def candidate_tradeoff_table(run_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "internal_metrics.csv")
    internal = (
        metrics.groupby(["dataset", "reduction", "clusterer", "k"], as_index=False)
        .agg(
            silhouette=("silhouette", "mean"),
            calinski_harabasz=("calinski_harabasz", "mean"),
            davies_bouldin=("davies_bouldin", "mean"),
            minimum_cluster_fraction=("minimum_cluster_fraction", "min"),
        )
    )
    stability = full_refit_summary(run_dir)
    keep = [
        "dataset",
        "reduction",
        "clusterer",
        "k",
        "mean_subsample_ari",
        "minimum_subsample_ari",
        "p10_subsample_ari",
    ]
    return internal.merge(
        stability[keep],
        on=["dataset", "reduction", "clusterer", "k"],
        how="inner",
        validate="one_to_one",
    )


def model_label(row: pd.Series) -> str:
    reduction = "PCA" if str(row["reduction"]).lower() == "pca" else "raw"
    clusterer = str(row["clusterer"]).replace("agglomerative", "agg.")
    return f"{reduction} {clusterer} k={int(row['k'])}"


def make_figure2(
    stroke_run: Path,
    sepsis_run: Path,
    methods_dir: Path,
    output_dir: Path,
) -> list[Path]:
    tables = {
        "Stroke": candidate_tradeoff_table(stroke_run),
        "Sepsis": candidate_tradeoff_table(sepsis_run),
    }
    null_tables = {
        "Stroke": pd.read_csv(methods_dir / "null_reference_silhouette_extended_stroke.csv"),
        "Sepsis": pd.read_csv(methods_dir / "null_reference_silhouette_extended_sepsis.csv"),
    }
    null_summary = pd.read_csv(
        methods_dir / "null_reference_silhouette_extended_summary.csv"
    ).set_index("dataset")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    for col, cohort in enumerate(["Stroke", "Sepsis"]):
        ax = axes[0, col]
        table = tables[cohort].copy()
        ax.scatter(
            table["silhouette"],
            table["mean_subsample_ari"],
            s=55,
            alpha=0.75,
        )
        for _, row in table.iterrows():
            selected = (
                str(row["reduction"]).lower() == "pca"
                and str(row["clusterer"]).lower() == "kmeans"
                and int(row["k"]) == 3
            )
            if selected:
                ax.scatter(
                    [row["silhouette"]],
                    [row["mean_subsample_ari"]],
                    s=180,
                    marker="*",
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.8,
                )
            ax.annotate(
                model_label(row),
                (row["silhouette"], row["mean_subsample_ari"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
        ax.set_xlabel("Silhouette score")
        ax.set_ylabel("Mean full-refit subsample ARI")
        ax.set_title(f"{chr(65 + col)}  {cohort}: separation vs stability", loc="left", fontweight="bold")
        ax.grid(alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax2 = axes[1, col]
        null = null_tables[cohort]["silhouette"].dropna()
        real = float(null_summary.loc[cohort.lower(), "real_silhouette"])
        ax2.hist(null, bins=18, alpha=0.65, edgecolor="white")
        ax2.axvline(real, linewidth=2.2, linestyle="--", label=f"Real = {real:.3f}")
        ax2.set_xlabel("Silhouette score")
        ax2.set_ylabel("Null replicates")
        ax2.set_title(
            f"{chr(67 + col)}  {cohort}: Gaussian-copula null reference",
            loc="left",
            fontweight="bold",
        )
        ax2.legend(frameon=False, fontsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.2, w_pad=1.8, h_pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure2_model_selection_and_null.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def dynamic_cluster_labels(run_dir: Path) -> list[str]:
    sizes = pd.read_csv(
        run_dir / "manuscript_outputs" / "tables" / "table_cluster_sizes.csv"
    ).sort_values("primary_cluster")
    labels = []
    for _, row in sizes.iterrows():
        short = str(row.get("primary_cluster_short_label", f"C{int(row['primary_cluster'])}"))
        short = short.replace("C0 ", "").replace("C1 ", "").replace("C2 ", "")
        labels.append(f"{short}\n(n={int(row['n']):,})")
    return labels


def draw_heatmap_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    panel: str,
    cluster_labels: list[str],
    limit: float,
):
    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_title(f"{panel}  {title}", loc="left", fontsize=10.5, fontweight="bold", pad=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8.3)
    ax.set_xticks(range(len(cluster_labels)))
    ax.set_xticklabels(cluster_labels, fontsize=7.9)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def make_figure3(stroke_run: Path, sepsis_run: Path, output_dir: Path) -> list[Path]:
    stroke_labels = dynamic_cluster_labels(stroke_run)
    sepsis_labels = dynamic_cluster_labels(sepsis_run)
    specs = [
        ("A", "Stroke: continuous features", _profile_table(stroke_run, "continuous"), 12, "continuous", stroke_labels),
        ("B", "Stroke: binary features", _profile_table(stroke_run, "binary"), 12, "binary", stroke_labels),
        ("C", "Sepsis: continuous features", _profile_table(sepsis_run, "continuous"), 16, "continuous", sepsis_labels),
        ("D", "Sepsis: binary features", _profile_table(sepsis_run, "binary"), 10, "binary", sepsis_labels),
    ]
    prepared = []
    domain_limit = {"continuous": 0.5, "binary": 0.5}
    for panel, title, profile, top_n, domain, labels in specs:
        matrix, _ = _heatmap_matrix(profile, top_n)
        domain_limit[domain] = max(
            domain_limit[domain], float(np.nanmax(np.abs(matrix.to_numpy())))
        )
        prepared.append((panel, title, matrix, domain, labels))

    fig = plt.figure(figsize=(12.8, 10.0))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.0, 1.0, 0.055],
        width_ratios=[1.0, 1.0],
        hspace=0.38, wspace=0.78,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax_cont = fig.add_subplot(gs[2, 0])
    cax_binary = fig.add_subplot(gs[2, 1])
    images = {}
    for ax, item in zip(axes, prepared):
        panel, title, matrix, domain, labels = item
        images[domain] = draw_heatmap_panel(
            ax, matrix, title, panel, labels, domain_limit[domain]
        )
    cb_cont = fig.colorbar(images["continuous"], cax=cax_cont, orientation="horizontal")
    cb_cont.set_label("Median difference divided by cohort IQR", fontsize=8.5)
    cb_binary = fig.colorbar(images["binary"], cax=cax_binary, orientation="horizontal")
    cb_binary.set_label("Standardized prevalence difference", fontsize=8.5)
    cb_cont.ax.tick_params(labelsize=8)
    cb_binary.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.975, bottom=0.08)

    outputs = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure3_phenotype_profiles_revised.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def make_figure4(methods_dir: Path, output_dir: Path) -> list[Path]:
    balanced = pd.read_csv(methods_dir / "balanced_block_summary.csv")
    representation_stroke = pd.read_csv(methods_dir / "representation_k_sensitivity_stroke.csv")
    representation_sepsis = pd.read_csv(methods_dir / "representation_k_sensitivity_sepsis.csv")
    redundancy = pd.read_csv(methods_dir / "sepsis_wbc_redundancy_sensitivity.csv")

    def raw_ari(table: pd.DataFrame) -> float:
        row = table[
            table["reduction"].astype(str).str.lower().eq("none")
            & table["clusterer"].astype(str).str.lower().eq("kmeans")
            & (pd.to_numeric(table["k"], errors="coerce") == 3)
        ].iloc[0]
        return float(row["ari_vs_pca_k3"])

    stroke_bal = float(
        balanced.loc[balanced["dataset"].eq("stroke"), "ari_vs_reference"].iloc[0]
    )
    sepsis_bal = float(
        balanced.loc[balanced["dataset"].eq("sepsis"), "ari_vs_reference"].iloc[0]
    )
    rows = pd.DataFrame(
        [
            {"cohort": "Stroke", "sensitivity": "Raw vs PCA", "ari": raw_ari(representation_stroke)},
            {"cohort": "Stroke", "sensitivity": "Balanced blocks", "ari": stroke_bal},
            {"cohort": "Sepsis", "sensitivity": "Raw vs PCA", "ari": raw_ari(representation_sepsis)},
            {"cohort": "Sepsis", "sensitivity": "Balanced blocks", "ari": sepsis_bal},
            {
                "cohort": "Sepsis",
                "sensitivity": "Drop WBC percentages",
                "ari": float(
                    redundancy.loc[
                        redundancy["variant"].eq("drop_wbc_percentages_keep_counts"),
                        "ari_vs_reference",
                    ].iloc[0]
                ),
            },
            {
                "cohort": "Sepsis",
                "sensitivity": "Drop WBC counts",
                "ari": float(
                    redundancy.loc[
                        redundancy["variant"].eq("drop_wbc_counts_keep_percentages"),
                        "ari_vs_reference",
                    ].iloc[0]
                ),
            },
        ]
    )
    rows.to_csv(output_dir / "figure4_sensitivity_values.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True)
    for ax, cohort, panel in zip(axes, ["Stroke", "Sepsis"], ["A", "B"]):
        sub = rows[rows["cohort"].eq(cohort)].copy()
        y = np.arange(len(sub))
        ax.barh(y, sub["ari"])
        ax.set_yticks(y)
        ax.set_yticklabels(sub["sensitivity"], fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.axvline(0.8, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("ARI versus primary PCA k=3 solution")
        ax.set_title(f"{panel}  {cohort}", loc="left", fontweight="bold")
        for yi, value in zip(y, sub["ari"]):
            ax.text(min(value + 0.02, 0.94), yi, f"{value:.3f}", va="center", fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.tight_layout(pad=1.2, w_pad=2.2)

    outputs = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure4_specification_sensitivity.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final reviewer-driven reporting outputs without refitting clustering models."
    )
    parser.add_argument("--stroke-config", default="configs/stroke_methods_revision.yaml")
    parser.add_argument("--sepsis-config", default="configs/sepsis.yaml")
    parser.add_argument("--methods-dir", default="results/methods_revision")
    parser.add_argument("--output-dir", default="results/methods_revision_reporting")
    args = parser.parse_args()

    stroke_cfg = load_yaml(args.stroke_config)
    sepsis_cfg = load_yaml(args.sepsis_config)
    stroke_run = latest_reporting_run(stroke_cfg["output_dir"])
    sepsis_run = latest_reporting_run(sepsis_cfg["output_dir"])
    methods_dir = Path(args.methods_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for cohort, run in [("stroke", stroke_run), ("sepsis", sepsis_run)]:
        audit = candidate_audit(run, cohort)
        path = output_dir / f"tableS_candidate_model_audit_{cohort}.csv"
        audit.to_csv(path, index=False)
        created.append(path)

        src = run / "manuscript_outputs" / "tables" / "table_cluster_sizes.csv"
        dst = output_dir / f"table_cluster_sizes_{cohort}.csv"
        pd.read_csv(src).to_csv(dst, index=False)
        created.append(dst)

    for source_name in [
        "balanced_block_summary.csv",
        "sepsis_wbc_redundancy_sensitivity.csv",
        "null_reference_summary.csv",
        "null_reference_silhouette_extended_summary.csv",
        "representation_k_sensitivity_stroke.csv",
        "representation_k_sensitivity_sepsis.csv",
    ]:
        source = methods_dir / source_name
        if source.exists():
            destination = output_dir / source_name
            pd.read_csv(source).to_csv(destination, index=False)
            created.append(destination)

    created.extend(make_figure2(stroke_run, sepsis_run, methods_dir, output_dir))
    created.extend(make_figure3(stroke_run, sepsis_run, output_dir))
    created.extend(make_figure4(methods_dir, output_dir))

    manifest = {
        "stroke_run": str(stroke_run),
        "sepsis_run": str(sepsis_run),
        "methods_dir": str(methods_dir),
        "created": [str(p) for p in created],
        "note": (
            "Reporting-only generation after reviewer-driven analyses. "
            "No clustering grid or robustness analysis is refit by this script."
        ),
    }
    manifest_path = output_dir / "reporting_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(created)} outputs to {output_dir}")
    print(manifest_path)


if __name__ == "__main__":
    main()
