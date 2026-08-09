from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _latest_run(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:8].isdigit())
    if not runs:
        raise FileNotFoundError(f"No timestamped run directories found under {root}")
    return runs[-1]


def _display_value(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    return str(value)


def _categorical_summary(series: pd.Series, cohort: str, variable: str) -> list[dict]:
    counts = series.fillna("Missing").value_counts(dropna=False)
    n = len(series)
    rows: list[dict] = []
    for level, count in counts.items():
        rows.append(
            {
                "cohort": cohort,
                "variable": variable,
                "level": _display_value(level),
                "n": int(count),
                "percent": round(100.0 * count / n, 1) if n else np.nan,
                "summary": f"{int(count):,} ({100.0 * count / n:.1f}%)" if n else "",
            }
        )
    return rows


def _continuous_summary(series: pd.Series, cohort: str, variable: str) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return {
            "cohort": cohort,
            "variable": variable,
            "level": "Median (IQR)",
            "n": 0,
            "percent": np.nan,
            "summary": "NA",
        }
    q1 = valid.quantile(0.25)
    med = valid.median()
    q3 = valid.quantile(0.75)
    return {
        "cohort": cohort,
        "variable": variable,
        "level": "Median (IQR)",
        "n": int(valid.size),
        "percent": np.nan,
        "summary": f"{med:.1f} ({q1:.1f}-{q3:.1f})",
    }


def make_demographic_table(stroke_cfg: dict, sepsis_cfg: dict, output_dir: Path) -> Path:
    rows: list[dict] = []
    specifications = [
        ("Stroke", stroke_cfg, "AGE_AT_STROKE"),
        ("Sepsis", sepsis_cfg, "AGE_AT_SEPSIS"),
    ]
    categorical_candidates = ["SEX", "RACE", "HISPANIC", "PAT_PREF_LANGUAGE_SPOKEN", "RUCA_CODE"]
    continuous_candidates = ["Pct_Urban", "ADI_NAT_RANK22", "SVI_OVERAL_RANK22", "enc_duration"]

    for cohort, cfg, age_col in specifications:
        frame = pd.read_parquet(cfg["input_path"])
        rows.append(
            {
                "cohort": cohort,
                "variable": "Analytic cohort",
                "level": "N",
                "n": int(len(frame)),
                "percent": 100.0,
                "summary": f"{len(frame):,}",
            }
        )
        if age_col in frame.columns:
            rows.append(_continuous_summary(frame[age_col], cohort, "Age"))
        for column in categorical_candidates:
            if column in frame.columns:
                rows.extend(_categorical_summary(frame[column], cohort, column))
        for column in continuous_candidates:
            if column in frame.columns:
                rows.append(_continuous_summary(frame[column], cohort, column))

    table = pd.DataFrame(rows)
    output = output_dir / "table1_cohort_characteristics_long.csv"
    table.to_csv(output, index=False)
    return output


def _candidate_table(run_dir: Path) -> pd.DataFrame:
    table_dir = run_dir / "manuscript_outputs" / "tables"
    candidates = sorted(table_dir.glob("*candidate*model*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find candidate-model summary under {table_dir}")
    table = pd.read_csv(candidates[0])
    return table


def _column(table: pd.DataFrame, choices: list[str]) -> str:
    for choice in choices:
        if choice in table.columns:
            return choice
    raise KeyError(f"None of {choices} found in columns: {list(table.columns)}")


def make_model_selection_figure(
    stroke_run: Path,
    sepsis_run: Path,
    output_dir: Path,
) -> list[Path]:
    datasets = [("Stroke", _candidate_table(stroke_run)), ("Sepsis", _candidate_table(sepsis_run))]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)

    for ax, (title, table) in zip(axes, datasets):
        sil_col = _column(table, ["silhouette", "silhouette_mean"])
        stability_col = _column(table, ["subsample_ari_mean", "mean_subsample_ari", "subsample_ari"])
        reduction_col = _column(table, ["reduction"])
        clusterer_col = _column(table, ["clusterer"])
        k_col = _column(table, ["k"])

        plot = table.copy()
        plot[sil_col] = pd.to_numeric(plot[sil_col], errors="coerce")
        plot[stability_col] = pd.to_numeric(plot[stability_col], errors="coerce")
        plot = plot.dropna(subset=[sil_col, stability_col])
        ax.scatter(plot[sil_col], plot[stability_col], s=30, alpha=0.45)

        selected = plot[
            (plot[reduction_col].astype(str).str.lower() == "pca")
            & (plot[clusterer_col].astype(str).str.lower() == "kmeans")
            & (pd.to_numeric(plot[k_col], errors="coerce") == 3)
        ]
        if not selected.empty:
            row = selected.sort_values(stability_col, ascending=False).iloc[0]
            ax.scatter([row[sil_col]], [row[stability_col]], s=125, marker="*", edgecolors="black", linewidths=0.8)
            ax.annotate(
                "Selected\nPCA + k-means, k=3",
                (row[sil_col], row[stability_col]),
                xytext=(8, -5),
                textcoords="offset points",
                fontsize=9,
                va="top",
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Silhouette score")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Repeated-subsample ARI")
    fig.suptitle("Model selection balanced separation and reproducibility", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Each point is a candidate model summary. The selected solution was not chosen by visual separation in a two-dimensional embedding.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=1.4)
    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure2_model_selection.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _profile_table(run_dir: Path, kind: str) -> pd.DataFrame:
    table_dir = run_dir / "manuscript_outputs" / "tables"
    if kind == "continuous":
        path = table_dir / "table_continuous_cluster_profiles_processed_scale.csv"
    else:
        path = table_dir / "table_binary_cluster_profiles.csv"
    return pd.read_csv(path)


def _heatmap_matrix(profile: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, str]:
    value_col = "median_difference_iqr" if "median_difference_iqr" in profile.columns else "standardised_difference"
    label_col = "feature_label" if "feature_label" in profile.columns else "feature"
    profile = profile.copy()
    profile["label"] = profile[label_col].astype(str).str.replace("_", " ", regex=False)
    profile["abs"] = pd.to_numeric(profile[value_col], errors="coerce").abs()
    top = profile.groupby("feature")["abs"].max().nlargest(top_n).index
    matrix = profile[profile["feature"].isin(top)].pivot_table(index="label", columns="cluster", values=value_col, aggfunc="first")
    matrix = matrix.reindex(columns=[0, 1, 2])
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values().index]
    return matrix, value_col


def make_compact_heatmap_figure(stroke_run: Path, sepsis_run: Path, output_dir: Path) -> list[Path]:
    specs = [
        ("Stroke: continuous features", _profile_table(stroke_run, "continuous"), 12),
        ("Stroke: binary features", _profile_table(stroke_run, "binary"), 12),
        ("Sepsis: continuous features", _profile_table(sepsis_run, "continuous"), 16),
        ("Sepsis: binary features", _profile_table(sepsis_run, "binary"), 10),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.0))
    for ax, (title, profile, top_n) in zip(axes.ravel(), specs):
        matrix, value_col = _heatmap_matrix(profile, top_n)
        limit = max(0.5, float(np.nanmax(np.abs(matrix.to_numpy()))))
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Cluster 0", "Cluster 1", "Cluster 2"], fontsize=8.5)
        ax.tick_params(length=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Median difference / IQR" if value_col == "median_difference_iqr" else "Standardized prevalence difference", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.07, wspace=0.30, hspace=0.34)
    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure3_phenotype_heatmaps_compact.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def make_compact_supplement_tables(stroke_run: Path, sepsis_run: Path, output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for cohort, run in [("stroke", stroke_run), ("sepsis", sepsis_run)]:
        table_dir = run / "manuscript_outputs" / "tables"

        miss = table_dir / "tableS_missingness_and_repository_imputation.csv"
        if miss.exists():
            df = pd.read_csv(miss)
            numeric_event_cols = [c for c in df.columns if any(token in c.lower() for token in ["missing", "invalid", "replaced", "negative"])]
            keep_mask = pd.Series(False, index=df.index)
            for col in numeric_event_cols:
                values = pd.to_numeric(df[col], errors="coerce").fillna(0)
                keep_mask |= values > 0
            concise = df.loc[keep_mask].copy()
            out = output_dir / f"supplement_missingness_events_{cohort}.csv"
            concise.to_csv(out, index=False, float_format="%.3g")
            outputs.append(out)

        anchor = table_dir / "tableS_anchor_nonanchor_feature_summary.csv"
        if anchor.exists():
            df = pd.read_csv(anchor)
            if "evidence_role" in df.columns:
                df = df[df["evidence_role"].astype(str).str.contains("non_anchor", na=False)].copy()
            rank_col = next((c for c in df.columns if "rank" in c.lower()), None)
            if rank_col:
                ranks = pd.to_numeric(df[rank_col], errors="coerce")
                df = df[ranks <= 5]
            drop_cols = [c for c in df.columns if c in {"interpretation_note", "anchor_role", "absolute_effect"}]
            df = df.drop(columns=drop_cols, errors="ignore")
            out = output_dir / f"supplement_nonanchor_top_features_{cohort}.csv"
            df.to_csv(out, index=False, float_format="%.3f")
            outputs.append(out)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript-revision tables and figures without rerunning clustering.")
    parser.add_argument("--stroke-config", default="configs/stroke.yaml")
    parser.add_argument("--sepsis-config", default="configs/sepsis.yaml")
    parser.add_argument("--stroke-run")
    parser.add_argument("--sepsis-run")
    parser.add_argument("--output-dir", default="results/author_revision_materials")
    args = parser.parse_args()

    stroke_cfg = _load_yaml(Path(args.stroke_config))
    sepsis_cfg = _load_yaml(Path(args.sepsis_config))
    stroke_run = Path(args.stroke_run) if args.stroke_run else _latest_run(stroke_cfg["output_dir"])
    sepsis_run = Path(args.sepsis_run) if args.sepsis_run else _latest_run(sepsis_cfg["output_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    created.append(make_demographic_table(stroke_cfg, sepsis_cfg, output_dir))
    created.extend(make_model_selection_figure(stroke_run, sepsis_run, output_dir))
    created.extend(make_compact_heatmap_figure(stroke_run, sepsis_run, output_dir))
    created.extend(make_compact_supplement_tables(stroke_run, sepsis_run, output_dir))

    manifest = {
        "stroke_run": str(stroke_run),
        "sepsis_run": str(sepsis_run),
        "created": [str(p) for p in created],
        "note": "These products summarize existing analysis outputs; clustering models are not refit.",
    }
    manifest_path = output_dir / "author_revision_materials_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(created)} products to {output_dir}")
    print(manifest_path)


if __name__ == "__main__":
    main()
