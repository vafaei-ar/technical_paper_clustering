from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec

from generate_author_revision_materials import (
    _latest_run,
    _load_yaml,
    _profile_table,
    _heatmap_matrix,
    make_model_selection_figure,
    make_compact_supplement_tables,
)

SEX_MAP = {0.0: "Female", 1.0: "Male"}
HISPANIC_MAP = {0.0: "Hispanic/Latino", 1.0: "Not Hispanic/Latino", 2.0: "Unknown/other"}
RACE_MAP = {
    0.0: "White",
    1.0: "Black/African American",
    2.0: "Asian",
    3.0: "American Indian/Alaska Native",
    4.0: "Native Hawaiian/Other Pacific Islander",
    5.0: "Multiple race",
    6.0: "Other",
}


def _fmt_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def _continuous_row(series: pd.Series, label: str) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {"characteristic": label, "summary": "NA"}
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    return {"characteristic": label, "summary": f"{_fmt_num(med)} ({_fmt_num(q1)}-{_fmt_num(q3)})"}


def _categorical_rows(series: pd.Series, label: str, mapping: dict[float, str]) -> list[dict]:
    x = pd.to_numeric(series, errors="coerce")
    n = len(series)
    rows = []
    for value, count in x.value_counts(dropna=False).items():
        if pd.isna(value):
            level = "Missing"
        else:
            level = mapping.get(float(value), f"Code {value:g}")
        rows.append(
            {
                "characteristic": f"{label}: {level}",
                "summary": f"{int(count):,} ({100.0 * count / n:.1f}%)",
            }
        )
    return rows


def make_manuscript_demographic_table(stroke_cfg: dict, sepsis_cfg: dict, output_dir: Path) -> Path:
    cohorts = [
        ("Stroke", stroke_cfg, "AGE_AT_STROKE"),
        ("Sepsis", sepsis_cfg, "AGE_AT_SEPSIS"),
    ]
    by_cohort: dict[str, list[dict]] = {}
    ordered_characteristics: list[str] = []

    for cohort, cfg, age_col in cohorts:
        df = pd.read_parquet(cfg["input_path"])
        rows = [{"characteristic": "Analytic cohort, n", "summary": f"{len(df):,}"}]
        if age_col in df.columns:
            rows.append(_continuous_row(df[age_col], "Age, median (IQR), years"))
        if "SEX" in df.columns:
            rows.extend(_categorical_rows(df["SEX"], "Sex", SEX_MAP))
        if "RACE" in df.columns:
            rows.extend(_categorical_rows(df["RACE"], "Race", RACE_MAP))
        if "HISPANIC" in df.columns:
            rows.extend(_categorical_rows(df["HISPANIC"], "Ethnicity", HISPANIC_MAP))
        if "enc_duration" in df.columns:
            rows.append(_continuous_row(df["enc_duration"], "Encounter duration, median (IQR), days"))

        by_cohort[cohort] = rows
        for row in rows:
            if row["characteristic"] not in ordered_characteristics:
                ordered_characteristics.append(row["characteristic"])

    lookup = {
        cohort: {row["characteristic"]: row["summary"] for row in rows}
        for cohort, rows in by_cohort.items()
    }
    out = pd.DataFrame(
        [
            {
                "Characteristic": characteristic,
                "Stroke": lookup.get("Stroke", {}).get(characteristic, ""),
                "Sepsis": lookup.get("Sepsis", {}).get(characteristic, ""),
            }
            for characteristic in ordered_characteristics
        ]
    )
    path = output_dir / "table1_cohort_characteristics_manuscript_ready.csv"
    out.to_csv(path, index=False)
    return path


def make_final_heatmap_figure(stroke_run: Path, sepsis_run: Path, output_dir: Path) -> list[Path]:
    specs = [
        ("Stroke: continuous features", _profile_table(stroke_run, "continuous"), 12, "continuous"),
        ("Stroke: binary features", _profile_table(stroke_run, "binary"), 12, "binary"),
        ("Sepsis: continuous features", _profile_table(sepsis_run, "continuous"), 16, "continuous"),
        ("Sepsis: binary features", _profile_table(sepsis_run, "binary"), 10, "binary"),
    ]

    prepared = []
    domain_limit = {"continuous": 0.5, "binary": 0.5}
    for title, profile, top_n, domain in specs:
        matrix, value_col = _heatmap_matrix(profile, top_n)
        local = float(np.nanmax(np.abs(matrix.to_numpy())))
        domain_limit[domain] = max(domain_limit[domain], local)
        prepared.append((title, matrix, value_col, domain))

    fig = plt.figure(figsize=(11.5, 10.0))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.055], wspace=0.48, hspace=0.34)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    caxes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]

    images = {}
    for ax, (title, matrix, value_col, domain) in zip(axes, prepared):
        limit = domain_limit[domain]
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        images[domain] = im
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Cluster 0", "Cluster 1", "Cluster 2"], fontsize=8.5)
        ax.tick_params(length=0)

    cb1 = fig.colorbar(images["continuous"], cax=caxes[0])
    cb1.ax.tick_params(labelsize=8)
    cb1.set_label("Median difference / IQR", fontsize=8)
    cb2 = fig.colorbar(images["binary"], cax=caxes[1])
    cb2.ax.tick_params(labelsize=8)
    cb2.set_label("Standardized prevalence difference", fontsize=8)

    fig.subplots_adjust(left=0.15, right=0.94, top=0.96, bottom=0.07)
    outputs = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure3_phenotype_heatmaps_final.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


HIGH_VALUE_PATHS = [
    "sepsis_extraction_scripts/01_setup_and_config.py",
    "sepsis_extraction_scripts/02_extract_sepsis_patients.py",
    "sepsis_extraction_scripts/05_create_final_cohort.py",
    "sepsis_extraction_scripts/06_data_preparation.py",
    "sepsis_extraction_scripts/config.yaml",
    "stroke_extraction_scripts/05_create_final_cohort.py",
    "stroke_extraction_scripts/06_data_preparation.py",
    "stroke_extraction_scripts/21_make_fully_compatible.py",
    "configs/psu.yaml",
    "clustering/PROMIS-Clustering/README.md",
    "clustering/PROMIS-Clustering/Result_Analysis.py",
    "clustering/PROMIS-Clustering/dimensionality_reduction.py",
]

TOPIC_PRIORITY = [
    "study_period",
    "inclusion_exclusion",
    "index_event",
    "identifier_semantics",
    "missingness_imputation",
    "duration_los",
    "ed_inpatient",
    "discharge_codes",
    "demographic_coding",
    "dimensionality_reduction",
]


def _score_evidence(row: pd.Series) -> int:
    file = str(row.get("file", ""))
    text = str(row.get("matched_text", ""))
    score = 0
    for rank, path in enumerate(HIGH_VALUE_PATHS[::-1], start=1):
        if file.endswith(path) or path in file:
            score += rank * 10
    if re.search(r"first sepsis|first stroke|groupby\(['\"]PATID|enc_duration|dt\.days|Emergency/Inpatient|race_mapping|SEX|HISPANIC|UMAP|t[-_ ]?SNE|autoencoder", text, re.I):
        score += 25
    if "README" in file:
        score += 3
    return score


def make_concise_methods_evidence(provenance_csv: Path, output_dir: Path) -> list[Path]:
    df = pd.read_csv(provenance_csv)
    df["evidence_score"] = df.apply(_score_evidence, axis=1)
    selected = []
    for topic in TOPIC_PRIORITY:
        block = df[df["topic"].astype(str) == topic].copy()
        if block.empty:
            continue
        for repository in sorted(block["repository"].dropna().unique()):
            sub = block[block["repository"] == repository].sort_values(["evidence_score", "file", "line"], ascending=[False, True, True])
            selected.append(sub.head(4))
    concise = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    keep = [c for c in ["repository", "topic", "file", "line", "matched_text", "context", "evidence_score"] if c in concise.columns]
    concise = concise[keep]
    path = output_dir / "upstream_methods_evidence_concise.csv"
    concise.to_csv(path, index=False)

    # Developmental dimensionality-reduction evidence: document methods without claiming all were successful.
    dr = df[df["topic"].astype(str) == "dimensionality_reduction"].copy()
    methods = {
        "PCA": r"\bPCA\b|\bpca\b",
        "t-SNE": r"t[-_ ]?SNE|\bTSNE\b|\btsne\b",
        "UMAP": r"\bUMAP\b|\bumap\b",
        "Autoencoder/deep learning": r"autoencoder|deep learning|\bVAE\b|denoising",
    }
    rows = []
    for method, pattern in methods.items():
        hits = dr[
            dr["matched_text"].astype(str).str.contains(pattern, case=False, regex=True, na=False)
            | dr["context"].astype(str).str.contains(pattern, case=False, regex=True, na=False)
        ].copy()
        if not hits.empty:
            hit = hits.sort_values("evidence_score", ascending=False).iloc[0]
            rows.append(
                {
                    "method": method,
                    "repository": hit["repository"],
                    "file": hit["file"],
                    "line": hit["line"],
                    "evidence": hit["matched_text"],
                    "interpretation": "Documented in earlier workflow/development code; do not imply successful final-model use without additional run-level evidence.",
                }
            )
    dr_path = output_dir / "developmental_dimensionality_reduction_evidence.csv"
    pd.DataFrame(rows).to_csv(dr_path, index=False)

    summary_path = output_dir / "upstream_methods_evidence_summary.json"
    summary = {
        "source": str(provenance_csv),
        "n_selected_evidence_rows": int(len(concise)),
        "topics": concise.groupby(["repository", "topic"]).size().reset_index(name="n").to_dict("records") if not concise.empty else [],
        "caution": "Rows are evidence candidates for manuscript drafting and require contextual review before asserting a method detail.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return [path, dr_path, summary_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final pre-manuscript author revision materials without rerunning clustering.")
    parser.add_argument("--stroke-config", default="configs/stroke.yaml")
    parser.add_argument("--sepsis-config", default="configs/sepsis.yaml")
    parser.add_argument("--stroke-run")
    parser.add_argument("--sepsis-run")
    parser.add_argument("--provenance-csv", required=True)
    parser.add_argument("--output-dir", default="results/author_revision_materials_v3")
    args = parser.parse_args()

    stroke_cfg = _load_yaml(Path(args.stroke_config))
    sepsis_cfg = _load_yaml(Path(args.sepsis_config))
    stroke_run = Path(args.stroke_run) if args.stroke_run else _latest_run(stroke_cfg["output_dir"])
    sepsis_run = Path(args.sepsis_run) if args.sepsis_run else _latest_run(sepsis_cfg["output_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    created.append(make_manuscript_demographic_table(stroke_cfg, sepsis_cfg, output_dir))
    created.extend(make_model_selection_figure(stroke_run, sepsis_run, output_dir))
    created.extend(make_final_heatmap_figure(stroke_run, sepsis_run, output_dir))
    created.extend(make_compact_supplement_tables(stroke_run, sepsis_run, output_dir))
    created.extend(make_concise_methods_evidence(Path(args.provenance_csv), output_dir))

    manifest = {
        "stroke_run": str(stroke_run),
        "sepsis_run": str(sepsis_run),
        "provenance_csv": str(args.provenance_csv),
        "created": [str(p) for p in created],
        "demographic_mapping_source_note": "Numeric SEX/RACE/HISPANIC mappings follow the compatibility logic documented in PROMIS-ML-pipeline stroke_extraction_scripts/21_make_fully_compatible.py.",
        "note": "These products summarize existing cohorts and analysis outputs; clustering models are not refit.",
    }
    manifest_path = output_dir / "author_revision_materials_v3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(created)} products to {output_dir}")
    print(manifest_path)


if __name__ == "__main__":
    main()
