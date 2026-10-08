"""
Real-data loader for the Clustro paper figures.

This file keeps the aligned visual code data-agnostic. Every number used in
Figures 1-5 is loaded from the reviewer-driven analysis outputs already
produced by this repository.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTING_DIR = Path(
    os.environ.get(
        "CLUSTRO_REPORTING_DIR",
        REPO_ROOT / "results" / "methods_revision_reporting",
    )
)
METHODS_DIR = Path(
    os.environ.get(
        "CLUSTRO_METHODS_DIR",
        REPO_ROOT / "results" / "methods_revision",
    )
)


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required figure input: {path}\n"
            "Run regenerate_methods_revision_reporting.sh first."
        )
    return path


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_required(REPORTING_DIR / name))


# --------------------------------------------------------------------------- #
# Cohorts / cluster sizes
# --------------------------------------------------------------------------- #
_stroke_sizes = _read_csv("table_cluster_sizes_stroke.csv").sort_values(
    "primary_cluster"
)
_sepsis_sizes = _read_csv("table_cluster_sizes_sepsis.csv").sort_values(
    "primary_cluster"
)
N_STROKE = int(_stroke_sizes["n"].sum())
N_SEPSIS = int(_sepsis_sizes["n"].sum())


def _count_map(df: pd.DataFrame) -> dict[int, int]:
    return {
        int(r["primary_cluster"]): int(r["n"])
        for _, r in df.iterrows()
    }


_stroke_n = _count_map(_stroke_sizes)
_sepsis_n = _count_map(_sepsis_sizes)

# --------------------------------------------------------------------------- #
# Figure 2 -- model-selection landscape
# --------------------------------------------------------------------------- #
_FAMILY = {
    ("pca", "kmeans"): "PCA k-means",
    ("none", "kmeans"): "Raw k-means",
    ("none", "agglomerative"): "Raw agglomerative",
}


def _candidate_rows(cohort: str):
    df = _read_csv(f"tableS_candidate_model_audit_{cohort}.csv")
    df = df[df["mean_subsample_ari"].notna()].copy()
    rows = []
    for _, r in df.iterrows():
        key = (str(r["reduction"]).lower(), str(r["clusterer"]).lower())
        if key not in _FAMILY:
            continue
        k = int(r["k"])
        if k not in (2, 3, 4) and key != ("none", "agglomerative"):
            continue
        if key == ("none", "agglomerative") and k not in (2, 3):
            continue
        rows.append(
            (
                _FAMILY[key],
                k,
                float(r["silhouette_mean"]),
                float(r["mean_subsample_ari"]),
            )
        )
    return rows


STROKE_CANDIDATES = _candidate_rows("stroke")
SEPSIS_CANDIDATES = _candidate_rows("sepsis")
STROKE_SELECTED = ("PCA k-means", 3)
SEPSIS_SELECTED = ("PCA k-means", 3)


def point(candidates, family: str, k: int) -> tuple[float, float]:
    for fam, kk, sil, ari in candidates:
        if fam == family and kk == k:
            return float(sil), float(ari)
    raise KeyError(f"Candidate not found: {family}, k={k}")


# --------------------------------------------------------------------------- #
# Figure 3 -- actual 100-replicate null reference
# --------------------------------------------------------------------------- #
_null_summary = _read_csv(
    "null_reference_silhouette_extended_summary.csv"
).set_index("dataset")


def _null_spec(dataset: str) -> dict:
    row = _null_summary.loc[dataset]
    rep_path = METHODS_DIR / f"null_reference_silhouette_extended_{dataset}.csv"
    reps = None
    if rep_path.exists():
        rep = pd.read_csv(rep_path)
        if "silhouette" in rep.columns:
            reps = rep["silhouette"].dropna().to_numpy(dtype=float)
    return {
        "mean": float(row["null_silhouette_mean"]),
        "sd": float(row["null_silhouette_sd"]),
        "observed": float(row["real_silhouette"]),
        "n_ge": int(row["n_null_ge_real"]),
        "n_rep": int(row["null_repeats"]),
        "p": float(row["empirical_p_silhouette"]),
        "values": reps,
    }


NULL_STROKE = _null_spec("stroke")
NULL_SEPSIS = _null_spec("sepsis")


def null_density(spec, n_grid=600, pad=3.5):
    """Smooth density from the actual null replicates.

    If the replicate CSV is unavailable, fall back to a Gaussian density using
    the reported null mean and standard deviation. No synthetic skew-normal
    distribution is used.
    """
    values = spec.get("values")
    if values is not None and len(values) >= 5:
        values = np.asarray(values, dtype=float)
        sd = values.std(ddof=1)
        bw = max(1e-4, 1.06 * sd * len(values) ** (-1 / 5))
        xmin = min(values.min(), spec["observed"]) - pad * bw
        xmax = max(values.max(), spec["observed"]) + pad * bw
        x = np.linspace(xmin, xmax, n_grid)
        z = (x[:, None] - values[None, :]) / bw
        dens = np.exp(-0.5 * z * z).mean(axis=1) / (
            bw * np.sqrt(2 * np.pi)
        )
        return x, dens

    sd = max(float(spec["sd"]), 1e-4)
    x = np.linspace(
        min(spec["mean"] - pad * sd, spec["observed"] - 2 * sd),
        max(spec["mean"] + pad * sd, spec["observed"] + 2 * sd),
        n_grid,
    )
    dens = np.exp(-0.5 * ((x - spec["mean"]) / sd) ** 2) / (
        sd * np.sqrt(2 * np.pi)
    )
    return x, dens


# --------------------------------------------------------------------------- #
# Figure 4 -- real phenotype profile values
# --------------------------------------------------------------------------- #
def _resolve_run(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _run_dirs() -> tuple[Path, Path]:
    manifest_path = REPORTING_DIR / "reporting_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return (
            _resolve_run(manifest["stroke_run"]),
            _resolve_run(manifest["sepsis_run"]),
        )

    def latest(root: Path):
        runs = [
            p for p in root.iterdir()
            if p.is_dir()
            and (
                p
                / "manuscript_outputs"
                / "tables"
                / "table_continuous_cluster_profiles_processed_scale.csv"
            ).exists()
        ]
        if not runs:
            raise FileNotFoundError(f"No reporting-ready runs under {root}")
        return max(runs, key=lambda p: p.stat().st_mtime)

    return (
        latest(REPO_ROOT / "results" / "stroke_methods_revision"),
        latest(REPO_ROOT / "results" / "sepsis"),
    )


_STROKE_RUN, _SEPSIS_RUN = _run_dirs()


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _profile_tables(run_dir: Path):
    table_dir = run_dir / "manuscript_outputs" / "tables"
    continuous = pd.read_csv(
        _required(
            table_dir
            / "table_continuous_cluster_profiles_processed_scale.csv"
        )
    )
    binary = pd.read_csv(
        _required(table_dir / "table_binary_cluster_profiles.csv")
    )
    return continuous, binary


_STROKE_CONT, _STROKE_BIN = _profile_tables(_STROKE_RUN)
_SEPSIS_CONT, _SEPSIS_BIN = _profile_tables(_SEPSIS_RUN)


def _feature_values(
    frame: pd.DataFrame,
    aliases: list[str],
    value_col: str,
) -> list[float]:
    label_cols = [c for c in ("feature", "feature_label") if c in frame.columns]
    lookup = {}
    for idx, row in frame.iterrows():
        for col in label_cols:
            lookup.setdefault(_normalise(row[col]), idx)

    match_idx = None
    for alias in aliases:
        key = _normalise(alias)
        if key in lookup:
            match_idx = lookup[key]
            break

    if match_idx is None:
        available = sorted(
            {
                str(v)
                for c in label_cols
                for v in frame[c].dropna().unique()
            }
        )
        raise KeyError(
            f"Could not match any of {aliases}. "
            f"Available features include: {available[:80]}"
        )

    feature_name = frame.loc[match_idx, "feature"]
    subset = frame[frame["feature"].eq(feature_name)].copy()
    values = []
    for cluster in (0, 1, 2):
        row = subset[subset["cluster"].astype(int).eq(cluster)]
        if row.empty:
            raise KeyError(
                f"Missing cluster {cluster} for feature {feature_name}"
            )
        values.append(float(row.iloc[0][value_col]))
    return values


def _cont(frame, *aliases):
    return _feature_values(
        frame, list(aliases), "median_difference_iqr"
    )


def _bin(frame, *aliases):
    value_col = (
        "standardised_difference"
        if "standardised_difference" in frame.columns
        else "standardized_difference"
    )
    return _feature_values(frame, list(aliases), value_col)


STROKE_CLUSTERS = [
    ("Preserved\nhematologic", _stroke_n[0], "blue"),
    ("Renal-anemic", _stroke_n[1], "orange"),
    ("Hyperglycemic", _stroke_n[2], "red"),
]

STROKE_ROWS = [
    ("Demographics", "people", [
        ("Age at stroke", _cont(_STROKE_CONT, "AGE_AT_STROKE", "Age at stroke")),
    ]),
    ("Hematology", "droplet", [
        ("Erythrocytes", _cont(_STROKE_CONT, "Erythrocytes")),
        ("Hemoglobin", _cont(_STROKE_CONT, "Hemoglobin")),
        ("Hematocrit", _cont(_STROKE_CONT, "Hematocrit")),
        ("Platelets", _cont(_STROKE_CONT, "Platelets")),
    ]),
    ("Renal /\nmetabolic", "kidneys", [
        ("Creatinine", _cont(_STROKE_CONT, "Creatinine")),
        ("Glucose", _cont(_STROKE_CONT, "Glucose")),
    ]),
    ("Cardiovascular\ncomorbidity", "heart", [
        ("Diabetes mellitus", _bin(_STROKE_BIN, "Diabetes_Mellitus", "Diabetes Mellitus")),
        ("Chronic kidney disease", _bin(_STROKE_BIN, "Chronic_kidney_disease", "Chronic kidney disease")),
        ("Heart failure", _bin(_STROKE_BIN, "Heart_failure", "Heart failure")),
        ("Atrial fibrillation", _bin(_STROKE_BIN, "Atrial_Fibrillation", "Atrial Fibrillation")),
        ("Hypertension", _bin(_STROKE_BIN, "Hypertension")),
        ("Dyslipidemia", _bin(_STROKE_BIN, "Dyslipidemia")),
    ]),
]

SEPSIS_CLUSTERS = [
    ("Reference /\nneutrophil", _sepsis_n[0], "blue"),
    ("IG-high", _sepsis_n[1], "orange"),
    ("Eosinophil-lymphocyte", _sepsis_n[2], "red"),
]

SEPSIS_ROWS = [
    ("Organ\ndysfunction", "gear", [
        ("Anion gap", _cont(_SEPSIS_CONT, "Anion gap 3", "Anion gap")),
        ("Creatinine", _cont(_SEPSIS_CONT, "Creatinine")),
        ("Albumin (BCG)", _cont(_SEPSIS_CONT, "Albumin (BCG)", "Albumin BCG")),
    ]),
    ("WBC\ncomposition", "cells", [
        ("WBC #", _cont(_SEPSIS_CONT, "WBC #", "WBC")),
        ("Neutrophils %", _cont(_SEPSIS_CONT, "Neutrophils %")),
        ("Lymphocytes %", _cont(_SEPSIS_CONT, "Lymphocytes %")),
        ("Eosinophils %", _cont(_SEPSIS_CONT, "Eosinophils %")),
        ("Eosinophils #", _cont(_SEPSIS_CONT, "Eosinophils #")),
        ("IG %", _cont(_SEPSIS_CONT, "IG %")),
        ("IG #", _cont(_SEPSIS_CONT, "IG #")),
    ]),
    ("Acute\nfindings", "lungs", [
        ("Acute respiratory failure", _bin(_SEPSIS_BIN, "acute respiratory failure")),
        ("Acute kidney failure", _bin(_SEPSIS_BIN, "acute kidney failure")),
        ("Acute hypotension", _bin(_SEPSIS_BIN, "acute hypotension")),
        ("Pulmonary imaging findings", _bin(_SEPSIS_BIN, "pulmonary imaging findings")),
        ("Acute WBC elevation", _bin(_SEPSIS_BIN, "acute WBC elevated", "acute WBC elevation")),
    ]),
]

_all_profile_values = [
    abs(v)
    for groups in (STROKE_ROWS, SEPSIS_ROWS)
    for _, _, rows in groups
    for _, vals in rows
    for v in vals
]
HEAT_VMAX = max(2.5, min(5.0, float(np.ceil(max(_all_profile_values) * 2) / 2)))


# --------------------------------------------------------------------------- #
# Figure 5 -- real robustness / sensitivity results
# --------------------------------------------------------------------------- #
K_GRID = [2, 3, 4]
K_SELECTED = 3


def _pca_k_values(cohort: str, column: str) -> list[float]:
    df = _read_csv(f"tableS_candidate_model_audit_{cohort}.csv")
    q = df[
        df["reduction"].astype(str).str.lower().eq("pca")
        & df["clusterer"].astype(str).str.lower().eq("kmeans")
        & df["k"].isin(K_GRID)
    ].set_index("k").reindex(K_GRID)
    return [float(v) for v in q[column]]


SIL_BY_K = {
    "Stroke": _pca_k_values("stroke", "silhouette_mean"),
    "Sepsis": _pca_k_values("sepsis", "silhouette_mean"),
}
ARI_BY_K = {
    "Stroke": _pca_k_values("stroke", "mean_subsample_ari"),
    "Sepsis": _pca_k_values("sepsis", "mean_subsample_ari"),
}

_sens = _read_csv("figure4_sensitivity_values.csv")


def _sens_value(cohort: str, name: str) -> float:
    row = _sens[
        _sens["cohort"].eq(cohort)
        & _sens["sensitivity"].eq(name)
    ]
    if row.empty:
        raise KeyError(f"Missing sensitivity value: {cohort} / {name}")
    return float(row.iloc[0]["ari"])


ARI_RAW_VS_PCA = {
    "Stroke": _sens_value("Stroke", "Raw vs PCA"),
    "Sepsis": _sens_value("Sepsis", "Raw vs PCA"),
}
ARI_BALANCED_BLOCK = {
    "Stroke": _sens_value("Stroke", "Balanced blocks"),
    "Sepsis": _sens_value("Sepsis", "Balanced blocks"),
}
ARI_WBC_REDUNDANCY = {
    "Drop WBC counts": _sens_value("Sepsis", "Drop WBC counts"),
    "Drop WBC percentages": _sens_value("Sepsis", "Drop WBC percentages"),
}
