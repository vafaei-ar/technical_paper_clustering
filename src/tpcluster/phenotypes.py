from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str:
    lookup = {_normalise(column): str(column) for column in columns}
    for alias in aliases:
        match = lookup.get(_normalise(alias))
        if match is not None:
            return match
    raise KeyError(
        "None of the expected columns were found. "
        f"Expected one of {list(aliases)}; available columns include {list(columns)}"
    )


def canonical_cluster_mapping(
    cohort: str,
    frame: pd.DataFrame,
    labels: pd.Series,
) -> dict[int, int]:
    """Map arbitrary three-cluster IDs to stable clinical phenotype IDs.

    Canonical IDs are:
      stroke: 0 preserved/lower-comorbidity, 1 renal-anaemic, 2 hyperglycaemic
      sepsis: 0 neutrophil/lower-acuity, 1 IG-high organ dysfunction,
              2 eosinophil-lymphocyte enriched
    """
    label_series = pd.Series(labels, index=frame.index, name="_raw_cluster")
    unique = sorted(int(value) for value in label_series.dropna().unique())
    if len(unique) != 3:
        raise ValueError(
            f"Canonical phenotype labelling requires exactly three clusters; found {unique}"
        )

    profiled = frame.copy()
    profiled["_raw_cluster"] = label_series.astype(int)
    profiles = profiled.groupby("_raw_cluster").median(numeric_only=True)

    if cohort == "stroke":
        glucose = _find_column(
            profiles.columns,
            ["Glucose", "glucose", "GLUCOSE", "Glucose, Blood"],
        )
        haematocrit = _find_column(
            profiles.columns,
            ["Hematocrit", "Haematocrit", "HCT", "hematocrit"],
        )
        creatinine = _find_column(
            profiles.columns,
            ["Creatinine", "creatinine", "CREATININE"],
        )

        hyperglycaemic = int(profiles[glucose].idxmax())
        remaining = [cluster for cluster in unique if cluster != hyperglycaemic]

        # Anaemia is the primary anchor; creatinine deterministically breaks a tie.
        renal_score = pd.DataFrame(
            {
                "hct": profiles.loc[remaining, haematocrit],
                "creatinine": profiles.loc[remaining, creatinine],
            }
        ).sort_values(["hct", "creatinine"], ascending=[True, False])
        renal_anaemic = int(renal_score.index[0])
        preserved = int(next(cluster for cluster in remaining if cluster != renal_anaemic))
        return {preserved: 0, renal_anaemic: 1, hyperglycaemic: 2}

    if cohort == "sepsis":
        ig_count = _find_column(
            profiles.columns,
            ["IG #", "Immature Granulocytes #", "Immature Granulocyte Count", "IG count"],
        )
        eosinophils = _find_column(
            profiles.columns,
            ["Eosinophils %", "Eosinophil %", "Eosinophils", "EOS %"],
        )
        lymphocytes = _find_column(
            profiles.columns,
            ["Lymphocytes %", "Lymphocyte %", "Lymphocytes", "LYMPH %"],
        )

        ig_high = int(profiles[ig_count].idxmax())
        remaining = [cluster for cluster in unique if cluster != ig_high]

        # Eosinophils are the primary anchor; lymphocytes break a tie.
        enriched_score = pd.DataFrame(
            {
                "eosinophils": profiles.loc[remaining, eosinophils],
                "lymphocytes": profiles.loc[remaining, lymphocytes],
            }
        ).sort_values(["eosinophils", "lymphocytes"], ascending=[False, False])
        eosinophil_lymphocyte = int(enriched_score.index[0])
        neutrophil_lower_acuity = int(
            next(cluster for cluster in remaining if cluster != eosinophil_lymphocyte)
        )
        return {
            neutrophil_lower_acuity: 0,
            ig_high: 1,
            eosinophil_lymphocyte: 2,
        }

    raise ValueError(f"Unsupported cohort for phenotype canonicalisation: {cohort}")


def apply_canonical_mapping(
    labels: pd.Series,
    mapping: dict[int, int],
) -> pd.Series:
    mapped = pd.Series(labels).map(mapping)
    if mapped.isna().any():
        missing = sorted(pd.Series(labels)[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"Cluster mapping did not cover raw cluster IDs: {missing}")
    return mapped.astype(int)


def remap_profile_clusters(
    profile: pd.DataFrame,
    mapping: dict[int, int],
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    result = profile.copy()
    result["raw_cluster"] = result[cluster_column].astype(int)
    result[cluster_column] = result["raw_cluster"].map(mapping)
    if result[cluster_column].isna().any():
        missing = sorted(result.loc[result[cluster_column].isna(), "raw_cluster"].unique())
        raise ValueError(f"Profile table contains unmapped raw cluster IDs: {missing}")
    result[cluster_column] = result[cluster_column].astype(int)
    return result.sort_values([cluster_column]).reset_index(drop=True)
