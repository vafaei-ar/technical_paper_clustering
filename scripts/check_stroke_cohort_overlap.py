#!/usr/bin/env python3
"""Aggregate-only identity and cluster-overlap audit for two Penn State stroke studies.

Reads local patient-level Parquet/CSV files but writes only counts, optional
cluster contingency tables, and ARI. Never exports patient identifiers.
No row-order matching is allowed: both label files must contain PATID.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower()==".tsv" else ",",
                           usecols=columns, dtype={columns[0]: "string"})
    raise ValueError(f"Unsupported file type: {path}")


def clean_ids(df: pd.DataFrame, name: str, key: str) -> pd.DataFrame:
    if key not in df.columns:
        raise ValueError(f"{name} lacks {key}")
    df = df.copy()
    df[key] = df[key].astype("string").str.strip()
    if df[key].isna().any() or df[key].eq("").any():
        raise ValueError(f"{name} contains missing/blank identifiers")
    if df[key].duplicated().any():
        raise ValueError(f"{name} contains duplicate patient identifiers")
    return df


def add_labels(base: pd.DataFrame, path: Path, base_key: str, labels_key: str,
               label_col: str, display: str) -> pd.DataFrame:
    labels = clean_ids(load(path, [labels_key, label_col]), display, labels_key)
    labels = labels.rename(columns={labels_key:base_key})
    labels[label_col] = labels[label_col].astype("string")
    if labels[label_col].isna().any():
        raise ValueError(f"{display} has missing cluster labels")
    overlap = base[[base_key]].merge(labels, how="left", on=base_key, validate="one_to_one")
    if overlap[label_col].isna().any():
        raise ValueError(f"{display} lacks labels for {int(overlap[label_col].isna().sum())} cohort patients")
    return overlap


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--technical-cohort", type=Path, required=True)
    p.add_argument("--clinical-cohort", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--id-col", default="PATID")
    p.add_argument("--ischemic-col", default="Ischemic_Stroke")
    p.add_argument("--hemorrhagic-col", default="Hemorrhagic_Stroke")
    p.add_argument("--technical-labels", type=Path)
    p.add_argument("--clinical-labels", type=Path)
    p.add_argument("--technical-label-column")
    p.add_argument("--clinical-label-column")
    p.add_argument("--technical-label-id-column", default="PATID")
    p.add_argument("--clinical-label-id-column", default="PATID")
    p.add_argument("--expected-technical", type=int, default=9835)
    p.add_argument("--expected-clinical", type=int, default=7688)
    args = p.parse_args()

    flags = [args.ischemic_col, args.hemorrhagic_col]
    technical = clean_ids(load(args.technical_cohort, [args.id_col]+flags), "technical cohort", args.id_col)
    clinical = clean_ids(load(args.clinical_cohort, [args.id_col]), "clinical cohort", args.id_col)
    if len(technical) != args.expected_technical or len(clinical) != args.expected_clinical:
        raise ValueError(f"Unexpected sizes: technical={len(technical)}, clinical={len(clinical)}")

    t_ids, c_ids = set(technical[args.id_col]), set(clinical[args.id_col])
    shared = t_ids.intersection(c_ids)
    ischemic = pd.to_numeric(technical[args.ischemic_col], errors="coerce").eq(1)
    hemorrhagic = pd.to_numeric(technical[args.hemorrhagic_col], errors="coerce").eq(1)
    flag_sets = {
        "ischemic_only": set(technical.loc[ischemic & ~hemorrhagic, args.id_col]),
        "hemorrhagic_only": set(technical.loc[~ischemic & hemorrhagic, args.id_col]),
        "dual_coded": set(technical.loc[ischemic & hemorrhagic, args.id_col]),
        "neither": set(technical.loc[~ischemic & ~hemorrhagic, args.id_col]),
    }
    result = {
        "technical_n":len(technical), "clinical_n":len(clinical),
        "shared_n":len(shared), "clinical_subset_of_technical":c_ids.issubset(t_ids),
        "clinical_only_n":len(c_ids-t_ids), "technical_only_n":len(t_ids-c_ids),
        "technical_indicator_counts":{k:len(v) for k,v in flag_sets.items()},
        "clinical_exactly_equals_technical_ischemic_only": c_ids == flag_sets["ischemic_only"],
        "clinical_non_ischemic_only_n":len(c_ids-flag_sets["ischemic_only"]),
    }
    if (args.technical_labels is None) != (args.clinical_labels is None):
        raise ValueError("Supply both label files or neither")
    if args.technical_labels:
        if not args.technical_label_column or not args.clinical_label_column:
            raise ValueError("Specify --technical-label-column and --clinical-label-column")
        tl = add_labels(technical[[args.id_col]], args.technical_labels,
                        args.id_col, args.technical_label_id_column, args.technical_label_column, "technical labels")
        cl = add_labels(clinical[[args.id_col]], args.clinical_labels,
                        args.id_col, args.clinical_label_id_column, args.clinical_label_column, "clinical labels")
        # Normalize label-file identifier names before joining.
        tl = tl.rename(columns={args.technical_label_column:"technical_cluster"})
        cl = cl.rename(columns={args.clinical_label_column:"clinical_cluster"})
        joint = tl.merge(cl, how="inner", on=args.id_col, validate="one_to_one")
        from sklearn.metrics import adjusted_rand_score
        result["shared_labeled_n"] = len(joint)
        result["cluster_ari_shared_patients"] = float(
            adjusted_rand_score(joint["technical_cluster"],joint["clinical_cluster"]))
        contingency = pd.crosstab(joint["technical_cluster"],joint["clinical_cluster"])
        result["cluster_contingency"] = {
            str(k):{str(kk):int(v) for kk,v in vals.items()} for k,vals in contingency.to_dict("index").items()
        }
    args.outdir.mkdir(parents=True, exist_ok=True)
    outfile = args.outdir/"stroke_cohort_overlap_aggregate.json"
    outfile.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    print(f"Aggregate results only: {outfile}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
