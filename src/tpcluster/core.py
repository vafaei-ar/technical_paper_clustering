from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def apply_plausibility_limits(
    frame: pd.DataFrame,
    limits_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    rows: list[dict[str, Any]] = []
    if not limits_path:
        return result, pd.DataFrame(rows)
    limits = _load_yaml(limits_path)
    for feature, bounds in limits.items():
        if feature not in result.columns:
            continue
        values = pd.to_numeric(result[feature], errors="coerce")
        invalid = pd.Series(False, index=values.index)
        if bounds.get("min") is not None:
            invalid |= values < float(bounds["min"])
        if bounds.get("max") is not None:
            invalid |= values > float(bounds["max"])
        rows.append({
            "feature": feature,
            "replaced_n": int(invalid.sum()),
            "minimum": bounds.get("min"),
            "maximum": bounds.get("max"),
        })
        result.loc[invalid, feature] = np.nan
    return result, pd.DataFrame(rows)


def prepare_matrix(
    frame: pd.DataFrame,
    features: list[str],
    preprocessing: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = preprocessing or {}
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise KeyError(f"Missing configured features: {missing}")

    raw = frame[features].copy()
    processed = raw.apply(pd.to_numeric, errors="coerce")
    processed, plausibility_report = apply_plausibility_limits(
        processed,
        cfg.get("plausibility_limits_path"),
    )

    log_rows: list[dict[str, Any]] = []
    for feature in cfg.get("log1p_features", []):
        if feature not in processed.columns:
            continue
        values = pd.to_numeric(processed[feature], errors="coerce")
        invalid = values < 0
        values = values.mask(invalid)
        processed[feature] = np.log1p(values)
        log_rows.append({"feature": feature, "negative_to_missing_n": int(invalid.sum())})
    log_report = pd.DataFrame(log_rows)

    processed = processed.replace([np.inf, -np.inf], np.nan)
    processed = processed.fillna(processed.median(numeric_only=True))
    if processed.isna().any().any():
        bad = processed.columns[processed.isna().any()].tolist()
        raise ValueError(f"Features remain missing after median imputation: {bad}")

    removed: list[str] = []
    if cfg.get("remove_zero_variance", True):
        removed = [column for column in processed if processed[column].nunique(dropna=False) <= 1]
        processed = processed.drop(columns=removed)

    clip_rows: list[dict[str, Any]] = []
    quantiles = cfg.get("clip_quantiles")
    if quantiles:
        lower_q, upper_q = map(float, quantiles)
        for feature in processed.columns:
            lower = float(processed[feature].quantile(lower_q))
            upper = float(processed[feature].quantile(upper_q))
            below = int((processed[feature] < lower).sum())
            above = int((processed[feature] > upper).sum())
            processed[feature] = processed[feature].clip(lower, upper)
            clip_rows.append({
                "feature": feature,
                "lower_quantile": lower_q,
                "upper_quantile": upper_q,
                "lower_value": lower,
                "upper_value": upper,
                "clipped_below_n": below,
                "clipped_above_n": above,
            })
    clipping_report = pd.DataFrame(clip_rows)

    scaler_name = str(cfg.get("scaler", "robust")).lower()
    if scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name in {"none", "identity"}:
        x_scaled = processed.to_numpy(dtype=float)
        return raw, processed, x_scaled, removed, plausibility_report, log_report, clipping_report
    else:
        raise ValueError(f"Unsupported scaler: {scaler_name}")

    x_scaled = scaler.fit_transform(processed)
    return raw, processed, x_scaled, removed, plausibility_report, log_report, clipping_report


def reduce_matrix(
    matrix: np.ndarray,
    reduction: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    method = reduction.lower()
    if method == "none":
        return matrix, {"reduction": "none", "n_components": matrix.shape[1]}
    if method != "pca":
        raise ValueError(f"Unsupported reduction: {reduction}")
    max_components = min(matrix.shape[0] - 1, matrix.shape[1])
    pca = PCA(n_components=0.90, svd_solver="full", random_state=seed)
    transformed = pca.fit_transform(matrix)
    return transformed, {
        "reduction": "pca",
        "n_components": int(transformed.shape[1]),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "maximum_possible_components": int(max_components),
    }


def fit_clusterer(
    matrix: np.ndarray,
    clusterer: str,
    k: int,
    seed: int,
) -> np.ndarray:
    method = clusterer.lower()
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(matrix)
    if method == "gmm":
        return GaussianMixture(n_components=k, n_init=5, random_state=seed).fit_predict(matrix)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=k).fit_predict(matrix)
    raise ValueError(f"Unsupported clusterer: {clusterer}")
