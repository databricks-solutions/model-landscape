from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-10


def _histogram(reference: np.ndarray, current: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return np.array([]), np.array([])
    edges = np.histogram_bin_edges(ref, bins=n_bins)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_props = ref_counts / ref_counts.sum() + EPSILON
    cur_props = cur_counts / cur_counts.sum() + EPSILON
    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()
    return ref_props, cur_props


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 20) -> float:
    ref_props, cur_props = _histogram(reference, current, n_bins)
    if len(ref_props) == 0:
        return float("nan")
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def compute_kl(reference: np.ndarray, current: np.ndarray, n_bins: int = 20) -> float:
    ref_props, cur_props = _histogram(reference, current, n_bins)
    if len(ref_props) == 0:
        return float("nan")
    return float(np.sum(cur_props * np.log(cur_props / ref_props)))


def compute_js(reference: np.ndarray, current: np.ndarray, n_bins: int = 20) -> float:
    ref_props, cur_props = _histogram(reference, current, n_bins)
    if len(ref_props) == 0:
        return float("nan")
    midpoint = 0.5 * (ref_props + cur_props)
    return float(
        0.5 * np.sum(ref_props * np.log2(ref_props / midpoint))
        + 0.5 * np.sum(cur_props * np.log2(cur_props / midpoint))
    )


def _numeric_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    available_features = [feature for feature in features if feature in df.columns]
    if not available_features:
        return pd.DataFrame()
    frame = df.loc[:, available_features].copy()
    for feature in available_features:
        if pd.api.types.is_numeric_dtype(frame[feature]):
            continue
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame


def compute_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    ref_numeric = _numeric_feature_frame(reference_df, features)
    cur_numeric = _numeric_feature_frame(current_df, features)
    shared_features = [feature for feature in features if feature in ref_numeric.columns and feature in cur_numeric.columns]
    rows: list[dict] = []
    for feature in shared_features:
        ref_values = ref_numeric[feature].to_numpy(dtype=float, copy=False)
        cur_values = cur_numeric[feature].to_numpy(dtype=float, copy=False)
        if (~np.isnan(ref_values)).sum() < 2 or (~np.isnan(cur_values)).sum() < 2:
            continue
        rows.append({
            "feature_name": feature,
            "psi": round(compute_psi(ref_values, cur_values), 6),
            "kl_divergence": round(compute_kl(ref_values, cur_values), 6),
            "js_divergence": round(compute_js(ref_values, cur_values), 6),
            "ref_mean": float(np.nanmean(ref_values)),
            "cur_mean": float(np.nanmean(cur_values)),
            "ref_std": float(np.nanstd(ref_values)),
            "cur_std": float(np.nanstd(cur_values)),
            "ref_null_pct": round(float(np.isnan(ref_values).mean() * 100), 2),
            "cur_null_pct": round(float(np.isnan(cur_values).mean() * 100), 2),
            "ref_count": int((~np.isnan(ref_values)).sum()),
            "cur_count": int((~np.isnan(cur_values)).sum()),
        })
    return pd.DataFrame(rows)
