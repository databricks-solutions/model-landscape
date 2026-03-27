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


def compute_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref_values = pd.to_numeric(reference_df[feature], errors="coerce").to_numpy()
        cur_values = pd.to_numeric(current_df[feature], errors="coerce").to_numpy()
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

