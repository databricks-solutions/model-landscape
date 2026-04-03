from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-10


def _stable_histogram_edges(reference: np.ndarray, current: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.histogram_bin_edges(reference, bins=n_bins).astype(float, copy=True)
    if edges.ndim != 1 or len(edges) < 2:
        return np.array([])
    current_min = float(np.nanmin(current))
    current_max = float(np.nanmax(current))
    edges[0] = min(edges[0], current_min)
    edges[-1] = max(edges[-1], current_max)
    if np.all(np.diff(edges) > 0):
        return edges
    lower = float(min(np.nanmin(reference), current_min))
    upper = float(max(np.nanmax(reference), current_max))
    if lower == upper:
        padding = max(abs(lower) * 0.01, 0.5)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, num=max(2, n_bins + 1), dtype=float)


def _histogram(reference: np.ndarray, current: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return np.array([]), np.array([])
    edges = _stable_histogram_edges(ref, cur, n_bins)
    if len(edges) == 0:
        return np.array([]), np.array([])
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return np.array([]), np.array([])
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


def _categorical_distribution(reference: pd.Series, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ref = reference.astype("string").fillna("__NULL__")
    cur = current.astype("string").fillna("__NULL__")
    categories = sorted(set(ref.tolist()) | set(cur.tolist()))
    if not categories:
        return np.array([]), np.array([])
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return np.array([]), np.array([])
    ref_props = ref_counts / ref_counts.sum() + EPSILON
    cur_props = cur_counts / cur_counts.sum() + EPSILON
    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()
    return ref_props, cur_props


def _categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    ref_props, cur_props = _categorical_distribution(reference, current)
    if len(ref_props) == 0:
        return float("nan")
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def _categorical_kl(reference: pd.Series, current: pd.Series) -> float:
    ref_props, cur_props = _categorical_distribution(reference, current)
    if len(ref_props) == 0:
        return float("nan")
    return float(np.sum(cur_props * np.log(cur_props / ref_props)))


def _categorical_js(reference: pd.Series, current: pd.Series) -> float:
    ref_props, cur_props = _categorical_distribution(reference, current)
    if len(ref_props) == 0:
        return float("nan")
    midpoint = 0.5 * (ref_props + cur_props)
    return float(
        0.5 * np.sum(ref_props * np.log2(ref_props / midpoint))
        + 0.5 * np.sum(cur_props * np.log2(cur_props / midpoint))
    )


def compute_categorical_distribution_metrics(
    reference_counts: dict[str, float],
    current_counts: dict[str, float],
) -> tuple[float, float, float]:
    categories = sorted(set(reference_counts) | set(current_counts))
    if not categories:
        return float("nan"), float("nan"), float("nan")
    ref_counts = np.array([float(reference_counts.get(category, 0.0)) for category in categories], dtype=float)
    cur_counts = np.array([float(current_counts.get(category, 0.0)) for category in categories], dtype=float)
    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    ref_props = ref_counts / ref_counts.sum() + EPSILON
    cur_props = cur_counts / cur_counts.sum() + EPSILON
    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()
    midpoint = 0.5 * (ref_props + cur_props)
    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    kl = float(np.sum(cur_props * np.log(cur_props / ref_props)))
    js = float(
        0.5 * np.sum(ref_props * np.log2(ref_props / midpoint))
        + 0.5 * np.sum(cur_props * np.log2(cur_props / midpoint))
    )
    return psi, kl, js


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    ref_numeric = _numeric_feature_frame(reference_df, features)
    cur_numeric = _numeric_feature_frame(current_df, features)
    categorical_set = set(categorical_features or [])
    shared_features = [
        feature
        for feature in features
        if (
            feature in categorical_set
            and feature in reference_df.columns
            and feature in current_df.columns
        )
        or (
            feature not in categorical_set
            and feature in ref_numeric.columns
            and feature in cur_numeric.columns
        )
    ]
    rows: list[dict] = []
    for feature in shared_features:
        if feature in categorical_set:
            ref_values = reference_df[feature] if feature in reference_df.columns else pd.Series(dtype="string")
            cur_values = current_df[feature] if feature in current_df.columns else pd.Series(dtype="string")
            if ref_values.empty or cur_values.empty:
                continue
            rows.append({
                "feature_name": feature,
                "psi": round(_categorical_psi(ref_values, cur_values), 6),
                "kl_divergence": round(_categorical_kl(ref_values, cur_values), 6),
                "js_divergence": round(_categorical_js(ref_values, cur_values), 6),
                "ref_mean": float("nan"),
                "cur_mean": float("nan"),
                "ref_std": float("nan"),
                "cur_std": float("nan"),
                "ref_null_pct": round(float(ref_values.isna().mean() * 100), 2),
                "cur_null_pct": round(float(cur_values.isna().mean() * 100), 2),
                "ref_count": int(ref_values.notna().sum()),
                "cur_count": int(cur_values.notna().sum()),
            })
            continue
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
