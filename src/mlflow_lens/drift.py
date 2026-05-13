"""Training-run drift detection across MLflow experiments.

Computes feature drift (via PSI, KL, JS) and prediction shift between a
reference run and the current run. Scoped to training-run drift, not
production inference monitoring.

Delegates to model_landscape.analytics.drift for the actual computation --
same engine that powers production monitoring, applied to training runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from mlflow_lens._artifacts import log_json_artifact
from mlflow_lens._version import __version__
from mlflow_lens.analytics.drift import (
    compute_js,
    compute_kl,
    compute_psi as _analytics_psi,
)

DRIFT_ARTIFACT_PATH = "lens/drift.json"

PSI_THRESHOLDS = {"low": 0.1, "moderate": 0.25}


def log_drift(
    reference_run_id: str,
    features: pd.DataFrame,
    predictions: pd.Series | np.ndarray | None = None,
    reference_features: pd.DataFrame | None = None,
    reference_predictions: pd.Series | np.ndarray | None = None,
    n_bins: int = 20,
) -> dict:
    """Log training-run drift relative to a reference run.

    If *reference_features* / *reference_predictions* are not provided, they
    are loaded from the reference run's ``lens/drift_snapshot.json`` artifact.

    Stores the result as ``lens/drift.json`` on the active run and a
    data snapshot as ``lens/drift_snapshot.json`` for future comparisons.
    """
    if reference_features is None:
        reference_features = _load_snapshot_features(reference_run_id)

    feature_drift = _compute_feature_drift(reference_features, features, n_bins)

    prediction_shift = None
    if predictions is not None:
        preds = np.asarray(predictions)
        if reference_predictions is not None:
            ref_preds = np.asarray(reference_predictions)
        else:
            ref_preds = _load_snapshot_predictions(reference_run_id)

        if ref_preds is not None:
            prediction_shift = _compute_prediction_shift(ref_preds, preds, n_bins)

    payload = {
        "lens_version": __version__,
        "schema_version": "2",
        "reference_run_id": reference_run_id,
        "features": feature_drift,
        "prediction_shift": prediction_shift,
    }

    mlflow.set_tag("lens.version", __version__)
    mlflow.set_tag("lens.has_drift", "true")

    log_json_artifact(payload, artifact_path="lens", filename="drift.json")
    _save_snapshot(features, predictions)

    return payload


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 20) -> float:
    """Population Stability Index between two 1-D distributions.

    Uses stable histogram edges from model_landscape.analytics.drift.
    """
    return _analytics_psi(reference, current, n_bins)


def classify_drift(psi: float) -> str:
    if psi < PSI_THRESHOLDS["low"]:
        return "low"
    elif psi < PSI_THRESHOLDS["moderate"]:
        return "moderate"
    return "high"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_feature_drift(
    ref_df: pd.DataFrame, cur_df: pd.DataFrame, n_bins: int
) -> list[dict]:
    results = []
    shared_cols = [c for c in ref_df.columns if c in cur_df.columns]
    for col in shared_cols:
        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        if not np.issubdtype(ref_vals.dtype, np.number):
            continue

        ref_arr = np.asarray(ref_vals, dtype=float)
        cur_arr = np.asarray(cur_vals, dtype=float)

        psi = _analytics_psi(ref_arr, cur_arr, n_bins)
        kl = compute_kl(ref_arr, cur_arr, n_bins)
        js = compute_js(ref_arr, cur_arr, n_bins)

        results.append({
            "name": col,
            "psi": round(psi, 4),
            "kl_divergence": round(kl, 4),
            "js_divergence": round(js, 4),
            "status": classify_drift(psi),
            "ref_mean": round(float(np.mean(ref_vals)), 4),
            "cur_mean": round(float(np.mean(cur_vals)), 4),
            "mean_delta": round(float(np.mean(cur_vals) - np.mean(ref_vals)), 4),
        })
    return results


def _compute_prediction_shift(
    ref_preds: np.ndarray, cur_preds: np.ndarray, n_bins: int
) -> dict:
    psi = _analytics_psi(ref_preds, cur_preds, n_bins)
    kl = compute_kl(ref_preds, cur_preds, n_bins)
    js = compute_js(ref_preds, cur_preds, n_bins)
    return {
        "psi": round(psi, 4),
        "kl_divergence": round(kl, 4),
        "js_divergence": round(js, 4),
        "status": classify_drift(psi),
        "mean_delta": round(float(np.mean(cur_preds) - np.mean(ref_preds)), 4),
        "ref_mean": round(float(np.mean(ref_preds)), 4),
        "cur_mean": round(float(np.mean(cur_preds)), 4),
    }


def _save_snapshot(
    features: pd.DataFrame,
    predictions: pd.Series | np.ndarray | None,
) -> None:
    """Save a feature/prediction snapshot so future runs can compare against this one."""
    snapshot: dict = {
        "lens_version": __version__,
        "schema_version": "1",
        "feature_stats": {},
        "prediction_stats": None,
    }
    for col in features.select_dtypes(include=[np.number]).columns:
        vals = features[col].dropna().values
        snapshot["feature_stats"][col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals.tolist(),
        }

    if predictions is not None:
        preds = np.asarray(predictions)
        snapshot["prediction_stats"] = {
            "mean": float(np.mean(preds)),
            "std": float(np.std(preds)),
            "values": preds.tolist(),
        }

    log_json_artifact(snapshot, artifact_path="lens", filename="drift_snapshot.json")


def _load_snapshot_features(run_id: str) -> pd.DataFrame:
    """Load feature data from a previous run's drift snapshot."""
    try:
        client = mlflow.tracking.MlflowClient()
        path = client.download_artifacts(run_id, "lens/drift_snapshot.json")
        snapshot = json.loads(Path(path).read_text())
        data = {}
        for col, stats in snapshot.get("feature_stats", {}).items():
            data[col] = stats["values"]
        return pd.DataFrame(data)
    except Exception:
        raise ValueError(
            f"No drift snapshot found for run {run_id}. "
            "Pass reference_features explicitly or ensure the reference run "
            "was created with log_drift()."
        )


def _load_snapshot_predictions(run_id: str) -> np.ndarray | None:
    try:
        client = mlflow.tracking.MlflowClient()
        path = client.download_artifacts(run_id, "lens/drift_snapshot.json")
        snapshot = json.loads(Path(path).read_text())
        pstats = snapshot.get("prediction_stats")
        if pstats and "values" in pstats:
            return np.array(pstats["values"])
    except Exception:
        pass
    return None
