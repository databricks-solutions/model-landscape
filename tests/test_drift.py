from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from model_landscape.analytics.drift import (
    compute_feature_drift,
    compute_js,
    compute_kl,
    compute_psi,
)


def test_numeric_drift_metrics_remain_finite_when_current_values_fall_outside_reference_range() -> (
    None
):
    reference = np.linspace(0.0, 10.0, 101)
    current = np.linspace(100.0, 110.0, 101)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        psi = compute_psi(reference, current)
        kl = compute_kl(reference, current)
        js = compute_js(reference, current)

    assert np.isfinite(psi)
    assert np.isfinite(kl)
    assert np.isfinite(js)
    assert psi > 0.0
    assert kl > 0.0
    assert js > 0.0


def test_compute_feature_drift_handles_sparse_numeric_windows_without_runtime_warnings() -> None:
    reference_df = pd.DataFrame({"amount": [0.0, 1.0]})
    current_df = pd.DataFrame({"amount": [10.0, 11.0]})

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        drift = compute_feature_drift(reference_df, current_df, ["amount"])

    assert list(drift["feature_name"]) == ["amount"]
    assert np.isfinite(float(drift.iloc[0]["psi"]))
    assert np.isfinite(float(drift.iloc[0]["kl_divergence"]))
    assert np.isfinite(float(drift.iloc[0]["js_divergence"]))


def test_numeric_drift_metrics_return_nan_only_when_no_comparable_data_exists() -> None:
    assert np.isnan(compute_psi(np.array([np.nan, np.nan]), np.array([1.0, 2.0])))
    assert np.isnan(compute_kl(np.array([1.0, 2.0]), np.array([np.nan, np.nan])))
    assert np.isnan(compute_js(np.array([]), np.array([1.0, 2.0])))

    drift = compute_feature_drift(
        pd.DataFrame({"amount": [np.nan, np.nan]}),
        pd.DataFrame({"amount": [1.0, np.nan]}),
        ["amount"],
    )

    assert drift.empty
