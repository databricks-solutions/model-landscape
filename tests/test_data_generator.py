"""Tests for tutorial data generator — shape, column, and PSI calibration verification."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add demo source to path so we can import the generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tutorial"))

from _resources.data_generator import (
    FRAUD_BASELINE,
    generate_fraud_inference,
    generate_fraud_labels,
    generate_fraud_training_set,
    generate_maintenance_inference,
    generate_maintenance_labels,
)
from mlflow_lens.analytics.drift import compute_psi


# ---------------------------------------------------------------------------
# Reproducibility + train/serving consistency
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_inference_is_deterministic_under_fixture_now(self, monkeypatch) -> None:
        monkeypatch.setenv("MODEL_LANDSCAPE_FIXTURE_NOW", "2026-05-27")
        a = generate_fraud_inference(n_days=10, rows_per_day=100, seed=42)
        b = generate_fraud_inference(n_days=10, rows_per_day=100, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_labels_are_deterministic_under_fixture_now(self, monkeypatch) -> None:
        monkeypatch.setenv("MODEL_LANDSCAPE_FIXTURE_NOW", "2026-05-27")
        infer = generate_fraud_inference(n_days=10, rows_per_day=100, seed=42)
        a = generate_fraud_labels(infer, seed=42)
        b = generate_fraud_labels(infer, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_training_set_is_deterministic(self) -> None:
        a = generate_fraud_training_set(n=5_000, seed=42)
        b = generate_fraud_training_set(n=5_000, seed=42)
        pd.testing.assert_frame_equal(a, b)


class TestTrainingSet:
    @pytest.fixture(scope="class")
    def train_df(self) -> pd.DataFrame:
        return generate_fraud_training_set(n=20_000, seed=42)

    def test_required_columns(self, train_df: pd.DataFrame) -> None:
        required = {
            "transaction_id", "timestamp", "transaction_amount",
            "device_trust_score", "distance_from_home_km", "velocity_24h",
            "account_age_days", "hour_of_day", "is_weekend",
            "merchant_category", "region", "is_fraud",
        }
        assert required.issubset(train_df.columns)

    def test_fraud_rate_realistic(self, train_df: pd.DataFrame) -> None:
        rate = train_df["is_fraud"].mean()
        # Signal weights + noise yield ~3-10% positive rate
        assert 0.02 < rate < 0.15, f"unexpected fraud rate {rate:.3f}"

    def test_distribution_matches_inference_baseline(
        self, train_df: pd.DataFrame, monkeypatch
    ) -> None:
        """The whole point of this refactor: training distribution must match
        the inference baseline period so day-0 PSI is near-zero."""
        monkeypatch.setenv("MODEL_LANDSCAPE_FIXTURE_NOW", "2026-05-27")
        infer = generate_fraud_inference(n_days=14, rows_per_day=500, seed=42)
        # Compare numeric feature means within 10% of baseline target
        for feat, (mu, _) in FRAUD_BASELINE.items():
            train_mean = float(train_df[feat].mean())
            infer_mean = float(infer[feat].mean())
            assert abs(train_mean - infer_mean) / max(abs(mu), 1.0) < 0.15, (
                f"{feat}: train mean {train_mean:.2f} vs baseline-period infer "
                f"mean {infer_mean:.2f} — train/serving skew at baseline"
            )


# ---------------------------------------------------------------------------
# Fraud scenario tests
# ---------------------------------------------------------------------------


class TestFraudInference:
    @pytest.fixture(scope="class")
    def fraud_df(self) -> pd.DataFrame:
        return generate_fraud_inference(n_days=60, rows_per_day=500, seed=42)

    def test_shape(self, fraud_df: pd.DataFrame) -> None:
        assert len(fraud_df) > 25000
        assert len(fraud_df) < 35000

    def test_required_columns(self, fraud_df: pd.DataFrame) -> None:
        required = {"event_ts", "prediction", "prediction_proba", "entity_id", "model_version"}
        assert required.issubset(fraud_df.columns)

    def test_feature_columns(self, fraud_df: pd.DataFrame) -> None:
        features = {
            "transaction_amount", "merchant_category", "device_trust_score",
            "distance_from_home_km", "velocity_24h", "hour_of_day",
            "is_weekend", "account_age_days", "region",
        }
        assert features.issubset(fraud_df.columns)

    def test_time_range(self, fraud_df: pd.DataFrame) -> None:
        ts = pd.to_datetime(fraud_df["event_ts"])
        assert (ts.max() - ts.min()).days >= 58

    def test_prediction_values(self, fraud_df: pd.DataFrame) -> None:
        assert fraud_df["prediction"].isin([0, 1]).all()
        proba = fraud_df["prediction_proba"]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_categorical_values(self, fraud_df: pd.DataFrame) -> None:
        assert set(fraud_df["merchant_category"].dropna().unique()) == {
            "retail", "online", "dining", "travel", "grocery"
        }
        assert set(fraud_df["region"].dropna().unique()) == {"na", "eu", "apac", "latam"}

    def test_device_trust_null_spike(self, fraud_df: pd.DataFrame) -> None:
        """device_trust_score should have elevated nulls during days 30-45."""
        fraud_df = fraud_df.copy()
        fraud_df["day"] = pd.to_datetime(fraud_df["event_ts"]).dt.date
        days_sorted = sorted(fraud_df["day"].unique())

        early_nulls = fraud_df[fraud_df["day"].isin(days_sorted[:14])]["device_trust_score"].isna().mean()
        spike_nulls = fraud_df[fraud_df["day"].isin(days_sorted[32:42])]["device_trust_score"].isna().mean()

        assert spike_nulls > early_nulls * 2, f"Expected null spike: early={early_nulls:.3f}, spike={spike_nulls:.3f}"


class TestFraudLabels:
    def test_labels_generated(self) -> None:
        inference_df = generate_fraud_inference(n_days=60, rows_per_day=100, seed=42)
        labels_df = generate_fraud_labels(inference_df, seed=42)
        assert len(labels_df) > 0
        assert set(labels_df.columns) == {"entity_id", "label_timestamp", "label"}
        assert labels_df["label"].isin([0, 1]).all()

    def test_labels_delayed(self) -> None:
        inference_df = generate_fraud_inference(n_days=60, rows_per_day=100, seed=42)
        labels_df = generate_fraud_labels(inference_df, seed=42)
        # All label timestamps should be after the corresponding event
        merged = labels_df.merge(
            inference_df[["entity_id", "event_ts"]], on="entity_id", how="left"
        )
        assert (pd.to_datetime(merged["label_timestamp"]) >= pd.to_datetime(merged["event_ts"])).all()


class TestFraudPSICalibration:
    """Verify that drift magnitudes produce PSI values that cross thresholds at the right times."""

    @pytest.fixture(scope="class")
    def fraud_df(self) -> pd.DataFrame:
        return generate_fraud_inference(n_days=60, rows_per_day=500, seed=42)

    @pytest.fixture(scope="class")
    def daily_data(self, fraud_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
        fraud_df = fraud_df.copy()
        fraud_df["day_idx"] = (
            pd.to_datetime(fraud_df["event_ts"]).dt.date - pd.to_datetime(fraud_df["event_ts"]).dt.date.min()
        ).apply(lambda d: d.days)
        return {day: group for day, group in fraud_df.groupby("day_idx")}

    def _baseline_window(self, daily_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Days 0-6 as baseline reference."""
        return pd.concat([daily_data[d] for d in range(7) if d in daily_data])

    def _compute_feature_psi(
        self, baseline: pd.DataFrame, current: pd.DataFrame, feature: str
    ) -> float:
        ref = baseline[feature].dropna().to_numpy(dtype=float)
        cur = current[feature].dropna().to_numpy(dtype=float)
        if len(ref) < 10 or len(cur) < 10:
            return 0.0
        return compute_psi(ref, cur)

    def test_baseline_psi_low(self, daily_data: dict[int, pd.DataFrame]) -> None:
        """During baseline period (days 7-14), PSI should be low (< 0.1)."""
        baseline = self._baseline_window(daily_data)
        comparison = pd.concat([daily_data[d] for d in range(7, 14) if d in daily_data])
        for feat in ["transaction_amount", "distance_from_home_km", "velocity_24h"]:
            psi = self._compute_feature_psi(baseline, comparison, feat)
            assert psi < 0.1, f"Baseline PSI too high for {feat}: {psi:.4f}"

    def test_sudden_drift_psi_critical(self, daily_data: dict[int, pd.DataFrame]) -> None:
        """During sudden shift (days 28-35), at least one feature should cross critical (0.25)."""
        baseline = self._baseline_window(daily_data)
        comparison = pd.concat([daily_data[d] for d in range(28, 35) if d in daily_data])
        psi_values = {}
        for feat in ["transaction_amount", "distance_from_home_km", "velocity_24h"]:
            psi_values[feat] = self._compute_feature_psi(baseline, comparison, feat)
        max_psi = max(psi_values.values())
        assert max_psi >= 0.25, f"No feature crossed critical threshold during sudden drift: {psi_values}"

    def test_recovery_psi_decreases(self, daily_data: dict[int, pd.DataFrame]) -> None:
        """During recovery (days 46-55), PSI should be lower than peak drift."""
        baseline = self._baseline_window(daily_data)
        peak = pd.concat([daily_data[d] for d in range(28, 35) if d in daily_data])
        recovery = pd.concat([daily_data[d] for d in range(46, 55) if d in daily_data])
        for feat in ["transaction_amount", "velocity_24h"]:
            peak_psi = self._compute_feature_psi(baseline, peak, feat)
            recovery_psi = self._compute_feature_psi(baseline, recovery, feat)
            assert recovery_psi < peak_psi, (
                f"{feat}: recovery PSI ({recovery_psi:.4f}) should be < peak ({peak_psi:.4f})"
            )


# ---------------------------------------------------------------------------
# Maintenance scenario tests
# ---------------------------------------------------------------------------


class TestMaintenanceInference:
    @pytest.fixture(scope="class")
    def maint_df(self) -> pd.DataFrame:
        return generate_maintenance_inference(n_days=90, rows_per_day=280, seed=137)

    def test_shape(self, maint_df: pd.DataFrame) -> None:
        assert len(maint_df) > 20000
        assert len(maint_df) < 30000

    def test_required_columns(self, maint_df: pd.DataFrame) -> None:
        required = {"event_ts", "prediction", "entity_id", "model_version"}
        assert required.issubset(maint_df.columns)

    def test_feature_columns(self, maint_df: pd.DataFrame) -> None:
        features = {
            "vibration_mm_s", "temperature_c", "pressure_kpa", "rpm",
            "oil_viscosity", "power_output_kw", "ambient_temp_c",
            "operating_hours_since_service", "load_factor",
            "equipment_class", "site",
        }
        assert features.issubset(maint_df.columns)

    def test_prediction_range(self, maint_df: pd.DataFrame) -> None:
        """RUL predictions should be positive and bounded."""
        assert maint_df["prediction"].min() >= 10.0
        assert maint_df["prediction"].max() <= 5000.0

    def test_oil_viscosity_null_for_plant_east(self, maint_df: pd.DataFrame) -> None:
        """oil_viscosity should be 100% null for plant_east during days 45-55."""
        maint_df = maint_df.copy()
        start = pd.to_datetime(maint_df["event_ts"]).dt.date.min()
        maint_df["day_idx"] = (pd.to_datetime(maint_df["event_ts"]).dt.date - start).apply(lambda d: d.days)

        outage = maint_df[(maint_df["day_idx"] >= 45) & (maint_df["day_idx"] <= 55) & (maint_df["site"] == "plant_east")]
        assert outage["oil_viscosity"].isna().mean() == 1.0, "plant_east oil_viscosity should be fully null during outage"

        # Other sites should be fine
        other = maint_df[(maint_df["day_idx"] >= 45) & (maint_df["day_idx"] <= 55) & (maint_df["site"] != "plant_east")]
        assert other["oil_viscosity"].isna().mean() < 0.05

    def test_equipment_ids(self, maint_df: pd.DataFrame) -> None:
        assert maint_df["entity_id"].str.startswith("turbine_").all()

    def test_categorical_values(self, maint_df: pd.DataFrame) -> None:
        assert set(maint_df["equipment_class"].unique()) == {"A", "B", "C"}
        assert set(maint_df["site"].unique()) == {"plant_north", "plant_south", "plant_east"}


class TestMaintenancePSICalibration:
    @pytest.fixture(scope="class")
    def maint_df(self) -> pd.DataFrame:
        return generate_maintenance_inference(n_days=90, rows_per_day=280, seed=137)

    @pytest.fixture(scope="class")
    def daily_data(self, maint_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
        maint_df = maint_df.copy()
        start = pd.to_datetime(maint_df["event_ts"]).dt.date.min()
        maint_df["day_idx"] = (pd.to_datetime(maint_df["event_ts"]).dt.date - start).apply(lambda d: d.days)
        return {day: group for day, group in maint_df.groupby("day_idx")}

    def _baseline(self, daily_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
        return pd.concat([daily_data[d] for d in range(7) if d in daily_data])

    def test_vibration_drifts_during_aging(self, daily_data: dict[int, pd.DataFrame]) -> None:
        """vibration_mm_s should show significant drift during equipment aging (days 45-55)."""
        baseline = self._baseline(daily_data)
        aging = pd.concat([daily_data[d] for d in range(45, 55) if d in daily_data])
        ref = baseline["vibration_mm_s"].dropna().to_numpy(dtype=float)
        cur = aging["vibration_mm_s"].dropna().to_numpy(dtype=float)
        psi = compute_psi(ref, cur)
        assert psi >= 0.1, f"vibration PSI during aging should be >= warning: {psi:.4f}"

    def test_baseline_stable(self, daily_data: dict[int, pd.DataFrame]) -> None:
        """Early period should show low PSI."""
        baseline = self._baseline(daily_data)
        early = pd.concat([daily_data[d] for d in range(7, 14) if d in daily_data])
        for feat in ["vibration_mm_s", "temperature_c", "oil_viscosity"]:
            ref = baseline[feat].dropna().to_numpy(dtype=float)
            cur = early[feat].dropna().to_numpy(dtype=float)
            psi = compute_psi(ref, cur)
            assert psi < 0.1, f"Early PSI too high for {feat}: {psi:.4f}"


class TestMaintenanceLabels:
    def test_sparse_labels(self) -> None:
        inference_df = generate_maintenance_inference(n_days=90, rows_per_day=100, seed=137)
        labels_df = generate_maintenance_labels(inference_df, seed=137)
        # Should be sparse: roughly 3% of rows
        ratio = len(labels_df) / len(inference_df)
        assert 0.01 < ratio < 0.10, f"Label ratio should be ~3%: got {ratio:.3f}"
        assert (labels_df["label"] >= 0).all()
