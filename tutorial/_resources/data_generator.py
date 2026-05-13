# Databricks notebook source
"""
Synthetic inference data generator for Model Landscape tutorial.

Generates realistic production inference tables with baked-in drift patterns,
calibrated to cross Model Landscape's PSI thresholds at specific narrative moments.

Two scenarios:
  - Fraud detection (binary classification): gradual → sudden → recovery drift
  - Predictive maintenance (regression): seasonal + equipment aging drift

All generation is pure NumPy/Pandas — no model training, no external data.
Each scenario generates in under 5 seconds.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core engine: day-by-day generation with time-varying parameters
# ---------------------------------------------------------------------------


def _spread_timestamps(day: date, n: int, rng: np.random.Generator) -> list[datetime]:
    """Spread n timestamps across business hours of a single day."""
    base = datetime(day.year, day.month, day.day, 6, 0, 0)
    offsets = np.sort(rng.integers(0, 16 * 3600, size=n))  # 6am-10pm
    return [base + timedelta(seconds=int(s)) for s in offsets]


def _inject_nulls(
    df: pd.DataFrame,
    day_index: int,
    null_schedule: dict[str, Callable[[int], float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Set values to NaN according to per-feature null probability schedule."""
    for col, prob_fn in null_schedule.items():
        if col not in df.columns:
            continue
        prob = prob_fn(day_index)
        if prob <= 0:
            continue
        mask = rng.random(len(df)) < prob
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Scenario 1: Fraud Detection (Binary Classification)
# ---------------------------------------------------------------------------

# Drift functions: return (mean_shift, std_scale) for each feature by day
# Calibrated so PSI crosses warning (0.1) around day 18 and critical (0.25) around day 28

def _fraud_amount_shift(day: int) -> tuple[float, float]:
    """transaction_amount: gradual then sudden shift."""
    if day < 15:
        return 0.0, 1.0
    if day < 26:
        t = (day - 15) / 10
        return 0.3 * t, 1.0 + 0.1 * t
    if day < 36:
        t = (day - 26) / 10
        return 0.3 + 0.7 * t, 1.1 + 0.3 * t
    if day < 46:
        t = (day - 36) / 10
        return 1.0 - 0.4 * t, 1.4 - 0.2 * t
    return 0.6, 1.2  # new normal


def _fraud_distance_shift(day: int) -> tuple[float, float]:
    """distance_from_home_km: widens during drift period."""
    if day < 15:
        return 0.0, 1.0
    if day < 26:
        t = (day - 15) / 10
        return 0.2 * t, 1.0 + 0.15 * t
    if day < 36:
        return 0.2 + 0.5 * min(1.0, (day - 26) / 10), 1.15 + 0.35 * min(1.0, (day - 26) / 10)
    if day < 46:
        t = (day - 36) / 10
        return 0.7 - 0.3 * t, 1.5 - 0.2 * t
    return 0.4, 1.3


def _fraud_velocity_shift(day: int) -> tuple[float, float]:
    """velocity_24h: spikes during sudden shift."""
    if day < 26:
        return 0.0, 1.0
    if day < 36:
        t = (day - 26) / 10
        return 0.6 * t, 1.0 + 0.4 * t
    if day < 46:
        t = (day - 36) / 10
        return 0.6 - 0.45 * t, 1.4 - 0.3 * t
    return 0.15, 1.1


def _fraud_merchant_weights(day: int) -> dict[str, float]:
    """merchant_category distribution: travel surges during shift."""
    base = {"retail": 0.30, "online": 0.25, "dining": 0.20, "travel": 0.10, "grocery": 0.15}
    if day < 26:
        return base
    if day < 36:
        t = min(1.0, (day - 26) / 10)
        return {
            "retail": 0.30 - 0.10 * t,
            "online": 0.25 - 0.05 * t,
            "dining": 0.20 - 0.05 * t,
            "travel": 0.10 + 0.25 * t,
            "grocery": 0.15 - 0.05 * t,
        }
    if day < 46:
        t = (day - 36) / 10
        return {
            "retail": 0.20 + 0.05 * t,
            "online": 0.20 + 0.03 * t,
            "dining": 0.15 + 0.03 * t,
            "travel": 0.35 - 0.15 * t,
            "grocery": 0.10 + 0.04 * t,
        }
    return {"retail": 0.25, "online": 0.23, "dining": 0.18, "travel": 0.20, "grocery": 0.14}


def _fraud_region_weights(day: int) -> dict[str, float]:
    """region distribution: latam surges during shift."""
    base = {"na": 0.45, "eu": 0.30, "apac": 0.20, "latam": 0.05}
    if day < 26:
        return base
    if day < 36:
        t = min(1.0, (day - 26) / 10)
        return {
            "na": 0.45 - 0.15 * t,
            "eu": 0.30 - 0.05 * t,
            "apac": 0.20 - 0.05 * t,
            "latam": 0.05 + 0.25 * t,
        }
    if day < 46:
        t = (day - 36) / 10
        return {
            "na": 0.30 + 0.05 * t,
            "eu": 0.25 + 0.02 * t,
            "apac": 0.15 + 0.03 * t,
            "latam": 0.30 - 0.10 * t,
        }
    return {"na": 0.35, "eu": 0.27, "apac": 0.18, "latam": 0.20}


def generate_fraud_inference(
    n_days: int = 60,
    rows_per_day: int = 500,
    start_date: date | None = None,
    seed: int = 42,
    model_version: str = "1",
) -> pd.DataFrame:
    """Generate fraud detection inference table with baked-in drift narrative.

    Returns a DataFrame with columns matching the InferenceContract:
    event_ts, prediction, prediction_proba, entity_id, model_version,
    plus feature columns and categorical slice columns.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=n_days)

    # Baseline feature parameters (mean, std)
    baseline = {
        "transaction_amount": (150.0, 80.0),
        "device_trust_score": (0.72, 0.15),
        "distance_from_home_km": (25.0, 30.0),
        "velocity_24h": (3.0, 2.0),
        "account_age_days": (450.0, 300.0),
    }

    # Feature weights for the latent fraud signal (how each feature contributes)
    signal_weights = {
        "transaction_amount": 0.25,
        "device_trust_score": -0.35,
        "distance_from_home_km": 0.20,
        "velocity_24h": 0.30,
        "account_age_days": -0.10,
    }

    null_schedule = {
        "device_trust_score": lambda d: 0.02 if d < 30 or d > 45 else 0.02 + 0.10 * min(1.0, (d - 30) / 5),
    }

    shift_fns: dict[str, Callable[[int], tuple[float, float]]] = {
        "transaction_amount": _fraud_amount_shift,
        "distance_from_home_km": _fraud_distance_shift,
        "velocity_24h": _fraud_velocity_shift,
    }

    all_rows: list[pd.DataFrame] = []

    for day_idx in range(n_days):
        rng = np.random.default_rng(seed + day_idx)
        current_date = start_date + timedelta(days=day_idx)
        n = rows_per_day + rng.integers(-20, 20)

        # Generate numeric features
        features: dict[str, np.ndarray] = {}
        for feat, (mu, sigma) in baseline.items():
            shift_fn = shift_fns.get(feat)
            if shift_fn:
                mean_shift, std_scale = shift_fn(day_idx)
                features[feat] = rng.normal(mu + mean_shift * sigma, sigma * std_scale, size=n)
            else:
                features[feat] = rng.normal(mu, sigma, size=n)

        # Clip to realistic ranges
        features["transaction_amount"] = np.clip(features["transaction_amount"], 1.0, 5000.0)
        features["device_trust_score"] = np.clip(features["device_trust_score"], 0.0, 1.0)
        features["distance_from_home_km"] = np.clip(features["distance_from_home_km"], 0.0, 500.0)
        features["velocity_24h"] = np.clip(features["velocity_24h"], 0.0, 50.0)
        features["account_age_days"] = np.clip(features["account_age_days"], 1.0, 3000.0)

        # Derived features
        features["hour_of_day"] = rng.integers(0, 24, size=n).astype(float)
        features["is_weekend"] = rng.choice([0.0, 1.0], size=n, p=[5 / 7, 2 / 7])

        # Categorical features
        merchant_weights = _fraud_merchant_weights(day_idx)
        categories = list(merchant_weights.keys())
        probs = np.array([merchant_weights[c] for c in categories])
        probs = probs / probs.sum()
        merchant = rng.choice(categories, size=n, p=probs)

        region_weights = _fraud_region_weights(day_idx)
        region_cats = list(region_weights.keys())
        region_probs = np.array([region_weights[c] for c in region_cats])
        region_probs = region_probs / region_probs.sum()
        region = rng.choice(region_cats, size=n, p=region_probs)

        # Compute latent fraud signal from features
        signal = np.zeros(n)
        for feat, weight in signal_weights.items():
            mu, sigma = baseline[feat]
            normalized = (features[feat] - mu) / max(sigma, 1e-6)
            signal += weight * normalized

        # Add noise and convert to probability via sigmoid
        signal += rng.normal(0, 0.3, size=n)
        fraud_prob = 1.0 / (1.0 + np.exp(-signal))

        # During drift periods, model calibration degrades
        if 26 <= day_idx < 36:
            # Model sees unfamiliar patterns — predictions become less calibrated
            noise_scale = 0.1 * min(1.0, (day_idx - 26) / 5)
            fraud_prob += rng.normal(0, noise_scale, size=n)
            fraud_prob = np.clip(fraud_prob, 0.01, 0.99)

        prediction = (fraud_prob > 0.5).astype(int)

        day_df = pd.DataFrame({
            "event_ts": _spread_timestamps(current_date, n, rng),
            "prediction": prediction,
            "prediction_proba": np.round(fraud_prob, 4),
            "entity_id": [f"txn_{day_idx:03d}_{i:05d}" for i in range(n)],
            "model_version": model_version,
            "transaction_amount": np.round(features["transaction_amount"], 2),
            "merchant_category": merchant,
            "device_trust_score": np.round(features["device_trust_score"], 4),
            "distance_from_home_km": np.round(features["distance_from_home_km"], 1),
            "velocity_24h": np.round(features["velocity_24h"], 2),
            "hour_of_day": features["hour_of_day"].astype(int),
            "is_weekend": features["is_weekend"].astype(int),
            "account_age_days": features["account_age_days"].astype(int),
            "region": region,
        })

        day_df = _inject_nulls(day_df, day_idx, null_schedule, rng)
        all_rows.append(day_df)

    return pd.concat(all_rows, ignore_index=True)


def generate_fraud_labels(
    inference_df: pd.DataFrame,
    label_delay_days: tuple[int, int] = (2, 5),
    label_noise: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate delayed fraud labels from inference data.

    Labels are a noisy version of the true signal (not the prediction).
    Only rows whose labels have "arrived" (delay < days since event) are included.
    """
    rng = np.random.default_rng(seed + 9999)
    today = pd.Timestamp.now().normalize()

    event_ts = pd.to_datetime(inference_df["event_ts"])
    delays = rng.integers(label_delay_days[0], label_delay_days[1] + 1, size=len(inference_df))
    label_ts = event_ts + pd.to_timedelta(delays, unit="D")

    # Labels arrive: only include rows where label_ts <= today
    arrived_mask = label_ts <= today

    # True label = prediction with some noise (simulates imperfect model)
    predictions = inference_df["prediction"].values
    flip_mask = rng.random(len(predictions)) < label_noise
    labels = predictions.copy()
    labels[flip_mask] = 1 - labels[flip_mask]

    label_df = pd.DataFrame({
        "entity_id": inference_df["entity_id"].values,
        "label_timestamp": label_ts,
        "label": labels,
    })

    return label_df[arrived_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scenario 2: Predictive Maintenance (Regression)
# ---------------------------------------------------------------------------

def _maint_vibration_shift(day: int) -> tuple[float, float]:
    """vibration_mm_s: equipment aging causes rightward shift."""
    if day < 22:
        return 0.0, 1.0
    if day < 41:
        t = (day - 22) / 18
        return 0.15 * t, 1.0 + 0.05 * t
    if day < 56:
        t = (day - 41) / 14
        return 0.15 + 0.45 * t, 1.05 + 0.25 * t
    if day < 71:
        t = (day - 56) / 14
        return 0.60 - 0.15 * t, 1.30 - 0.05 * t
    return 0.45, 1.25


def _maint_temperature_shift(day: int) -> tuple[float, float]:
    """temperature_c: seasonal shift (summer onset)."""
    if day < 22:
        return 0.0, 1.0
    if day < 41:
        t = (day - 22) / 18
        return 0.4 * t, 1.0 + 0.1 * t  # +8C over baseline std of 20
    return 0.4, 1.1  # stays warm


def _maint_oil_viscosity_shift(day: int) -> tuple[float, float]:
    """oil_viscosity: drops as equipment ages."""
    if day < 41:
        return 0.0, 1.0
    if day < 56:
        t = (day - 41) / 14
        return -0.35 * t, 1.0 + 0.15 * t
    return -0.35, 1.15


def generate_maintenance_inference(
    n_days: int = 90,
    rows_per_day: int = 280,
    start_date: date | None = None,
    seed: int = 137,
) -> pd.DataFrame:
    """Generate predictive maintenance inference table with sensor drift.

    Regression problem: predict remaining useful life (RUL) in hours.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=n_days)

    baseline = {
        "vibration_mm_s": (4.5, 1.8),
        "temperature_c": (65.0, 20.0),
        "pressure_kpa": (320.0, 40.0),
        "rpm": (1500.0, 200.0),
        "oil_viscosity": (45.0, 8.0),
        "power_output_kw": (280.0, 50.0),
        "ambient_temp_c": (22.0, 5.0),
        "operating_hours_since_service": (500.0, 300.0),
        "load_factor": (0.65, 0.15),
    }

    shift_fns: dict[str, Callable[[int], tuple[float, float]]] = {
        "vibration_mm_s": _maint_vibration_shift,
        "temperature_c": _maint_temperature_shift,
        "oil_viscosity": _maint_oil_viscosity_shift,
    }

    # RUL signal weights (higher vibration/temp = lower RUL)
    rul_weights = {
        "vibration_mm_s": -80.0,
        "temperature_c": -15.0,
        "pressure_kpa": 2.0,
        "oil_viscosity": 20.0,
        "operating_hours_since_service": -1.5,
        "load_factor": -200.0,
    }

    equipment_ids = [f"turbine_{i:03d}" for i in range(12)]
    sites = ["plant_north", "plant_south", "plant_east"]
    equipment_classes = ["A", "B", "C"]
    equipment_site = {eid: sites[i % 3] for i, eid in enumerate(equipment_ids)}
    equipment_class = {eid: equipment_classes[i % 3] for i, eid in enumerate(equipment_ids)}

    null_schedule = {
        "oil_viscosity": lambda d: 1.0 if 45 <= d <= 55 else 0.0,  # applied conditionally per site
    }

    all_rows: list[pd.DataFrame] = []

    for day_idx in range(n_days):
        rng = np.random.default_rng(seed + day_idx)
        current_date = start_date + timedelta(days=day_idx)
        n = rows_per_day + rng.integers(-10, 10)

        # Assign equipment IDs round-robin
        eids = rng.choice(equipment_ids, size=n)
        site_arr = np.array([equipment_site[e] for e in eids])
        class_arr = np.array([equipment_class[e] for e in eids])

        features: dict[str, np.ndarray] = {}
        for feat, (mu, sigma) in baseline.items():
            shift_fn = shift_fns.get(feat)
            if shift_fn:
                mean_shift, std_scale = shift_fn(day_idx)
                features[feat] = rng.normal(mu + mean_shift * sigma, sigma * std_scale, size=n)
            else:
                features[feat] = rng.normal(mu, sigma, size=n)

        # Clip
        features["vibration_mm_s"] = np.clip(features["vibration_mm_s"], 0.5, 25.0)
        features["temperature_c"] = np.clip(features["temperature_c"], 10.0, 150.0)
        features["pressure_kpa"] = np.clip(features["pressure_kpa"], 100.0, 600.0)
        features["rpm"] = np.clip(features["rpm"], 500.0, 3000.0)
        features["oil_viscosity"] = np.clip(features["oil_viscosity"], 10.0, 80.0)
        features["power_output_kw"] = np.clip(features["power_output_kw"], 50.0, 500.0)
        features["ambient_temp_c"] = np.clip(features["ambient_temp_c"], 5.0, 45.0)
        features["operating_hours_since_service"] = np.clip(
            features["operating_hours_since_service"], 0.0, 3000.0
        )
        features["load_factor"] = np.clip(features["load_factor"], 0.1, 1.0)

        # Apply seasonal ambient temp shift alongside temperature_c
        if day_idx >= 22:
            t = min(1.0, (day_idx - 22) / 18)
            features["ambient_temp_c"] += 8.0 * t

        # Compute RUL prediction (noisy regression)
        rul_base = 2000.0
        rul_signal = np.full(n, rul_base)
        for feat, weight in rul_weights.items():
            mu, sigma = baseline[feat]
            normalized = (features[feat] - mu) / max(sigma, 1e-6)
            rul_signal += weight * normalized

        # Add noise (model imperfection increases during drift)
        noise_std = 40.0
        if 41 <= day_idx < 56:
            noise_std = 40.0 + 30.0 * min(1.0, (day_idx - 41) / 10)
        rul_signal += rng.normal(0, noise_std, size=n)
        prediction = np.clip(rul_signal, 10.0, 5000.0)

        day_df = pd.DataFrame({
            "event_ts": _spread_timestamps(current_date, n, rng),
            "prediction": np.round(prediction, 1),
            "entity_id": eids,
            "model_version": "1",
            "vibration_mm_s": np.round(features["vibration_mm_s"], 3),
            "temperature_c": np.round(features["temperature_c"], 1),
            "pressure_kpa": np.round(features["pressure_kpa"], 1),
            "rpm": np.round(features["rpm"], 0).astype(int),
            "oil_viscosity": np.round(features["oil_viscosity"], 1),
            "power_output_kw": np.round(features["power_output_kw"], 1),
            "ambient_temp_c": np.round(features["ambient_temp_c"], 1),
            "operating_hours_since_service": np.round(features["operating_hours_since_service"], 0).astype(int),
            "load_factor": np.round(features["load_factor"], 3),
            "equipment_class": class_arr,
            "site": site_arr,
        })

        # oil_viscosity null only for plant_east during sensor outage
        if 45 <= day_idx <= 55:
            east_mask = day_df["site"] == "plant_east"
            day_df.loc[east_mask, "oil_viscosity"] = np.nan

        all_rows.append(day_df)

    return pd.concat(all_rows, ignore_index=True)


def generate_maintenance_labels(
    inference_df: pd.DataFrame,
    failure_rate: float = 0.03,
    seed: int = 137,
) -> pd.DataFrame:
    """Generate sparse maintenance labels (only at failure events).

    Only ~3% of rows get labels (when equipment actually fails).
    Label = actual RUL at time of failure (noisy ground truth).
    """
    rng = np.random.default_rng(seed + 8888)

    # Sparse: only a fraction of rows have labels
    has_label = rng.random(len(inference_df)) < failure_rate
    subset = inference_df[has_label].copy()

    if len(subset) == 0:
        return pd.DataFrame(columns=["entity_id", "label_timestamp", "label"])

    # True RUL = prediction + noise (ground truth is close to but not identical to prediction)
    noise = rng.normal(0, 30, size=len(subset))
    true_rul = np.clip(subset["prediction"].values + noise, 0.0, 5000.0)

    # Labels arrive 1-3 days after event
    event_ts = pd.to_datetime(subset["event_ts"])
    delays = rng.integers(1, 4, size=len(subset))
    label_ts = event_ts + pd.to_timedelta(delays, unit="D")

    today = pd.Timestamp.now().normalize()
    arrived_mask = label_ts <= today

    label_df = pd.DataFrame({
        "entity_id": subset["entity_id"].values,
        "label_timestamp": label_ts,
        "label": np.round(true_rul, 1),
    })

    return label_df[arrived_mask].reset_index(drop=True)
