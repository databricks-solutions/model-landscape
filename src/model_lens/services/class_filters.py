from __future__ import annotations

from typing import Any

import pandas as pd


POSITIVE_CLASS_TOKENS = frozenset({"1", "1.0", "true", "t", "yes", "y", "positive", "pos"})
NEGATIVE_CLASS_TOKENS = frozenset({"0", "0.0", "false", "f", "no", "n", "negative", "neg"})


def normalize_class_filter(class_basis: str | None, class_value: str | None) -> tuple[str, str, bool]:
    normalized_basis = str(class_basis or "all").strip().lower()
    normalized_value = str(class_value or "all").strip().lower()
    active = normalized_basis in {"actual", "predicted"} and normalized_value in {"positive", "negative"}
    return normalized_basis, normalized_value, active


def normalize_binary_value(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
    text = str(value).strip().lower()
    if not text:
        return None
    if text in POSITIVE_CLASS_TOKENS:
        return 1
    if text in NEGATIVE_CLASS_TOKENS:
        return 0
    return None


def normalized_binary_series(values: pd.Series) -> pd.Series:
    return values.map(normalize_binary_value).astype("Int64")


def resolved_prediction_binary_series(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    prediction_score_col: str | None = None,
) -> pd.Series:
    discrete = normalized_binary_series(frame[prediction_col])
    if discrete.notna().all():
        return discrete
    result = discrete.copy()
    if prediction_score_col and prediction_score_col in frame.columns:
        scores = pd.to_numeric(frame[prediction_score_col], errors="coerce")
        valid = result.isna() & scores.notna() & scores.between(0.0, 1.0, inclusive="both")
        if valid.any():
            result.loc[valid] = (scores.loc[valid] >= 0.5).astype(int).astype("Int64")
        return result
    numeric = pd.to_numeric(frame[prediction_col], errors="coerce")
    valid = result.isna() & numeric.notna() & numeric.between(0.0, 1.0, inclusive="both")
    if valid.any():
        result.loc[valid] = (numeric.loc[valid] >= 0.5).astype(int).astype("Int64")
    return result


def supports_binary_class_filters(config: object | None) -> bool:
    if config is None:
        return False
    contract = getattr(config, "contract", None)
    label_col = getattr(contract, "label_col", None)
    prediction_col = getattr(contract, "prediction_col", None)
    prediction_score_col = getattr(contract, "prediction_score_col", None)
    problem_type = str(getattr(config, "problem_type", "classification") or "classification").strip().lower()
    return bool(label_col) and bool(prediction_col or prediction_score_col) and problem_type == "classification"
