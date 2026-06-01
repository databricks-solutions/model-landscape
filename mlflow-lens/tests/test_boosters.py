"""End-to-end smoke tests for non-sklearn ML packages.

mlflow-lens panels are meant to work against:
* sklearn-compatible estimators (the common case)
* low-level XGBoost ``Booster`` (from ``xgb.train``)
* low-level LightGBM ``Booster`` (from ``lgb.train``)

These tests confirm a Booster instance flows through the classifier and
regressor panels end-to-end without the user having to drop down to
``.from_scores`` / ``.from_predictions``.

Both libraries are part of mlflow-lens' ``[dev]`` extras so the tests
run by default; if either is missing the tests skip cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split

xgb = pytest.importorskip("xgboost")
lgb = pytest.importorskip("lightgbm")

# Importing the panels has to come *after* importorskip so the module-level
# skip kicks in cleanly on environments without xgboost / lightgbm.
from mlflow_lens.classifier import (  # noqa: E402
    class_prediction_error,
    classification_report,
    confusion_matrix,
    precision_recall,
    roc_auc,
)
from mlflow_lens.regressor import prediction_error, residuals  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures — train Boosters once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def xgb_binary_booster():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.3)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    booster = xgb.train(
        {"objective": "binary:logistic", "verbosity": 0},
        dtrain,
        num_boost_round=20,
    )
    return booster, X_te, y_te


@pytest.fixture(scope="session")
def xgb_multiclass_booster():
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.3)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": 3, "verbosity": 0},
        dtrain,
        num_boost_round=20,
    )
    return booster, X_te, y_te


@pytest.fixture(scope="session")
def xgb_regression_booster():
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.3)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    booster = xgb.train(
        {"objective": "reg:squarederror", "verbosity": 0},
        dtrain,
        num_boost_round=30,
    )
    return booster, X_te, y_te, X_tr, y_tr


@pytest.fixture(scope="session")
def lgb_binary_booster():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.3)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1},
        dtrain,
        num_boost_round=20,
    )
    return booster, X_te, y_te


@pytest.fixture(scope="session")
def lgb_regression_booster():
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.3)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    booster = lgb.train(
        {"objective": "regression", "verbose": -1},
        dtrain,
        num_boost_round=30,
    )
    return booster, X_te, y_te


# ---------------------------------------------------------------------------
# XGBoost Booster — classification
# ---------------------------------------------------------------------------


def test_xgb_booster_roc_auc(xgb_binary_booster):
    booster, X, y = xgb_binary_booster
    fig = roc_auc(booster, X, y)
    assert len(fig.data) >= 2  # at least one ROC trace + chance diagonal


def test_xgb_booster_multiclass_roc_auc(xgb_multiclass_booster):
    booster, X, y = xgb_multiclass_booster
    fig = roc_auc(booster, X, y)
    # 3 per-class + micro + diagonal
    assert len(fig.data) == 5


def test_xgb_booster_confusion_matrix(xgb_binary_booster):
    booster, X, y = xgb_binary_booster
    fig = confusion_matrix(booster, X, y)
    cm = fig.data[0].z
    # row sums must equal class counts in the test split
    assert sum(cm[0]) + sum(cm[1]) == len(y)


def test_xgb_booster_classification_report(xgb_multiclass_booster):
    booster, X, y = xgb_multiclass_booster
    fig = classification_report(booster, X, y)
    assert fig.data[0].type == "heatmap"


def test_xgb_booster_class_prediction_error(xgb_multiclass_booster):
    booster, X, y = xgb_multiclass_booster
    fig = class_prediction_error(booster, X, y)
    assert all(t.type == "bar" for t in fig.data)


def test_xgb_booster_precision_recall(xgb_binary_booster):
    booster, X, y = xgb_binary_booster
    fig = precision_recall(booster, X, y)
    assert len(fig.data) >= 1


def test_xgb_booster_logs_artifacts(experiment_id, xgb_binary_booster):
    mlflow.set_experiment(experiment_id=experiment_id)
    booster, X, y = xgb_binary_booster
    with mlflow.start_run() as run:
        roc_auc(booster, X, y, log=True)
        confusion_matrix(booster, X, y, log=True, labels=[0, 1])

    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(run.info.run_id, "lens/panels/roc_curve.json")
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "roc_curve"
    assert payload["data"]["n_classes"] == 2


# ---------------------------------------------------------------------------
# XGBoost Booster — regression
# ---------------------------------------------------------------------------


def test_xgb_booster_prediction_error(xgb_regression_booster):
    booster, X, y, _, _ = xgb_regression_booster
    fig = prediction_error(booster, X, y)
    # scatter + best-fit + identity
    assert len(fig.data) == 3


def test_xgb_booster_residuals_with_train_overlay(xgb_regression_booster):
    booster, X_te, y_te, X_tr, y_tr = xgb_regression_booster
    fig = residuals(booster, X_te, y_te, X_train=X_tr, y_train=y_tr)
    assert len(fig.data) == 2  # train + test


# ---------------------------------------------------------------------------
# LightGBM Booster
# ---------------------------------------------------------------------------


def test_lgb_booster_roc_auc(lgb_binary_booster):
    booster, X, y = lgb_binary_booster
    fig = roc_auc(booster, X, y)
    assert len(fig.data) >= 2


def test_lgb_booster_confusion_matrix(lgb_binary_booster):
    booster, X, y = lgb_binary_booster
    fig = confusion_matrix(booster, X, y)
    cm = fig.data[0].z
    assert sum(cm[0]) + sum(cm[1]) == len(y)


def test_lgb_booster_prediction_error(lgb_regression_booster):
    booster, X, y = lgb_regression_booster
    fig = prediction_error(booster, X, y)
    assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# Unsupported model types still emit a useful error
# ---------------------------------------------------------------------------


class _BareModel:
    """No predict, no predict_proba, no decision_function, not a Booster."""


def test_unsupported_model_gives_actionable_error():
    with pytest.raises(AttributeError, match="from_scores"):
        roc_auc(_BareModel(), np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_unsupported_regressor_gives_actionable_error():
    with pytest.raises(AttributeError, match="from_predictions"):
        prediction_error(_BareModel(), np.zeros((4, 2)), np.array([1.0, 2.0, 3.0, 4.0]))
