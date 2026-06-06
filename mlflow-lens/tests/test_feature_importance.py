from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

pytest.importorskip("shap")

from mlflow_lens import feature_importance  # noqa: E402
from mlflow_lens.feature_importance import _is_tree_model  # noqa: E402


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    names = [f"f{i}" for i in range(X.shape[1])]
    return X, y, names


@pytest.fixture(scope="module")
def fitted_rf(data):
    X, y, _ = data
    return RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)


def test_autodetect_tree_vs_kernel(data, fitted_rf):
    X, y, _ = data
    lr = LogisticRegression(max_iter=500).fit(X, y)
    assert _is_tree_model(fitted_rf) is True
    assert _is_tree_model(lr) is False


def test_tree_explainer_multiclass(data, fitted_rf):
    X, _, names = data
    fig = feature_importance.shap_importance(fitted_rf, X, feature_names=names, max_samples=40)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    xs = list(fig.data[0].x)
    assert len(xs) == len(names)
    assert all(v >= 0 for v in xs)  # mean |SHAP| is non-negative
    assert xs == sorted(xs, reverse=True)  # sorted descending


def test_kernel_explainer_fallback(data):
    X, y, names = data
    lr = LogisticRegression(max_iter=500).fit(X, y)
    fig = feature_importance.shap_importance(
        lr, X, feature_names=names, max_samples=10, background_samples=20, nsamples=64
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == len(names)


def test_from_shap_values_handles_list_and_3d():
    names = ["a", "b", "c"]
    # per-class list of (n_samples, n_features)
    sv_list = [np.ones((5, 3)), np.full((5, 3), 2.0)]
    fig = feature_importance.shap_importance.from_shap_values(names, sv_list, top_n=2)
    assert len(fig.data[0].y) == 2
    # 3-D (n_samples, n_features, n_classes)
    sv_3d = np.ones((5, 3, 2))
    fig2 = feature_importance.shap_importance.from_shap_values(names, sv_3d)
    assert len(fig2.data[0].y) == 3


def test_explicit_explainer_and_top_n(data, fitted_rf):
    X, _, names = data
    fig = feature_importance.shap_importance(
        fitted_rf, X, feature_names=names, explainer="tree", top_n=3
    )
    assert len(fig.data[0].x) == 3


def test_array_without_feature_names_raises(data, fitted_rf):
    X, _, _ = data
    with pytest.raises(ValueError):
        feature_importance.shap_importance(fitted_rf, X, max_samples=10)


def test_shap_importance_logs_artifacts(experiment_id, data, fitted_rf):
    X, _, names = data
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run() as run:
        feature_importance.shap_importance(
            fitted_rf, X, feature_names=names, max_samples=20, top_n=4, log=True
        )
    client = mlflow.tracking.MlflowClient()
    json_path = client.download_artifacts(
        run.info.run_id, "lens/panels/feature_importance.json"
    )
    payload = json.loads(Path(json_path).read_text())
    assert payload["type"] == "feature_importance"
    assert payload["top_n"] == 4
    assert len(payload["data"]) == 4
    client.download_artifacts(run.info.run_id, "lens/panels/feature_importance.html")
