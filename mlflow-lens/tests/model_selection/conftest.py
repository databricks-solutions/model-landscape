from __future__ import annotations

import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="session")
def cls_dataset():
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer().feature_names.tolist()
    return X, y, feature_names


@pytest.fixture(scope="session")
def fitted_rf(cls_dataset):
    X, y, _ = cls_dataset
    return RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
