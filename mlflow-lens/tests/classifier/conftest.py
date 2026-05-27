from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def binary_data():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.3)
    model = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    return model, X_te, y_te


@pytest.fixture(scope="session")
def multiclass_data():
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.3)
    model = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    return model, X_te, y_te


@pytest.fixture(scope="session")
def binary_scores(binary_data):
    model, X_te, y_te = binary_data
    return np.asarray(y_te), model.predict_proba(X_te)


@pytest.fixture(scope="session")
def binary_predictions(binary_data):
    model, X_te, y_te = binary_data
    return np.asarray(y_te), model.predict(X_te)
