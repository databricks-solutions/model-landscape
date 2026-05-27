from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def reg_data():
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.3)
    model = LinearRegression().fit(X_tr, y_tr)
    return model, X_te, y_te, X_tr, y_tr


@pytest.fixture(scope="session")
def reg_predictions(reg_data):
    model, X_te, y_te, *_ = reg_data
    return np.asarray(y_te, dtype=float), model.predict(X_te)


@pytest.fixture(scope="session")
def ridge_estimator():
    return Ridge()
