from mlflow_lens._compat import has_logged_models, mlflow_major


def test_mlflow_major():
    assert mlflow_major() >= 2


def test_has_logged_models_returns_bool():
    result = has_logged_models()
    assert isinstance(result, bool)
