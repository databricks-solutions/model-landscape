from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

import mlflow_lens
import model_landscape


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.2.0"


def test_product_versions_align() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    assert payload["project"]["version"] == EXPECTED_VERSION
    assert model_landscape.__version__ == EXPECTED_VERSION
    assert mlflow_lens.__version__ == EXPECTED_VERSION

    for distribution in ("model-landscape", "mlflow-lens"):
        try:
            installed_version = version(distribution)
        except PackageNotFoundError:
            installed_version = None

        if installed_version is not None:
            assert installed_version == EXPECTED_VERSION
