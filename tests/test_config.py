from __future__ import annotations

import importlib
import logging

import model_landscape.config as config_module


def _reload_config():
    return importlib.reload(config_module)


def test_settings_invalid_integer_env_falls_back_without_import_error(monkeypatch, caplog) -> None:
    monkeypatch.setenv("DATABRICKS_APP_PORT", "not-a-number")
    caplog.set_level(logging.WARNING, logger="model_landscape.config")

    module = _reload_config()

    assert module.settings.app_port == 8080
    assert "DATABRICKS_APP_PORT='not-a-number' is not a valid integer" in caplog.text
    monkeypatch.delenv("DATABRICKS_APP_PORT", raising=False)
    _reload_config()


def test_settings_below_minimum_env_falls_back_without_import_error(monkeypatch, caplog) -> None:
    monkeypatch.setenv("REFRESH_STALE_RUN_MINUTES", "0")
    caplog.set_level(logging.WARNING, logger="model_landscape.config")

    module = _reload_config()

    assert module.settings.refresh_stale_run_minutes == 75
    assert "REFRESH_STALE_RUN_MINUTES='0' is below minimum 1" in caplog.text
    monkeypatch.delenv("REFRESH_STALE_RUN_MINUTES", raising=False)
    _reload_config()
