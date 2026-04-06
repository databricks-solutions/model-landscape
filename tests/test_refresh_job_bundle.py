from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundle_refresh_job_uses_named_parameters() -> None:
    text = (REPO_ROOT / "resources" / "jobs.yml").read_text()

    assert "named_parameters:" in text
    assert "scope: scheduler" in text
    assert "python_wheel_task:" in text
