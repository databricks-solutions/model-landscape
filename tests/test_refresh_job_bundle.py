from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundle_refresh_job_uses_job_parameters_and_pushdown() -> None:
    text = (REPO_ROOT / "resources" / "jobs.yml").read_text()

    assert "parameters:" in text
    assert "named_parameters:" in text
    assert 'scope: "{{job.parameters.scope}}"' in text
    assert 'model-key: "{{job.parameters.model_key}}"' in text
    assert "default: scheduler" in text
    assert "python_wheel_task:" in text
    assert "job_clusters:" in text
    assert "job_cluster_key: refresh_compute" in text
    assert "timeout_seconds: ${var.refresh_timeout_seconds}" in text
