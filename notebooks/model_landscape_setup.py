"""Control plane setup entrypoint for the Model Landscape refresh job.

Deployed by the DABs bundle as a python_file task.  Creates or migrates
the control plane tables (monitor_configs, drift_metrics, quality_metrics,
refresh_runs, etc.) in the configured Unity Catalog namespace.

Invoked by:  ``databricks bundle run refresh_control_plane`` (setup task)
See also:    model_landscape/workflows/setup_control_plane.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_repo_root() -> None:
    repo_root = Path.cwd()
    if not (repo_root / "model_landscape").is_dir():
        raise RuntimeError(
            "Run this helper from the Model Landscape repo root or set "
            "PYTHONPATH to the repo root."
        )
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def main() -> int:
    _bootstrap_repo_root()
    from model_landscape.workflows.setup_control_plane import main as setup_main

    return int(setup_main() or 0)

if __name__ == "__main__":
    main()
