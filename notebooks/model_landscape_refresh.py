"""Scheduled refresh job entrypoint for Model Landscape.

Deployed by the DABs bundle as a python_file task on a cron schedule.
Runs the refresh cycle: computes drift metrics, quality profiles,
performance metrics, incidents, and comparison windows for all active
monitors from their source inference tables.

Invoked by:  ``databricks bundle run refresh_control_plane`` (refresh task)
See also:    model_landscape/workflows/refresh_job.py
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
    from model_landscape.workflows.refresh_job import main as refresh_main

    return int(refresh_main() or 0)

if __name__ == "__main__":
    main()
