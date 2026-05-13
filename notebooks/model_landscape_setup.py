"""Control plane setup entrypoint for the Model Landscape refresh job.

Deployed by the DABs bundle as a python_file task.  Creates or migrates
the control plane tables (monitor_configs, drift_metrics, quality_metrics,
refresh_runs, etc.) in the configured Unity Catalog namespace.

Invoked by:  ``databricks bundle run refresh_control_plane`` (setup task)
See also:    src/model_landscape/workflows/setup_control_plane.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_repo_src() -> None:
    src = Path.cwd() / "src"
    if not src.is_dir():
        raise RuntimeError("Run this helper from the Model Landscape repo root or set PYTHONPATH=src.")
    src_path = str(src)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> int:
    _bootstrap_repo_src()
    from model_landscape.workflows.setup_control_plane import main as setup_main

    return int(setup_main() or 0)

if __name__ == "__main__":
    main()
