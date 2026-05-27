"""Build a deployable source tree for workspaces that cannot use DABs.

Alternative deployment path for constrained environments where an admin
pre-creates the Databricks App and refresh job manually.  Outputs:

  <output-dir>/
    app.yaml            -- pre-filled with warehouse, catalog, schema, job IDs
    src/                -- application source code
    dist/*.whl          -- built wheel for the refresh job environment
    refresh-job.json    -- (optional) REST API payload for job creation

Usage::

    uv run python notebooks/prepare_existing_app_source.py \\
        --app-name model-landscape \\
        --sql-warehouse-id <id> \\
        --control-plane-catalog <catalog> \\
        --control-plane-schema <schema> \\
        --output-dir /tmp/model-landscape-deploy

See also:  docs/existing_app.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_landscape.manual_setup import (
    MANUAL_SOURCE_ITEMS,
    ManualAppSettings,
    ManualRefreshJobSettings,
    build_manual_app_yaml,
    build_manual_refresh_job_payload,
    find_built_wheel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a deployable source tree for an existing Databricks App and "
            "optionally emit a refresh job payload that avoids bundle-managed app resources."
        )
    )
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--sql-warehouse-id", required=True)
    parser.add_argument("--control-plane-catalog", required=True)
    parser.add_argument("--control-plane-schema", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refresh-job-id", default="")
    parser.add_argument("--refresh-job-name", default="")
    parser.add_argument("--bootstrap-refresh-job-id", default="")
    parser.add_argument("--bootstrap-refresh-job-name", default="")
    parser.add_argument("--workspace-source-path", default="")
    parser.add_argument("--lakebase-instance-name", default="")
    parser.add_argument("--lakebase-database-name", default="")
    parser.add_argument("--lakebase-pguser", default="")
    parser.add_argument("--lakebase-schema", default="model_landscape_ui")
    parser.add_argument("--genie-space-id", default="")
    parser.add_argument(
        "--use-lakebase-read-model",
        action="store_true",
        help="Emit the optional Lakebase flags in the generated refresh job payload.",
    )
    return parser.parse_args()


def _copy_source_items(output_dir: Path) -> None:
    for relative in MANUAL_SOURCE_ITEMS:
        source = REPO_ROOT / relative
        target = output_dir / relative
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _write_manual_app_yaml(output_dir: Path, args: argparse.Namespace) -> None:
    settings = ManualAppSettings(
        app_name=args.app_name,
        sql_warehouse_id=args.sql_warehouse_id,
        control_plane_catalog=args.control_plane_catalog,
        control_plane_schema=args.control_plane_schema,
        refresh_job_id=args.refresh_job_id,
        refresh_job_name=args.refresh_job_name,
        bootstrap_refresh_job_id=args.bootstrap_refresh_job_id,
        bootstrap_refresh_job_name=args.bootstrap_refresh_job_name,
        lakebase_instance_name=args.lakebase_instance_name,
        lakebase_database_name=args.lakebase_database_name,
        lakebase_schema=args.lakebase_schema,
        genie_space_id=args.genie_space_id,
    )
    (output_dir / "app.yaml").write_text(build_manual_app_yaml(settings))


def _copy_built_wheel(output_dir: Path) -> Path | None:
    wheel = find_built_wheel(REPO_ROOT / "dist")
    if wheel is None:
        return None
    target_dir = output_dir / "dist"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / wheel.name
    shutil.copy2(wheel, target)
    return target


def _write_refresh_job_payload(
    output_dir: Path, args: argparse.Namespace, wheel_name: str
) -> Path:
    workspace_source_path = args.workspace_source_path.rstrip("/")
    wheel_workspace_path = f"{workspace_source_path}/dist/{wheel_name}"
    payload = build_manual_refresh_job_payload(
        ManualRefreshJobSettings(
            app_name=args.app_name,
            wheel_workspace_path=wheel_workspace_path,
            sql_warehouse_id=args.sql_warehouse_id,
            control_plane_catalog=args.control_plane_catalog,
            control_plane_schema=args.control_plane_schema,
            lakebase_instance_name=args.lakebase_instance_name,
            lakebase_database_name=args.lakebase_database_name,
            lakebase_pguser=args.lakebase_pguser,
            lakebase_schema=args.lakebase_schema,
            use_lakebase_read_model=args.use_lakebase_read_model,
        )
    )
    output_path = output_dir / "refresh-job.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_source_items(output_dir)
    _write_manual_app_yaml(output_dir, args)
    copied_wheel = _copy_built_wheel(output_dir)

    if args.workspace_source_path:
        if copied_wheel is None:
            raise SystemExit(
                "No built wheel found in dist/. Run `python3 -m pip wheel --no-deps "
                "--no-build-isolation --wheel-dir dist .` before generating refresh-job.json."
            )
        job_payload_path = _write_refresh_job_payload(
            output_dir, args, copied_wheel.name
        )
        print(f"Wrote refresh job payload to {job_payload_path}")

    print(f"Prepared existing-app source tree at {output_dir}")


if __name__ == "__main__":
    main()
