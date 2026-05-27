# Deploy into an existing app

Use this guide when the workspace already has a Databricks App and you want to
deploy Model Landscape into it without creating a new app resource.

## When to use this path

- You have an existing Databricks App that must keep its URL and service principal
- You have an approved SQL warehouse
- You have (or can create) a shared refresh job
- You cannot or prefer not to use `databricks bundle deploy` for the app resource

If you want Databricks to create a new app automatically, use
[Deploy to a Workspace](deploy.md) instead.

## Prerequisites

Collect these values before starting:

| Value | Example |
|-------|---------|
| Existing app name | `ml-drift-monitor` |
| SQL warehouse ID | `abc123def456` |
| Control-plane catalog | `main` or `hive_metastore` |
| Control-plane schema | `model_landscape_control_plane` |
| Workspace upload path | `/Workspace/Users/<email>/model-landscape-app` |
| Refresh job ID (if reusing) | `12345678` |
| Spark node type (if creating job) | `i3.xlarge` |

## Quick path

For operators who want the shortest working deployment:

```bash
# 1. Build
uv run pytest
uv build --wheel --out-dir dist

# 2. Generate deployable source tree
uv run python notebooks/prepare_existing_app_source.py \
  --app-name <app-name> \
  --sql-warehouse-id <warehouse-id> \
  --control-plane-catalog <catalog> \
  --control-plane-schema <schema> \
  --output-dir /tmp/ml-deploy \
  --workspace-source-path /Workspace/Users/<email>/model-landscape-app

# 3. Upload to workspace
databricks workspace mkdirs /Workspace/Users/<email>/model-landscape-app
databricks workspace import-dir /tmp/ml-deploy \
  /Workspace/Users/<email>/model-landscape-app --overwrite

# 4. Create or reset the refresh job (set node_type_id first)
databricks jobs create --json @/tmp/ml-deploy/refresh-job.json
# Record the job_id from the output

# 5. Regenerate with exact job ID, re-upload, deploy
uv run python notebooks/prepare_existing_app_source.py \
  --app-name <app-name> \
  --sql-warehouse-id <warehouse-id> \
  --control-plane-catalog <catalog> \
  --control-plane-schema <schema> \
  --refresh-job-id <job-id> \
  --output-dir /tmp/ml-deploy \
  --workspace-source-path /Workspace/Users/<email>/model-landscape-app

databricks workspace import-dir /tmp/ml-deploy \
  /Workspace/Users/<email>/model-landscape-app --overwrite

databricks apps start <app-name>
databricks apps deploy <app-name> \
  --source-code-path /Workspace/Users/<email>/model-landscape-app

# 6. Grant permissions (see Permissions section below)
```

## Two deployment tracks

### Track A: bundle-bound existing app

Use when you have `Can manage` on the existing app and SQL warehouse.

```bash
databricks bundle deployment bind \
  -t warehouse_only \
  --var "app_name=<app-name>" \
  --var "sql_warehouse_id=<warehouse-id>" \
  --var "refresh_node_type_id=<node-type>" \
  --var "control_plane_catalog=<catalog>" \
  --var "control_plane_schema=<schema>" \
  control_plane <app-resource-id>

databricks bundle deploy -t warehouse_only \
  --var "app_name=<app-name>" \
  --var "sql_warehouse_id=<warehouse-id>" \
  --var "refresh_node_type_id=<node-type>" \
  --var "control_plane_catalog=<catalog>" \
  --var "control_plane_schema=<schema>"
```

If this fails with a `sql_warehouse` permission error, switch to Track B.

### Track B: manual deployment (no app-resource management)

Use when the operator cannot manage the app's `sql_warehouse` resource.
This is the recommended path for constrained customer workspaces.

The helper script `notebooks/prepare_existing_app_source.py` generates:
- `app.yaml` with literal `SQL_WAREHOUSE_ID`
- `src/` application source
- `dist/*.whl` for the refresh job
- `refresh-job.json` REST API payload for job creation

See the Quick Path above for the full command sequence.

## Refresh job contract

The shared refresh job must match:

- One Python wheel task with package `model_landscape`, entry point `model-landscape-refresh`
- Task key: `refresh_control_plane`
- Job parameters: `warehouse_id`, `control_plane_catalog`, `control_plane_schema`, `scope`, `model_key`
- `scope` defaults to `scheduler`
- UC-capable access mode: `USER_ISOLATION` or `SINGLE_USER`
- Active hourly schedule for `scheduler_only` pickup

Before `jobs create` or `jobs reset`, set `job_clusters[0].new_cluster.node_type_id`
to a workspace-approved node type.

To reuse an existing job instead of creating a new one:

```bash
databricks jobs reset --json @/tmp/ml-deploy/refresh-job-reset.json
```

## Permissions

### App service principal

| Scope | Grants |
|-------|--------|
| SQL warehouse | `CAN_USE` |
| Source data | `USE CATALOG`, `USE SCHEMA`, `SELECT` |
| Control plane | `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY` |
| Control plane (if setup creates objects) | `CREATE TABLE`, `CREATE SCHEMA` |
| Refresh job (recommended) | `CAN_MANAGE` for full in-app job management |
| Refresh job (minimum for direct trigger) | `CAN_MANAGE_RUN` |

### Refresh job run-as identity

| Scope | Grants |
|-------|--------|
| SQL warehouse | `CAN_USE` |
| Source data | `USE CATALOG`, `USE SCHEMA`, `SELECT` |
| Control plane | `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY` |

## Verification

After deployment, open the app and confirm:

1. `SQL_WAREHOUSE_ID` is populated in Monitor Settings
2. `REFRESH_JOB_ID` is populated
3. Control-plane namespace matches your configuration
4. `Validate Workspace Wiring` reports `fully_ready` or `scheduler_only`
5. `Setup Control Plane` creates the control-plane tables
6. First monitor saves and triggers (or queues for scheduled pickup)

## Scheduler-only mode

`scheduler_only` is a supported operating mode, not a deployment failure.
It applies when the app service principal does not have `CAN_MANAGE_RUN`
on the refresh job. Monitors save normally and the shared hourly job picks
them up on its next scheduled run.

## Troubleshooting

**App opens but cannot query data:**
Check that `SQL_WAREHOUSE_ID` is set and the app SP has `CAN_USE` on the warehouse.

**Monitor saves but never refreshes:**
Check that `REFRESH_JOB_ID` points at the correct job, the schedule is unpaused,
and the job Run-As identity has data-plane access.

**Setup shows "Verification unavailable":**
The app could not inspect workflow ACLs. Try `Run First Refresh` or
`databricks jobs run-now <job-id>` to test directly.

**Bundle deploy fails with warehouse permission error:**
Switch from Track A to Track B.
