# Architecture

This document describes the deployed Model Lens architecture across both supported deployment modes.

## Design Goals

Model Lens is optimized for four things:

1. Keep customer data inside Databricks.
2. Make deployment simple enough for customer workspaces.
3. Use one monitoring contract instead of custom per-model code paths.
4. Keep monitoring state durable and auditable.
5. Keep the UI responsive without turning Lakebase into the system of record.

## Deployment Modes

Model Lens supports two runtime shapes:

1. `warehouse_only`
   - Unity Catalog + SQL warehouse only
   - no bundle-managed Lakebase resource attached to the app
   - app reads summaries and incidents directly from the warehouse-backed repository

2. `dev` / `prod` with Lakebase
   - Unity Catalog remains the system of record
   - the refresh workflow syncs a Lakebase read model for fast monitor and incident views
   - the app can use the same Lakebase read model when the operator provides Lakebase instance/database values in the session or app env

## Architecture Overview

```mermaid
flowchart LR
  User["Operator in Browser"] --> App["Databricks App: Model Lens"]
  App -->|scan source tables + write configs| Warehouse["Databricks SQL Warehouse"]
  App -->|optional hot summary + incident reads| Lakebase["Lakebase Read Model"]

  Refresh["Serverless Refresh Workflow"] -->|read inference + labels| Warehouse
  Warehouse --> Source["Unity Catalog Source Tables"]
  Refresh -->|write drift, quality, performance, incidents| Control["Unity Catalog Control-Plane Tables"]
  Refresh -->|sync monitor summary + incidents| Lakebase

  Control --> Warehouse
  App -->|deep reads and scans| Control
```

## Components

### 1. Databricks App

The app is the operator control plane.

Responsibilities:

- provide a route-based operator shell with persistent model selection and page navigation
- let operators move from overview cards straight into model-specific drift analysis
- use a staged onboarding wizard so workspace setup, source discovery, contract review, and final activation are separated into explicit steps
- initialize the control-plane schema
- let operators override the control-plane catalog/schema used by the app session
- discover monitor drafts from source-table schema, preview rows, optional labels tables, and optional MLflow metadata
- preview external labels tables during discovery, infer join/label/order columns, and validate matched vs unmatched source rows before activation
- let operators override inferred columns only when the draft is ambiguous
- let operators choose either a rolling baseline window or a fixed known-good baseline date range
- save monitor configs
- trigger the initial refresh workflow asynchronously during monitor activation
- render monitor summaries and incidents from Lakebase when configured
- recommend the Lakebase-enabled target when running warehouse-only in a workspace that appears to have Lakebase available

Primary code:

- `src/model_lens/app.py`
- `src/model_lens/backend.py`
- `src/model_lens/callbacks.py`
- `src/model_lens/pages/`
- `src/model_lens/ui/`

### 2. Databricks SQL Warehouse

The SQL warehouse is the access layer for:

- source table scans
- source data reads during refresh
- writes into control-plane Delta tables
- deep drilldowns and fallback reads from the app

The warehouse is the compute access layer and the authoritative read/write path for system-of-record state.

### 3. Unity Catalog Control Plane

The durable monitoring state lives in Delta tables under a customer-selected Unity Catalog namespace:

- catalog: deployment input, default `model_observability`
- schema: deployment input, default `control_plane`

Current tables:

- `monitor_configs`
- `drift_metrics`
- `quality_metrics`
- `quality_history`
- `performance_metrics`
- `incidents`
- `incident_history`
- `refresh_runs`
- `comparison_windows`

This is the system of record.

### 4. Lakebase Read Model

Lakebase is used as a projection for the app, not as the canonical store.

Current Lakebase projection tables:

- `monitor_inventory`
- `monitor_summary`
- `open_incidents`

The app uses Lakebase for hot UI reads when configured. If Lakebase is unavailable or not configured, Model Lens falls back to the warehouse-backed queries.

### 5. Refresh Workflow

The refresh workflow is a serverless Databricks job.

The app does not run the heavy first refresh inline. During activation it saves the monitor config, resolves the workflow from `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`, and triggers the job asynchronously so the UI stays responsive. That means the app service principal also needs `CAN MANAGE RUN` on the refresh job, while the job's Run as identity still needs the source-data and control-plane privileges required for the actual computation.

Responsibilities:

- enumerate active monitors
- read source inference data
- optionally join labels from an external table with deterministic dedupe
- scope shared source tables down to one monitored model/version when configured
- backfill all valid daily rolling or fixed-baseline comparison windows on the first run
- append only new daily windows on later runs by default
- compute drift, quality, and degradation summaries across those windows
- record one refresh-run row per model execution with requested mode, effective mode, counts, status, and data range
- persist one comparison-window row per logical baseline/current pairing
- persist one quality-history row per comparison window for row-count, null-rate, and prediction-stat trends
- persist incident lifecycle rows (`opened`, `ongoing`, `escalated`, `downgraded`, `recovered`) per comparison window while keeping `incidents` as the current open-incident projection
- replace or append persisted metric windows depending on refresh mode
- sync the current UI projection into Lakebase

Packaging/runtime shape:

- the bundle builds a wheel artifact from the repo
- the workflow runs a `python_wheel_task`
- this avoids workspace-file import issues in Databricks serverless

Primary code:

- `resources/jobs.yml`
- `src/model_lens/workflows/refresh_job.py`
- `src/model_lens/services/refresh_runner.py`
- `src/model_lens/services/refresh_engine.py`

## Data Flow

### Onboarding Flow

1. The operator enters a source inference table and can optionally add a labels table plus an MLflow experiment or registered model.
2. The app loads schema metadata and sample rows through the SQL warehouse.
3. The discovery service infers the monitoring contract, feature set, slices, model scope candidates, and optional MLflow lineage. It accepts timestamp-like ISO strings, prioritizes shared-name shared-type join keys for external labels, and can fall back to identifier-like `model_version` values when a dedicated `model_id` column is absent.
   It also treats a 0-row labels join as a review-blocking warning and keeps the full numeric feature set selected by default rather than silently shrinking the first refresh to a small subset.
4. In the contract step, the operator reviews the inferred draft and only opens `Advanced` when overrides are needed.
5. If the source table contains multiple model IDs, the operator confirms or pins one `model_id_value`.
6. If the labels table is not unique on the join key, the operator confirms or provides a label ordering column.
7. In the review step, the app summarizes the final namespace, feature set, model scope, labels strategy, and MLflow linkage before activation.
8. The app writes one active row into `monitor_configs`.
9. If Lakebase mode is active, the repository syncs the projected monitor inventory into Lakebase.
10. The app can immediately trigger the first refresh.

### Refresh Flow

1. The workflow loads active monitor configs.
2. For each config, it reads source data from the inference table.
3. If configured, it applies `model_id_value` / `model_version_value` filters before analysis.
4. If configured, it joins an external labels table and uses the configured order column to dedupe repeated label keys.
5. It generates all valid daily comparison windows for the configured baseline policy within the comparison horizon.
6. In `auto` mode, it backfills full history when no matching history exists and appends only new windows when history is already aligned.
7. It computes numeric drift metrics, windowed quality history, incident lifecycle rows, and performance contributors for each comparison window, while still keeping `quality_metrics` as the latest-summary compatibility row and `incidents` as the current open-incident projection.
8. It writes `refresh_runs` and `comparison_windows` provenance rows alongside the metric facts.
9. It replaces or appends persisted rows for that model without duplicating logical windows.
10. If Lakebase mode is active, it refreshes the Lakebase monitor summary and open-incident projection.

Current limitation:

- the app UI still emphasizes current/open incidents; there is not yet a dedicated historical incident timeline page even though warehouse incident history is now persisted
- the remaining incident readback/productization work is tracked in [Historical Backfill Plan](/Users/volo.vragov/Desktop/work/model-lens/docs/HISTORICAL_BACKFILL_PLAN.md)

### Readback Flow

1. In Lakebase mode, the app reads monitor summary and incident inbox data from Lakebase.
2. In warehouse-only mode, or if Lakebase is unavailable, it reads those views from the warehouse-backed repository.
3. The app renders the current state for operators.

## Why The System Of Record Stays In Unity Catalog

Model Lens keeps the authoritative monitoring state in Unity Catalog because it is:

- durable
- auditable
- directly queryable by customer data teams
- aligned with how the refresh workflow already computes metrics

Lakebase improves interaction speed, but it is intentionally only a projection.

## Why Lakebase Exists

Model Lens uses Lakebase for the UI because the app has a different access pattern than the analytics layer.

Good Lakebase candidates:

- monitor inventory
- latest summary per model
- open incident inbox
- saved operator preferences later

Bad Lakebase candidates:

- raw inference data
- historical metric fact tables as the only durable copy
- refresh computation state

## Read/Write Split

The product now follows this split:

- `writes`: always land in Unity Catalog control-plane tables first
- `refresh compute`: always runs against the lakehouse / warehouse path
- `hot reads`: prefer Lakebase
- `fallback reads`: use the warehouse when Lakebase is missing or unavailable

## Operational Notes

- The app does not perform DDL on normal reads. Control-plane creation is an explicit setup step.
- If the control-plane catalog should be created by Model Lens itself, setup must be run by an identity with catalog-create privileges.
- If Lakebase is configured for the app but unavailable at runtime, or if the projection is empty, the UI falls back to warehouse reads instead of crashing or going blank.
- The refresh workflow can also sync the Lakebase projection when it has the Lakebase instance/database inputs.
- The scheduled job identity must be permitted to connect to the target Lakebase database if you want the projection kept fresh by the workflow rather than only by app-driven refreshes.
- On Databricks CLI `v0.260.0`, the app resource can bind a SQL warehouse but not an app-level `database` or `job` resource. Model Lens therefore treats Lakebase app reads as session/app-env configuration instead of Terraform-managed app resource wiring.
- Existing-app deployments have two supported wiring modes:
  - bundle-managed app resource mode, where [app.yaml](/Users/volo.vragov/Desktop/work/model-lens/app.yaml) resolves `SQL_WAREHOUSE_ID` from `valueFrom: sql_warehouse`
  - manual existing-app mode, where [prepare_existing_app_source.py](/Users/volo.vragov/Desktop/work/model-lens/scripts/prepare_existing_app_source.py) generates an alternate `app.yaml` with a literal `SQL_WAREHOUSE_ID` so constrained operators do not need permission to manage app resources
- `databricks bundle deploy` creates the app resource, but `databricks apps deploy ... --source-code-path ...` is still required to deploy the app source onto compute.
