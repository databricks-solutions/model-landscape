# Architecture

This document describes the deployed Model Lens architecture.

## Design Goals

Model Lens is optimized for four things:

1. Keep customer data inside Databricks.
2. Make deployment simple enough for customer workspaces.
3. Use one monitoring contract instead of custom per-model code paths.
4. Keep monitoring state durable and auditable.
5. Keep the UI responsive without turning Lakebase into the system of record.

## Architecture Overview

```mermaid
flowchart LR
  User["Operator in Browser"] --> App["Databricks App: Model Lens"]
  App -->|scan source tables + write configs| Warehouse["Databricks SQL Warehouse"]
  App -->|hot summary + incident reads| Lakebase["Lakebase Read Model"]

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

- initialize the control-plane schema
- scan source tables
- map source columns into the monitoring contract
- save monitor configs
- trigger refreshes
- render monitor summaries and incidents from Lakebase when configured

Primary code:

- `src/model_lens/app.py`

### 2. Databricks SQL Warehouse

The SQL warehouse is the access layer for:

- source table scans
- source data reads during refresh
- writes into control-plane Delta tables
- deep drilldowns and fallback reads from the app

The warehouse is the compute access layer and the authoritative read/write path for system-of-record state.

### 3. Unity Catalog Control Plane

The durable monitoring state lives in Delta tables under:

- catalog: `model_observability`
- schema: `control_plane`

Current tables:

- `monitor_configs`
- `drift_metrics`
- `quality_metrics`
- `performance_metrics`
- `incidents`

This is the system of record.

### 4. Lakebase Read Model

Lakebase is used as a projection for the app, not as the canonical store.

Current Lakebase projection tables:

- `monitor_inventory`
- `monitor_summary`
- `open_incidents`

The app uses Lakebase for hot UI reads when configured. If Lakebase is unavailable, Model Lens falls back to the warehouse-backed queries.

### 5. Refresh Workflow

The refresh workflow is a serverless Databricks job.

Responsibilities:

- enumerate active monitors
- read source inference data
- optionally join labels from an external table
- split baseline vs current windows
- compute drift, quality, and degradation summaries
- replace the current persisted snapshot for the refreshed model
- sync the current UI projection into Lakebase

Primary code:

- `resources/jobs.yml`
- `src/model_lens/workflows/refresh_job.py`
- `src/model_lens/services/refresh_runner.py`
- `src/model_lens/services/refresh_engine.py`

## Data Flow

### Onboarding Flow

1. The operator scans a source table from the app.
2. The app loads schema metadata and sample rows through the SQL warehouse.
3. The operator maps fields into the monitoring contract.
4. The app writes one active row into `monitor_configs`.
5. The repository syncs the projected monitor inventory into Lakebase.
6. The app can immediately trigger the first refresh.

### Refresh Flow

1. The workflow loads active monitor configs.
2. For each config, it reads source data from the inference table.
3. If configured, it joins an external labels table.
4. It builds baseline and current windows from the configured baseline policy.
5. It computes numeric drift metrics, quality metrics, performance contributors, and incident rows.
6. It replaces the persisted current snapshot for that model.
7. It refreshes the Lakebase monitor summary and open-incident projection.

### Readback Flow

1. The app reads monitor summary and incident inbox data from Lakebase.
2. If the Lakebase projection is unavailable, it falls back to warehouse-backed reads.
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

- If Lakebase is configured for the app but unavailable at runtime, the UI falls back to warehouse reads instead of crashing.
- The refresh workflow can also sync the Lakebase projection when it has the Lakebase instance/database inputs.
- The scheduled job identity must be permitted to connect to the target Lakebase database if you want the projection kept fresh by the workflow rather than only by app-driven refreshes.
