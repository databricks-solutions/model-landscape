# Historical Backfill Plan

This document defines the next backend expansion for Model Lens so the product can show useful drift, performance, quality, and incident history immediately after the first monitor activation.

## Status

Current local implementation status:

- Phase 1 UX hardening is implemented
- Phase 2 existing-table multi-window backfill/incremental refresh is implemented
- Phase 3 provenance hardening via `refresh_runs` and `comparison_windows` is implemented
- Phase 4 quality-history rows, queries, and charts are implemented
- Phase 5 incident-history rows and recovery lifecycle events are implemented in the warehouse path
- shared refresh hardening is implemented: refresh audit rows are created before source-range discovery, `quality_metrics` is rebuilt from persisted `daily_quality_profiles`, stale `running` rows are reclaimed after a timeout, and canonical `performance_bin_specs` now keep incremental performance repair comparable to bootstrap

The remaining work in this document is now primarily:

- incident-history readback/product surfaces in the app
- any later normalization or UI features that build on the recorded window/run provenance

## Remaining Problem

Model Lens now backfills drift, performance, quality, and incident history on first refresh, but incident history is still thinner in the app than it is in the warehouse.

That means the first refresh now gives useful warehouse history across all four surfaces, but the product story is still incomplete:

- drift timelines and granularity views are populated immediately
- performance trends are populated immediately
- quality trends are populated immediately
- incident lifecycle rows are persisted immediately
- the app still emphasizes current/open incidents rather than a dedicated historical incident experience

## Goal

Make the first refresh backfill all available historical windows, then make scheduled refreshes append only new windows.

After first activation, the app should already be able to show:

- drift heatmaps across multiple historical windows
- trend lines for drift and performance
- useful daily/weekly/monthly granularity switching
- incident history over time
- quality trends across multiple windows

## Design Principles

1. Persist daily comparison windows as the canonical grain.
2. Keep Unity Catalog as the system of record.
3. Make first refresh a backfill run.
4. Make scheduled refreshes incremental and idempotent.
5. Keep the frontend query-only wherever possible.

## Revised Delivery Strategy

After reviewing the deployment team's expansion draft, the plan should be staged more aggressively:

1. Do not block the first useful release on a full schema migration.
2. Ship multi-window history first on top of the existing metric tables.
3. Add `auto` / `backfill` / `incremental` refresh modes immediately.
4. Keep daily windows as the stored canonical grain and aggregate upward in the backend.
5. Build later history features on top of explicit run/window provenance.

This staged approach shortened time-to-value materially: customers now get populated timelines after the first refresh without waiting for a broader warehouse migration.

## Current Gaps

### Drift

- only one `window_start` / `window_end` pair is computed per run
- timeline charts have little or no history after onboarding

### Performance

- performance rows exist for one window only unless the job has run many times
- stable windows need to render as “no significant degradation,” not as missing data

### Quality

- `quality_metrics` still exists as the latest-summary compatibility row, but it is now rebuilt from all persisted `daily_quality_profiles` rather than from only the latest bounded refresh slice
- `quality_history` now stores one row per comparison window with row-count, prediction-stat, and null-rate trend data

### Incidents

- `incidents` remains the current open-incident projection used by Overview and the Lakebase read model
- `incident_history` now persists onset / persistence / severity changes / recovery across comparison windows
- there is not yet a dedicated incident-history page or richer timeline readback in the app

## Initial Shipping Model

The first multi-window implementation can use the current metric tables without breaking existing monitors:

- `drift_metrics` stores many `window_start` / `window_end` pairs per model
- `performance_metrics` stores many `window_start` / `window_end` pairs per model
- `quality_metrics` remains as the latest-summary compatibility row and is rebuilt from persisted daily profiles after each refresh write
- `quality_history` stores many `window_start` / `window_end` pairs per model
- `incidents` can stay as the current open-incident projection
- `incident_history` stores lifecycle rows per feature / metric / comparison window

The important change is write semantics, not table count:

- backfill writes many daily windows in one refresh
- incremental runs append only missing or repaired windows
- writes remain idempotent per logical comparison window

For the first cut, uniqueness should not rely on `window_end` alone. Use a logical window key built from:

- `model_key`
- `baseline_kind`
- `baseline_start`
- `baseline_end`
- `window_start`
- `window_end`

That is stricter than the deployment draft and avoids collisions when fixed baselines or repair horizons are involved.

## Target Data Model

Add two new system-of-record tables:

### `refresh_runs`

One row per refresh execution:

- `run_id`
- `model_key`
- `run_kind` (`backfill`, `incremental`, `manual`)
- `started_at`
- `completed_at`
- `status`
- `window_count`
- `data_min_date`
- `data_max_date`

### `comparison_windows`

One row per logical baseline/current comparison window:

- `window_id`
- `model_key`
- `window_grain` (`daily`)
- `window_start`
- `window_end`
- `baseline_start`
- `baseline_end`
- `baseline_kind`
- `created_at`
- `source_run_id`

The metric tables should then reference `window_id` logically, even if the first implementation keeps denormalized dates for compatibility.

## Refresh Workflow Changes

### First Run

For a rolling baseline with `baseline_days = 7`:

- compute all valid trailing daily windows across the available data
- example with 21 days:
  - baseline `1-7`, current `8-14`
  - baseline `2-8`, current `9-15`
  - ...
  - baseline `8-14`, current `15-21`

For a fixed baseline:

- keep baseline fixed
- compare it against every valid trailing current window of equal length after the baseline period

### Scheduled Run

- detect the latest persisted window for the model
- append only new daily windows
- optionally recompute a short repair horizon to correct late-arriving labels

### Idempotency

Use deterministic uniqueness for writes:

- drift: `model_key + window_end + baseline_end + feature_name + metric_name`
- performance: `model_key + window_end + feature_name + bin_label + metric_name`
- quality: `model_key + window_id`
- incidents: current open-incident projection by business key
- incident history: `model_key + window_id + feature_name + metric_name + event_type`

The deployment draft was right to push idempotency earlier, but too loose on `window_end`-only replacement. The implementation should delete/replace by full logical window identity, not just the trailing date.

## Backend Query Rules

The deployment draft also surfaced a useful shortcut: daily windows should remain canonical, and weekly/monthly views should be aggregated in the backend rather than materialized separately.

That said, the aggregation policy cannot be "blindly take `last` or `max` for every column."

Use these rules:

- drift severity metrics (`metric_value`) aggregate to weekly/monthly using a worst-case policy such as `max`
- counts (`ref_count`, `cur_count`) aggregate with `sum`
- descriptive overlays like `ref_mean`, `cur_mean`, `ref_std`, `cur_std` should come from the latest constituent daily window in the bucket unless and until we introduce a more principled rolled-up summary model
- sparse-history states should still render one point and an explicit note

This keeps the weekly/monthly view useful for monitoring without pretending those rolled-up descriptive statistics are mathematically exact.

## Query Layer Changes

### Drift

- query all daily windows for a model
- aggregate to weekly/monthly in SQL or backend code
- show a clear sparse-history message when fewer than two windows exist

### Performance

- query all historical performance windows
- compute timeline series from persisted rows
- keep bin-level tables visible even when all degradation contributions are zero

### Quality

- keep latest-summary quality plus existing `daily_volume` JSON so the dashboard remains backward compatible
- persist first-class windowed quality rows for row-count, null-rate, and prediction-stat trends
- preserve latest-summary queries for Overview while the quality page reads `quality_history`

### Incidents

- persist incident timeline rows across windows
- keep Overview on the current open-incident projection while future pages can query onset, persistence, severity changes, and recovery from `incident_history`

## Frontend Implications

The frontend should remain thin:

- Overview reads latest summaries only
- Drift, Performance, and Quality pages read precomputed history
- Granularity switches aggregate persisted daily windows
- First-run UX shows real trends immediately after onboarding

## Implementation Phases

### Phase 1: UX Hardening

- show sparse-history messages when only one window exists
- keep single-point charts rendering
- keep zero-delta performance visible with a stable-state note and bin table

### Phase 2: Existing-Table Multi-Window Engine

- replace single-window splitting with `generate_window_pairs()`
- support:
  - rolling baselines across all valid trailing windows
  - fixed baselines against every valid trailing current window
- add `auto` / `backfill` / `incremental` refresh modes
- append only missing windows on scheduled runs
- support a short repair horizon for late-arriving labels
- keep current tables and read paths working

### Phase 3: Historical Window Schema Hardening

- add `refresh_runs`
- add `comparison_windows`
- move from implicit logical keys to explicit `window_id`
- capture provenance, run kind, status, and repair/backfill lineage

### Phase 4: Shared-Job Scheduler And Runtime State

- keep one shared refresh workflow as the default deployment contract
- add per-monitor cadence presets to `monitor_configs`
- add `monitor_runtime_state` for:
  - `bootstrap_status`
  - last drift/performance refresh timestamps
  - next due timestamps
  - latest label watermark
  - latest run status/error
- treat app-side `Run now` as optional acceleration only
- let the shared hourly workflow pick up pending bootstraps and overdue monitors even when the app cannot trigger runs directly, but only after that shared workflow has been created in the workspace and the app has been wired to it with `REFRESH_JOB_ID` or `REFRESH_JOB_NAME`

### Phase 4: Repository / Backend Queries

- query persisted history directly
- aggregate daily windows into weekly/monthly views
- preserve latest-summary reads for Overview

### Phase 5: Incident History

- persist incident evolution over time
- keep incremental refresh continuity by seeding the first new window from the repository’s current open incidents
- leave richer incident readback and UI surfaces as the next follow-up

### Phase 6: Daily Profile Foundations

- persist `daily_quality_profiles`
- persist `daily_feature_profiles`
- persist `daily_performance_profiles`
- persist `performance_bin_specs`
- use bounded range reads and the shared refresh job to materialize those facts without requiring a full-table pandas load
- derive the persisted comparison-window tables from that per-run daily-profile layer so refresh no longer reloads every logical window from the warehouse
- on incremental runs, merge already-persisted daily facts for the affected date span so derivation can reuse prior history instead of depending entirely on the current bounded load
- reuse persisted canonical performance-bin specs on incremental/performance-repair runs so daily performance profiles remain comparable across runs
- keep the current window tables as the stable UI/read contract

### Phase 7: Operationalization

- add a `--mode` argument to the refresh workflow entry point
- default scheduled jobs to `auto`
- keep `backfill` available for manual recompute
- add or document a daily schedule only after the backfill path is stable in workspace tests

## Acceptance Criteria

1. A newly onboarded monitor with enough history shows multiple drift/performance windows after the first refresh.
2. Drift granularity changes produce visibly different daily vs weekly/monthly views.
3. Performance pages render stable windows as healthy/stable, not blank.
4. The first refresh can be rerun idempotently without duplicating historical rows.
5. Scheduled refreshes append new windows instead of recomputing full history unnecessarily.
6. Daily profile facts exist and are already used inside refresh to derive the persisted window/history tables, even though the UI still reads the stable window/history tables.
7. The first implementation can ship without forcing a frontend rewrite for those daily facts.
