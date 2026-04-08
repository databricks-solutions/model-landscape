# App Testing Triage

This file records the implementation status of the highest-confidence items from the external app-testing report.

## Fixed In This Batch

- `#1`, `#2`, `#10`: Overview now keeps models with no drift history in `Computing/Pending`, excludes them from healthy/warning/critical counts, and uses monitor-contract feature counts instead of drift-row counts.
- `#3`, `#4`, `#14`, `#15`: drift/null-rate thresholds now come from one shared internal threshold contract used by Overview, charts, incidents, and Spark incident derivation.
- `#13`, `#17`, `#18`, `#23`: Drift heatmap now respects the ranked Top N subset, uses `max` aggregation consistently, keeps server-side Top N normalization, and includes the active granularity in the title.
- `#38`, `#39`, `#40`, `#43`, `#45`: Data Quality now skips malformed null-rate JSON values, renders missing prediction mean as `N/A`, shows prediction std as a KPI, deduplicates period history by latest `computed_at`, and always shows the null-rate critical marker when data exists.
- `#51`, `#52`, `#53`, `#54`, `#55`, `#57`, `#58`, `#59`, `#61`: Feature Deep Dive now catches backend errors, tolerates mixed/non-numeric distribution payloads, removes the unbounded full-table fallback, refreshes on reload, defaults to the most historically drifted feature, labels the data source/window context, and shows missing dimension groups explicitly.
- `#28`, `#29`: Performance page now degrades cleanly when the selected metric column is absent and no longer crashes when only partial window metadata is available.
- `#73`: onboarding scan failure now clears stale `scan-data` state instead of leaving old discovery values on screen.

## Already Fixed Or Stale

- `#31`: `perf-model-banner` is already wired.
- `#49`: Data Quality callback coverage already exists; the remaining gap was edge-case hardening, which is covered in this batch.
- `#60`, `#81`, `#82`: loading states and Reference/selector synchronization were already fixed on the current local branch before this batch.

## Not A Bug / Semantic Clarification

- `#75`: `incidents` is the current open-incident projection. Severe historical drift can coexist with `incidents = 0` when the latest window recovered; historical lifecycle is stored in `incident_history`.
- `#67`, `#68`, `#69`: external label fields are already validated against the labels table during save-time validation.

## Deferred

- Lower-value polish and broader UX items from the original report remain deferred for later passes, especially the visual/wording-only items and larger IA follow-ups that do not affect correctness or safety.
