-- Model Lens deterministic scratch dataset
--
-- Update the catalog below if `main` is not writable in your workspace.
-- This script creates:
--   main.model_lens_demo.inference_logs
--   main.model_lens_demo.labels

USE CATALOG main;
CREATE SCHEMA IF NOT EXISTS model_lens_demo;
USE SCHEMA model_lens_demo;

CREATE OR REPLACE TABLE inference_logs AS
WITH day_series AS (
  SELECT explode(sequence(DATE'2026-01-01', DATE'2026-01-21', INTERVAL 1 DAY)) AS event_date
),
row_series AS (
  SELECT explode(sequence(1, 40)) AS row_id
),
base AS (
  SELECT
    event_date,
    row_id,
    concat(
      'ent_',
      date_format(event_date, 'yyyyMMdd'),
      '_',
      lpad(CAST(row_id AS STRING), 3, '0')
    ) AS entity_id
  FROM day_series
  CROSS JOIN row_series
)
SELECT
  to_timestamp(
    concat(
      CAST(event_date AS STRING),
      ' ',
      lpad(CAST(row_id % 24 AS STRING), 2, '0'),
      ':',
      lpad(CAST((row_id * 7) % 60 AS STRING), 2, '0'),
      ':00'
    )
  ) AS event_ts,
  'fraud_model_v1' AS model_id,
  CASE
    WHEN event_date <= DATE'2026-01-07' THEN 0.08 + ((row_id % 6) * 0.03)
    ELSE 0.62 + ((row_id % 6) * 0.04)
  END AS prediction,
  CASE
    WHEN event_date <= DATE'2026-01-07' THEN 45 + (row_id % 7)
    ELSE 95 + (row_id % 13)
  END AS amount,
  CASE
    WHEN event_date <= DATE'2026-01-07' THEN 1.10 + ((row_id % 5) * 0.07)
    ELSE 3.50 + ((row_id % 5) * 0.19)
  END AS velocity_7d,
  CASE
    WHEN event_date <= DATE'2026-01-07' THEN 0.15 + ((row_id % 8) * 0.03)
    ELSE 0.48 + ((row_id % 8) * 0.04)
  END AS device_score,
  CASE
    WHEN event_date <= DATE'2026-01-07' THEN CASE row_id % 3
      WHEN 0 THEN 'na'
      WHEN 1 THEN 'eu'
      ELSE 'apac'
    END
    ELSE CASE row_id % 3
      WHEN 0 THEN 'latam'
      WHEN 1 THEN 'latam'
      ELSE 'apac'
    END
  END AS region,
  CASE
    WHEN row_id % 4 IN (0, 1) THEN 'enterprise'
    ELSE 'smb'
  END AS merchant_segment,
  entity_id
FROM base;

CREATE OR REPLACE TABLE labels AS
SELECT
  entity_id,
  CASE
    WHEN prediction >= 0.74 THEN 1
    WHEN prediction >= 0.54
      AND CAST(regexp_extract(entity_id, '_(\\d+)$', 1) AS INT) % 4 = 0 THEN 1
    ELSE 0
  END AS label
FROM inference_logs;

-- Quick validation queries:
-- SELECT COUNT(*) AS inference_rows FROM main.model_lens_demo.inference_logs;
-- SELECT COUNT(*) AS label_rows FROM main.model_lens_demo.labels;
-- SELECT MIN(DATE(event_ts)) AS min_date, MAX(DATE(event_ts)) AS max_date
-- FROM main.model_lens_demo.inference_logs;
