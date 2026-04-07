from __future__ import annotations

from model_lens.services.table_names import TableNames


def monitor_config_migration_columns() -> dict[str, str]:
    return {
        "baseline_start": "DATE",
        "baseline_end": "DATE",
        "model_id_value": "STRING",
        "model_version_value": "STRING",
        "labels_order_col": "STRING",
        "performance_metric_names": "ARRAY<STRING>",
        "default_performance_metric": "STRING",
        "drift_cadence_preset": "STRING",
        "performance_cadence_preset": "STRING",
        "schedule_enabled": "BOOLEAN",
        "mlflow_experiment_name": "STRING",
        "mlflow_experiment_id": "STRING",
        "mlflow_run_id": "STRING",
        "mlflow_registered_model_name": "STRING",
        "mlflow_model_version": "STRING",
    }


def refresh_run_migration_columns() -> dict[str, str]:
    return {
        "scope": "STRING",
        "scheduled_at": "TIMESTAMP",
        "range_start": "DATE",
        "range_end": "DATE",
        "rows_scanned": "BIGINT",
        "label_rows_scanned": "BIGINT",
    }


def runtime_state_migration_columns() -> dict[str, str]:
    return {
        "last_run_started_at": "TIMESTAMP",
        "last_run_completed_at": "TIMESTAMP",
        "backoff_until": "TIMESTAMP",
        "consecutive_failures": "INT",
    }


def daily_performance_profile_migration_columns() -> dict[str, str]:
    return {
        "volume_pct": "DOUBLE",
    }


def drift_metric_migration_columns() -> dict[str, str]:
    return {
        "window_id": "STRING",
    }


def performance_metric_migration_columns() -> dict[str, str]:
    return {
        "window_id": "STRING",
    }


def ddl(table_names: TableNames) -> dict[str, str]:
    return {
        "monitor_configs": f"""
            CREATE TABLE IF NOT EXISTS {table_names.monitor_configs} (
                model_key STRING,
                display_name STRING,
                source_table STRING,
                timestamp_col STRING,
                model_id_col STRING,
                model_id_value STRING,
                prediction_col STRING,
                model_version_col STRING,
                model_version_value STRING,
                prediction_score_col STRING,
                label_col STRING,
                entity_id_col STRING,
                feature_columns ARRAY<STRING>,
                slice_columns ARRAY<STRING>,
                categorical_columns ARRAY<STRING>,
                baseline_kind STRING,
                baseline_n_days INT,
                baseline_start DATE,
                baseline_end DATE,
                baseline_max_comparison_days INT,
                problem_type STRING,
                labels_table STRING,
                labels_join_col STRING,
                labels_order_col STRING,
                performance_metric_names ARRAY<STRING>,
                default_performance_metric STRING,
                drift_cadence_preset STRING,
                performance_cadence_preset STRING,
                schedule_enabled BOOLEAN,
                mlflow_experiment_name STRING,
                mlflow_experiment_id STRING,
                mlflow_run_id STRING,
                mlflow_registered_model_name STRING,
                mlflow_model_version STRING,
                created_by STRING,
                status STRING,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "drift_metrics": f"""
            CREATE TABLE IF NOT EXISTS {table_names.drift_metrics} (
                model_key STRING,
                window_id STRING,
                feature_name STRING,
                metric_name STRING,
                metric_value DOUBLE,
                window_start DATE,
                window_end DATE,
                baseline_start DATE,
                baseline_end DATE,
                ref_mean DOUBLE,
                cur_mean DOUBLE,
                ref_std DOUBLE,
                cur_std DOUBLE,
                ref_null_pct DOUBLE,
                cur_null_pct DOUBLE,
                ref_count BIGINT,
                cur_count BIGINT,
                computed_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "quality_metrics": f"""
            CREATE TABLE IF NOT EXISTS {table_names.quality_metrics} (
                model_key STRING,
                total_rows BIGINT,
                min_date DATE,
                max_date DATE,
                prediction_mean DOUBLE,
                prediction_std DOUBLE,
                daily_volume STRING,
                null_rates STRING,
                computed_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "quality_history": f"""
            CREATE TABLE IF NOT EXISTS {table_names.quality_history} (
                model_key STRING,
                window_id STRING,
                window_start DATE,
                window_end DATE,
                baseline_start DATE,
                baseline_end DATE,
                row_count BIGINT,
                prediction_mean DOUBLE,
                prediction_std DOUBLE,
                null_rates STRING,
                computed_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "daily_quality_profiles": f"""
            CREATE TABLE IF NOT EXISTS {table_names.daily_quality_profiles} (
                model_key STRING,
                profile_date DATE,
                row_count BIGINT,
                prediction_mean DOUBLE,
                prediction_std DOUBLE,
                null_rates STRING,
                label_row_count BIGINT,
                computed_at TIMESTAMP,
                source_run_id STRING
            ) USING DELTA
        """.strip(),
        "daily_feature_profiles": f"""
            CREATE TABLE IF NOT EXISTS {table_names.daily_feature_profiles} (
                model_key STRING,
                profile_date DATE,
                feature_name STRING,
                feature_kind STRING,
                row_count BIGINT,
                non_null_count BIGINT,
                null_pct DOUBLE,
                mean DOUBLE,
                std DOUBLE,
                min_value DOUBLE,
                max_value DOUBLE,
                distribution_json STRING,
                computed_at TIMESTAMP,
                source_run_id STRING
            ) USING DELTA
        """.strip(),
        "performance_metrics": f"""
            CREATE TABLE IF NOT EXISTS {table_names.performance_metrics} (
                model_key STRING,
                window_id STRING,
                feature_name STRING,
                bin_label STRING,
                baseline_metric DOUBLE,
                current_metric DOUBLE,
                delta DOUBLE,
                volume_pct DOUBLE,
                contribution DOUBLE,
                metric_name STRING,
                window_start DATE,
                window_end DATE,
                computed_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "daily_performance_profiles": f"""
            CREATE TABLE IF NOT EXISTS {table_names.daily_performance_profiles} (
                model_key STRING,
                profile_date DATE,
                feature_name STRING,
                bin_label STRING,
                metric_name STRING,
                metric_value DOUBLE,
                row_count BIGINT,
                volume_pct DOUBLE,
                computed_at TIMESTAMP,
                source_run_id STRING
            ) USING DELTA
        """.strip(),
        "performance_bin_specs": f"""
            CREATE TABLE IF NOT EXISTS {table_names.performance_bin_specs} (
                model_key STRING,
                feature_name STRING,
                edges_json STRING,
                computed_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "incidents": f"""
            CREATE TABLE IF NOT EXISTS {table_names.incidents} (
                model_key STRING,
                feature_name STRING,
                metric_name STRING,
                severity STRING,
                status STRING,
                metric_value DOUBLE,
                window_end DATE,
                observed_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "incident_history": f"""
            CREATE TABLE IF NOT EXISTS {table_names.incident_history} (
                model_key STRING,
                feature_name STRING,
                metric_name STRING,
                event_type STRING,
                severity STRING,
                status STRING,
                metric_value DOUBLE,
                window_id STRING,
                window_start DATE,
                window_end DATE,
                baseline_start DATE,
                baseline_end DATE,
                observed_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "refresh_runs": f"""
            CREATE TABLE IF NOT EXISTS {table_names.refresh_runs} (
                run_id STRING,
                model_key STRING,
                requested_mode STRING,
                run_kind STRING,
                scope STRING,
                status STRING,
                started_at TIMESTAMP,
                scheduled_at TIMESTAMP,
                completed_at TIMESTAMP,
                window_count INT,
                data_min_date DATE,
                data_max_date DATE,
                range_start DATE,
                range_end DATE,
                drift_row_count BIGINT,
                quality_row_count BIGINT,
                performance_row_count BIGINT,
                incident_row_count BIGINT,
                rows_scanned BIGINT,
                label_rows_scanned BIGINT,
                error_message STRING
            ) USING DELTA
        """.strip(),
        "comparison_windows": f"""
            CREATE TABLE IF NOT EXISTS {table_names.comparison_windows} (
                window_id STRING,
                model_key STRING,
                window_grain STRING,
                window_start DATE,
                window_end DATE,
                baseline_start DATE,
                baseline_end DATE,
                baseline_kind STRING,
                created_at TIMESTAMP,
                source_run_id STRING
            ) USING DELTA
        """.strip(),
        "monitor_runtime_state": f"""
            CREATE TABLE IF NOT EXISTS {table_names.monitor_runtime_state} (
                model_key STRING,
                bootstrap_status STRING,
                last_drift_refresh_at TIMESTAMP,
                last_performance_refresh_at TIMESTAMP,
                next_drift_due_at TIMESTAMP,
                next_performance_due_at TIMESTAMP,
                last_label_watermark STRING,
                last_run_status STRING,
                last_run_error STRING,
                last_run_started_at TIMESTAMP,
                last_run_completed_at TIMESTAMP,
                backoff_until TIMESTAMP,
                consecutive_failures INT
            ) USING DELTA
        """.strip(),
    }
