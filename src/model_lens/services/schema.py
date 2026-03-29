from __future__ import annotations

from model_lens.services.table_names import TableNames


def monitor_config_migration_columns() -> dict[str, str]:
    return {
        "model_id_value": "STRING",
        "model_version_value": "STRING",
        "labels_order_col": "STRING",
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
                baseline_max_comparison_days INT,
                problem_type STRING,
                labels_table STRING,
                labels_join_col STRING,
                labels_order_col STRING,
                created_by STRING,
                status STRING,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            ) USING DELTA
        """.strip(),
        "drift_metrics": f"""
            CREATE TABLE IF NOT EXISTS {table_names.drift_metrics} (
                model_key STRING,
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
        "performance_metrics": f"""
            CREATE TABLE IF NOT EXISTS {table_names.performance_metrics} (
                model_key STRING,
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
    }
