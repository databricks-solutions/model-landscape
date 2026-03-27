from ml_drift_monitor_next.services.contracts import build_contract


def test_build_contract_preserves_full_feature_list() -> None:
    columns = ["event_ts", "model_id", "prediction", "f1", "f2", "f3", "country"]
    contract = build_contract(
        columns=columns,
        timestamp_col="event_ts",
        model_id_col="model_id",
        prediction_col="prediction",
        feature_columns=["f1", "f2", "f3", "country"],
        categorical_columns=["country"],
        slice_columns=["country"],
    )
    assert contract.feature_columns == ("f1", "f2", "f3", "country")
    assert contract.slice_columns == ("country",)


def test_build_contract_rejects_reserved_overlap() -> None:
    columns = ["event_ts", "model_id", "prediction", "f1"]
    try:
        build_contract(
            columns=columns,
            timestamp_col="event_ts",
            model_id_col="model_id",
            prediction_col="prediction",
            feature_columns=["prediction", "f1"],
        )
    except ValueError as error:
        assert "reserved fields" in str(error)
    else:
        raise AssertionError("expected validation failure")
