from types import SimpleNamespace

from model_lens.services import refresh_jobs


def test_build_refresh_job_params_uses_namespace_and_model_key(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="",
            refresh_job_name="model-lens-refresh",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    params = refresh_jobs.build_refresh_job_params(
        model_key="fraud_model_demo",
        control_plane_catalog="model_observability",
        control_plane_schema="control_plane",
    )

    assert params == [
        "--warehouse-id",
        "wh-123",
        "--catalog",
        "model_observability",
        "--schema",
        "control_plane",
        "--scope",
        "scheduler",
        "--model-key",
        "fraud_model_demo",
    ]


def test_build_refresh_job_params_includes_lakebase_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="",
            refresh_job_name="model-lens-refresh",
            lakebase_host="lakebase.example.internal",
            lakebase_port=5432,
            lakebase_pguser="lakebase-user",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    params = refresh_jobs.build_refresh_job_params(
        model_key="fraud_model_demo",
        control_plane_catalog="model_observability",
        control_plane_schema="control_plane",
        lakebase_database_name="model_lens_ui",
    )

    assert "--use-lakebase-read-model" in params
    assert "--lakebase-database-name" in params
    assert "model_lens_ui" in params
    assert "--lakebase-host" in params
    assert "lakebase.example.internal" in params


def test_trigger_refresh_job_uses_configured_job_id(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="321",
            refresh_job_name="model-lens-refresh",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    class FakeJobs:
        def __init__(self) -> None:
            self.run_call = None

        def run_now(self, **kwargs):
            self.run_call = kwargs
            return refresh_jobs.make_fake_run_response(999)

    fake_jobs = FakeJobs()
    fake_workspace = SimpleNamespace(jobs=fake_jobs)

    trigger = refresh_jobs.trigger_refresh_job(
        model_key="fraud_model_demo",
        control_plane_catalog="model_observability",
        control_plane_schema="control_plane",
        workspace_client=fake_workspace,
    )

    assert trigger.job_id == 321
    assert trigger.run_id == 999
    assert fake_jobs.run_call["job_id"] == 321
    assert fake_jobs.run_call["python_params"][-4:] == ["--scope", "bootstrap", "--model-key", "fraud_model_demo"]


def test_trigger_refresh_job_falls_back_to_named_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="",
            refresh_job_name="model-lens-refresh",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    class FakeJobs:
        def __init__(self) -> None:
            self.list_name = None
            self.run_call = None

        def list(self, *, name=None, limit=None):
            self.list_name = name
            return [SimpleNamespace(job_id=654, settings=SimpleNamespace(name="model-lens-refresh"))]

        def run_now(self, **kwargs):
            self.run_call = kwargs
            return refresh_jobs.make_fake_run_response(111)

    fake_jobs = FakeJobs()
    fake_workspace = SimpleNamespace(jobs=fake_jobs)

    trigger = refresh_jobs.trigger_refresh_job(
        model_key="fraud_model_demo",
        control_plane_catalog="model_observability",
        control_plane_schema="control_plane",
        workspace_client=fake_workspace,
    )

    assert fake_jobs.list_name == "model-lens-refresh"
    assert trigger.job_id == 654
    assert trigger.run_id == 111


def test_trigger_refresh_job_resolves_dab_prefixed_name_by_suffix(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="",
            refresh_job_name="model-lens-refresh",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    class FakeJobs:
        def __init__(self) -> None:
            self.list_calls = []
            self.run_call = None

        def list(self, *, name=None, limit=None):
            self.list_calls.append({"name": name, "limit": limit})
            if name is not None:
                return []
            return [
                SimpleNamespace(job_id=777, settings=SimpleNamespace(name="[dev volo_vragov] model-lens-refresh"))
            ]

        def run_now(self, **kwargs):
            self.run_call = kwargs
            return refresh_jobs.make_fake_run_response(333)

    fake_jobs = FakeJobs()
    fake_workspace = SimpleNamespace(jobs=fake_jobs)

    trigger = refresh_jobs.trigger_refresh_job(
        model_key="fraud_model_demo",
        control_plane_catalog="model_observability",
        control_plane_schema="control_plane",
        workspace_client=fake_workspace,
    )

    assert fake_jobs.list_calls == [
        {"name": "model-lens-refresh", "limit": 25},
        {"name": None, "limit": 100},
    ]
    assert trigger.job_id == 777
    assert trigger.run_id == 333


def test_resolve_refresh_job_id_requires_unambiguous_name(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="",
            refresh_job_name="model-lens-refresh",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    fake_workspace = SimpleNamespace(
        jobs=SimpleNamespace(
            list=lambda **_: [
                SimpleNamespace(job_id=1, settings=SimpleNamespace(name="model-lens-refresh")),
                SimpleNamespace(job_id=2, settings=SimpleNamespace(name="model-lens-refresh")),
            ]
        )
    )

    try:
        refresh_jobs.resolve_refresh_job_id(fake_workspace)
    except RuntimeError as error:
        assert "Multiple refresh jobs" in str(error)
    else:
        raise AssertionError("Expected ambiguous refresh job name to raise")


def test_resolve_refresh_job_id_requires_unambiguous_suffix_match(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="",
            refresh_job_name="model-lens-refresh",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    fake_workspace = SimpleNamespace(
        jobs=SimpleNamespace(
            list=lambda **kwargs: (
                []
                if kwargs.get("name") is not None
                else [
                    SimpleNamespace(job_id=1, settings=SimpleNamespace(name="[dev alice] model-lens-refresh")),
                    SimpleNamespace(job_id=2, settings=SimpleNamespace(name="[dev bob] model-lens-refresh")),
                ]
            )
        )
    )

    try:
        refresh_jobs.resolve_refresh_job_id(fake_workspace)
    except RuntimeError as error:
        assert "end with" in str(error)
    else:
        raise AssertionError("Expected ambiguous suffix refresh job name to raise")
