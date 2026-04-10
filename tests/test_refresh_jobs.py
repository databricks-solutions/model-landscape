from types import SimpleNamespace

from model_lens.services import refresh_jobs


def _fake_run_response(run_id: int | None) -> SimpleNamespace:
    return SimpleNamespace(response=SimpleNamespace(run_id=run_id))


def test_build_refresh_job_named_params_uses_namespace_and_model_key(monkeypatch) -> None:
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

    params = refresh_jobs.build_refresh_job_named_params(
        model_key="fraud_model_demo",
        control_plane_catalog="model_observability",
        control_plane_schema="control_plane",
    )

    assert params == {
        "warehouse_id": "wh-123",
        "control_plane_catalog": "model_observability",
        "control_plane_schema": "control_plane",
        "scope": "scheduler",
        "model_key": "fraud_model_demo",
    }


def test_build_refresh_job_named_params_includes_lakebase_when_configured(monkeypatch) -> None:
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

    params = refresh_jobs.build_refresh_job_named_params(
        model_key="fraud_model_demo",
        control_plane_catalog="model_observability",
        control_plane_schema="control_plane",
        lakebase_database_name="model_lens_ui",
    )

    assert params["use_lakebase_read_model"] == "true"
    assert params["lakebase_database_name"] == "model_lens_ui"
    assert params["lakebase_host"] == "lakebase.example.internal"


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
            return _fake_run_response(999)

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
    assert trigger.workflow_kind == "shared"
    assert trigger.used_shared_fallback is True
    assert fake_jobs.run_call["job_id"] == 321
    assert fake_jobs.run_call["job_parameters"]["scope"] == "bootstrap"
    assert fake_jobs.run_call["job_parameters"]["model_key"] == "fraud_model_demo"


def test_trigger_refresh_job_uses_separate_bootstrap_job_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="321",
            refresh_job_name="model-lens-refresh",
            bootstrap_refresh_job_id="654",
            bootstrap_refresh_job_name="model-lens-bootstrap-refresh",
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
            return _fake_run_response(222)

    fake_jobs = FakeJobs()
    fake_workspace = SimpleNamespace(jobs=fake_jobs)

    trigger = refresh_jobs.trigger_refresh_job(
        model_key="fraud_model_demo",
        control_plane_catalog="model_observability",
        control_plane_schema="control_plane",
        workspace_client=fake_workspace,
    )

    assert trigger.job_id == 654
    assert trigger.run_id == 222
    assert trigger.workflow_kind == "bootstrap"
    assert trigger.used_shared_fallback is False
    assert fake_jobs.run_call["job_id"] == 654


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
            return _fake_run_response(111)

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
            return _fake_run_response(333)

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
        {"name": None, "limit": None},
    ]
    assert trigger.job_id == 777
    assert trigger.run_id == 333


def test_trigger_refresh_job_falls_back_to_substring_match(monkeypatch) -> None:
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
                SimpleNamespace(
                    job_id=888,
                    settings=SimpleNamespace(name="[dev volo_vragov] model-lens-refresh bootstrap"),
                )
            ]

        def run_now(self, **kwargs):
            self.run_call = kwargs
            return _fake_run_response(444)

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
        {"name": None, "limit": None},
    ]
    assert trigger.job_id == 888
    assert trigger.run_id == 444


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


def test_resolve_refresh_job_id_reports_missing_shared_job_clearly(monkeypatch) -> None:
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

    fake_workspace = SimpleNamespace(jobs=SimpleNamespace(list=lambda **_: []))

    try:
        refresh_jobs.resolve_refresh_job_id(fake_workspace)
    except RuntimeError as error:
        message = str(error)
        assert "matching 'model-lens-refresh'" in message
        assert "existing shared workflow" in message
        assert "redeploy the app" in message
    else:
        raise AssertionError("Expected missing refresh job lookup to raise")


def test_validate_workspace_readiness_reports_missing_workflow_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="",
            refresh_job_name="",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    readiness = refresh_jobs.validate_workspace_readiness(control_plane_ready=True, workspace_client=SimpleNamespace())

    assert readiness.overall_mode == "not_ready"
    assert readiness.refresh_workflow_configured is False
    assert any("REFRESH_JOB_ID" in issue for issue in readiness.blocking_issues)


def test_validate_workspace_readiness_reports_scheduler_only_when_run_now_is_unconfirmed(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="321",
            refresh_job_name="model-lens-refresh",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    fake_job = SimpleNamespace(
        job_id=321,
        settings=SimpleNamespace(
            name="model-lens-refresh",
            schedule=SimpleNamespace(pause_status="UNPAUSED"),
            trigger=None,
            continuous=None,
            queue=SimpleNamespace(enabled=True),
            max_concurrent_runs=1,
        ),
    )
    fake_workspace = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda **_: fake_job,
            get_permissions=lambda *_: (_ for _ in ()).throw(RuntimeError("permission read denied")),
        ),
        current_user=SimpleNamespace(me=lambda: SimpleNamespace(user_name="svc@app", display_name="svc@app")),
    )

    readiness = refresh_jobs.validate_workspace_readiness(control_plane_ready=True, workspace_client=fake_workspace)

    assert readiness.overall_mode == "scheduler_only"
    assert readiness.refresh_workflow_resolved is True
    assert readiness.scheduler_path_available is True
    assert readiness.run_now_available is None
    assert any("Direct trigger may still work" in warning for warning in readiness.warnings)
    assert not any("CAN_MANAGE_RUN on job 321" in warning for warning in readiness.warnings)


def test_validate_workspace_readiness_reports_explicit_run_now_grant_when_manage_run_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="321",
            refresh_job_name="model-lens-refresh",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    fake_job = SimpleNamespace(
        job_id=321,
        settings=SimpleNamespace(
            name="model-lens-refresh",
            schedule=SimpleNamespace(pause_status="UNPAUSED"),
            trigger=None,
            continuous=None,
            queue=SimpleNamespace(enabled=True),
            max_concurrent_runs=1,
        ),
    )
    fake_workspace = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda **_: fake_job,
            get_permissions=lambda *_: SimpleNamespace(
                access_control_list=[
                    SimpleNamespace(
                        user_name="svc@app",
                        service_principal_name=None,
                        display_name="svc@app",
                        all_permissions=[SimpleNamespace(permission_level="CAN_VIEW")],
                    )
                ]
            ),
        ),
        current_user=SimpleNamespace(me=lambda: SimpleNamespace(user_name="svc@app", display_name="svc@app")),
    )

    readiness = refresh_jobs.validate_workspace_readiness(control_plane_ready=True, workspace_client=fake_workspace)

    assert readiness.overall_mode == "scheduler_only"
    assert readiness.run_now_available is False
    assert any("Grant the app service principal CAN_MANAGE_RUN on job 321." == warning for warning in readiness.warnings)


def test_validate_workspace_readiness_treats_separate_bootstrap_lane_as_optional(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="321",
            refresh_job_name="model-lens-refresh",
            bootstrap_refresh_job_id="654",
            bootstrap_refresh_job_name="model-lens-bootstrap-refresh",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    shared_job = SimpleNamespace(
        job_id=321,
        settings=SimpleNamespace(
            name="model-lens-refresh",
            schedule=SimpleNamespace(pause_status="UNPAUSED"),
            trigger=None,
            continuous=None,
            queue=SimpleNamespace(enabled=True),
            max_concurrent_runs=1,
        ),
    )
    bootstrap_job = SimpleNamespace(
        job_id=654,
        settings=SimpleNamespace(
            name="model-lens-bootstrap-refresh",
            schedule=None,
            trigger=None,
            continuous=None,
            queue=SimpleNamespace(enabled=True),
            max_concurrent_runs=1,
        ),
    )

    def get_job(*, job_id):
        return {321: shared_job, 654: bootstrap_job}[job_id]

    def get_permissions(job_id):
        if str(job_id) == "321":
            return SimpleNamespace(
                access_control_list=[
                    SimpleNamespace(
                        user_name="svc@app",
                        service_principal_name=None,
                        display_name="svc@app",
                        all_permissions=[SimpleNamespace(permission_level="CAN_MANAGE_RUN")],
                    )
                ]
            )
        return SimpleNamespace(access_control_list=[])

    fake_workspace = SimpleNamespace(
        jobs=SimpleNamespace(get=get_job, get_permissions=get_permissions),
        current_user=SimpleNamespace(me=lambda: SimpleNamespace(user_name="svc@app", display_name="svc@app")),
    )

    readiness = refresh_jobs.validate_workspace_readiness(control_plane_ready=True, workspace_client=fake_workspace)

    assert readiness.overall_mode == "fully_ready"
    assert readiness.bootstrap_workflow_mode == "separate"
    assert readiness.bootstrap_workflow_resolved is True
    assert readiness.bootstrap_workflow_id == 654
    assert readiness.bootstrap_run_now_available is None
    assert not any("no schedule or trigger configured" in issue.lower() for issue in readiness.blocking_issues)
    assert any("Direct trigger may still work" in warning for warning in readiness.warnings)
    assert not any("CAN_MANAGE_RUN on job 654" in warning for warning in readiness.warnings)


def test_validate_workspace_readiness_reports_fully_ready_with_direct_manage_run_access(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="321",
            refresh_job_name="model-lens-refresh",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    fake_job = SimpleNamespace(
        job_id=321,
        settings=SimpleNamespace(
            name="model-lens-refresh",
            schedule=SimpleNamespace(pause_status="UNPAUSED"),
            trigger=None,
            continuous=None,
            queue=SimpleNamespace(enabled=True),
            max_concurrent_runs=1,
        ),
    )
    fake_workspace = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda **_: fake_job,
            get_permissions=lambda *_: SimpleNamespace(
                access_control_list=[
                    SimpleNamespace(
                        user_name="svc@app",
                        service_principal_name=None,
                        display_name="svc@app",
                        all_permissions=[SimpleNamespace(permission_level="CAN_MANAGE_RUN")],
                    )
                ]
            ),
        ),
        current_user=SimpleNamespace(me=lambda: SimpleNamespace(user_name="svc@app", display_name="svc@app")),
    )

    readiness = refresh_jobs.validate_workspace_readiness(control_plane_ready=True, workspace_client=fake_workspace)

    assert readiness.overall_mode == "fully_ready"
    assert readiness.run_now_available is True


def test_validate_workspace_readiness_blocks_paused_shared_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_jobs,
        "settings",
        SimpleNamespace(
            sql_warehouse_id="wh-123",
            refresh_job_id="321",
            refresh_job_name="model-lens-refresh",
            lakebase_instance_name="",
            lakebase_database_name="",
            lakebase_host="",
            lakebase_port=5432,
            lakebase_pguser="",
            lakebase_sslmode="require",
            lakebase_schema="model_lens_ui",
        ),
    )

    fake_job = SimpleNamespace(
        job_id=321,
        settings=SimpleNamespace(
            name="model-lens-refresh",
            schedule=SimpleNamespace(pause_status="PAUSED"),
            trigger=None,
            continuous=None,
            queue=SimpleNamespace(enabled=True),
            max_concurrent_runs=1,
        ),
    )
    fake_workspace = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda **_: fake_job,
            get_permissions=lambda *_: SimpleNamespace(access_control_list=[]),
        ),
        current_user=SimpleNamespace(me=lambda: SimpleNamespace(user_name="svc@app", display_name="svc@app")),
    )

    readiness = refresh_jobs.validate_workspace_readiness(control_plane_ready=True, workspace_client=fake_workspace)

    assert readiness.overall_mode == "not_ready"
    assert readiness.scheduler_path_available is False
    assert any("paused" in issue.lower() for issue in readiness.blocking_issues)


def test_resolve_shared_workflow_schedule_status_detects_supported_interval_and_manage_access(monkeypatch) -> None:
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

    fake_job = SimpleNamespace(
        job_id=321,
        settings=SimpleNamespace(
            name="model-lens-refresh",
            schedule=SimpleNamespace(
                pause_status="UNPAUSED",
                quartz_cron_expression="0 0 */6 * * ?",
                timezone_id="UTC",
            ),
            trigger=None,
            continuous=None,
            queue=SimpleNamespace(enabled=True),
            max_concurrent_runs=1,
        ),
    )
    fake_workspace = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda **_: fake_job,
            get_permissions=lambda *_: SimpleNamespace(
                access_control_list=[
                    SimpleNamespace(
                        user_name="svc@app",
                        service_principal_name=None,
                        display_name="svc@app",
                        all_permissions=[SimpleNamespace(permission_level="CAN_MANAGE")],
                    )
                ]
            ),
        ),
        current_user=SimpleNamespace(me=lambda: SimpleNamespace(user_name="svc@app", display_name="svc@app")),
    )

    status = refresh_jobs.resolve_shared_workflow_schedule_status(fake_workspace)

    assert status.configured is True
    assert status.resolved is True
    assert status.current_interval_hours == 6
    assert status.current_label == "Every 6 Hours"
    assert status.editable is True
    assert status.supported is True


def test_update_shared_workflow_schedule_updates_only_schedule_field(monkeypatch) -> None:
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

    fake_job = SimpleNamespace(
        job_id=321,
        settings=SimpleNamespace(
            name="model-lens-refresh",
            schedule=SimpleNamespace(
                pause_status="UNPAUSED",
                quartz_cron_expression="0 0 * * * ?",
                timezone_id="UTC",
            ),
            trigger=None,
            continuous=None,
            queue=SimpleNamespace(enabled=True),
            max_concurrent_runs=1,
        ),
    )

    class FakeJobs:
        def __init__(self) -> None:
            self.update_call = None

        def get(self, **_) -> object:
            return fake_job

        def get_permissions(self, *_):
            return SimpleNamespace(
                access_control_list=[
                    SimpleNamespace(
                        user_name="svc@app",
                        service_principal_name=None,
                        display_name="svc@app",
                        all_permissions=[SimpleNamespace(permission_level="CAN_MANAGE")],
                    )
                ]
            )

        def update(self, **kwargs):
            self.update_call = kwargs
            fake_job.settings.schedule = kwargs["new_settings"].schedule

    fake_jobs = FakeJobs()
    fake_workspace = SimpleNamespace(
        jobs=fake_jobs,
        current_user=SimpleNamespace(me=lambda: SimpleNamespace(user_name="svc@app", display_name="svc@app")),
    )

    status = refresh_jobs.update_shared_workflow_schedule(12, workspace_client=fake_workspace)

    assert fake_jobs.update_call is not None
    assert fake_jobs.update_call["job_id"] == 321
    assert fake_jobs.update_call["new_settings"].schedule.quartz_cron_expression == "0 0 */12 * * ?"
    assert status.current_interval_hours == 12
    assert status.current_label == "Every 12 Hours"
