from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from uuid import uuid4

from model_lens.config import settings


class RefreshJobConfigError(RuntimeError):
    pass


class RefreshJobLookupError(RuntimeError):
    pass


@dataclass(frozen=True)
class RefreshWorkflowStatus:
    configured: bool
    resolved: bool
    configured_via: str
    configured_value: str
    job_id: int | None
    job_name: str
    scheduler_path_available: bool
    scheduler_mode: str
    queue_enabled: bool | None
    max_concurrent_runs: int | None
    run_now_available: bool | None
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkspaceReadiness:
    control_plane_ready: bool
    warehouse_ready: bool
    refresh_workflow_configured: bool
    refresh_workflow_resolved: bool
    refresh_workflow_name: str
    refresh_workflow_id: int | None
    refresh_workflow_configured_via: str
    refresh_workflow_configured_value: str
    scheduler_path_available: bool
    scheduler_mode: str
    queue_enabled: bool | None
    max_concurrent_runs: int | None
    run_now_available: bool | None
    lakebase_ready: bool
    overall_mode: str
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def run_now_permission_guidance(job_id: int | None) -> str | None:
    if job_id is None:
        return None
    return f"Grant the app service principal CAN_MANAGE_RUN on job {int(job_id)}."


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalized_job_name(value: object) -> str:
    return " ".join(_clean(value).split()).casefold()


def _job_name(job: object) -> str:
    return _clean(getattr(getattr(job, "settings", None), "name", None))


def _jobs_with_exact_name(jobs: list[object], refresh_job_name: str) -> list[object]:
    normalized_target = _normalized_job_name(refresh_job_name)
    return [
        job
        for job in jobs
        if getattr(job, "job_id", None) and _normalized_job_name(_job_name(job)) == normalized_target
    ]


def _jobs_with_suffix_name(jobs: list[object], refresh_job_name: str) -> list[object]:
    normalized_target = _normalized_job_name(refresh_job_name)
    return [
        job
        for job in jobs
        if getattr(job, "job_id", None) and _normalized_job_name(_job_name(job)).endswith(normalized_target)
    ]


def _jobs_with_contains_name(jobs: list[object], refresh_job_name: str) -> list[object]:
    normalized_target = _normalized_job_name(refresh_job_name)
    return [
        job
        for job in jobs
        if getattr(job, "job_id", None) and normalized_target in _normalized_job_name(_job_name(job))
    ]


def is_refresh_job_configuration_error(error: Exception) -> bool:
    return isinstance(error, (RefreshJobConfigError, RefreshJobLookupError))


def _configured_job_selector() -> tuple[str, str]:
    configured_job_id = _clean(settings.refresh_job_id)
    if configured_job_id:
        return "id", configured_job_id
    configured_job_name = _clean(settings.refresh_job_name)
    if configured_job_name:
        return "name", configured_job_name
    return "none", ""


def _load_job_by_id(workspace_client, job_id: int) -> object:
    if hasattr(workspace_client.jobs, "get"):
        return workspace_client.jobs.get(job_id=job_id)
    for job in list(workspace_client.jobs.list()):
        if getattr(job, "job_id", None) and int(job.job_id) == int(job_id):
            return job
    raise RefreshJobLookupError(
        f"Configured refresh job ID {job_id} does not exist in this workspace. "
        "Set REFRESH_JOB_ID or REFRESH_JOB_NAME to a deployed shared refresh workflow and redeploy the app."
    )


def _pause_status(resource: object | None) -> str:
    raw = getattr(resource, "pause_status", None)
    return _clean(getattr(raw, "value", raw)).upper()


def _scheduler_path_status(job: object) -> tuple[bool, str, str | None]:
    job_settings = getattr(job, "settings", None)
    if job_settings is None:
        return False, "missing", "Resolved refresh workflow has no settings payload."

    schedule = getattr(job_settings, "schedule", None)
    if schedule is not None:
        if _pause_status(schedule) == "PAUSED":
            return False, "schedule", "Resolved refresh workflow has a cron schedule, but it is paused."
        return True, "schedule", None

    trigger = getattr(job_settings, "trigger", None)
    if trigger is not None:
        if _pause_status(trigger) == "PAUSED":
            return False, "trigger", "Resolved refresh workflow has a trigger, but it is paused."
        return True, "trigger", None

    continuous = getattr(job_settings, "continuous", None)
    if continuous is not None:
        if _pause_status(continuous) == "PAUSED":
            return False, "continuous", "Resolved refresh workflow is configured for continuous execution, but it is paused."
        return True, "continuous", None

    return False, "missing", "Resolved refresh workflow has no schedule or trigger configured."


def _direct_run_now_permission(workspace_client, job_id: int) -> tuple[bool | None, str | None]:
    try:
        identity = workspace_client.current_user.me()
    except Exception:
        return None, "Could not confirm immediate Run now permission; scheduler-only mode assumed."

    principal_candidates = {
        _normalized_job_name(getattr(identity, "user_name", None)),
        _normalized_job_name(getattr(identity, "display_name", None)),
    }
    principal_candidates.discard("")

    if not principal_candidates:
        return None, "Could not identify the current app principal; scheduler-only mode assumed."

    try:
        permissions = workspace_client.jobs.get_permissions(str(job_id))
    except Exception:
        return None, "Could not read refresh workflow permissions; scheduler-only mode assumed."

    for acl in getattr(permissions, "access_control_list", None) or []:
        principal_names = {
            _normalized_job_name(getattr(acl, "user_name", None)),
            _normalized_job_name(getattr(acl, "service_principal_name", None)),
            _normalized_job_name(getattr(acl, "display_name", None)),
        }
        principal_names.discard("")
        if not principal_candidates.intersection(principal_names):
            continue
        permission_levels = {
            _clean(getattr(getattr(permission, "permission_level", None), "value", getattr(permission, "permission_level", None))).upper()
            for permission in (getattr(acl, "all_permissions", None) or [])
        }
        if permission_levels.intersection({"IS_OWNER", "CAN_MANAGE", "CAN_MANAGE_RUN"}):
            return True, None
        return False, None

    return None, "Could not confirm immediate Run now permission from the refresh workflow ACL; scheduler-only mode assumed."


def resolve_refresh_workflow_status(workspace_client=None) -> RefreshWorkflowStatus:
    configured_via, configured_value = _configured_job_selector()
    if configured_via == "none":
        return RefreshWorkflowStatus(
            configured=False,
            resolved=False,
            configured_via="none",
            configured_value="",
            job_id=None,
            job_name="",
            scheduler_path_available=False,
            scheduler_mode="missing",
            queue_enabled=None,
            max_concurrent_runs=None,
            run_now_available=None,
            blocking_issues=("Set REFRESH_JOB_ID (preferred) or REFRESH_JOB_NAME in the app environment.",),
            warnings=(),
        )

    client = workspace_client or _workspace_client()
    warnings: list[str] = []
    if configured_via == "name":
        warnings.append("REFRESH_JOB_NAME works, but REFRESH_JOB_ID is more reliable for production wiring.")

    try:
        job_id = resolve_refresh_job_id(client)
        job = _load_job_by_id(client, job_id)
    except Exception as error:
        return RefreshWorkflowStatus(
            configured=True,
            resolved=False,
            configured_via=configured_via,
            configured_value=configured_value,
            job_id=None,
            job_name="",
            scheduler_path_available=False,
            scheduler_mode="missing",
            queue_enabled=None,
            max_concurrent_runs=None,
            run_now_available=None,
            blocking_issues=(str(error),),
            warnings=tuple(warnings),
        )

    scheduler_path_available, scheduler_mode, scheduler_issue = _scheduler_path_status(job)
    blocking_issues: list[str] = []
    if scheduler_issue:
        blocking_issues.append(scheduler_issue)

    job_settings = getattr(job, "settings", None)
    queue_enabled = getattr(getattr(job_settings, "queue", None), "enabled", None)
    max_concurrent_runs = getattr(job_settings, "max_concurrent_runs", None)
    if queue_enabled is not True:
        warnings.append("The shared refresh workflow should enable queueing so overlapping runs wait instead of failing.")
    if max_concurrent_runs != 1:
        warnings.append("The shared refresh workflow should set max_concurrent_runs=1 to avoid overlap.")

    run_now_available, run_now_warning = _direct_run_now_permission(client, int(job_id))
    if run_now_warning:
        warnings.append(run_now_warning)
    if run_now_available is not True:
        guidance = run_now_permission_guidance(int(job_id))
        if guidance:
            warnings.append(guidance)

    return RefreshWorkflowStatus(
        configured=True,
        resolved=True,
        configured_via=configured_via,
        configured_value=configured_value,
        job_id=int(job_id),
        job_name=_job_name(job),
        scheduler_path_available=scheduler_path_available,
        scheduler_mode=scheduler_mode,
        queue_enabled=queue_enabled,
        max_concurrent_runs=max_concurrent_runs,
        run_now_available=run_now_available,
        blocking_issues=tuple(blocking_issues),
        warnings=tuple(warnings),
    )


def validate_workspace_readiness(
    *,
    control_plane_ready: bool,
    lakebase_requested: bool = False,
    workspace_client=None,
) -> WorkspaceReadiness:
    warehouse_ready = bool(_clean(settings.sql_warehouse_id))
    workflow_status = resolve_refresh_workflow_status(workspace_client=workspace_client)
    lakebase_ready = True
    if lakebase_requested:
        lakebase_ready = bool(
            _clean(settings.lakebase_database_name)
            and (_clean(settings.lakebase_instance_name) or _clean(settings.lakebase_host))
        )

    blocking_issues: list[str] = []
    warnings: list[str] = list(workflow_status.warnings)
    if not control_plane_ready:
        blocking_issues.append("Run Setup Control Plane successfully before onboarding a monitor.")
    if not warehouse_ready:
        blocking_issues.append("SQL_WAREHOUSE_ID is not configured for this app.")
    blocking_issues.extend(workflow_status.blocking_issues)
    if lakebase_requested and not lakebase_ready:
        warnings.append("Lakebase fields are only partially configured; the app will operate in warehouse mode until they are completed.")

    if blocking_issues:
        overall_mode = "not_ready"
    elif workflow_status.run_now_available is True:
        overall_mode = "fully_ready"
    else:
        overall_mode = "scheduler_only"

    return WorkspaceReadiness(
        control_plane_ready=control_plane_ready,
        warehouse_ready=warehouse_ready,
        refresh_workflow_configured=workflow_status.configured,
        refresh_workflow_resolved=workflow_status.resolved,
        refresh_workflow_name=workflow_status.job_name,
        refresh_workflow_id=workflow_status.job_id,
        refresh_workflow_configured_via=workflow_status.configured_via,
        refresh_workflow_configured_value=workflow_status.configured_value,
        scheduler_path_available=workflow_status.scheduler_path_available,
        scheduler_mode=workflow_status.scheduler_mode,
        queue_enabled=workflow_status.queue_enabled,
        max_concurrent_runs=workflow_status.max_concurrent_runs,
        run_now_available=workflow_status.run_now_available,
        lakebase_ready=lakebase_ready,
        overall_mode=overall_mode,
        blocking_issues=tuple(blocking_issues),
        warnings=tuple(warnings),
    )


def workspace_readiness_payload(readiness: WorkspaceReadiness) -> dict[str, object]:
    payload = asdict(readiness)
    payload["blocking_issues"] = list(readiness.blocking_issues)
    payload["warnings"] = list(readiness.warnings)
    return payload


def build_refresh_job_named_params(
    *,
    model_key: str,
    control_plane_catalog: str,
    control_plane_schema: str,
    lakebase_instance_name: str = "",
    lakebase_database_name: str = "",
    lakebase_schema: str = "",
    scope: str = "scheduler",
) -> dict[str, str]:
    warehouse_id = _clean(settings.sql_warehouse_id)
    if not warehouse_id:
        raise RuntimeError("SQL_WAREHOUSE_ID is not configured for refresh job execution.")

    params: dict[str, str] = {
        "warehouse-id": warehouse_id,
        "catalog": _clean(control_plane_catalog),
        "schema": _clean(control_plane_schema),
        "scope": _clean(scope) or "scheduler",
    }
    if _clean(model_key):
        params["model-key"] = _clean(model_key)

    resolved_lakebase_schema = _clean(lakebase_schema) or _clean(settings.lakebase_schema)
    resolved_lakebase_instance = _clean(lakebase_instance_name)
    resolved_lakebase_database = _clean(lakebase_database_name)
    resolved_lakebase_host = _clean(settings.lakebase_host)

    if resolved_lakebase_database and (resolved_lakebase_instance or resolved_lakebase_host):
        params["use-lakebase-read-model"] = "true"
        if resolved_lakebase_instance:
            params["lakebase-instance-name"] = resolved_lakebase_instance
        if resolved_lakebase_database:
            params["lakebase-database-name"] = resolved_lakebase_database
        if resolved_lakebase_host:
            params["lakebase-host"] = resolved_lakebase_host
        if settings.lakebase_port:
            params["lakebase-port"] = str(settings.lakebase_port)
        if _clean(settings.lakebase_pguser):
            params["lakebase-pguser"] = _clean(settings.lakebase_pguser)
        if _clean(settings.lakebase_sslmode):
            params["lakebase-sslmode"] = _clean(settings.lakebase_sslmode)
        if resolved_lakebase_schema:
            params["lakebase-schema"] = resolved_lakebase_schema

    return params


def resolve_refresh_job_id(workspace_client=None) -> int:
    configured_job_id = _clean(settings.refresh_job_id)
    if configured_job_id:
        try:
            return int(configured_job_id)
        except ValueError as error:
            raise RefreshJobConfigError(f"REFRESH_JOB_ID must be an integer, got {configured_job_id!r}.") from error

    refresh_job_name = _clean(settings.refresh_job_name)
    if not refresh_job_name:
        raise RefreshJobConfigError("Neither REFRESH_JOB_ID nor REFRESH_JOB_NAME is configured.")

    client = workspace_client or _workspace_client()
    exact_matches = _jobs_with_exact_name(list(client.jobs.list(name=refresh_job_name, limit=25)), refresh_job_name)
    if len(exact_matches) > 1:
        raise RefreshJobLookupError(
            f"Multiple refresh jobs are named {refresh_job_name!r}. "
            "Set REFRESH_JOB_ID explicitly so the app triggers the correct workflow."
        )
    if len(exact_matches) == 1:
        return int(exact_matches[0].job_id)

    all_jobs = list(client.jobs.list())
    all_exact_matches = _jobs_with_exact_name(all_jobs, refresh_job_name)
    if len(all_exact_matches) > 1:
        raise RefreshJobLookupError(
            f"Multiple refresh jobs are named {refresh_job_name!r}. "
            "Set REFRESH_JOB_ID explicitly so the app triggers the correct workflow."
        )
    if len(all_exact_matches) == 1:
        return int(all_exact_matches[0].job_id)

    suffix_matches = _jobs_with_suffix_name(all_jobs, refresh_job_name)
    if len(suffix_matches) == 1:
        return int(suffix_matches[0].job_id)
    if len(suffix_matches) > 1:
        raise RefreshJobLookupError(
            f"Multiple refresh jobs end with {refresh_job_name!r}. "
            "Set REFRESH_JOB_ID explicitly so the app triggers the correct workflow."
        )

    contains_matches = _jobs_with_contains_name(all_jobs, refresh_job_name)
    if len(contains_matches) == 1:
        return int(contains_matches[0].job_id)
    if len(contains_matches) > 1:
        raise RefreshJobLookupError(
            f"Multiple refresh jobs contain {refresh_job_name!r}. "
            "Set REFRESH_JOB_ID explicitly so the app triggers the correct workflow."
        )

    raise RefreshJobLookupError(
        f"Could not find a refresh job matching {refresh_job_name!r}. "
        "The app only triggers an existing shared workflow; set REFRESH_JOB_ID or REFRESH_JOB_NAME to the deployed workflow and redeploy the app."
    )


@dataclass(frozen=True)
class RefreshJobTrigger:
    job_id: int
    run_id: int | None


def trigger_refresh_job(
    *,
    model_key: str,
    control_plane_catalog: str,
    control_plane_schema: str,
    lakebase_instance_name: str = "",
    lakebase_database_name: str = "",
    lakebase_schema: str = "",
    scope: str = "bootstrap",
    workspace_client=None,
) -> RefreshJobTrigger:
    client = workspace_client or _workspace_client()
    job_id = resolve_refresh_job_id(client)
    named_params = build_refresh_job_named_params(
        model_key=model_key,
        control_plane_catalog=control_plane_catalog,
        control_plane_schema=control_plane_schema,
        lakebase_instance_name=lakebase_instance_name,
        lakebase_database_name=lakebase_database_name,
        lakebase_schema=lakebase_schema,
        scope=scope,
    )
    waiter = client.jobs.run_now(
        job_id=job_id,
        python_named_params=named_params,
        idempotency_token=str(uuid4()),
    )
    response = getattr(waiter, "response", None)
    run_id = getattr(response, "run_id", None)
    return RefreshJobTrigger(job_id=job_id, run_id=int(run_id) if run_id is not None else None)


def _workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def make_fake_run_response(run_id: int | None) -> SimpleNamespace:
    return SimpleNamespace(response=SimpleNamespace(run_id=run_id))
