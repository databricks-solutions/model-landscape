from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    bootstrap_workflow_mode: str
    bootstrap_workflow_resolved: bool
    bootstrap_workflow_name: str
    bootstrap_workflow_id: int | None
    bootstrap_workflow_configured_via: str
    bootstrap_workflow_configured_value: str
    bootstrap_run_now_available: bool | None
    lakebase_ready: bool
    overall_mode: str
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SharedWorkflowScheduleStatus:
    configured: bool
    resolved: bool
    job_id: int | None
    job_name: str
    scheduler_mode: str
    current_expression: str
    current_interval_hours: int | None
    current_label: str
    timezone_id: str
    paused: bool
    editable: bool
    supported: bool
    checked_at: str
    management_available: bool | None = None
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


SCHEDULE_INTERVAL_OPTIONS: tuple[int, ...] = (1, 3, 6, 12, 24)


def run_now_permission_guidance(job_id: int | None) -> str | None:
    if job_id is None:
        return None
    return f"Grant the app service principal CAN_MANAGE_RUN on job {int(job_id)}."


def _workflow_env_names(workflow_kind: str) -> tuple[str, str]:
    if workflow_kind == "bootstrap":
        return "BOOTSTRAP_REFRESH_JOB_ID", "BOOTSTRAP_REFRESH_JOB_NAME"
    return "REFRESH_JOB_ID", "REFRESH_JOB_NAME"


def _workflow_display_name(workflow_kind: str) -> str:
    if workflow_kind == "bootstrap":
        return "bootstrap refresh workflow"
    return "shared refresh workflow"


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


def _configured_job_selector(workflow_kind: str = "shared") -> tuple[str, str]:
    if workflow_kind == "bootstrap":
        configured_job_id = _clean(getattr(settings, "bootstrap_refresh_job_id", ""))
        if configured_job_id:
            return "id", configured_job_id
        configured_job_name = _clean(getattr(settings, "bootstrap_refresh_job_name", ""))
        if configured_job_name:
            return "name", configured_job_name
        return "none", ""

    configured_job_id = _clean(settings.refresh_job_id)
    if configured_job_id:
        return "id", configured_job_id
    configured_job_name = _clean(settings.refresh_job_name)
    if configured_job_name:
        return "name", configured_job_name
    return "none", ""


def _load_job_by_id(workspace_client, job_id: int, *, workflow_kind: str = "shared") -> object:
    workflow_name = _workflow_display_name(workflow_kind)
    job_id_env_name, job_name_env_name = _workflow_env_names(workflow_kind)
    if hasattr(workspace_client.jobs, "get"):
        return workspace_client.jobs.get(job_id=job_id)
    for job in list(workspace_client.jobs.list()):
        if getattr(job, "job_id", None) and int(job.job_id) == int(job_id):
            return job
    raise RefreshJobLookupError(
        f"Configured {workflow_name} ID {job_id} does not exist in this workspace. "
        f"Set {job_id_env_name} or {job_name_env_name} to a deployed workflow and redeploy the app."
    )


def _pause_status(resource: object | None) -> str:
    raw = getattr(resource, "pause_status", None)
    return _clean(getattr(raw, "value", raw)).upper()


def _checked_at_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _schedule_interval_label(interval_hours: int | None) -> str:
    if interval_hours == 1:
        return "Every 1 Hour"
    if interval_hours == 24:
        return "Every 24 Hours"
    if interval_hours in SCHEDULE_INTERVAL_OPTIONS:
        return f"Every {int(interval_hours)} Hours"
    return "Custom / Unsupported"


def _schedule_interval_to_quartz(interval_hours: int) -> str:
    interval = int(interval_hours)
    if interval not in SCHEDULE_INTERVAL_OPTIONS:
        raise RefreshJobConfigError(
            f"Shared refresh schedule must be one of {', '.join(str(value) for value in SCHEDULE_INTERVAL_OPTIONS)} hours."
        )
    if interval == 1:
        return "0 0 * * * ?"
    if interval == 24:
        return "0 0 0 * * ?"
    return f"0 0 */{interval} * * ?"


def _quartz_interval_hours(expression: str) -> int | None:
    parts = [part.strip() for part in _clean(expression).split() if part.strip()]
    if len(parts) not in {6, 7}:
        return None
    _, minute, hour, day_of_month, _, day_of_week = parts[:6]
    if minute != "0":
        return None
    if day_of_month != "*" or day_of_week not in {"?", "*"}:
        return None
    normalized_hour = hour.lower()
    if normalized_hour == "*":
        return 1
    if normalized_hour.startswith("*/"):
        try:
            interval = int(normalized_hour.split("/", 1)[1])
        except ValueError:
            return None
        return interval if interval in SCHEDULE_INTERVAL_OPTIONS else None
    if normalized_hour.startswith("0/"):
        try:
            interval = int(normalized_hour.split("/", 1)[1])
        except ValueError:
            return None
        return interval if interval in SCHEDULE_INTERVAL_OPTIONS else None
    return None


def _principal_candidates(workspace_client) -> tuple[set[str], str | None]:
    try:
        identity = workspace_client.current_user.me()
    except Exception:
        return set(), "Could not verify the current app principal against refresh workflow permissions."

    principal_candidates = {
        _normalized_job_name(getattr(identity, "user_name", None)),
        _normalized_job_name(getattr(identity, "display_name", None)),
    }
    principal_candidates.discard("")
    if not principal_candidates:
        return set(), "Could not identify the current app principal to verify refresh workflow permissions."
    return principal_candidates, None


def _job_permission_levels(workspace_client, job_id: int) -> tuple[set[str] | None, str | None]:
    principal_candidates, identity_warning = _principal_candidates(workspace_client)
    if not principal_candidates:
        return None, identity_warning

    try:
        permissions = workspace_client.jobs.get_permissions(str(job_id))
    except Exception:
        return None, "Could not read refresh workflow permissions."

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
        return permission_levels, None
    return None, "Could not confirm refresh workflow permissions from the job ACL."


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
    permission_levels, warning = _job_permission_levels(workspace_client, job_id)
    if permission_levels is None:
        return None, (
            "Could not verify immediate Run now permission. Direct trigger may still work, but the app could not inspect the current principal."
            if warning is None
            else f"{warning} Direct trigger may still work."
        )
    if permission_levels.intersection({"IS_OWNER", "CAN_MANAGE", "CAN_MANAGE_RUN"}):
        return True, None
    return False, None


def _direct_schedule_manage_permission(workspace_client, job_id: int) -> tuple[bool | None, str | None]:
    permission_levels, warning = _job_permission_levels(workspace_client, job_id)
    if permission_levels is None:
        return None, (
            "Could not verify shared workflow schedule-edit permission."
            if warning is None
            else f"{warning} Schedule editing may still work."
        )
    if permission_levels.intersection({"IS_OWNER", "CAN_MANAGE"}):
        return True, None
    return False, None


def resolve_refresh_workflow_status(
    workspace_client=None,
    *,
    workflow_kind: str = "shared",
    require_scheduler_path: bool = True,
) -> RefreshWorkflowStatus:
    configured_via, configured_value = _configured_job_selector(workflow_kind)
    job_id_env_name, job_name_env_name = _workflow_env_names(workflow_kind)
    workflow_name = _workflow_display_name(workflow_kind)
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
            blocking_issues=(
                (f"Set {job_id_env_name} (preferred) or {job_name_env_name} in the app environment.",)
                if workflow_kind == "shared"
                else ()
            ),
            warnings=(),
        )

    client = workspace_client or _workspace_client()
    warnings: list[str] = []
    if configured_via == "name":
        warnings.append(f"{job_name_env_name} works, but {job_id_env_name} is more reliable for production wiring.")

    try:
        job_id = resolve_refresh_job_id(client, workflow_kind=workflow_kind)
        job = _load_job_by_id(client, job_id, workflow_kind=workflow_kind)
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
    if require_scheduler_path and scheduler_issue:
        blocking_issues.append(scheduler_issue)

    job_settings = getattr(job, "settings", None)
    queue_enabled = getattr(getattr(job_settings, "queue", None), "enabled", None)
    max_concurrent_runs = getattr(job_settings, "max_concurrent_runs", None)
    if workflow_kind == "shared":
        if queue_enabled is not True:
            warnings.append("The shared refresh workflow should enable queueing so overlapping runs wait instead of failing.")
        if max_concurrent_runs != 1:
            warnings.append("The shared refresh workflow should set max_concurrent_runs=1 to avoid overlap.")

    run_now_available, run_now_warning = _direct_run_now_permission(client, int(job_id))
    if run_now_warning:
        warnings.append(run_now_warning)
    if run_now_available is False:
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
    workflow_status = resolve_refresh_workflow_status(workspace_client=workspace_client, workflow_kind="shared")
    bootstrap_selector_via, _ = _configured_job_selector("bootstrap")
    bootstrap_mode = "shared_default"
    bootstrap_status: RefreshWorkflowStatus | None = None
    if bootstrap_selector_via != "none":
        bootstrap_mode = "separate"
        bootstrap_status = resolve_refresh_workflow_status(
            workspace_client=workspace_client,
            workflow_kind="bootstrap",
            require_scheduler_path=False,
        )
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
        blocking_issues.append(
            "SQL_WAREHOUSE_ID is not configured for this app. For Git-based app deploys, set a literal SQL_WAREHOUSE_ID value in app.yaml."
        )
    blocking_issues.extend(workflow_status.blocking_issues)
    if bootstrap_status is not None:
        warnings.extend(bootstrap_status.blocking_issues)
        warnings.extend(bootstrap_status.warnings)
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
        bootstrap_workflow_mode=bootstrap_mode,
        bootstrap_workflow_resolved=(
            bootstrap_status.resolved if bootstrap_status is not None else workflow_status.resolved
        ),
        bootstrap_workflow_name=(
            bootstrap_status.job_name if bootstrap_status is not None else workflow_status.job_name
        ),
        bootstrap_workflow_id=(
            bootstrap_status.job_id if bootstrap_status is not None else workflow_status.job_id
        ),
        bootstrap_workflow_configured_via=(
            bootstrap_status.configured_via if bootstrap_status is not None else workflow_status.configured_via
        ),
        bootstrap_workflow_configured_value=(
            bootstrap_status.configured_value if bootstrap_status is not None else workflow_status.configured_value
        ),
        bootstrap_run_now_available=(
            bootstrap_status.run_now_available if bootstrap_status is not None else workflow_status.run_now_available
        ),
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


def resolve_shared_workflow_schedule_status(workspace_client=None) -> SharedWorkflowScheduleStatus:
    workflow_status = resolve_refresh_workflow_status(
        workspace_client=workspace_client,
        workflow_kind="shared",
        require_scheduler_path=False,
    )
    checked_at = _checked_at_text()
    warnings = list(workflow_status.warnings)
    blocking_issues = list(workflow_status.blocking_issues)
    if not workflow_status.configured or not workflow_status.resolved or workflow_status.job_id is None:
        return SharedWorkflowScheduleStatus(
            configured=workflow_status.configured,
            resolved=workflow_status.resolved,
            job_id=workflow_status.job_id,
            job_name=workflow_status.job_name,
            scheduler_mode=workflow_status.scheduler_mode,
            current_expression="",
            current_interval_hours=None,
            current_label="Unavailable",
            timezone_id="UTC",
            paused=False,
            editable=False,
            supported=False,
            checked_at=checked_at,
            management_available=None,
            blocking_issues=tuple(blocking_issues),
            warnings=tuple(warnings),
        )

    client = workspace_client or _workspace_client()
    job = _load_job_by_id(client, workflow_status.job_id, workflow_kind="shared")
    job_settings = getattr(job, "settings", None)
    schedule = getattr(job_settings, "schedule", None)
    if schedule is None:
        current_label = {
            "trigger": "Shared workflow uses a trigger, not a cron schedule.",
            "continuous": "Shared workflow uses continuous execution, not a cron schedule.",
            "missing": "Shared workflow has no editable cron schedule.",
        }.get(workflow_status.scheduler_mode, "Shared workflow does not expose a cron schedule.")
        manage_available, manage_warning = _direct_schedule_manage_permission(client, workflow_status.job_id)
        if manage_warning:
            warnings.append(manage_warning)
        return SharedWorkflowScheduleStatus(
            configured=True,
            resolved=True,
            job_id=workflow_status.job_id,
            job_name=workflow_status.job_name,
            scheduler_mode=workflow_status.scheduler_mode,
            current_expression="",
            current_interval_hours=None,
            current_label=current_label,
            timezone_id="UTC",
            paused=False,
            editable=False,
            supported=False,
            checked_at=checked_at,
            management_available=manage_available,
            blocking_issues=tuple(blocking_issues),
            warnings=tuple(warnings),
        )

    current_expression = _clean(getattr(schedule, "quartz_cron_expression", None))
    timezone_id = _clean(getattr(schedule, "timezone_id", None)) or "UTC"
    paused = _pause_status(schedule) == "PAUSED"
    interval_hours = _quartz_interval_hours(current_expression)
    manage_available, manage_warning = _direct_schedule_manage_permission(client, workflow_status.job_id)
    if manage_warning:
        warnings.append(manage_warning)
    supported = interval_hours in SCHEDULE_INTERVAL_OPTIONS
    if not supported:
        warnings.append(
            "Shared workflow uses a custom cron schedule. The app can display it, but only the standard 1/3/6/12/24-hour options are editable here."
        )
    return SharedWorkflowScheduleStatus(
        configured=True,
        resolved=True,
        job_id=workflow_status.job_id,
        job_name=workflow_status.job_name,
        scheduler_mode=workflow_status.scheduler_mode,
        current_expression=current_expression,
        current_interval_hours=interval_hours,
        current_label=_schedule_interval_label(interval_hours) if supported else f"Custom cron: {current_expression or '(unset)'}",
        timezone_id=timezone_id,
        paused=paused,
        editable=bool(manage_available) and supported,
        supported=supported,
        checked_at=checked_at,
        management_available=manage_available,
        blocking_issues=tuple(blocking_issues),
        warnings=tuple(warnings),
    )


def update_shared_workflow_schedule(interval_hours: int, workspace_client=None) -> SharedWorkflowScheduleStatus:
    client = workspace_client or _workspace_client()
    status = resolve_shared_workflow_schedule_status(client)
    if not status.configured or not status.resolved or status.job_id is None:
        raise RefreshJobConfigError("Shared refresh workflow is not configured or could not be resolved.")
    if not status.supported:
        raise RefreshJobConfigError(
            "Shared workflow schedule editing is only supported when the job already uses a cron schedule."
        )
    if status.management_available is False:
        raise RefreshJobConfigError(
            f"The app principal cannot edit the shared workflow schedule for job {status.job_id}. Grant CAN_MANAGE or update the job externally."
        )

    from databricks.sdk.service import jobs

    pause_status = jobs.PauseStatus.PAUSED if status.paused else jobs.PauseStatus.UNPAUSED
    new_schedule = jobs.CronSchedule(
        quartz_cron_expression=_schedule_interval_to_quartz(int(interval_hours)),
        timezone_id=status.timezone_id or "UTC",
        pause_status=pause_status,
    )
    client.jobs.update(
        job_id=int(status.job_id),
        new_settings=jobs.JobSettings(schedule=new_schedule),
    )
    return resolve_shared_workflow_schedule_status(client)


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
        raise RuntimeError(
            "SQL_WAREHOUSE_ID is not configured for refresh job execution. For Git-based app deploys, set a literal SQL_WAREHOUSE_ID value in app.yaml."
        )

    params: dict[str, str] = {
        "warehouse_id": warehouse_id,
        "control_plane_catalog": _clean(control_plane_catalog),
        "control_plane_schema": _clean(control_plane_schema),
        "scope": _clean(scope) or "scheduler",
    }
    if _clean(model_key):
        params["model_key"] = _clean(model_key)

    resolved_lakebase_schema = _clean(lakebase_schema) or _clean(settings.lakebase_schema)
    resolved_lakebase_instance = _clean(lakebase_instance_name)
    resolved_lakebase_database = _clean(lakebase_database_name)
    resolved_lakebase_host = _clean(settings.lakebase_host)

    if resolved_lakebase_database and (resolved_lakebase_instance or resolved_lakebase_host):
        params["use_lakebase_read_model"] = "true"
        if resolved_lakebase_instance:
            params["lakebase_instance_name"] = resolved_lakebase_instance
        if resolved_lakebase_database:
            params["lakebase_database_name"] = resolved_lakebase_database
        if resolved_lakebase_host:
            params["lakebase_host"] = resolved_lakebase_host
        if settings.lakebase_port:
            params["lakebase_port"] = str(settings.lakebase_port)
        if _clean(settings.lakebase_pguser):
            params["lakebase_pguser"] = _clean(settings.lakebase_pguser)
        if _clean(settings.lakebase_sslmode):
            params["lakebase_sslmode"] = _clean(settings.lakebase_sslmode)
        if resolved_lakebase_schema:
            params["lakebase_schema"] = resolved_lakebase_schema

    return params


def resolve_refresh_job_id(workspace_client=None, *, workflow_kind: str = "shared", allow_shared_fallback: bool = False) -> int:
    configured_job_id, refresh_job_name = "", ""
    configured_via, configured_value = _configured_job_selector(workflow_kind)
    job_id_env_name, job_name_env_name = _workflow_env_names(workflow_kind)
    workflow_name = _workflow_display_name(workflow_kind)

    if configured_via == "none" and workflow_kind == "bootstrap" and allow_shared_fallback:
        return resolve_refresh_job_id(workspace_client, workflow_kind="shared", allow_shared_fallback=False)

    configured_job_id = configured_value if configured_via == "id" else ""
    if configured_job_id:
        try:
            return int(configured_job_id)
        except ValueError as error:
            raise RefreshJobConfigError(f"{job_id_env_name} must be an integer, got {configured_job_id!r}.") from error

    refresh_job_name = configured_value if configured_via == "name" else ""
    if not refresh_job_name:
        raise RefreshJobConfigError(f"Neither {job_id_env_name} nor {job_name_env_name} is configured.")

    client = workspace_client or _workspace_client()
    exact_matches = _jobs_with_exact_name(list(client.jobs.list(name=refresh_job_name, limit=25)), refresh_job_name)
    if len(exact_matches) > 1:
        if workflow_kind == "shared":
            raise RefreshJobLookupError(
                f"Multiple refresh jobs are named {refresh_job_name!r}. "
                f"Set {job_id_env_name} explicitly so the app triggers the correct workflow."
            )
        raise RefreshJobLookupError(
            f"Multiple {workflow_name}s are named {refresh_job_name!r}. "
            f"Set {job_id_env_name} explicitly so the app triggers the correct workflow."
        )
    if len(exact_matches) == 1:
        return int(exact_matches[0].job_id)

    all_jobs = list(client.jobs.list())
    all_exact_matches = _jobs_with_exact_name(all_jobs, refresh_job_name)
    if len(all_exact_matches) > 1:
        if workflow_kind == "shared":
            raise RefreshJobLookupError(
                f"Multiple refresh jobs are named {refresh_job_name!r}. "
                f"Set {job_id_env_name} explicitly so the app triggers the correct workflow."
            )
        raise RefreshJobLookupError(
            f"Multiple {workflow_name}s are named {refresh_job_name!r}. "
            f"Set {job_id_env_name} explicitly so the app triggers the correct workflow."
        )
    if len(all_exact_matches) == 1:
        return int(all_exact_matches[0].job_id)

    suffix_matches = _jobs_with_suffix_name(all_jobs, refresh_job_name)
    if len(suffix_matches) == 1:
        return int(suffix_matches[0].job_id)
    if len(suffix_matches) > 1:
        if workflow_kind == "shared":
            raise RefreshJobLookupError(
                f"Multiple refresh jobs end with {refresh_job_name!r}. "
                f"Set {job_id_env_name} explicitly so the app triggers the correct workflow."
            )
        raise RefreshJobLookupError(
            f"Multiple {workflow_name}s end with {refresh_job_name!r}. "
            f"Set {job_id_env_name} explicitly so the app triggers the correct workflow."
        )

    contains_matches = _jobs_with_contains_name(all_jobs, refresh_job_name)
    if len(contains_matches) == 1:
        return int(contains_matches[0].job_id)
    if len(contains_matches) > 1:
        if workflow_kind == "shared":
            raise RefreshJobLookupError(
                f"Multiple refresh jobs contain {refresh_job_name!r}. "
                f"Set {job_id_env_name} explicitly so the app triggers the correct workflow."
            )
        raise RefreshJobLookupError(
            f"Multiple {workflow_name}s contain {refresh_job_name!r}. "
            f"Set {job_id_env_name} explicitly so the app triggers the correct workflow."
        )

    if workflow_kind == "shared":
        raise RefreshJobLookupError(
            f"Could not find a refresh job matching {refresh_job_name!r}. "
            f"The app only triggers an existing shared workflow; set {job_id_env_name} or {job_name_env_name} to the deployed workflow and redeploy the app."
        )
    raise RefreshJobLookupError(
        f"Could not find a {workflow_name} matching {refresh_job_name!r}. "
        f"The app only triggers an existing workflow; set {job_id_env_name} or {job_name_env_name} to the deployed workflow and redeploy the app."
    )


@dataclass(frozen=True)
class RefreshJobTrigger:
    job_id: int
    run_id: int | None
    workflow_kind: str = "shared"
    used_shared_fallback: bool = False


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
    resolved_scope = _clean(scope).lower() or "scheduler"
    requested_workflow_kind = "bootstrap" if resolved_scope in {"bootstrap", "backfill"} else "shared"
    configured_bootstrap = _configured_job_selector("bootstrap")[0] != "none"
    workflow_kind = requested_workflow_kind
    used_shared_fallback = False
    if requested_workflow_kind == "bootstrap" and not configured_bootstrap:
        workflow_kind = "shared"
        used_shared_fallback = True

    job_id = resolve_refresh_job_id(
        client,
        workflow_kind=workflow_kind,
        allow_shared_fallback=requested_workflow_kind == "bootstrap",
    )
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
        job_parameters=named_params,
        idempotency_token=str(uuid4()),
    )
    response = getattr(waiter, "response", None)
    run_id = getattr(response, "run_id", None)
    return RefreshJobTrigger(
        job_id=job_id,
        run_id=int(run_id) if run_id is not None else None,
        workflow_kind=workflow_kind,
        used_shared_fallback=used_shared_fallback,
    )


def _workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()
