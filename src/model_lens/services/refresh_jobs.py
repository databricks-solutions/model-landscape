from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

from model_lens.config import settings


class RefreshJobConfigError(RuntimeError):
    pass


class RefreshJobLookupError(RuntimeError):
    pass


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


def build_refresh_job_named_params(
    *,
    model_key: str,
    control_plane_catalog: str,
    control_plane_schema: str,
    lakebase_instance_name: str = "",
    lakebase_database_name: str = "",
    lakebase_schema: str = "",
    scope: str = "scheduler",
) -> list[str]:
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
