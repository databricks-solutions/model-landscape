"""Build, cache, and load structured run summaries from MLflow experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.entities import Run

from mlflow_lens._artifacts import log_json_artifact
from mlflow_lens._compat import has_logged_models
from mlflow_lens._version import __version__

SUMMARY_ARTIFACT_PATH = "lens/summary.json"


def log(
    task: str,
    primary_metric: str,
    score: float,
    dataset: str = "",
    notes: str = "",
    extra: dict | None = None,
) -> dict:
    """Log a structured summary artifact to the active MLflow run.

    The summary is stored as a JSON artifact at ``lens/summary.json``
    and enables fast ranking, filtering, and comparison across runs.
    """
    payload: dict = {
        "lens_version": __version__,
        "schema_version": "1",
        "task": task,
        "primary_metric": primary_metric,
        "score": score,
        "dataset": dataset,
        "notes": notes,
    }
    if extra:
        payload.update(extra)

    mlflow.set_tag("lens.version", __version__)
    mlflow.set_tag("lens.has_summary", "true")

    log_json_artifact(payload, artifact_path="lens", filename="summary.json")
    return payload


def build(
    experiment_ids: list[str],
    tags: dict[str, str] | None = None,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Build or update a summary index for runs across experiments.

    If *cache_path* points to an existing index, only new runs are processed
    (append-only). Returns a DataFrame of all indexed run envelopes.
    """
    cached = load(cache_path) if cache_path else pd.DataFrame()
    cached_run_ids = set(cached["run_id"].tolist()) if not cached.empty else set()

    envelopes: list[dict] = []
    for eid in experiment_ids:
        for run in _search_all_runs(eid, tags):
            if run.info.run_id in cached_run_ids:
                continue
            envelopes.append(_run_to_envelope(run))

    if not envelopes:
        return cached

    new_df = pd.DataFrame(envelopes)
    combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
    return combined


_JSON_COLUMNS = {"metrics", "params", "tags", "summary"}


def cache(index: pd.DataFrame, path: str) -> None:
    """Persist the summary index to a parquet file (append-only semantics).

    Dict columns are serialized as JSON strings to avoid pyarrow struct issues.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = index.copy()
    for col in _JSON_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    df.to_parquet(path, index=False)


def load(path: str | None) -> pd.DataFrame:
    """Load a previously cached summary index. Returns empty DataFrame if missing."""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        for col in _JSON_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PAGE_SIZE = 1000


def _search_all_runs(
    experiment_id: str, tags: dict[str, str] | None = None
) -> list[Run]:
    """Paginate through all runs in an experiment, defeating the 1000-run cap."""
    filter_parts: list[str] = []
    if tags:
        for k, v in tags.items():
            filter_parts.append(f"tags.`{k}` = '{v}'")
    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    all_runs: list[Run] = []
    page_token: str | None = None
    while True:
        result = mlflow.tracking.MlflowClient().search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=_PAGE_SIZE,
            page_token=page_token,
        )
        all_runs.extend(result)
        page_token = result.token if hasattr(result, "token") else None
        if not page_token or len(result) < _PAGE_SIZE:
            break

    return all_runs


def _run_to_envelope(run: Run) -> dict:
    """Convert an MLflow Run into a Lens run envelope."""
    info = run.info
    data = run.data

    summary_data = _load_summary_artifact(info.run_id)
    logged_model_id = _get_logged_model_id(run) if has_logged_models() else None

    envelope = {
        "lens_version": __version__,
        "schema_version": "1",
        "run_id": info.run_id,
        "experiment_id": info.experiment_id,
        "run_name": info.run_name or "",
        "status": info.status,
        "start_time": info.start_time,
        "end_time": info.end_time,
        "duration_s": (
            (info.end_time - info.start_time) / 1000.0
            if info.end_time and info.start_time
            else None
        ),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "has_lens_artifacts": summary_data is not None,
        "metrics": data.metrics,
        "params": data.params,
        "tags": {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")},
        "summary": summary_data,
    }

    if logged_model_id:
        envelope["logged_model_id"] = logged_model_id

    return envelope


def _load_summary_artifact(run_id: str) -> dict | None:
    """Attempt to download and parse the lens/summary.json artifact from a run."""
    try:
        client = mlflow.tracking.MlflowClient()
        tmp_path = client.download_artifacts(run_id, "lens/summary.json")
        return json.loads(Path(tmp_path).read_text())
    except Exception:
        return None


def _get_logged_model_id(run: Run) -> str | None:
    """Extract the LoggedModel ID from an MLflow 3.x run, if any."""
    try:
        tags = run.data.tags
        # MLflow 3.x stores model info in tags
        for key, val in tags.items():
            if key.startswith("mlflow.loggedModel"):
                return val
    except Exception:
        pass
    return None
