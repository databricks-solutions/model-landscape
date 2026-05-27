"""Decorator that adds MLflow logging plumbing to panel quick functions.

Every quick function returns a tuple ``(fig, data)`` from its implementation;
the :func:`quickfn` wrapper handles the ``log=True`` path (writes the existing
JSON panel artifact via :func:`panels.log_panel` plus a new interactive
HTML figure via :func:`mlflow.log_figure`) and returns the figure to the caller.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from mlflow_lens._version import __version__
from mlflow_lens.panels import PANEL_TYPES, log_panel


def quickfn(panel_type: str) -> Callable[[Callable[..., tuple]], Callable[..., Any]]:
    """Wrap a panel implementation with ``log=``/``run_id=`` plumbing.

    The wrapped function must return ``(fig, data)`` where ``fig`` is a Plotly
    figure and ``data`` is the payload accepted by :func:`panels.log_panel`.

    The returned callable accepts the same args/kwargs as the implementation,
    plus:

    * ``log: bool = False`` — log the panel JSON and HTML figure to the run.
    * ``run_id: str | None = None`` — log into that run instead of the active one.

    Reserved kwargs ``top_n``, ``labels``, ``extra`` are forwarded *both* to
    the implementation (so it can render them) *and* to :func:`log_panel`.
    """
    if panel_type not in PANEL_TYPES:
        raise ValueError(
            f"Unknown panel type {panel_type!r}. "
            f"Add it to mlflow_lens.panels.PANEL_TYPES first."
        )

    def decorator(impl: Callable[..., tuple]) -> Callable[..., Any]:
        @functools.wraps(impl)
        def runner(
            *args: Any,
            log: bool = False,
            run_id: str | None = None,
            **kwargs: Any,
        ) -> Any:
            fig, data = impl(*args, **kwargs)
            if log:
                _log_with_figure(
                    panel_type,
                    fig,
                    data,
                    run_id=run_id,
                    top_n=kwargs.get("top_n"),
                    labels=kwargs.get("labels"),
                    extra=kwargs.get("extra"),
                )
            return fig

        runner._panel_type = panel_type  # type: ignore[attr-defined]
        runner.__wrapped_impl__ = impl  # type: ignore[attr-defined]
        return runner

    return decorator


def _log_with_figure(
    panel_type: str,
    fig: Any,
    data: Any,
    *,
    run_id: str | None,
    top_n: int | None,
    labels: list[str] | None,
    extra: dict[str, Any] | None,
) -> None:
    """Log the panel JSON payload and its interactive Plotly HTML."""
    import mlflow

    if run_id is None and mlflow.active_run() is None:
        raise RuntimeError(
            "mlflow-lens: log=True but no active MLflow run. "
            "Wrap the call in `with mlflow.start_run():` or pass run_id=..."
        )

    if run_id is not None and (
        mlflow.active_run() is None or mlflow.active_run().info.run_id != run_id
    ):
        with mlflow.start_run(run_id=run_id, nested=mlflow.active_run() is not None):
            _do_log(panel_type, fig, data, top_n=top_n, labels=labels, extra=extra)
    else:
        _do_log(panel_type, fig, data, top_n=top_n, labels=labels, extra=extra)


def _do_log(
    panel_type: str,
    fig: Any,
    data: Any,
    *,
    top_n: int | None,
    labels: list[str] | None,
    extra: dict[str, Any] | None,
) -> None:
    import mlflow

    log_panel(panel_type, data, top_n=top_n, labels=labels, extra=extra)
    mlflow.log_figure(fig, f"lens/panels/{panel_type}.html")
    mlflow.set_tag(f"lens.figure.{panel_type}", "true")
    mlflow.set_tag("lens.version", __version__)
