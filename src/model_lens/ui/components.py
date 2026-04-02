from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from model_lens.ui.styles import (
    CARD_STYLE,
    COLORS,
    WIZARD_STEP_ACTIVE,
    WIZARD_STEP_COMPLETE,
    WIZARD_STEP_INACTIVE,
)


DEFAULT_THRESHOLDS = {
    "psi": {"warning": 0.1, "critical": 0.2},
    "js_divergence": {"warning": 0.05, "critical": 0.15},
    "kl_divergence": {"warning": 0.1, "critical": 0.3},
    "null_rate": {"warning": 1.0, "critical": 5.0},
}


def get_thresholds(metric: str = "psi") -> tuple[float, float]:
    thresholds = DEFAULT_THRESHOLDS.get(metric, {"warning": 0.1, "critical": 0.2})
    return thresholds["warning"], thresholds["critical"]


def get_drift_status(value: float | int | None, metric: str = "psi") -> tuple[str, str, str]:
    warning, critical = get_thresholds(metric)
    numeric = float(value or 0)
    if numeric > critical:
        return "Critical", "danger", COLORS["high"]
    if numeric > warning:
        return "Warning", "warning", COLORS["moderate"]
    return "Healthy", "success", COLORS["low"]


def make_metric_card(title: str, value: str, subtitle: str = "", color: str = "primary"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(title, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
                html.H3(value, className=f"text-{color} mb-0"),
                html.Small(subtitle, className="text-muted") if subtitle else None,
            ]
        ),
        style=CARD_STYLE,
        className="h-100",
    )


def make_chart_card(figure, class_name: str = "mb-3"):
    return dbc.Card(
        dbc.CardBody(dcc.Graph(figure=figure, config={"displayModeBar": False})),
        style=CARD_STYLE,
        className=class_name,
    )


def make_model_status_card(
    *,
    model_name: str,
    model_id: str,
    max_psi: float,
    avg_psi: float,
    drifting_count: int,
    total_features: int,
    description: str = "",
    max_null_rate: float | None = None,
    has_labels: bool = False,
    metric_label: str = "PSI",
    metric_key: str = "psi",
    computing: bool = False,
    freshness_status: str = "fresh",
    last_run_status: str = "",
):
    if computing:
        border = COLORS["cyan"]
        return dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5(model_name, className="mb-1 text-light"),
                                    dbc.Badge(
                                        [html.I(className="fas fa-spinner fa-spin me-1"), "Computing..."],
                                        color="info",
                                        className="mb-2",
                                    ),
                                ]
                            ),
                            dbc.Col(
                                html.I(
                                    className="fas fa-cog fa-spin fa-2x",
                                    style={"color": border, "opacity": "0.5"},
                                ),
                                width="auto",
                            ),
                        ],
                        className="align-items-start",
                    ),
                    html.P(description, className="text-muted mb-2", style={"fontSize": "0.75rem"})
                    if description
                    else None,
                    html.Hr(style={"borderColor": COLORS["grid"], "margin": "8px 0"}),
                    html.P(
                        [html.I(className="fas fa-clock me-2"), "Drift calculations are running. Results will appear shortly."],
                        className="text-muted mb-0",
                        style={"fontSize": "0.85rem"},
                    ),
                ]
            ),
            style={**CARD_STYLE, "borderLeft": f"4px solid {border}", "cursor": "pointer"},
            className="h-100",
        )

    status, badge_color, border = get_drift_status(max_psi, metric_key)
    badges = [dbc.Badge(status, color=badge_color, className="mb-2")]
    freshness_badges = {
        "pending_bootstrap": dbc.Badge("Pending Bootstrap", color="secondary", className="ms-1 mb-2"),
        "stale": dbc.Badge("Refresh Overdue", color="warning", className="ms-1 mb-2"),
        "failed": dbc.Badge("Last Run Failed", color="danger", className="ms-1 mb-2"),
        "manual": dbc.Badge("Manual Schedule", color="dark", className="ms-1 mb-2"),
    }
    if freshness_status in freshness_badges:
        badges.append(freshness_badges[freshness_status])
    if has_labels:
        badges.append(
            dbc.Badge([html.I(className="fas fa-tag me-1"), "Labels"], color="info", className="ms-1 mb-2")
        )
    if max_null_rate is not None and max_null_rate > 0:
        null_status, null_badge_color, _ = get_drift_status(max_null_rate, "null_rate")
        if null_status != "Healthy":
            badges.append(
                dbc.Badge(
                    [html.I(className="fas fa-exclamation-triangle me-1"), f"Nulls {max_null_rate:.1f}%"],
                    color=null_badge_color,
                    className="ms-1 mb-2",
                )
            )

    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col([html.H5(model_name, className="mb-1 text-light"), html.Div(badges)]),
                        dbc.Col(
                            html.I(className="fas fa-robot fa-2x", style={"color": border, "opacity": "0.7"}),
                            width="auto",
                        ),
                    ],
                    className="align-items-start",
                ),
                html.P(description, className="text-muted mb-2", style={"fontSize": "0.75rem"})
                if description
                else None,
                html.Small(
                    f"Last shared-job status: {last_run_status or 'not started'}",
                    className="text-muted d-block mb-2",
                ),
                html.Hr(style={"borderColor": COLORS["grid"], "margin": "8px 0"}),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small(f"Max {metric_label}", className="text-muted d-block"),
                                html.Span(
                                    f"{max_psi:.4f}",
                                    style={"color": border, "fontWeight": "600", "fontSize": "1.1rem"},
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.Small(f"Avg {metric_label}", className="text-muted d-block"),
                                html.Span(f"{avg_psi:.4f}", style={"fontWeight": "600", "fontSize": "1.1rem"}),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.Small("Drifting", className="text-muted d-block"),
                                html.Span(
                                    f"{drifting_count}/{total_features}",
                                    style={"fontWeight": "600", "fontSize": "1.1rem"},
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        style={**CARD_STYLE, "borderLeft": f"4px solid {border}", "cursor": "pointer"},
        className="h-100",
    )


def make_empty_state(message: str, icon: str = "fas fa-info-circle"):
    return html.Div(
        [
            html.I(className=f"{icon} fa-3x mb-3", style={"color": COLORS["muted"]}),
            html.P(message, className="text-muted"),
        ],
        className="text-center py-5",
    )


def make_wizard_step(step_num: int, label: str, current_step: int):
    if step_num < current_step:
        style = WIZARD_STEP_COMPLETE
        icon = html.I(className="fas fa-check", style={"fontSize": "0.8rem"})
    elif step_num == current_step:
        style = WIZARD_STEP_ACTIVE
        icon = html.Span(str(step_num))
    else:
        style = WIZARD_STEP_INACTIVE
        icon = html.Span(str(step_num))

    return html.Div(
        [
            html.Div(icon, style=style),
            html.Small(label, className="text-muted mt-1", style={"fontSize": "0.7rem"}),
        ],
        className="d-flex flex-column align-items-center mx-2",
    )
