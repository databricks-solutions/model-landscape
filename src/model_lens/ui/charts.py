from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_lens.domain.performance_metrics import performance_metric_label
from model_lens.services.thresholds import drift_severity, get_thresholds
from model_lens.ui.styles import COLORS


LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
    margin=dict(l=80, r=40, t=70, b=60),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)

PERFORMANCE_METRIC_COLORS = {
    "precision": COLORS["blue"],
    "recall": COLORS["highlight"],
    "f1": COLORS["cyan"],
    "accuracy": COLORS["purple"],
    "mae": COLORS["blue"],
    "rmse": COLORS["highlight"],
}

DRIFT_HEATMAP_ROBUST_PERCENTILE = 95.0
DRIFT_HEATMAP_MIN_POSITIVE_CELLS = 8
DRIFT_HEATMAP_MAX_HEIGHT = 1000


def _apply_layout(fig, **kwargs):
    fig.update_layout(**{**LAYOUT_DEFAULTS, **kwargs})
    return fig


def _metric_color(
    value: float | int | None,
    metric: str,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> str:
    severity = drift_severity(value, metric, thresholds)
    if severity == "critical":
        return COLORS["high"]
    if severity == "warning":
        return COLORS["moderate"]
    return COLORS["low"]


def _numeric_array(values) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values) if not isinstance(values, pd.Series) else values), errors="coerce").dropna()
    return series.to_numpy(dtype=float)


def _metric_tickformat(metric: str, values: pd.Series | np.ndarray | list[float]) -> str:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    max_abs = float(series.abs().max()) if not series.empty else 0.0
    if metric != "null_rate" and max_abs > 0 and max_abs < 0.01:
        return ".2e"
    if metric == "null_rate":
        return ".2f"
    return ".4f"


def _format_metric_value(metric: str, value: float, values: pd.Series | np.ndarray | list[float]) -> str:
    tickformat = _metric_tickformat(metric, values)
    if tickformat == ".2e":
        return f"{float(value):.2e}"
    if tickformat == ".2f":
        return f"{float(value):.2f}"
    return f"{float(value):.4f}"


def _apply_percentile_trim(values: np.ndarray, percentile: float | None) -> np.ndarray:
    if percentile is None or percentile <= 0 or percentile >= 50 or values.size == 0:
        return values
    lower = np.nanpercentile(values, percentile)
    upper = np.nanpercentile(values, 100 - percentile)
    return values[(values >= lower) & (values <= upper)]


def _apply_iqr_fence(values: np.ndarray, multiplier: float | None) -> np.ndarray:
    if multiplier is None or multiplier <= 0 or values.size == 0:
        return values
    q1 = np.nanpercentile(values, 25.0)
    q3 = np.nanpercentile(values, 75.0)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return values
    lower = q1 - (multiplier * iqr)
    upper = q3 + (multiplier * iqr)
    return values[(values >= lower) & (values <= upper)]


def _apply_outlier_filter(values: np.ndarray, mode: str = "off", value: float | None = None) -> np.ndarray:
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode == "percentile_clip":
        return _apply_percentile_trim(values, value)
    if normalized_mode == "iqr_fence":
        return _apply_iqr_fence(values, value)
    return values


def _heatmap_colorscale(
    metric: str,
    *,
    zmax: float,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> list[list[float | str]]:
    warning, critical = get_thresholds(metric, thresholds)
    safe_max = max(float(zmax), float(critical) * 1.5, 1e-9)
    warning_stop = min(max(warning / safe_max, 0.0), 1.0)
    critical_stop = min(max(critical / safe_max, warning_stop), 1.0)
    return [
        [0.0, COLORS["low"]],
        [warning_stop, COLORS["low"]],
        [warning_stop, COLORS["moderate"]],
        [critical_stop, COLORS["moderate"]],
        [critical_stop, COLORS["high"]],
        [1.0, COLORS["high"]],
    ]


def _neutral_heatmap_colorscale() -> list[list[float | str]]:
    return [
        [0.0, "#0f1b2d"],
        [0.35, "#1f4e79"],
        [0.7, COLORS["blue"]],
        [1.0, COLORS["cyan"]],
    ]


def describe_drift_heatmap_scale(
    df: pd.DataFrame,
    metric: str = "psi",
    *,
    show_thresholds: bool = False,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    if df.empty:
        _, critical = get_thresholds(metric, thresholds)
        zmax = critical * 1.5 if show_thresholds else 1.0
        return {
            "zmax": zmax,
            "colorscale": _heatmap_colorscale(metric, zmax=zmax, thresholds=thresholds)
            if show_thresholds
            else _neutral_heatmap_colorscale(),
            "tickformat": _metric_tickformat(metric, []),
            "clip_note": "",
            "clip_cap": None,
            "clip_count": 0,
        }

    pivot = df.pivot_table(index="feature", columns="period", values=metric, aggfunc="max").sort_index()
    values = pd.to_numeric(pd.Series(pivot.values.ravel()), errors="coerce").dropna()
    positive_values = values[values > 0]
    raw_max = float(values.max()) if not values.empty else 0.0
    visual_cap = raw_max
    clip_note = ""
    clip_cap: float | None = None
    clip_count = 0
    if positive_values.size >= DRIFT_HEATMAP_MIN_POSITIVE_CELLS:
        percentile_cap = float(np.nanpercentile(positive_values.to_numpy(dtype=float), DRIFT_HEATMAP_ROBUST_PERCENTILE))
        if np.isfinite(percentile_cap) and percentile_cap > 0 and percentile_cap < raw_max:
            visual_cap = percentile_cap
            clip_cap = percentile_cap
            clip_count = int((values > percentile_cap).sum())
            clip_note = (
                f"Heatmap color range capped at the {int(DRIFT_HEATMAP_ROBUST_PERCENTILE)}th percentile "
                f"({_format_metric_value(metric, percentile_cap, values)}). "
                f"{clip_count} cell{'s' if clip_count != 1 else ''} exceed the cap and render at the top color."
            )

    _, critical = get_thresholds(metric, thresholds)
    if show_thresholds:
        zmax = max(visual_cap, critical * 1.5) if raw_max > 0 else critical * 1.5
        colorscale = _heatmap_colorscale(metric, zmax=zmax, thresholds=thresholds)
    else:
        zmax = max(visual_cap, 1e-9) if raw_max > 0 else 1.0
        colorscale = _neutral_heatmap_colorscale()
    return {
        "zmax": zmax,
        "colorscale": colorscale,
        "tickformat": _metric_tickformat(metric, values),
        "clip_note": clip_note,
        "clip_cap": clip_cap,
        "clip_count": clip_count,
    }


def _ranked_bar_colors(count: int) -> list[str]:
    palette = px.colors.sequential.Tealgrn
    if count <= 0:
        return []
    if count <= len(palette):
        return palette[:count]
    return [palette[index % len(palette)] for index in range(count)]


def build_drift_heatmap(
    df: pd.DataFrame,
    metric: str = "psi",
    title: str = "Feature Drift Over Time",
    *,
    show_thresholds: bool = False,
    thresholds: dict[str, dict[str, float]] | None = None,
):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No drift data available", showarrow=False)
        return _apply_layout(fig, title=title)

    pivot = df.pivot_table(index="feature", columns="period", values=metric, aggfunc="max").sort_index()
    scale = describe_drift_heatmap_scale(
        df,
        metric=metric,
        show_thresholds=show_thresholds,
        thresholds=thresholds,
    )
    zmax = float(scale["zmax"])
    colorscale = scale["colorscale"]
    tickformat = str(scale["tickformat"])
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(column) for column in pivot.columns],
            y=pivot.index,
            zmin=0,
            zmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title=metric.upper(), tickfont=dict(color=COLORS["text"]), tickformat=tickformat),
            hovertemplate="<b>%{y}</b><br>Period: %{x}<br>" + f"{metric.upper()}: %{{z:{tickformat}}}<extra></extra>",
        )
    )
    return _apply_layout(
        fig,
        title=title,
        xaxis=dict(title="Time Period", gridcolor=COLORS["grid"], tickangle=-45),
        yaxis=dict(title="", gridcolor=COLORS["grid"], autorange="reversed", automargin=True),
        height=min(DRIFT_HEATMAP_MAX_HEIGHT, max(300, len(pivot.index) * 35 + 100)),
    )


def build_drift_timeline(
    df: pd.DataFrame,
    features: list[str],
    metric: str = "psi",
    *,
    show_thresholds: bool = False,
    thresholds: dict[str, dict[str, float]] | None = None,
    title: str | None = None,
):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No drift data available", showarrow=False)
        return _apply_layout(fig, title=f"{metric.upper()} Over Time")

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    selected_features = [feature for feature in features if feature in set(df["feature"].astype(str))]
    if not selected_features:
        selected_features = df["feature"].dropna().astype(str).drop_duplicates().tolist()
    period_count = int(df["period"].nunique()) if "period" in df.columns else 0
    tickformat = _metric_tickformat(metric, df.get(metric, pd.Series(dtype=float)))

    for index, feature in enumerate(selected_features):
        feature_frame = df[df["feature"] == feature].sort_values("period")
        if feature_frame.empty:
            continue
        short_name = feature[:30] + "..." if len(feature) > 30 else feature
        fig.add_trace(
            go.Scatter(
                x=feature_frame["period"].astype(str),
                y=feature_frame[metric],
                name=short_name,
                mode="markers" if int(feature_frame["period"].nunique()) < 2 else "lines+markers",
                line=dict(color=colors[index % len(colors)], width=2),
                marker=dict(size=8 if period_count < 2 else 5),
                hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>" + f"{metric.upper()}: %{{y:{tickformat}}}<extra></extra>",
            )
        )

    if not fig.data:
        fig.add_annotation(text="No feature timeline data available yet", showarrow=False)
    elif period_count < 2:
        fig.add_annotation(
            text="Only one comparison window is available. Run more refreshes to see trends over time.",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            xanchor="left",
            showarrow=False,
            font=dict(size=12, color=COLORS["muted"]),
        )
    if fig.data and show_thresholds:
        warning, critical = get_thresholds(metric, thresholds)
        fig.add_hline(
            y=warning,
            line_color=COLORS["moderate"],
            line_dash="dot",
            annotation_text=f"Warning ({warning:.4f})",
            annotation_position="bottom right",
            annotation_font_color=COLORS["text"],
            annotation_font_size=11,
            annotation_bgcolor=COLORS["card"],
        )
        fig.add_hline(
            y=critical,
            line_color=COLORS["high"],
            line_dash="dash",
            annotation_text=f"Critical ({critical:.4f})",
            annotation_position="top right",
            annotation_font_color=COLORS["text"],
            annotation_font_size=11,
            annotation_bgcolor=COLORS["card"],
        )

    return _apply_layout(
        fig,
        title=title or f"{metric.upper()} Over Time",
        xaxis_title="Time Period",
        yaxis=dict(title=metric.upper(), tickformat=tickformat, gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(size=9),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
        ),
        height=420,
        margin=dict(l=80, r=170 if len(selected_features) > 8 else 140, t=70, b=60),
    )


def build_top_drifters_bar(
    df: pd.DataFrame,
    metric: str = "psi",
    top_n: int = 10,
    title: str | None = None,
    *,
    show_thresholds: bool = False,
    thresholds: dict[str, dict[str, float]] | None = None,
):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No drift data available", showarrow=False)
        return _apply_layout(fig, title="Top Drifting Features")

    working = df[["feature", metric]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce").fillna(0.0)
    latest = (
        working.groupby("feature", as_index=False)[metric]
        .max()
        .nlargest(top_n, metric)
    )
    colors = (
        [_metric_color(value, metric, thresholds) for value in latest[metric]]
        if show_thresholds
        else _ranked_bar_colors(len(latest))
    )
    tickformat = _metric_tickformat(metric, latest[metric] if metric in latest.columns else pd.Series(dtype=float))
    fig = go.Figure(
        go.Bar(
            x=latest[metric],
            y=latest["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>" + f"{metric.upper()}: %{{x:{tickformat}}}<extra></extra>",
        )
    )
    return _apply_layout(
        fig,
        title=title or f"Top {top_n} Drifting Features (Historical Max)",
        xaxis=dict(title=metric.upper(), tickformat=tickformat, gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(autorange="reversed", gridcolor=COLORS["grid"], automargin=True),
        height=max(300, top_n * 35 + 80),
    )


def build_feature_distribution(
    reference,
    current,
    feature_name: str,
    *,
    binning_mode: str = "auto",
    n_bins: int = 40,
    custom_edges: list[float] | None = None,
    outlier_mode: str = "off",
    outlier_value: float | None = None,
):
    ref_clean = _numeric_array(reference)
    cur_clean = _numeric_array(current)
    ref_clean = _apply_outlier_filter(ref_clean, outlier_mode, outlier_value)
    cur_clean = _apply_outlier_filter(cur_clean, outlier_mode, outlier_value)
    if len(ref_clean) == 0 and len(cur_clean) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return _apply_layout(fig, title=f"Distribution: {feature_name}")

    fig = go.Figure()
    if binning_mode == "custom" and custom_edges and len(custom_edges) >= 2:
        baseline_hist, edges = np.histogram(ref_clean, bins=np.asarray(custom_edges, dtype=float), density=True)
        current_hist, _ = np.histogram(cur_clean, bins=np.asarray(custom_edges, dtype=float), density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        fig.add_trace(go.Bar(x=centers, y=baseline_hist, width=widths, name="Baseline", opacity=0.55, marker_color=COLORS["blue"]))
        fig.add_trace(go.Bar(x=centers, y=current_hist, width=widths, name="Current", opacity=0.55, marker_color=COLORS["highlight"]))
    else:
        histogram_kwargs = {} if binning_mode == "auto" else {"nbinsx": max(int(n_bins), 2)}
        fig.add_trace(
            go.Histogram(
                x=ref_clean,
                name="Baseline",
                opacity=0.6,
                marker_color=COLORS["blue"],
                histnorm="probability density",
                **histogram_kwargs,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=cur_clean,
                name="Current",
                opacity=0.6,
                marker_color=COLORS["highlight"],
                histnorm="probability density",
                **histogram_kwargs,
            )
        )
    return _apply_layout(
        fig,
        title=f"Distribution: {feature_name}",
        xaxis_title="Value",
        yaxis_title="Density",
        barmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=350,
    )


def build_volume_timeline(daily_volume: dict[str, int]):
    if not daily_volume:
        fig = go.Figure()
        fig.add_annotation(text="No daily monitoring rows available", showarrow=False)
        return _apply_layout(fig, title="Daily Monitoring Rows", height=300)
    dates = sorted(daily_volume.keys())
    volumes = [daily_volume[date] for date in dates]
    fig = go.Figure(go.Bar(x=[str(date) for date in dates], y=volumes, marker_color=COLORS["accent"]))
    return _apply_layout(fig, title="Daily Monitoring Rows", xaxis_title="Date", yaxis_title="Rows", height=300)


def build_quality_window_timeline(history_df: pd.DataFrame):
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No quality history available", showarrow=False)
        return _apply_layout(fig, title="Window Row Count")

    fig = go.Figure(
        go.Scatter(
            x=history_df["period"].astype(str),
            y=history_df["row_count"],
            mode="markers+text" if len(history_df) < 2 else "lines+markers",
            line=dict(color=COLORS["cyan"], width=3),
            marker=dict(size=10 if len(history_df) < 2 else 7),
            name="Rows",
            text=[f"{int(value):,}" for value in history_df["row_count"]] if len(history_df) < 2 else None,
            textposition="top center",
            fill="tozeroy",
            fillcolor="rgba(41,128,185,0.10)",
        )
    )
    if len(history_df) < 2:
        fig.add_annotation(
            text="Only one comparison window is available so far.",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            xanchor="left",
            showarrow=False,
            font=dict(size=12, color=COLORS["muted"]),
        )
    return _apply_layout(fig, title="Rows Per Comparison Window", xaxis_title="Window End", yaxis_title="Rows", height=300)


def build_null_rate_chart(
    null_rates: dict[str, float],
    *,
    show_thresholds: bool = False,
    thresholds: dict[str, dict[str, float]] | None = None,
):
    if not null_rates:
        fig = go.Figure()
        fig.add_annotation(text="No null rate data", showarrow=False)
        return _apply_layout(fig, title="Null Rates by Feature")

    sorted_items = sorted(null_rates.items(), key=lambda item: item[1], reverse=True)
    features = [item[0] for item in sorted_items[:15]]
    rates = [item[1] for item in sorted_items[:15]]
    colors = [_metric_color(rate, "null_rate", thresholds) for rate in rates]

    fig = go.Figure(
        go.Bar(
            x=rates,
            y=features,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Null Rate: %{x:.2f}%<extra></extra>",
        )
    )
    if rates and show_thresholds:
        _, critical = get_thresholds("null_rate", thresholds)
        fig.add_vline(x=critical, line_dash="dash", line_color=COLORS["high"])
    return _apply_layout(
        fig,
        title="Null Rates by Feature",
        xaxis_title="Null Rate (%)",
        yaxis=dict(autorange="reversed", gridcolor=COLORS["grid"], automargin=True),
        height=max(300, len(features) * 30 + 80),
    )


def build_null_rate_timeline(history_df: pd.DataFrame):
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No null-rate history available", showarrow=False)
        return _apply_layout(fig, title="Null Rate Trends")

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    single_period = len(history_df["period"].dropna().astype(str).unique()) < 2
    show_single_point_labels = single_period and history_df["feature"].dropna().nunique() <= 6
    for index, feature in enumerate(history_df["feature"].dropna().astype(str).drop_duplicates().tolist()):
        feature_frame = history_df[history_df["feature"] == feature].sort_values("period")
        fig.add_trace(
            go.Scatter(
                x=feature_frame["period"].astype(str),
                y=feature_frame["null_rate"],
                mode="markers+text" if single_period and show_single_point_labels else ("markers" if len(feature_frame) < 2 else "lines+markers"),
                name=feature,
                line=dict(color=colors[index % len(colors)], width=2),
                marker=dict(size=9 if len(feature_frame) < 2 else 6),
                text=[f"{float(value):.2f}%" for value in feature_frame["null_rate"]] if single_period and show_single_point_labels else None,
                textposition="top center",
            )
        )
    if single_period:
        fig.add_annotation(
            text="Only one comparison window is available so far.",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            xanchor="left",
            showarrow=False,
            font=dict(size=12, color=COLORS["muted"]),
        )
    return _apply_layout(fig, title="Null Rate Trends", xaxis_title="Window End", yaxis_title="Null Rate (%)", height=320)


def build_prediction_quality_timeline(history_df: pd.DataFrame):
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No prediction-history data available", showarrow=False)
        return _apply_layout(fig, title="Prediction Average Over Time")

    frame = history_df.sort_values("period").copy()
    frame["prediction_mean"] = pd.to_numeric(frame["prediction_mean"], errors="coerce")
    frame["prediction_std"] = pd.to_numeric(frame["prediction_std"], errors="coerce").fillna(0.0)
    lower = frame["prediction_mean"] - frame["prediction_std"]
    upper = frame["prediction_mean"] + frame["prediction_std"]

    fig = go.Figure()
    if len(frame) >= 2:
        fig.add_trace(
            go.Scatter(
                x=frame["period"].astype(str),
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["period"].astype(str),
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(26,188,156,0.12)",
                name="std band",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=frame["period"].astype(str),
            y=frame["prediction_mean"],
            mode="markers+text" if len(frame) < 2 else "lines+markers",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=10 if len(frame) < 2 else 7),
            name="prediction average",
            text=[f"{float(value):.4f}" for value in frame["prediction_mean"]] if len(frame) < 2 else None,
            textposition="top center",
        )
    )
    if len(frame) < 2:
        fig.add_annotation(
            text="Only one comparison window is available so far.",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            xanchor="left",
            showarrow=False,
            font=dict(size=12, color=COLORS["muted"]),
        )
    return _apply_layout(fig, title="Prediction Average Over Time", xaxis_title="Window End", yaxis_title="Prediction Average", height=320)


def build_multi_model_summary(model_drift_data: list[dict], metric: str = "psi"):
    if not model_drift_data:
        fig = go.Figure()
        fig.add_annotation(text="No multi-model data", showarrow=False)
        return _apply_layout(fig, title="Multi-Model Drift Summary")

    models = [item["model"] for item in model_drift_data]
    max_psi = [item["max_psi"] for item in model_drift_data]
    drifting_count = [item["drifting_features"] for item in model_drift_data]
    computing = [bool(item.get("computing")) for item in model_drift_data]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Max {metric.upper()} by Model", "Drifting Features Count"),
        horizontal_spacing=0.1,
    )
    colors = [
        COLORS["cyan"]
        if computing[index]
        else _metric_color(value, metric, model_drift_data[index].get("thresholds"))
        for index, value in enumerate(max_psi)
    ]
    max_trace = go.Bar(
        x=models,
        y=max_psi,
        marker_color=colors,
        name="Max PSI",
        text=["Computing" if is_computing else None for is_computing in computing],
        textposition="auto",
        cliponaxis=False,
    )
    fig.add_trace(max_trace, row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=drifting_count, marker_color=COLORS["blue"], name="Features > Warning"), row=1, col=2)
    fig.update_xaxes(tickangle=-20, automargin=True, row=1, col=1)
    fig.update_xaxes(tickangle=-20, automargin=True, row=1, col=2)
    return _apply_layout(fig, title="Multi-Model Drift Overview", showlegend=False, height=420)


def build_dimension_breakdown(breakdown_df: pd.DataFrame, feature_name: str, dimension_name: str):
    if breakdown_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No dimension data available", showarrow=False)
        return _apply_layout(fig, title=f"{feature_name} by {dimension_name}")

    working = breakdown_df.copy()
    for column in ("feature_average", "feature_p25", "feature_p50", "feature_p75"):
        if column not in working.columns:
            working[column] = np.nan
    if "row_count" not in working.columns:
        working["row_count"] = 0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=working["dimension_value"],
            y=working["feature_average"],
            name="Average",
            marker_color=COLORS["blue"],
            customdata=working[["feature_p25", "feature_p50", "feature_p75", "row_count"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Average: %{y:.4f}<br>"
                "P25: %{customdata[0]:.4f}<br>"
                "P50: %{customdata[1]:.4f}<br>"
                "P75: %{customdata[2]:.4f}<br>"
                "Rows: %{customdata[3]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=working["dimension_value"],
            y=working["feature_p50"],
            name="Median (P50)",
            mode="markers+lines+text" if len(working) <= 12 else "markers+lines",
            line=dict(color=COLORS["highlight"], width=2),
            marker=dict(size=7),
            text=(
                [
                    f"Median={p50:.4g}<br>P75={p75:.4g}<br>P25={p25:.4g}"
                    for p50, p75, p25 in zip(
                        working["feature_p50"].fillna(np.nan),
                        working["feature_p75"].fillna(np.nan),
                        working["feature_p25"].fillna(np.nan),
                    )
                ]
                if len(working) <= 12
                else None
            ),
            textposition="top center",
            textfont=dict(size=10, color=COLORS["text"]),
            cliponaxis=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=(working["feature_p75"] - working["feature_p50"]).clip(lower=0).fillna(0.0),
                arrayminus=(working["feature_p50"] - working["feature_p25"]).clip(lower=0).fillna(0.0),
                visible=True,
            ),
            customdata=working[["feature_average", "feature_p25", "feature_p75", "row_count"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Median (P50): %{y:.4f}<br>"
                "Average: %{customdata[0]:.4f}<br>"
                "P25: %{customdata[1]:.4f}<br>"
                "P75: %{customdata[2]:.4f}<br>"
                "Rows: %{customdata[3]:,.0f}<extra></extra>"
            ),
        )
    )
    return _apply_layout(
        fig,
        title=f"{feature_name} by {dimension_name}",
        xaxis=dict(title=dimension_name, tickangle=-35 if len(working) > 10 else 0, automargin=True),
        yaxis_title="Value",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=350,
    )


def build_performance_timeline(
    metrics_over_time: list[dict],
    metric_name: str = "f1",
    *,
    metric_names: list[str] | tuple[str, ...] | None = None,
):
    if not metrics_over_time:
        fig = go.Figure()
        fig.add_annotation(text="No performance data available", showarrow=False)
        return _apply_layout(fig, title=f"{metric_name.upper()} Over Time")

    frame = pd.DataFrame(metrics_over_time)
    requested_metrics = [
        str(value).strip().lower()
        for value in (metric_names or [metric_name])
        if str(value).strip()
    ]
    available_metrics = [value for value in requested_metrics if value in frame.columns]
    if not available_metrics:
        fig = go.Figure()
        fig.add_annotation(text=f"No {metric_name.upper()} values are available yet", showarrow=False)
        return _apply_layout(fig, title=f"{metric_name.upper()} Over Time")
    if not any(not pd.to_numeric(frame[current_metric], errors="coerce").dropna().empty for current_metric in available_metrics):
        fig = go.Figure()
        title = "Performance Metrics Over Time" if len(available_metrics) > 1 else f"{metric_name.upper()} Over Time"
        fig.add_annotation(
            text=(
                "Performance metrics are undefined for the available days"
                if len(available_metrics) > 1
                else f"{metric_name.upper()} is undefined for the available days"
            ),
            showarrow=False,
        )
        return _apply_layout(fig, title=title)
    fig = go.Figure()
    for index, current_metric in enumerate(available_metrics):
        current_values = pd.to_numeric(frame[current_metric], errors="coerce")
        if current_values.dropna().empty:
            continue
        trace_color = PERFORMANCE_METRIC_COLORS.get(current_metric, COLORS["cyan"])
        fig.add_trace(
            go.Scatter(
                x=frame["period"].astype(str),
                y=current_values,
                mode="markers+text" if len(frame) < 2 else "lines+markers",
                line=dict(color=trace_color, width=3 if index == 0 else 2),
                marker=dict(size=10 if len(frame) < 2 else 8, color=trace_color),
                name=performance_metric_label(current_metric),
                text=[f"{float(value):.4f}" if pd.notna(value) else "" for value in current_values] if len(frame) < 2 else None,
                textposition="top center",
                connectgaps=False,
            )
        )
    chart_title = "Performance Metrics Over Time" if len(available_metrics) > 1 else f"{metric_name.upper()} Over Time"
    if len(frame) < 2:
        fig.add_annotation(
            text="Only one comparison window is available. Run more refreshes to see trends over time.",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            xanchor="left",
            showarrow=False,
            font=dict(size=12, color=COLORS["muted"]),
        )
    return _apply_layout(
        fig,
        title=chart_title,
        xaxis_title="Period",
        yaxis_title="Metric Value" if len(available_metrics) > 1 else metric_name.upper(),
        height=350,
    )


def build_feature_bin_impact(degradation_df: pd.DataFrame, feature_contributors: pd.DataFrame):
    if degradation_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No degradation data", showarrow=False)
        return _apply_layout(fig, title="Feature Impact on Performance")

    contribution_values = pd.to_numeric(degradation_df.get("degradation_contribution"), errors="coerce").fillna(0.0)
    if not contribution_values.empty and float(contribution_values.abs().max()) < 1e-9:
        fig = go.Figure()
        fig.add_annotation(
            text="No significant degradation detected in the latest window",
            showarrow=False,
            font=dict(size=14, color=COLORS["text"]),
        )
        return _apply_layout(fig, title="Feature Impact on Performance")

    if not feature_contributors.empty:
        feature_order = feature_contributors.sort_values("weighted_delta")["feature"].tolist()
    else:
        totals = degradation_df.groupby("feature")["degradation_contribution"].sum()
        feature_order = totals.sort_values().index.tolist()

    fig = go.Figure()
    for feature in feature_order:
        feature_frame = degradation_df[degradation_df["feature"] == feature].copy()
        feature_frame["_bin_sort"] = feature_frame["bin_label"].apply(_bin_sort_key)
        feature_frame = feature_frame.sort_values(["_bin_sort", "bin_label"]).drop(columns=["_bin_sort"])
        for _, row in feature_frame.iterrows():
            delta = row["delta"]
            if delta < -0.02:
                color = COLORS["high"]
            elif delta < -0.005:
                color = "#e67e22"
            elif delta < 0:
                color = COLORS["moderate"]
            else:
                color = COLORS["low"]
            bar_width = abs(row["degradation_contribution"])
            fig.add_trace(
                go.Bar(
                    x=[bar_width if delta <= 0 else -bar_width],
                    y=[feature],
                    orientation="h",
                    marker_color=color,
                    marker_line=dict(color=COLORS["grid"], width=1.5),
                    opacity=0.96,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{feature}</b><br>"
                        f"Bin: {row['bin_label']}<br>"
                        f"Baseline: {row['baseline_metric']:.4f}<br>"
                        f"Current: {row['current_metric']:.4f}<br>"
                        f"Delta: {delta:+.4f}<br>"
                        f"Current Window Share: {row['current_volume_pct']:.1f}%<br>"
                        f"Weighted Contribution: {row['degradation_contribution']:.4f}"
                        "<extra></extra>"
                    ),
                )
            )

    for label, color in [
        ("Degraded (delta <= -2%)", COLORS["high"]),
        ("Degraded (-2% < delta < 0)", COLORS["moderate"]),
        ("Improved / stable (delta >= 0)", COLORS["low"]),
    ]:
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", marker_color=color, name=label, showlegend=True))

    fig.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
    fig.add_annotation(
        text="Colors classify raw slice delta. Bar position shows weighted contribution, so the zero line is the absolute guide.",
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        xanchor="left",
        showarrow=False,
        font=dict(size=11, color=COLORS["muted"]),
    )
    return _apply_layout(
        fig,
        title="Feature Impact on Performance — Per-Bin Breakdown",
        xaxis_title="Weighted Contribution to Metric Change",
        yaxis=dict(
            autorange="reversed",
            gridcolor=COLORS["grid"],
            categoryorder="array",
            categoryarray=feature_order,
            automargin=True,
        ),
        barmode="relative",
        height=max(350, len(feature_order) * 50 + 100),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def _bin_sort_key(label):
    import re

    match = re.search(r"[\[\(]([-\d.]+)", str(label))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return float("inf")
    return float("inf")


def build_bin_detail(bin_df: pd.DataFrame, feature_name: str, metric_name: str = "f1"):
    if bin_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No bin data", showarrow=False)
        return _apply_layout(fig, title=f"Bin Detail: {feature_name}")

    working = bin_df.copy()
    working["_sort"] = working["bin_label"].apply(_bin_sort_key)
    working = working.sort_values("_sort").drop(columns=["_sort"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=working["bin_label"], y=working["baseline_metric"], name="Baseline", marker_color=COLORS["blue"], opacity=0.7),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=working["bin_label"], y=working["current_metric"], name="Current", marker_color=COLORS["highlight"], opacity=0.7),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=working["bin_label"],
            y=working["current_volume_pct"],
            name="Volume %",
            mode="lines+markers",
            line=dict(color=COLORS["cyan"], width=2, dash="dot"),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text=metric_name.upper(), secondary_y=False)
    fig.update_yaxes(title_text="Volume %", secondary_y=True)
    return _apply_layout(
        fig,
        title=f"Bin Detail: {feature_name}",
        xaxis_title="Bin",
        barmode="group",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=400,
    )
def build_latest_window_metric_snapshot(snapshot: dict[str, object] | None):
    payload = snapshot or {}
    metrics = payload.get("metrics") or {}
    title = "Latest Window Performance Snapshot"
    if not payload.get("supported", True):
        fig = go.Figure()
        fig.add_annotation(
            text=str(payload.get("message") or "This monitor does not support latest-window performance metrics."),
            showarrow=False,
        )
        return _apply_layout(fig, title=title, height=300)
    if not metrics:
        fig = go.Figure()
        fig.add_annotation(
            text=str(payload.get("message") or "Latest-window performance metrics are unavailable until labeled daily facts exist."),
            showarrow=False,
        )
        return _apply_layout(fig, title=title, height=300)
    labels = ["Precision", "Recall", "F1", "Accuracy"]
    values = [metrics.get(label.lower()) for label in labels]
    bar_values = [float(value) if value is not None else 0.0 for value in values]
    colors = [COLORS["accent"] if value is not None else COLORS["muted"] for value in values]
    text = [f"{float(value):.4f}" if value is not None else "N/A" for value in values]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=bar_values,
            marker_color=colors,
            text=text,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Value: %{text}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    return _apply_layout(fig, title=title, xaxis_title="", yaxis_title="Metric Value", height=300)
