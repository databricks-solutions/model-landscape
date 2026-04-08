from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_lens.services.thresholds import drift_severity, get_thresholds
from model_lens.ui.styles import COLORS


LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
    margin=dict(l=60, r=20, t=50, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)


def _apply_layout(fig, **kwargs):
    fig.update_layout(**{**LAYOUT_DEFAULTS, **kwargs})
    return fig


def _metric_color(value: float | int | None, metric: str) -> str:
    severity = drift_severity(value, metric)
    if severity == "critical":
        return COLORS["high"]
    if severity == "warning":
        return COLORS["moderate"]
    return COLORS["low"]


def _numeric_array(values) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values) if not isinstance(values, pd.Series) else values), errors="coerce").dropna()
    return series.to_numpy(dtype=float)


def _heatmap_colorscale(metric: str, *, zmax: float) -> list[list[float | str]]:
    warning, critical = get_thresholds(metric)
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


def build_drift_heatmap(df: pd.DataFrame, metric: str = "psi", title: str = "Feature Drift Over Time"):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No drift data available", showarrow=False)
        return _apply_layout(fig, title=title)

    pivot = df.pivot_table(index="feature", columns="period", values=metric, aggfunc="max").sort_index()
    values = pd.to_numeric(pd.Series(pivot.values.ravel()), errors="coerce").dropna()
    warning, critical = get_thresholds(metric)
    zmax = max(values.max(), critical * 1.5) if not values.empty else critical * 1.5
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(column) for column in pivot.columns],
            y=pivot.index,
            zmin=0,
            zmax=zmax,
            colorscale=_heatmap_colorscale(metric, zmax=zmax),
            colorbar=dict(title=metric.upper(), tickfont=dict(color=COLORS["text"])),
            hovertemplate="<b>%{y}</b><br>Period: %{x}<br>" + f"{metric.upper()}: %{{z:.4f}}<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"Warning {warning:.2f} | Critical {critical:.2f}",
        xref="paper",
        yref="paper",
        x=1,
        y=1.1,
        xanchor="right",
        showarrow=False,
        font=dict(size=11, color=COLORS["muted"]),
    )
    return _apply_layout(
        fig,
        title=title,
        xaxis=dict(title="Time Period", gridcolor=COLORS["grid"], tickangle=-45),
        yaxis=dict(title="", gridcolor=COLORS["grid"], autorange="reversed"),
        height=max(300, len(pivot.index) * 35 + 100),
    )


def build_drift_timeline(df: pd.DataFrame, features: list[str], metric: str = "psi", thresholds: dict | None = None):
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
            )
        )

    if thresholds:
        if "warning" in thresholds:
            fig.add_hline(
                y=thresholds["warning"],
                line_dash="dash",
                line_color=COLORS["moderate"],
                annotation_text="Warning",
            )
        if "critical" in thresholds:
            fig.add_hline(
                y=thresholds["critical"],
                line_dash="dash",
                line_color=COLORS["high"],
                annotation_text="Critical",
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

    return _apply_layout(
        fig,
        title=f"{metric.upper()} Over Time",
        xaxis_title="Time Period",
        yaxis_title=metric.upper(),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=10)),
        height=400,
    )


def build_top_drifters_bar(df: pd.DataFrame, metric: str = "psi", top_n: int = 10):
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
    colors = [_metric_color(value, metric) for value in latest[metric]]
    fig = go.Figure(
        go.Bar(
            x=latest[metric],
            y=latest["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>" + f"{metric.upper()}: %{{x:.4f}}<extra></extra>",
        )
    )
    return _apply_layout(
        fig,
        title=f"Top {top_n} Drifting Features (Historical Max)",
        xaxis_title=metric.upper(),
        yaxis=dict(autorange="reversed", gridcolor=COLORS["grid"]),
        height=max(300, top_n * 35 + 80),
    )


def build_feature_distribution(reference, current, feature_name: str, n_bins: int = 40):
    ref_clean = _numeric_array(reference)
    cur_clean = _numeric_array(current)
    if len(ref_clean) == 0 and len(cur_clean) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return _apply_layout(fig, title=f"Distribution: {feature_name}")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=ref_clean,
            nbinsx=n_bins,
            name="Baseline",
            opacity=0.6,
            marker_color=COLORS["blue"],
            histnorm="probability density",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=cur_clean,
            nbinsx=n_bins,
            name="Current",
            opacity=0.6,
            marker_color=COLORS["highlight"],
            histnorm="probability density",
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
        fig.add_annotation(text="No daily volume data available", showarrow=False)
        return _apply_layout(fig, title="Daily Inference Volume", height=300)
    dates = sorted(daily_volume.keys())
    volumes = [daily_volume[date] for date in dates]
    fig = go.Figure(go.Bar(x=[str(date) for date in dates], y=volumes, marker_color=COLORS["accent"]))
    return _apply_layout(fig, title="Daily Inference Volume", xaxis_title="Date", yaxis_title="Transaction Count", height=300)


def build_quality_window_timeline(history_df: pd.DataFrame):
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No quality history available", showarrow=False)
        return _apply_layout(fig, title="Window Row Count")

    fig = go.Figure(
        go.Scatter(
            x=history_df["period"].astype(str),
            y=history_df["row_count"],
            mode="markers" if len(history_df) < 2 else "lines+markers",
            line=dict(color=COLORS["cyan"], width=3),
            marker=dict(size=10 if len(history_df) < 2 else 7),
            name="Rows",
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


def build_null_rate_chart(null_rates: dict[str, float]):
    if not null_rates:
        fig = go.Figure()
        fig.add_annotation(text="No null rate data", showarrow=False)
        return _apply_layout(fig, title="Null Rates by Feature")

    sorted_items = sorted(null_rates.items(), key=lambda item: item[1], reverse=True)
    features = [item[0] for item in sorted_items[:15]]
    rates = [item[1] for item in sorted_items[:15]]
    colors = [_metric_color(rate, "null_rate") for rate in rates]

    fig = go.Figure(
        go.Bar(
            x=rates,
            y=features,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Null Rate: %{x:.2f}%<extra></extra>",
        )
    )
    if rates:
        _, critical = get_thresholds("null_rate")
        fig.add_vline(x=critical, line_dash="dash", line_color=COLORS["high"])
    return _apply_layout(
        fig,
        title="Null Rates by Feature",
        xaxis_title="Null Rate (%)",
        yaxis=dict(autorange="reversed", gridcolor=COLORS["grid"]),
        height=max(300, len(features) * 30 + 80),
    )


def build_null_rate_timeline(history_df: pd.DataFrame):
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No null-rate history available", showarrow=False)
        return _apply_layout(fig, title="Null Rate Trends")

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for index, feature in enumerate(history_df["feature"].dropna().astype(str).drop_duplicates().tolist()):
        feature_frame = history_df[history_df["feature"] == feature].sort_values("period")
        fig.add_trace(
            go.Scatter(
                x=feature_frame["period"].astype(str),
                y=feature_frame["null_rate"],
                mode="markers" if len(feature_frame) < 2 else "lines+markers",
                name=feature,
                line=dict(color=colors[index % len(colors)], width=2),
                marker=dict(size=9 if len(feature_frame) < 2 else 6),
            )
        )
    if len(history_df["period"].dropna().astype(str).unique()) < 2:
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
        return _apply_layout(fig, title="Prediction Mean Over Time")

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
            mode="markers" if len(frame) < 2 else "lines+markers",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=10 if len(frame) < 2 else 7),
            name="prediction mean",
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
    return _apply_layout(fig, title="Prediction Mean Over Time", xaxis_title="Window End", yaxis_title="Prediction Mean", height=320)


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
        subplot_titles=("Max PSI by Model", "Drifting Features Count"),
        horizontal_spacing=0.15,
    )
    colors = [COLORS["cyan"] if computing[index] else _metric_color(value, metric) for index, value in enumerate(max_psi)]
    max_trace = go.Bar(
        x=models,
        y=max_psi,
        marker_color=colors,
        name="Max PSI",
        text=["Computing" if is_computing else None for is_computing in computing],
        textposition="outside",
    )
    fig.add_trace(max_trace, row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=drifting_count, marker_color=COLORS["blue"], name="Features > Warning"), row=1, col=2)
    return _apply_layout(fig, title="Multi-Model Drift Overview", showlegend=False, height=350)


def build_dimension_breakdown(breakdown_df: pd.DataFrame, feature_name: str, dimension_name: str):
    if breakdown_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No dimension data available", showarrow=False)
        return _apply_layout(fig, title=f"{feature_name} by {dimension_name}")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=breakdown_df["dimension_value"],
            y=breakdown_df["feature_mean"],
            name="Mean",
            marker_color=COLORS["blue"],
            error_y=dict(type="data", array=breakdown_df["feature_std"], visible=True),
        )
    )
    fig.add_trace(
        go.Bar(
            x=breakdown_df["dimension_value"],
            y=breakdown_df["null_pct"],
            name="Null %",
            marker_color=COLORS["highlight"],
            opacity=0.7,
            yaxis="y2",
        )
    )
    return _apply_layout(
        fig,
        title=f"{feature_name} by {dimension_name}",
        xaxis_title=dimension_name,
        yaxis_title="Mean Value",
        yaxis2=dict(title="Null %", overlaying="y", side="right", gridcolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        barmode="group",
        height=350,
    )


def build_performance_timeline(metrics_over_time: list[dict], metric_name: str = "f1"):
    if not metrics_over_time:
        fig = go.Figure()
        fig.add_annotation(text="No performance data available", showarrow=False)
        return _apply_layout(fig, title=f"{metric_name.upper()} Over Time")

    frame = pd.DataFrame(metrics_over_time)
    if metric_name not in frame.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"No {metric_name.upper()} values are available yet", showarrow=False)
        return _apply_layout(fig, title=f"{metric_name.upper()} Over Time")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["period"].astype(str),
            y=frame[metric_name],
            mode="markers" if len(frame) < 2 else "lines+markers",
            line=dict(color=COLORS["cyan"], width=3),
            marker=dict(size=10 if len(frame) < 2 else 8),
            name=metric_name.upper(),
            fill="tozeroy",
            fillcolor="rgba(26,188,156,0.1)",
        )
    )
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
    return _apply_layout(fig, title=f"{metric_name.upper()} Over Time", xaxis_title="Period", yaxis_title=metric_name.upper(), height=350)


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
        feature_frame = degradation_df[degradation_df["feature"] == feature].sort_values("bin_label")
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
                    marker_line=dict(color=COLORS["bg"], width=0.5),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{feature}</b><br>"
                        f"Bin: {row['bin_label']}<br>"
                        f"Baseline: {row['baseline_metric']:.4f}<br>"
                        f"Current: {row['current_metric']:.4f}<br>"
                        f"Delta: {delta:+.4f}<br>"
                        f"Volume: {row['current_volume_pct']:.1f}%<br>"
                        f"Impact: {row['degradation_contribution']:.4f}"
                        "<extra></extra>"
                    ),
                )
            )

    for label, color in [("Degraded (> 2%)", COLORS["high"]), ("Degraded (< 2%)", COLORS["moderate"]), ("Improved", COLORS["low"])]:
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", marker_color=color, name=label, showlegend=True))

    fig.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
    return _apply_layout(
        fig,
        title="Feature Impact on Performance — Per-Bin Breakdown",
        xaxis_title="Degradation Impact (wider = more impact, left = worse)",
        yaxis=dict(autorange="reversed", gridcolor=COLORS["grid"], categoryorder="array", categoryarray=feature_order),
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


def build_prediction_distribution(values, prob_column: str = "prediction"):
    series = pd.Series(values)
    if series.empty:
        fig = go.Figure()
        fig.add_annotation(text="No prediction data", showarrow=False)
        return _apply_layout(fig, title="Prediction Score Distribution")

    fig = go.Figure(
        go.Histogram(
            x=series.dropna(),
            nbinsx=50,
            marker_color=COLORS["cyan"],
            opacity=0.8,
            histnorm="probability density",
        )
    )
    return _apply_layout(fig, title="Prediction Score Distribution", xaxis_title=prob_column, yaxis_title="Density", height=300)
