from __future__ import annotations

COLORS = {
    "bg": "#1a1a2e",
    "card": "#16213e",
    "accent": "#0f3460",
    "highlight": "#e94560",
    "text": "#eee",
    "muted": "#888",
    "low": "#2ecc71",
    "moderate": "#f39c12",
    "high": "#e74c3c",
    "grid": "#2a2a4a",
    "blue": "#3498db",
    "cyan": "#1abc9c",
    "purple": "#9b59b6",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "border": f"1px solid {COLORS['grid']}",
    "borderRadius": "8px",
}

SIDEBAR_STYLE = {
    "backgroundColor": COLORS["card"],
    "padding": "20px",
    "borderRight": f"1px solid {COLORS['grid']}",
    "height": "100vh",
    "overflowY": "auto",
    "position": "fixed",
    "width": "280px",
}

CONTENT_STYLE = {
    "marginLeft": "300px",
    "padding": "20px",
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
}

DROPDOWN_STYLE = {
    "backgroundColor": "#1e2d42",
    "color": "#ccc",
    "borderColor": "#2a2a4a",
}

WIZARD_STEP_ACTIVE = {
    "backgroundColor": COLORS["accent"],
    "color": COLORS["text"],
    "borderRadius": "50%",
    "width": "32px",
    "height": "32px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "fontWeight": "bold",
}

WIZARD_STEP_INACTIVE = {
    **WIZARD_STEP_ACTIVE,
    "backgroundColor": COLORS["grid"],
    "color": COLORS["muted"],
}

WIZARD_STEP_COMPLETE = {
    **WIZARD_STEP_ACTIVE,
    "backgroundColor": COLORS["low"],
}

INDEX_STRING = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
    body { background-color: #1a1a2e; font-family: Inter, system-ui, sans-serif; }
    .model-lens-shell { min-height: 100vh; background-color: #1a1a2e; }
    .model-lens-sidebar {
        background-color: #16213e;
        border-right: 1px solid #2a2a4a;
        position: fixed;
        top: 0;
        left: 0;
        width: 280px;
        min-height: 100vh;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        z-index: 1000;
    }
    .model-lens-sidebar-nav {
        flex: 1 1 auto;
        overflow-y: auto;
        padding-right: 4px;
    }
    .model-lens-sidebar-footer { margin-top: auto; }
    .model-lens-content {
        margin-left: 300px;
        padding: 20px;
        min-height: 100vh;
        background-color: #1a1a2e;
    }
    .Select-control { background-color: #1e2d42 !important; border-color: #2a2a4a !important; }
    .Select-value-label, .Select-placeholder { color: #ccc !important; }
    .Select-menu-outer { background-color: #1e2d42 !important; border-color: #2a2a4a !important; }
    .VirtualizedSelectOption { background-color: #1e2d42 !important; color: #ccc !important; }
    .VirtualizedSelectFocusedOption { background-color: #0f3460 !important; color: #fff !important; }
    .Select-input > input { color: #ccc !important; }
    .Select-arrow { border-color: #888 transparent transparent !important; }
    .Select.is-open > .Select-control .Select-arrow { border-color: transparent transparent #888 !important; }
    .Select-clear { color: #888 !important; }
    .dash-dropdown .Select-menu-outer { z-index: 1000 !important; }
    .dash-dropdown .Select-control,
    .dash-dropdown .Select-multi-value-wrapper,
    .dash-dropdown .Select-input input { color: #ccc !important; }
    .DateInput_input { background-color: #1e2d42 !important; color: #ccc !important;
                       border-color: #2a2a4a !important; font-size: 0.85rem !important; }
    .DateRangePickerInput { background-color: #1e2d42 !important; }
    .DateRangePickerInput_arrow svg { fill: #888 !important; }
    .nav-link { color: #888 !important; padding: 10px 15px; border-radius: 6px; margin: 2px 0; }
    .nav-link:hover { color: #eee !important; background-color: #0f3460 !important; }
    .nav-link.active { color: #fff !important; background-color: #0f3460 !important; }
    @media (max-width: 991px) {
        .model-lens-sidebar {
            position: relative;
            width: 100%;
            min-height: auto;
            border-right: none;
            border-bottom: 1px solid #2a2a4a;
        }
        .model-lens-sidebar-nav {
            overflow-y: visible;
            padding-right: 0;
        }
        .model-lens-content {
            margin-left: 0;
            padding: 16px;
        }
    }
    @media (max-width: 575px) {
        .model-lens-sidebar { padding: 16px; }
        .model-lens-content { padding: 12px; }
    }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""
