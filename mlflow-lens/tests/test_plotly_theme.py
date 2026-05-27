from __future__ import annotations

from mlflow_lens._plotly_theme import LENS_COLORWAY, lens_layout


def test_lens_layout_defaults():
    layout = lens_layout()
    assert layout["template"] == "plotly_white"
    assert layout["colorway"] == list(LENS_COLORWAY)
    assert layout["paper_bgcolor"] == "white"
    assert "title" not in layout


def test_lens_layout_titles():
    layout = lens_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    assert layout["title"]["text"] == "<b>ROC Curve</b>"
    assert layout["xaxis"]["title"] == "False Positive Rate"
    assert layout["yaxis"]["title"] == "True Positive Rate"
    assert layout["xaxis"]["gridcolor"] == "#E5E7EB"


def test_lens_layout_overrides_merge_nested():
    layout = lens_layout(margin=dict(l=120), xaxis=dict(range=[0, 1]))
    assert layout["margin"]["l"] == 120
    assert layout["margin"]["r"] == 20
    assert layout["xaxis"]["range"] == [0, 1]
    assert layout["xaxis"]["gridcolor"] == "#E5E7EB"


def test_lens_layout_overrides_replace_scalar():
    layout = lens_layout(hovermode="x unified")
    assert layout["hovermode"] == "x unified"
