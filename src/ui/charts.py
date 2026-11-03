"""
charts.py — Plotly figures for the analytics dashboard.

All inputs come from AnalyticsService (real stored records). Colour roles:
  * diseases → fixed categorical slots (validated all-pairs; aqua < 3:1 → bars carry text labels)
  * screening statuses → reserved status colours, always with a text label
"""
from __future__ import annotations

import plotly.graph_objects as go

from src.clinical.translations import t

DISEASE_COLOURS = {"tb": "#2a78d6", "dr": "#eb6834", "oral": "#1baf7a"}
STATUS_COLOURS = {"SCREENING_FLAGGED": "#ec835a", "NO_FLAG_DETECTED": "#0ca30c", "INCONCLUSIVE": "#8a8f98"}
REJECT_COLOUR = "#0e2a47"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SHORT = {"tb": "TB", "dr": "DR", "oral": "Oral cancer"}


def _layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": INK}, "x": 0, "xanchor": "left"},
        height=height, margin={"l": 10, "r": 10, "t": 48, "b": 10},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"family": "Inter, system-ui, sans-serif", "color": MUTED, "size": 12},
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK}},
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def _empty(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 13, "color": MUTED}, x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title, 260)


def disease_status_chart(rows: list[dict], lang: str = "en") -> go.Figure:
    title = "Screenings by module and result"
    if not rows:
        return _empty(title, "No screenings recorded yet")
    diseases = [d for d in ("tb", "dr", "oral") if any(r["disease"] == d for r in rows)]
    fig = go.Figure()
    for status, colour in STATUS_COLOURS.items():
        counts = [sum(r["n"] for r in rows if r["disease"] == d and r["status"] == status) for d in diseases]
        label = t(f"status.{status}", lang).split(" — ")[0]
        fig.add_bar(x=[SHORT[d] for d in diseases], y=counts, name=label, marker_color=colour,
                    marker_line={"color": "#ffffff", "width": 2}, text=[c or "" for c in counts],
                    textposition="inside", insidetextfont={"color": "#ffffff"},
                    hovertemplate="%{x}<br>" + label + ": %{y}<extra></extra>")
    fig.update_layout(barmode="stack")
    return _layout(fig, title)


def rejection_chart(rows: list[dict]) -> go.Figure:
    title = "Rejected uploads by reason (share of all rejections)"
    if not rows:
        return _empty(title, "No rejected images recorded yet")
    rows = sorted(rows, key=lambda r: r["n"])
    labels = [r["code"].replace("_", " ").title() for r in rows]
    fig = go.Figure(go.Bar(
        x=[r["pct"] for r in rows], y=labels, orientation="h", marker_color=REJECT_COLOUR,
        text=[f"{r['pct']:.0f}%  ({r['n']})" for r in rows], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(ticksuffix="%", range=[0, max(r["pct"] for r in rows) * 1.25])
    return _layout(fig, title, max(260, 60 + 34 * len(rows)))


def trend_chart(rows: list[dict]) -> go.Figure:
    title = "Daily activity (last 60 days)"
    if not rows:
        return _empty(title, "No activity recorded yet")
    days = sorted({r["day"] for r in rows})
    fig = go.Figure()
    series = [("tb", SHORT["tb"], DISEASE_COLOURS["tb"]), ("dr", SHORT["dr"], DISEASE_COLOURS["dr"]),
              ("oral", SHORT["oral"], DISEASE_COLOURS["oral"]), ("rejected", "Rejected uploads", REJECT_COLOUR)]
    for key, name, colour in series:
        ys = [sum(r["n"] for r in rows if r["day"] == d and r["series"] == key) for d in days]
        if not any(ys):
            continue
        fig.add_scatter(x=days, y=ys, name=name, mode="lines+markers", line={"color": colour, "width": 2},
                        marker={"size": 8, "line": {"color": "#ffffff", "width": 2}},
                        hovertemplate=name + ": %{y}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    # Categorical days: a date axis with one or two points zooms into sub-second ticks.
    fig.update_xaxes(type="category")
    fig.update_yaxes(rangemode="tozero", dtick=1 if max(r["n"] for r in rows) < 10 else None)
    return _layout(fig, title)
