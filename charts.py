"""
charts.py
---------
Plotly figure builders — one function per chart.
All functions accept DataFrames from calculations.py
and return plotly.graph_objects.Figure objects.

FIN 511 - Investments I: Module 1, Lesson 1-5
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ── Design tokens ──────────────────────────────────────────────────────────────
COLORS = {
    "navy":    "#1F4E79",
    "blue":    "#2E75B6",
    "lblue":   "#BDD7EE",
    "green":   "#1E6B3A",
    "lgreen":  "#C6EFCE",
    "amber":   "#E8A020",
    "red":     "#C00000",
    "violet":  "#7B2D8B",
    "gray":    "#595959",
    "lgray":   "#F2F2F2",
    "white":   "#FFFFFF",
    "black":   "#000000",
    # region colours
    "efficient":  "#1E6B3A",
    "dominated":  "#C00000",
    "short_A1":   "#E8A020",
    "long_A1":    "#7B2D8B",
    "long_no_lev":"#2E75B6",
    "long_lev":   "#C00000",
    "short_cal":  "#7B2D8B",
}

RHO_COLORS = {
    -0.8: "#1F4E79",
    -0.4: "#2E75B6",
     0.0: "#1E6B3A",
     0.4: "#E8A020",
     0.8: "#C00000",
}

REGION_LABELS = {
    "efficient":   "Efficient Frontier",
    "dominated":   "Dominated",
    "short_A1":    "Short Asset 1 / Long Asset 2",
    "long_A1":     "Long Asset 1 / Short Asset 2",
    "long_no_lev": "Long (No Leverage)",
    "long_lev":    "Long With Leverage",
    "short":       "Short Risky Asset",
}


# ── Layout defaults ────────────────────────────────────────────────────────────
def _base_layout(title, xaxis_title, yaxis_title, subtitle=""):
    """Return a consistent Plotly layout dict."""
    full_title = (f"<b>{title}</b><br>"
                  f"<span style='font-size:12px;color:#595959'>{subtitle}</span>"
                  if subtitle else f"<b>{title}</b>")
    return dict(
        title=dict(text=full_title, font=dict(size=15, color=COLORS["navy"]),
                   x=0, xanchor="left"),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=12, color=COLORS["gray"])),
            tickfont=dict(size=11, color=COLORS["gray"]),
            gridcolor="#E8E8E8", showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=12, color=COLORS["gray"])),
            tickfont=dict(size=11, color=COLORS["gray"]),
            gridcolor="#E8E8E8", showgrid=True, zeroline=False,
        ),
        plot_bgcolor=COLORS["white"],
        paper_bgcolor=COLORS["white"],
        legend=dict(
            font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CCCCCC", borderwidth=1,
            x=1.02, y=1, xanchor="left",
            yanchor="top",
        ),
        margin=dict(t=80, b=110, l=70, r=220, autoexpand=True),
        hovermode="closest",
    )


def _hover_frontier(df_row_label=""):
    """Standard hover template for frontier charts."""
    return (
        "<b>%{customdata[0]}</b><br>"
        "Asset 1 Weight: %{customdata[1]:.1f}%<br>"
        "Asset 2 Weight: %{customdata[2]:.1f}%<br>"
        "<extra></extra>"
        "──────────────────<br>"
        "Exp. Return: %{y:.2f}%<br>"
        "Std. Dev.: %{x:.2f}%<br>"
        "Sharpe Ratio: %{customdata[3]:.3f}"
    )


def _hover_cal():
    """Standard hover template for CAL charts."""
    return (
        "<b>%{customdata[0]}</b><br>"
        "Risky Asset Weight: %{customdata[1]:.1f}%<br>"
        "Risk-Free Weight: %{customdata[2]:.1f}%<br>"
        "<extra></extra>"
        "──────────────────<br>"
        "Exp. Return: %{y:.2f}%<br>"
        "Std. Dev.: %{x:.2f}%<br>"
        "Sharpe Ratio: %{customdata[3]:.3f}"
    )


def _add_asset_markers(fig, df):
    """Add scatter markers for 100% Asset 1 and/or 100% Asset 2 if present in df.
    Uses wider tolerance (±0.02) to catch boundary points at region edges."""
    df_a1 = df[df["w_A1"].between(0.98, 1.02)]
    if not df_a1.empty:
        row = df_a1.iloc[(df_a1["w_A1"] - 1.0).abs().argsort().iloc[0]]
        fig.add_trace(go.Scatter(
            x=[row["sd"]], y=[row["ret"]], mode="markers+text",
            marker=dict(size=10, color=COLORS["navy"],
                        symbol="diamond", line=dict(width=1, color="white")),
            text=["(100% A1, 0% A2)"], textposition="top right",
            textfont=dict(size=10, color=COLORS["navy"]),
            name="(100% A1, 0% A2)",
            customdata=[[REGION_LABELS.get("efficient",""), 100.0, 0.0, row["sharpe"]]],
            hovertemplate=_hover_frontier(),
            showlegend=True,
        ))
    df_a2 = df[df["w_A1"].between(-0.02, 0.02)]
    if not df_a2.empty:
        row = df_a2.iloc[(df_a2["w_A1"] - 0.0).abs().argsort().iloc[0]]
        fig.add_trace(go.Scatter(
            x=[row["sd"]], y=[row["ret"]], mode="markers+text",
            marker=dict(size=10, color=COLORS["amber"],
                        symbol="diamond", line=dict(width=1, color="white")),
            text=["(0% A1, 100% A2)"], textposition="top right",
            textfont=dict(size=10, color=COLORS["amber"]),
            name="(0% A1, 100% A2)",
            customdata=[[REGION_LABELS.get("efficient",""), 0.0, 100.0, row["sharpe"]]],
            hovertemplate=_hover_frontier(),
            showlegend=True,
        ))
    return fig


def _add_mvp_marker(fig, mvp):
    """Add a star marker for the Minimum Variance Portfolio."""
    fig.add_trace(go.Scatter(
        x=[mvp["sd"]], y=[mvp["ret"]], mode="markers+text",
        marker=dict(size=14, color=COLORS["green"],
                    symbol="star", line=dict(width=1, color="white")),
        text=["MVP"], textposition="top left",
        textfont=dict(size=10, color=COLORS["green"]),
        name="Min. Variance Portfolio",
        customdata=[[
            "Min. Variance Portfolio",
            mvp["w_A1"] * 100,
            mvp["w_A2"] * 100,
            mvp["sharpe"],
        ]],
        hovertemplate=_hover_frontier(),
        showlegend=True,
    ))
    return fig


def _add_key_portfolio_markers(fig, frontier_df, chart_regions, max_sr, max_ret_lo, max_ret_lev, allow_short):
    """Add Max Sharpe, Max Return (Long Only), and Max Return (Leveraged) markers.
    Only plots a marker if its w_A1 row in frontier_df has a chart_region in chart_regions.
    chart_regions: set of strings e.g. {"chart2"} or {"chart2","chart3","chart4"} for Chart 1.
    Boundaries (w=0, w=1) belong to "chart2" only — Charts 3 & 4 pass {"chart3"} / {"chart4"}
    so they never inadvertently show long-only portfolio markers."""
    markers = [
        (max_sr,      "Max Sharpe",            COLORS["blue"],  "star-diamond",     "top right"),
        (max_ret_lo,  "Max Return (Long Only)", COLORS["green"], "star-triangle-up", "top right"),
    ]
    if allow_short and max_ret_lev is not None:
        markers.append(
            (max_ret_lev, "Max Return (Leveraged)", COLORS["red"], "star-triangle-up", "top right")
        )
    for port, label, color, symbol, tpos in markers:
        if port is None:
            continue
        w = port["w_A1"]
        match = frontier_df[frontier_df["w_A1"].between(w - 0.02, w + 0.02)]
        if match.empty:
            continue
        if match.iloc[0]["chart_region"] not in chart_regions:
            continue
        fig.add_trace(go.Scatter(
            x=[port["sd"]], y=[port["ret"]], mode="markers+text",
            marker=dict(size=12, color=color, symbol=symbol,
                        line=dict(width=1, color="white")),
            text=[label], textposition=tpos,
            textfont=dict(size=10, color=color),
            name=label,
            customdata=[[label, port["w_A1"]*100, port["w_A2"]*100, port["sharpe"]]],
            hovertemplate=_hover_frontier(),
            showlegend=True,
        ))
    return fig


def _add_dominated_line(fig, frontier_df):
    """Add red dashed dominated region line to any frontier chart."""
    df_dom = frontier_df[frontier_df["region"] == "dominated"].sort_values("w_A1", ascending=False)
    if df_dom.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=df_dom["sd"], y=df_dom["ret"],
        mode="lines",
        name=REGION_LABELS["dominated"],
        line=dict(color=COLORS["dominated"], width=2, dash="dash"),
        customdata=list(zip(
            [REGION_LABELS["dominated"]] * len(df_dom),
            df_dom["w_A1"] * 100,
            df_dom["w_A2"] * 100,
            df_dom["sharpe"],
        )),
        hovertemplate=_hover_frontier(),
    ))
    return fig


def _add_extreme_markers(fig, df):
    """Add markers for 200% Asset 1 (w_A1=2) and 200% Asset 2 (w_A1=-1)
    only if those weights exist in the passed df."""
    # 200% Asset 1: w_A1=2, w_A2=-1
    df_200a1 = df[df["w_A1"].between(1.95, 2.05)]
    if not df_200a1.empty:
        row = df_200a1.iloc[0]
        fig.add_trace(go.Scatter(
            x=[row["sd"]], y=[row["ret"]], mode="markers+text",
            marker=dict(size=9, color=COLORS["navy"], symbol="diamond-open",
                        line=dict(width=2, color=COLORS["navy"])),
            text=["(200% A1, -100% A2)"], textposition="bottom left",
            textfont=dict(size=10, color=COLORS["navy"]),
            name="(200% A1, -100% A2)",
            customdata=[[REGION_LABELS.get("long_A1",""), 200.0, -100.0, row["sharpe"]]],
            hovertemplate=_hover_frontier(),
            showlegend=True,
        ))
    # 200% Asset 2: w_A1=-1, w_A2=2
    df_200a2 = df[df["w_A1"].between(-1.05, -0.95)]
    if not df_200a2.empty:
        row = df_200a2.iloc[0]
        fig.add_trace(go.Scatter(
            x=[row["sd"]], y=[row["ret"]], mode="markers+text",
            marker=dict(size=9, color=COLORS["amber"], symbol="diamond-open",
                        line=dict(width=2, color=COLORS["amber"])),
            text=["(-100% A1, 200% A2)"], textposition="bottom left",
            textfont=dict(size=10, color=COLORS["amber"]),
            name="(-100% A1, 200% A2)",
            customdata=[[REGION_LABELS.get("short_A1",""), -100.0, 200.0, row["sharpe"]]],
            hovertemplate=_hover_frontier(),
            showlegend=True,
        ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO FRONTIER CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_frontier_all(frontier_df, r1, sd1, r2, sd2, mvp, max_sr, max_ret_lo, max_ret_lev, allow_short=False):
    """
    Chart 1 — All Allocations (full frontier, requires short-selling ON).
    Shows 3 colored segments by weight region: long_only, short_A1, long_A1.
    """
    fig = go.Figure()

    weight_region_styles = [
        ("long_only", "Long Only",                    COLORS["efficient"], "solid", 2.5),
        ("short_A1",  "Short Asset 1 / Long Asset 2", COLORS["short_A1"],  "solid", 2.5),
        ("long_A1",   "Long Asset 1 / Short Asset 2", COLORS["long_A1"],   "solid", 2.5),
    ]

    for wr, label, color, dash, width in weight_region_styles:
        df_r = frontier_df[frontier_df["weight_region"] == wr].sort_values("w_A1")
        if df_r.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_r["sd"], y=df_r["ret"],
            mode="lines",
            name=label,
            line=dict(color=color, width=width, dash=dash),
            customdata=list(zip(
                [label] * len(df_r),
                df_r["w_A1"] * 100,
                df_r["w_A2"] * 100,
                df_r["sharpe"],
            )),
            hovertemplate=_hover_frontier(),
        ))

    fig = _add_mvp_marker(fig, mvp)
    fig = _add_key_portfolio_markers(fig, frontier_df, {"chart2","chart3","chart4"}, max_sr, max_ret_lo, max_ret_lev, allow_short)
    fig = _add_asset_markers(fig, frontier_df)
    fig = _add_extreme_markers(fig, frontier_df)

    fig.update_layout(_base_layout(
        title="Portfolio Frontier — All Allocations",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Asset 1 Weight: −100% → +200%  |  Asset 2 Weight: +200% → −100%",
    ))
    return fig


def chart_frontier_long_only(frontier_df, r1, sd1, r2, sd2, mvp, max_sr, max_ret_lo, max_ret_lev=None, allow_short=False):
    """
    Chart 2 — Long Only (always visible).
    Upper portion = efficient frontier. Lower portion = dominated.
    Filtered to w_A1: 0% → 100% only.
    """
    fig = go.Figure()
    df = frontier_df[frontier_df["weight_region"] == "long_only"]

    for region in ["efficient", "dominated"]:
        df_r = df[df["region"] == region].sort_values("w_A1", ascending=False)
        if df_r.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_r["sd"], y=df_r["ret"],
            mode="lines",
            name=REGION_LABELS[region],
            line=dict(
                color=COLORS[region],
                width=3 if region == "efficient" else 2,
                dash="solid" if region == "efficient" else "dash",
            ),
            customdata=list(zip(
                [REGION_LABELS[region]] * len(df_r),
                df_r["w_A1"] * 100,
                df_r["w_A2"] * 100,
                df_r["sharpe"],
            )),
            hovertemplate=_hover_frontier(),
        ))

    fig = _add_mvp_marker(fig, mvp)
    fig = _add_key_portfolio_markers(fig, frontier_df, {"chart2"}, max_sr, max_ret_lo, max_ret_lev, allow_short)
    fig = _add_asset_markers(fig, df)

    # Vertical reference line through MVP
    fig.add_vline(
        x=mvp["sd"], line_dash="dot", line_color=COLORS["gray"],
        line_width=1, opacity=0.5,
        annotation_text="MVP",
        annotation_position="top",
        annotation_font=dict(size=10, color=COLORS["gray"]),
    )

    # Show dominated annotation if dominated region exists in long-only
    df_dom = frontier_df[frontier_df["weight_region"] == "long_only"]
    df_dom = df_dom[df_dom["region"] == "dominated"]
    if not df_dom.empty:
        fig.add_annotation(
            text="⚠ Dominated region — same risk as portfolios<br>above MVP but lower return",
            xref="paper", yref="paper", x=0.5, y=-0.18,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=11, color=COLORS["red"]),
            bgcolor="rgba(255,200,200,0.3)",
            bordercolor=COLORS["red"], borderwidth=1,
        )

    fig.update_layout(_base_layout(
        title="Long Only",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Asset 1 Weight: 0% → 100%  |  Asset 2 Weight: 100% → 0%",
    ))
    return fig


def chart_frontier_short_A1(frontier_df, r1, sd1, r2, sd2, max_sr, max_ret_lo, max_ret_lev=None, allow_short=False):
    """
    Chart 3 — Short Asset 1 / Long Asset 2 (requires short-selling ON).
    Shows efficient + dominated filtered to w_A1: −100% → 0% only.
    """
    fig = go.Figure()
    df = frontier_df[
        (frontier_df["weight_region"] == "short_A1") |
        (frontier_df["w_A1"].between(-0.02, 0.02))   # include w_A1=0 boundary
    ]

    for region in ["efficient", "dominated"]:
        df_r = df[df["region"] == region].sort_values("w_A1", ascending=False)
        if df_r.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_r["sd"], y=df_r["ret"],
            mode="lines",
            name=REGION_LABELS[region],
            line=dict(
                color=COLORS[region],
                width=3 if region == "efficient" else 2,
                dash="solid" if region == "efficient" else "dash",
            ),
            customdata=list(zip(
                [REGION_LABELS[region]] * len(df_r),
                df_r["w_A1"] * 100,
                df_r["w_A2"] * 100,
                df_r["sharpe"],
            )),
            hovertemplate=_hover_frontier(),
        ))

    fig = _add_key_portfolio_markers(fig, frontier_df, {"chart3"}, max_sr, max_ret_lo, max_ret_lev, allow_short)
    fig = _add_asset_markers(fig, df)
    fig = _add_extreme_markers(fig, df)

    # Dynamic annotation — only show if dominated rows exist in this weight range
    df_dom_c3 = df[df["region"] == "dominated"]
    if not df_dom_c3.empty:
        fig.add_annotation(
            text="⚠ Dominated region — same risk as portfolios<br>above MVP but lower return",
            xref="paper", yref="paper", x=0.5, y=-0.18,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=11, color=COLORS["red"]),
            bgcolor="rgba(255,200,200,0.3)",
            bordercolor=COLORS["red"], borderwidth=1,
        )

    fig.update_layout(_base_layout(
        title="Short Asset 1 / Long Asset 2",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Asset 1 Weight: −100% → 0%  |  Asset 2 Weight: 200% → 100%",
    ))
    return fig


def chart_frontier_long_A1(frontier_df, r1, sd1, r2, sd2, max_sr, max_ret_lo, max_ret_lev=None, allow_short=False):
    """
    Chart 4 — Long Asset 1 / Short Asset 2 (requires short-selling ON).
    Shows efficient + dominated filtered to w_A1: 100% → 200% only.
    """
    fig = go.Figure()
    df = frontier_df[
        (frontier_df["weight_region"] == "long_A1") |
        (frontier_df["w_A1"].between(0.98, 1.02))    # include w_A1=1 boundary
    ]

    for region in ["efficient", "dominated"]:
        df_r = df[df["region"] == region].sort_values("w_A1", ascending=False)
        if df_r.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_r["sd"], y=df_r["ret"],
            mode="lines",
            name=REGION_LABELS[region],
            line=dict(
                color=COLORS[region],
                width=3 if region == "efficient" else 2,
                dash="solid" if region == "efficient" else "dash",
            ),
            customdata=list(zip(
                [REGION_LABELS[region]] * len(df_r),
                df_r["w_A1"] * 100,
                df_r["w_A2"] * 100,
                df_r["sharpe"],
            )),
            hovertemplate=_hover_frontier(),
        ))

    fig = _add_key_portfolio_markers(fig, frontier_df, {"chart4"}, max_sr, max_ret_lo, max_ret_lev, allow_short)
    fig = _add_asset_markers(fig, df)
    fig = _add_extreme_markers(fig, df)

    # Dynamic annotation — only show if dominated rows exist in this weight range
    df_dom_c4 = df[df["region"] == "dominated"]
    if not df_dom_c4.empty:
        fig.add_annotation(
            text="⚠ Dominated region — same risk as portfolios<br>above MVP but lower return",
            xref="paper", yref="paper", x=0.5, y=-0.18,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=11, color=COLORS["red"]),
            bgcolor="rgba(255,200,200,0.3)",
            bordercolor=COLORS["red"], borderwidth=1,
        )

    fig.update_layout(_base_layout(
        title="Long Asset 1 / Short Asset 2",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Asset 1 Weight: 100% → 200%  |  Asset 2 Weight: 0% → −100%",
    ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAL CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _cal_traces(cal_df, regions, rf):
    """
    Helper: build Scatter traces for specified CAL regions.
    Returns list of go.Scatter objects.
    """
    traces = []
    for region in regions:
        df_r = cal_df[cal_df["region"] == region].sort_values("sd")
        if df_r.empty:
            continue
        traces.append(go.Scatter(
            x=df_r["sd"], y=df_r["ret"],
            mode="lines",
            name=REGION_LABELS.get(region, region),
            line=dict(color=COLORS.get(region, COLORS["blue"]), width=2.5,
                      dash="dash" if region == "short" else "solid"),
            customdata=list(zip(
                [REGION_LABELS.get(region, region)] * len(df_r),
                df_r["w_risky"] * 100,
                df_r["w_rf"] * 100,
                df_r["sharpe"],
            )),
            hovertemplate=_hover_cal(),
        ))
    return traces


def _add_cal_key_markers(fig, cal_df, r_risky, sd_risky, rf):
    """Add markers for 100% RF, 100% Risky, w=2 leverage, and -100% short if it exists in cal_df."""
    points = [
        (0.0,          rf,                 0.0,        "100% Risk-Free",               COLORS["green"]),
        (sd_risky,     r_risky,            sd_risky,   "100% Risky Asset",             COLORS["navy"]),
        (2*sd_risky,   2*r_risky-rf,       2*sd_risky, "200% Risky Asset (Leverage)",  COLORS["red"]),
    ]
    # Add -100% Risky Asset (Short-Selling) only if w=-1 exists in cal_df
    df_short = cal_df[cal_df["w_risky"].between(-1.02, -0.98)]
    if not df_short.empty:
        points.append(
            (sd_risky, -r_risky + 2*rf, sd_risky, "200% Risk Free, -100% Risky (Short-Selling)", COLORS["purple"] if "purple" in COLORS else "#7B2D8B")
        )
        # Compute properly from cal_df row
        row = df_short.iloc[0]
        w, w_rf = -1.0, 2.0
        sharpe = (r_risky - rf) / sd_risky if sd_risky > 0 else 0
        fig.add_trace(go.Scatter(
            x=[row["sd"]], y=[row["ret"]],
            mode="markers+text",
            marker=dict(size=10, color="#7B2D8B", symbol="circle",
                        line=dict(width=1, color="white")),
            text=["-100% Risky Asset (Short-Selling)"], textposition="top right",
            textfont=dict(size=10, color="#7B2D8B"),
            name="-100% Risky Asset (Short-Selling)",
            customdata=[["200% Risk Free, -100% Risky (Short-Selling)", w*100, w_rf*100, sharpe]],
            hovertemplate=_hover_cal(),
            showlegend=True,
        ))
        # Remove from points list to avoid double-plotting
        points = points[:-1]

    for sd_val, ret_val, _, label, color in points:
        w       = sd_val / sd_risky if sd_risky > 0 else 0
        w_rf    = 1 - w
        sharpe  = (r_risky - rf) / sd_risky if sd_risky > 0 else 0
        fig.add_trace(go.Scatter(
            x=[sd_val], y=[ret_val],
            mode="markers+text",
            marker=dict(size=10, color=color, symbol="circle",
                        line=dict(width=1, color="white")),
            text=[label], textposition="top right",
            textfont=dict(size=10, color=color),
            name=label,
            customdata=[[label, w*100, w_rf*100, sharpe]],
            hovertemplate=_hover_cal(),
            showlegend=True,
        ))
    return fig


def chart_cal_all(cal_df, r_risky, sd_risky, rf):
    """
    CAL Chart 1 — All Allocations (requires short-selling ON).
    Full CAL from w=−100% to w=+200%.
    """
    fig = go.Figure()
    for trace in _cal_traces(cal_df, ["short", "long_no_lev", "long_lev"], rf):
        fig.add_trace(trace)

    fig = _add_cal_key_markers(fig, cal_df, r_risky, sd_risky, rf)
    fig.add_hline(y=rf, line_dash="dot", line_color=COLORS["gray"],
                  line_width=1, opacity=0.5,
                  annotation_text=f"rf = {rf}%",
                  annotation_position="right",
                  annotation_font=dict(size=10, color=COLORS["gray"]))

    fig.update_layout(_base_layout(
        title="CAL — All Allocations",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Risky Asset Weight: −100% → +200%  |  Risk-Free Weight: +200% → −100%",
    ))
    return fig


def chart_cal_all_long(cal_df, r_risky, sd_risky, rf):
    """
    CAL Chart 2 — All Long (always visible, placed first when short-selling OFF).
    Long-only including leverage: w=0% to w=+200%.
    """
    df_filtered = cal_df[cal_df["w_risky"] >= 0].copy()
    fig = go.Figure()

    for trace in _cal_traces(df_filtered, ["long_no_lev", "long_lev"], rf):
        fig.add_trace(trace)

    fig = _add_cal_key_markers(fig, df_filtered, r_risky, sd_risky, rf)
    fig.add_hline(y=rf, line_dash="dot", line_color=COLORS["gray"],
                  line_width=1, opacity=0.5,
                  annotation_text=f"rf = {rf}%",
                  annotation_position="right",
                  annotation_font=dict(size=10, color=COLORS["gray"]))

    fig.update_layout(_base_layout(
        title="CAL — All Long (No Leverage + With Leverage)",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Risky Asset Weight: 0% → +200%  |  Risk-Free Weight: +100% → −100%",
    ))
    return fig


def chart_cal_long_no_leverage(cal_df, r_risky, sd_risky, rf):
    """
    CAL Chart 3 — Long Without Leverage (always visible).
    w=0% to w=100%: from pure T-Bills to 100% risky asset.
    """
    df_filtered = cal_df[
        (cal_df["w_risky"] >= 0) & (cal_df["w_risky"] <= 1)
    ].copy()
    fig = go.Figure()

    for trace in _cal_traces(df_filtered, ["long_no_lev"], rf):
        fig.add_trace(trace)

    # Only add RF and 100% Risky markers (no leverage point)
    for w_val, label, color in [
        (0.0, "100% Risk-Free",   COLORS["green"]),
        (1.0, "100% Risky Asset", COLORS["navy"]),
    ]:
        ret_val = w_val * r_risky + (1 - w_val) * rf
        sd_val  = w_val * sd_risky
        w_rf    = 1 - w_val
        sharpe  = (r_risky - rf) / sd_risky if sd_risky > 0 else 0
        fig.add_trace(go.Scatter(
            x=[sd_val], y=[ret_val], mode="markers+text",
            marker=dict(size=10, color=color, symbol="circle",
                        line=dict(width=1, color="white")),
            text=[label], textposition="top right",
            textfont=dict(size=10, color=color),
            name=label,
            customdata=[[label, w_val*100, w_rf*100, sharpe]],
            hovertemplate=_hover_cal(),
            showlegend=True,
        ))

    fig.update_layout(_base_layout(
        title="CAL — Long Without Leverage",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Risky Asset Weight: 0% → 100%  |  Risk-Free Weight: 100% → 0%",
    ))
    return fig


def chart_cal_long_with_leverage(cal_df, r_risky, sd_risky, rf):
    """
    CAL Chart 4 — Long With Leverage (always visible).
    w=100% to w=200%: borrowing at rf to amplify risky exposure.
    """
    df_filtered = cal_df[cal_df["w_risky"] >= 1.0].copy()
    fig = go.Figure()

    for trace in _cal_traces(df_filtered, ["long_lev"], rf):
        fig.add_trace(trace)

    # Add 100% Risky and w=2 markers
    for w_val, label, color in [
        (1.0, "100% Risky Asset (start)", COLORS["navy"]),
        (2.0, "200% Risky Asset (Max Leverage)", COLORS["red"]),
    ]:
        ret_val = w_val * r_risky + (1 - w_val) * rf
        sd_val  = w_val * sd_risky
        w_rf    = 1 - w_val
        sharpe  = (r_risky - rf) / sd_risky if sd_risky > 0 else 0
        fig.add_trace(go.Scatter(
            x=[sd_val], y=[ret_val], mode="markers+text",
            marker=dict(size=10, color=color, symbol="circle",
                        line=dict(width=1, color="white")),
            text=[label], textposition="top right",
            textfont=dict(size=10, color=color),
            name=label,
            customdata=[[label, w_val*100, w_rf*100, sharpe]],
            hovertemplate=_hover_cal(),
            showlegend=True,
        ))

    # Annotation explaining leverage
    fig.add_annotation(
        text=(f"Borrow at rf={rf}% to invest<br>"
              "more than 100% in risky asset.<br>"
              "Amplifies both return AND risk."),
        xref="paper", yref="paper", x=0.02, y=0.95,
        showarrow=False,
        font=dict(size=10, color=COLORS["red"]),
        bgcolor="rgba(255,200,200,0.3)",
        bordercolor=COLORS["red"], borderwidth=1,
        align="left",
    )

    fig.update_layout(_base_layout(
        title="CAL — Long With Leverage",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Risky Asset Weight: 100% → 200%  |  Risk-Free Weight: 0% → −100% (borrowing)",
    ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATION EFFECT CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_rho_effect(rho_frontiers, current_rho, r1, sd1, r2, sd2):
    """
    Tab 3 Chart 1 — Overlaid efficient frontiers for 5 correlation values.
    Current ρ is shown with thicker line and full opacity.
    All others are dashed with reduced opacity.
    """
    fig = go.Figure()

    for rho, df in rho_frontiers.items():
        is_current = abs(rho - current_rho) < 1e-9
        color      = RHO_COLORS.get(rho, COLORS["blue"])
        label      = f"ρ = {rho}"
        if is_current:
            label += "  ← current"

        fig.add_trace(go.Scatter(
            x=df["sd"], y=df["ret"],
            mode="lines",
            name=label,
            line=dict(
                color=color,
                width=3.5 if is_current else 1.5,
                dash="solid" if is_current else "dash",
            ),
            opacity=1.0 if is_current else 0.55,
            customdata=list(zip(
                [f"ρ={rho}"] * len(df),
                df["w_A1"] * 100,
                df["w_A2"] * 100,
                df["sharpe"],
            )),
            hovertemplate=_hover_frontier(),
        ))

    fig = _add_asset_markers(fig, df)

    fig.add_annotation(
        text="Long-only portfolios (Asset 1 & Asset 2 Weights: 0% → 100%)",
        xref="paper", yref="paper", x=0.0, y=-0.12,
        showarrow=False, font=dict(size=10, color=COLORS["gray"]),
        xanchor="left",
    )

    fig.update_layout(_base_layout(
        title="Effect of Correlation on the Efficient Frontier",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Lower ρ → frontier bows further left → more diversification benefit",
    ))
    return fig


def chart_rho_mvp_table(rho_mvp_df):
    """
    Tab 3 Chart 2 — MVP comparison table as a Plotly table figure.
    Highlights the current ρ row.
    """
    display_df = rho_mvp_df.drop(columns=["is_current"]).copy()
    display_df["Correlation (ρ)"] = rho_mvp_df.apply(
        lambda r: f"★ {r['Correlation (ρ)']}  ← current" if r["is_current"]
                  else str(r["Correlation (ρ)"]),
        axis=1
    )

    fill_colors = []
    for _, row in rho_mvp_df.iterrows():
        if row["is_current"]:
            fill_colors.append("#C6EFCE")
        else:
            fill_colors.append("#F8F9FA")

    n_cols = len(display_df.columns)

    fig = go.Figure(data=[go.Table(
        columnwidth=[180, 140, 140, 120, 130, 120],
        header=dict(
            values=[f"<b>{c}</b>" for c in display_df.columns],
            fill_color=COLORS["navy"],
            font=dict(color="white", size=12),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[display_df[c].tolist() for c in display_df.columns],
            fill_color=[fill_colors] * n_cols,
            font=dict(color=COLORS["gray"], size=11),
            align=["left"] + ["center"] * (n_cols - 1),
            height=32,
        ),
    )])

    fig.update_layout(
        title=dict(
            text="<b>Min. Variance Portfolio — Comparison Across Correlations</b>",
            font=dict(size=14, color=COLORS["navy"]),
            x=0, xanchor="left",
        ),
        margin=dict(t=60, b=20, l=0, r=0),
        paper_bgcolor=COLORS["white"],
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLES AS PLOTLY FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def chart_frontier_summary_table(summary_df):
    """Render the frontier summary table (bottom of Tab 1) as a Plotly table."""
    n_cols = len(summary_df.columns)

    fill_colors = []
    for _, row in summary_df.iterrows():
        if "⭐" in str(row.get("Portfolio", "")):
            fill_colors.append("#FFF3CD")
        else:
            fill_colors.append("#F8F9FA")

    fig = go.Figure(data=[go.Table(
        columnwidth=[250, 130, 130, 110, 110, 110],
        header=dict(
            values=[f"<b>{c}</b>" for c in summary_df.columns],
            fill_color=COLORS["navy"],
            font=dict(color="white", size=12),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[summary_df[c].tolist() for c in summary_df.columns],
            fill_color=[fill_colors] * n_cols,
            font=dict(color=COLORS["gray"], size=11),
            align=["left"] + ["center"] * (n_cols - 1),
            height=32,
        ),
    )])

    fig.update_layout(
        title=dict(
            text="<b>Portfolio Summary — Key Allocation Points</b>",
            font=dict(size=14, color=COLORS["navy"]),
            x=0, xanchor="left",
        ),
        margin=dict(t=60, b=20, l=0, r=0),
        paper_bgcolor=COLORS["white"],
    )
    return fig


def chart_cal_summary_table(cal_summary_df):
    """Render the CAL summary table (bottom of Tab 2) as a Plotly table."""
    n_cols = len(cal_summary_df.columns)

    fill_colors = []
    for _, row in cal_summary_df.iterrows():
        region = str(row.get("Region", ""))
        if "Leverage" in region:
            fill_colors.append("#FDE8E8")
        elif "Short" in region:
            fill_colors.append("#EDE7F6")
        else:
            fill_colors.append("#F8F9FA")

    fig = go.Figure(data=[go.Table(
        columnwidth=[280, 150, 140, 110, 110, 110, 160],
        header=dict(
            values=[f"<b>{c}</b>" for c in cal_summary_df.columns],
            fill_color=COLORS["navy"],
            font=dict(color="white", size=12),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[cal_summary_df[c].tolist() for c in cal_summary_df.columns],
            fill_color=[fill_colors] * n_cols,
            font=dict(color=COLORS["gray"], size=11),
            align=["left"] + ["center"] * (n_cols - 1),
            height=32,
        ),
    )])

    fig.update_layout(
        title=dict(
            text="<b>CAL Summary — Key Portfolio Points</b>",
            font=dict(size=14, color=COLORS["navy"]),
            x=0, xanchor="left",
        ),
        margin=dict(t=60, b=20, l=0, r=0),
        paper_bgcolor=COLORS["white"],
    )
    return fig
