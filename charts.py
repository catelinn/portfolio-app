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
    "short":       "200% Risk-Free, -100% Risky (Short-Selling)",
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
            customdata=[[REGION_LABELS[row["region"]], 100.0, 0.0, row["sharpe"]]],
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
            customdata=[[REGION_LABELS[row["region"]], 0.0, 100.0, row["sharpe"]]],
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


def _add_msp_marker(fig, msp, label="MSP", show_legend=True):
    """Add a diamond marker for the Max Sharpe Portfolio."""
    fig.add_trace(go.Scatter(
        x=[msp["sd"]], y=[msp["ret"]], mode="markers+text",
        marker=dict(size=14, color=COLORS["blue"],
                    symbol="star-diamond", line=dict(width=1, color="white")),
        text=[label], textposition="top right",
        textfont=dict(size=10, color=COLORS["blue"]),
        name="Max Sharpe Portfolio",
        customdata=[[
            "Max Sharpe Portfolio",
            msp["w_A1"] * 100,
            msp["w_A2"] * 100,
            msp["sharpe"],
        ]],
        hovertemplate=_hover_frontier(),
        showlegend=show_legend,
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
    # Add 200% Risk-Free, -100% Risky (Short-Selling) only if w=-1 exists in cal_df
    df_short = cal_df[cal_df["w_risky"].between(-1.02, -0.98)]
    if not df_short.empty:
        points.append(
            (sd_risky, -r_risky + 2*rf, sd_risky, "200% Risk-Free, -100% Risky (Short-Selling)", COLORS["purple"] if "purple" in COLORS else "#7B2D8B")
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
            text=["200% Risk-Free, -100% Risky (Short-Selling)"], textposition="top right",
            textfont=dict(size=10, color="#7B2D8B"),
            name="200% Risk-Free, -100% Risky (Short-Selling)",
            customdata=[["200% Risk-Free, -100% Risky (Short-Selling)", w*100, w_rf*100, sharpe]],
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

def chart_rho_effect(rho_frontiers, current_rho, r1, sd1, r2, sd2,
                     mvp_points=None, msp_points=None, allow_short=False):
    """
    Tab 2 Chart 1 — Overlaid efficient frontiers for 5 correlation values.
    Current ρ is shown with thicker line and full opacity.
    All others are dashed with reduced opacity.

    mvp_points : dict {rho: Series}  MVP row per frontier (optional)
    msp_points : dict {rho: Series}  MSP row per frontier (optional)

    When allow_short=True, frontiers extend to the full short-selling range
    and MVP/MSP markers reflect unconstrained weights.
    """
    fig = go.Figure()

    current_df = None
    for rho, df in rho_frontiers.items():
        is_current = abs(rho - current_rho) < 1e-9
        color      = RHO_COLORS.get(rho, "#7B2D8B" if is_current else COLORS["blue"])
        label      = f"ρ = {rho}"
        if is_current:
            label += "  ← current"
            current_df = df

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

    # MVP markers — one per frontier, legend shown only for first
    if mvp_points:
        for i, (rho, mvp) in enumerate(mvp_points.items()):
            is_current = abs(rho - current_rho) < 1e-9
            fig.add_trace(go.Scatter(
                x=[mvp["sd"]], y=[mvp["ret"]], mode="markers+text",
                marker=dict(size=10 if is_current else 7,
                            color=COLORS["green"],
                            symbol="star",
                            line=dict(width=1, color="white"),
                            opacity=1.0 if is_current else 0.5),
                text=["MVP"] if is_current else [""],
                textposition="top left",
                textfont=dict(size=10, color=COLORS["green"]),
                name="MVP" if i == 0 else None,
                showlegend=(i == 0),
                customdata=[["Min. Variance Portfolio", mvp["w_A1"] * 100,
                              mvp["w_A2"] * 100, mvp["sharpe"]]],
                hovertemplate=_hover_frontier(),
            ))

    # MSP markers — one per frontier, legend shown only for first
    if msp_points:
        for i, (rho, msp) in enumerate(msp_points.items()):
            is_current = abs(rho - current_rho) < 1e-9
            fig.add_trace(go.Scatter(
                x=[msp["sd"]], y=[msp["ret"]], mode="markers+text",
                marker=dict(size=10 if is_current else 7,
                            color=COLORS["blue"],
                            symbol="star-diamond",
                            line=dict(width=1, color="white"),
                            opacity=1.0 if is_current else 0.5),
                text=["MSP"] if is_current else [""],
                textposition="top right",
                textfont=dict(size=10, color=COLORS["blue"]),
                name="MSP" if i == 0 else None,
                showlegend=(i == 0),
                customdata=[["Max Sharpe Portfolio", msp["w_A1"] * 100,
                              msp["w_A2"] * 100, msp["sharpe"]]],
                hovertemplate=_hover_frontier(),
            ))

    fig = _add_asset_markers(fig, current_df if current_df is not None else df)

    if allow_short:
        annotation_text = "• Full frontier: Asset 1 Weight −100% → +200% (short-selling enabled — MVP weights unconstrained)"
    else:
        annotation_text = "• Long-only portfolios (Asset 1 & Asset 2 Weights: 0% → 100%)"

    fig.update_layout(_base_layout(
        title="Effect of Correlation on the Efficient Frontier",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle=(
            "Lower ρ → frontier bows further left → more diversification benefit (long-only region)  |  "
            "Beyond 100% A2 (short A1): effect reverses — higher ρ reduces variance"
            if allow_short
            else "Lower ρ → frontier bows further left → more diversification benefit"
        ),
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
        height=280,
        paper_bgcolor=COLORS["white"],
    )
    return fig


def chart_rho_msp_table(rho_msp_df):
    """
    Tab 2 — Max Sharpe Portfolio comparison table as a Plotly table figure.
    Highlights the current ρ row.
    """
    display_df = rho_msp_df.drop(columns=["is_current"]).copy()
    display_df["Correlation (ρ)"] = rho_msp_df.apply(
        lambda r: f"★ {r['Correlation (ρ)']}  ← current" if r["is_current"]
                  else str(r["Correlation (ρ)"]),
        axis=1
    )

    fill_colors = []
    for _, row in rho_msp_df.iterrows():
        if row["is_current"]:
            fill_colors.append("#DDEEFF")
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
            text="<b>Max Sharpe Portfolio — Comparison Across Correlations</b>",
            font=dict(size=14, color=COLORS["navy"]),
            x=0, xanchor="left",
        ),
        margin=dict(t=60, b=20, l=0, r=0),
        height=280,
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


def chart_frontier_with_solver(frontier_df, result_rows, constraint, mvp, allow_short=False):
    """
    Tab 4 — Portfolio Solver chart.

    Full frontier drawn as a muted gray background; the active constraint
    region is overlaid in bold colour; each solver result is marked with a
    star (gold = efficient region, red-orange = dominated region).

    Parameters
    ----------
    frontier_df : pd.DataFrame  full frontier from build_frontier()
    result_rows : list of (pd.Series, str)  each entry is (row, label);
                  label is e.g. "Efficient Region" or "Dominated Region"
    constraint  : str           active constraint key
    mvp         : pd.Series     Minimum Variance Portfolio row
    allow_short : bool
    """
    fig = go.Figure()

    # ── Background: full frontier in muted light gray ─────────────────────────
    all_df = frontier_df.sort_values("w_A1")
    fig.add_trace(go.Scatter(
        x=all_df["sd"], y=all_df["ret"],
        mode="lines",
        name="Full Frontier (background)",
        line=dict(color="#CCCCCC", width=1.5),
        hoverinfo="skip",
        showlegend=True,
    ))

    # ── Constraint region overlay — colored, bold ─────────────────────────────
    if constraint == "full":
        for wr, label, color in [
            ("long_only", "Long Only",                    COLORS["navy"]),
            ("short_A1",  "Short A1 / Long A2",           COLORS["amber"]),
            ("long_A1",   "Long A1 / Short A2",           COLORS["violet"]),
        ]:
            df_r = frontier_df[frontier_df["weight_region"] == wr].sort_values("w_A1")
            if df_r.empty:
                continue
            fig.add_trace(go.Scatter(
                x=df_r["sd"], y=df_r["ret"],
                mode="lines",
                name=f"{label} (Constraint)",
                line=dict(color=color, width=3),
                customdata=list(zip(
                    [label] * len(df_r),
                    df_r["w_A1"] * 100,
                    df_r["w_A2"] * 100,
                    df_r["sharpe"],
                )),
                hovertemplate=_hover_frontier(),
            ))
    else:
        constraint_styles = {
            "long_only": (
                frontier_df[frontier_df["weight_region"] == "long_only"],
                "Long Only (Constraint)", COLORS["navy"],
            ),
            "long_A1": (
                frontier_df[frontier_df["weight_region"] == "long_A1"],
                "Long A1 / Short A2 (Constraint)", COLORS["violet"],
            ),
            "short_A1": (
                frontier_df[frontier_df["weight_region"] == "short_A1"],
                "Short A1 / Long A2 (Constraint)", COLORS["amber"],
            ),
        }
        if constraint in constraint_styles:
            df_c, label_c, color_c = constraint_styles[constraint]
            df_c = df_c.sort_values("w_A1")
            if not df_c.empty:
                fig.add_trace(go.Scatter(
                    x=df_c["sd"], y=df_c["ret"],
                    mode="lines",
                    name=label_c,
                    line=dict(color=color_c, width=3),
                    customdata=list(zip(
                        [label_c] * len(df_c),
                        df_c["w_A1"] * 100,
                        df_c["w_A2"] * 100,
                        df_c["sharpe"],
                    )),
                    hovertemplate=_hover_frontier(),
                ))

    # ── MVP marker (context) ──────────────────────────────────────────────────
    fig = _add_mvp_marker(fig, mvp)

    # ── Asset endpoint markers ────────────────────────────────────────────────
    fig = _add_asset_markers(fig, frontier_df)

    # ── Solver result stars ───────────────────────────────────────────────────
    _star_color = {
        "Efficient Region": "#FF8C00",   # gold
        "Dominated Region": "#E63946",   # red-orange
    }
    for _rr, _rl in (result_rows or []):
        if _rr is None:
            continue
        _color = _star_color["Dominated Region"] if "Dominated Region" in _rl else _star_color["Efficient Region"]
        fig.add_trace(go.Scatter(
            x=[_rr["sd"]], y=[_rr["ret"]],
            mode="markers+text",
            marker=dict(size=18, color=_color, symbol="star",
                        line=dict(width=2, color="white")),
            text=[_rl], textposition="top right",
            textfont=dict(size=11, color=_color),
            name=_rl,
            customdata=[[
                _rl,
                _rr["w_A1"] * 100,
                _rr["w_A2"] * 100,
                _rr["sharpe"],
            ]],
            hovertemplate=_hover_frontier(),
            showlegend=True,
        ))

    fig.update_layout(_base_layout(
        title="Portfolio Solver — Result on Frontier",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Highlighted = active constraint region  |  Gold star = solver result",
    ))
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


# ══════════════════════════════════════════════════════════════════════════════
# N-ASSET PORTFOLIO CHARTS
# ══════════════════════════════════════════════════════════════════════════════

# Per-asset colour palette (up to 8 assets)
N_ASSET_COLORS = [
    "#1F4E79",  # navy
    "#E8A020",  # amber
    "#1E6B3A",  # green
    "#C00000",  # red
    "#7B2D8B",  # violet
    "#2E75B6",  # blue
    "#595959",  # gray
    "#B8860B",  # dark goldenrod
]

KAPPA_COLORS = {
    0.0:  "#1F4E79",
    0.25: "#2E75B6",
    0.5:  "#1E6B3A",
    0.75: "#E8A020",
    1.0:  "#C00000",
}


def chart_n_frontier(frontier_df, asset_names, mvp, max_sr, asset_mus=None, asset_sds=None, rf=None):
    """
    N-asset efficient frontier chart.

    Parameters
    ----------
    frontier_df : pd.DataFrame  output of build_n_frontier()
    asset_names : list of str
    mvp         : pd.Series     MVP row
    max_sr      : pd.Series     Max Sharpe row

    Returns
    -------
    go.Figure
    """
    fig    = go.Figure()
    n      = len(asset_names)
    w_cols = [f"w_{i+1}" for i in range(n)]
    eff_df = frontier_df.sort_values("ret")

    def _cd(df, label):
        out = []
        for _, row in df.iterrows():
            wstr = " | ".join(
                f"{asset_names[i]}: {row[w_cols[i]] * 100:.1f}%"
                for i in range(n)
            )
            out.append([label, wstr, row["sharpe"]])
        return out

    hover_tpl = (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "──────────────────<br>"
        "Exp. Return: %{y:.2f}%<br>"
        "Std. Dev.: %{x:.2f}%<br>"
        "Sharpe Ratio: %{customdata[2]:.3f}"
        "<extra></extra>"
    )

    if not eff_df.empty:
        fig.add_trace(go.Scatter(
            x=eff_df["sd"], y=eff_df["ret"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color=COLORS["efficient"], width=3),
            customdata=_cd(eff_df, "Efficient Frontier"),
            hovertemplate=hover_tpl,
        ))

    # Asset endpoint markers — always plot at each asset's own (σ, μ)
    if asset_mus is not None and asset_sds is not None:
        for i, name in enumerate(asset_names):
            color = N_ASSET_COLORS[i % len(N_ASSET_COLORS)]
            fig.add_trace(go.Scatter(
                x=[asset_sds[i]], y=[asset_mus[i]],
                mode="markers+text",
                marker=dict(size=10, color=color, symbol="circle",
                            line=dict(width=1, color="white")),
                text=[name], textposition="top right",
                textfont=dict(size=10, color=color),
                name=name,
                showlegend=True,
                hoverinfo="skip",
            ))

    # MVP
    fig.add_trace(go.Scatter(
        x=[mvp["sd"]], y=[mvp["ret"]],
        mode="markers+text",
        marker=dict(size=14, color=COLORS["green"], symbol="star",
                    line=dict(width=1, color="white")),
        text=["MVP"], textposition="top left",
        textfont=dict(size=10, color=COLORS["green"]),
        name="Min. Variance Portfolio",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Max Sharpe
    fig.add_trace(go.Scatter(
        x=[max_sr["sd"]], y=[max_sr["ret"]],
        mode="markers+text",
        marker=dict(size=14, color=COLORS["blue"], symbol="star-diamond",
                    line=dict(width=1, color="white")),
        text=["MSP"], textposition="top right",
        textfont=dict(size=10, color=COLORS["blue"]),
        name="Max Sharpe Portfolio",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Capital Market Line — from (0, rf) through tangency portfolio (MSP)
    if rf is not None and max_sr["sd"] > 0:
        slope   = max_sr["sharpe"]                       # = (ret_T - rf) / sd_T
        x_end   = eff_df["sd"].max() * 1.35
        y_start = rf
        y_end   = rf + slope * x_end
        fig.add_trace(go.Scatter(
            x=[0, x_end],
            y=[y_start, y_end],
            mode="lines",
            name="Capital Market Line (CML)",
            line=dict(color=COLORS.get("orange", "#E8883A"), width=1.8, dash="dot"),
            hovertemplate=(
                "<b>Capital Market Line</b><br>"
                "σ = %{x:.2f}%<br>"
                "E[R] = %{y:.2f}%<br>"
                f"Slope (Sharpe) = {slope:.3f}"
                "<extra></extra>"
            ),
        ))
        # Mark the risk-free rate intercept
        fig.add_trace(go.Scatter(
            x=[0], y=[rf],
            mode="markers+text",
            marker=dict(size=9, color=COLORS.get("orange", "#E8883A"), symbol="circle",
                        line=dict(width=1, color="white")),
            text=[f"rf={rf:.1f}%"], textposition="middle right",
            textfont=dict(size=9, color=COLORS.get("orange", "#E8883A")),
            name="Risk-Free Rate",
            showlegend=True,
            hoverinfo="skip",
        ))

    subtitle = "Green = Efficient Frontier  |  Stars = MVP & Max Sharpe Portfolio"
    if rf is not None:
        subtitle += "  |  Orange = Capital Market Line"
    fig.update_layout(_base_layout(
        title=f"N-Asset Efficient Frontier ({n} Assets)",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle=subtitle,
    ))
    return fig


def chart_n_weights_bar(portfolios, asset_names):
    """
    Stacked horizontal bar chart of asset weight allocations.

    Parameters
    ----------
    portfolios  : list of (label, pd.Series)
    asset_names : list of str

    Returns
    -------
    go.Figure
    """
    fig      = go.Figure()
    n        = len(asset_names)
    w_cols   = [f"w_{i+1}" for i in range(n)]
    p_labels = [p[0] for p in portfolios]

    for i, (name, col) in enumerate(zip(asset_names, w_cols)):
        weights = [round(float(row[col]) * 100, 2) for _, row in portfolios]
        color   = N_ASSET_COLORS[i % len(N_ASSET_COLORS)]
        fig.add_trace(go.Bar(
            name=name,
            x=weights,
            y=p_labels,
            orientation="h",
            marker_color=color,
            text=[f"{w:.1f}%" for w in weights],
            textposition="inside",
            insidetextanchor="middle",
        ))

    fig.update_layout(
        barmode="relative",
        title=dict(
            text="<b>Portfolio Weight Allocations</b>",
            font=dict(size=14, color=COLORS["navy"]),
            x=0, xanchor="left",
        ),
        xaxis=dict(title="Weight (%)", ticksuffix="%", gridcolor="#E8E8E8"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor=COLORS["white"],
        paper_bgcolor=COLORS["white"],
        legend=dict(title="Asset", font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#CCCCCC", borderwidth=1),
        margin=dict(t=60, b=60, l=150, r=60),
        height=max(220, 60 + len(portfolios) * 55),
    )
    return fig


def chart_n_heatmap(corr_matrix, asset_names):
    """
    Correlation matrix heatmap.

    Parameters
    ----------
    corr_matrix : array-like, shape (N, N)
    asset_names : list of str

    Returns
    -------
    go.Figure
    """
    corr = np.asarray(corr_matrix, dtype=float)
    n    = len(asset_names)
    text = [[f"{corr[i, j]:.2f}" for j in range(n)] for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=asset_names,
        y=asset_names,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[
            [0.0, "#C00000"],
            [0.5, "#FFFFFF"],
            [1.0, "#1F4E79"],
        ],
        zmin=-1, zmax=1,
        colorbar=dict(
            title="ρ",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0", "-0.5", "0.0", "+0.5", "+1.0"],
        ),
        hovertemplate=(
            "<b>%{y} × %{x}</b><br>"
            "Correlation: %{z:.3f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="<b>Correlation Matrix</b>",
                   font=dict(size=14, color=COLORS["navy"]),
                   x=0, xanchor="left"),
        xaxis=dict(side="bottom", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        margin=dict(t=60, b=60, l=80, r=60),
        paper_bgcolor=COLORS["white"],
        plot_bgcolor=COLORS["white"],
        height=max(300, 60 + n * 50),
    )
    return fig


def chart_n_kappa_effect(kappa_frontiers, current_kappa, mvp_points):
    """
    Overlaid N-asset frontiers for different correlation scalars κ.

    Parameters
    ----------
    kappa_frontiers : dict { κ: pd.DataFrame }
    current_kappa   : float
    mvp_points      : dict { κ: pd.Series }

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for kappa in sorted(kappa_frontiers.keys()):
        df     = kappa_frontiers[kappa]
        color  = KAPPA_COLORS.get(kappa, COLORS["gray"])
        is_cur = abs(kappa - current_kappa) < 1e-9
        width  = 3   if is_cur else 1.5
        opacity= 1.0 if is_cur else 0.55
        dash   = "solid" if is_cur else "dash"
        label  = f"κ = {kappa:.2f}"

        if df.empty:
            continue
        df_s = df.sort_values("ret")
        fig.add_trace(go.Scatter(
            x=df_s["sd"], y=df_s["ret"],
            mode="lines",
            name=label,
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Exp. Return: %{y:.2f}%<br>"
                "Std. Dev.: %{x:.2f}%"
                "<extra></extra>"
            ),
        ))

        mvp = mvp_points.get(kappa)
        if mvp is not None:
            fig.add_trace(go.Scatter(
                x=[mvp["sd"]], y=[mvp["ret"]],
                mode="markers",
                marker=dict(size=9 if is_cur else 7,
                            color=color, symbol="star",
                            line=dict(width=1, color="white")),
                name=f"MVP (κ={kappa:.2f})",
                showlegend=False,
                hovertemplate=(
                    f"<b>MVP — {label}</b><br>"
                    "Std. Dev.: %{x:.2f}%<br>"
                    "Exp. Return: %{y:.2f}%"
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(_base_layout(
        title="Correlation Scalar (κ) Effect on Efficient Frontier",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="κ = 0: uncorrelated  |  κ = 1: full correlation  |  Stars = MVP per κ",
    ))
    return fig


def chart_n_kappa_mvp_table(kappa_frontiers, mvp_points, current_kappa):
    """
    Plotly table: MVP statistics across κ values.

    Returns
    -------
    go.Figure
    """
    rows = []
    for k in sorted(kappa_frontiers.keys()):
        mvp = mvp_points.get(k)
        if mvp is None:
            continue
        rows.append({
            "κ (Corr. Scalar)": f"{k:.2f}",
            "MVP Std. Dev.":    f"{mvp['sd']:.2f}%",
            "MVP Exp. Return":  f"{mvp['ret']:.2f}%",
            "MVP Sharpe":       f"{mvp['sharpe']:.3f}",
            "is_current":       abs(k - current_kappa) < 1e-9,
        })

    if not rows:
        return go.Figure()

    df   = pd.DataFrame(rows)
    cols = [c for c in df.columns if c != "is_current"]
    fill = ["#FFF3CD" if r else "#F8F9FA" for r in df["is_current"]]

    fig = go.Figure(data=[go.Table(
        columnwidth=[130, 130, 140, 110],
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color=COLORS["navy"],
            font=dict(color="white", size=12),
            align="center", height=36,
        ),
        cells=dict(
            values=[df[c].tolist() for c in cols],
            fill_color=[fill] * len(cols),
            font=dict(color=COLORS["gray"], size=11),
            align="center", height=30,
        ),
    )])
    fig.update_layout(
        title=dict(text="<b>MVP Comparison Across Correlation Scalars</b>",
                   font=dict(size=13, color=COLORS["navy"]),
                   x=0, xanchor="left"),
        margin=dict(t=55, b=15, l=0, r=0),
        paper_bgcolor=COLORS["white"],
    )
    return fig


def chart_n_solver(frontier_df, result_row, mvp, asset_names):
    """
    N-asset solver result plotted on the efficient frontier.

    Parameters
    ----------
    frontier_df : pd.DataFrame
    result_row  : pd.Series or None
    mvp         : pd.Series
    asset_names : list of str

    Returns
    -------
    go.Figure
    """
    fig    = go.Figure()
    n      = len(asset_names)
    w_cols = [f"w_{i+1}" for i in range(n)]
    df_s   = frontier_df.sort_values("ret")

    fig.add_trace(go.Scatter(
        x=df_s["sd"], y=df_s["ret"],
        mode="lines", name="Efficient Frontier",
        line=dict(color=COLORS["efficient"], width=2),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[mvp["sd"]], y=[mvp["ret"]],
        mode="markers+text",
        marker=dict(size=12, color=COLORS["green"], symbol="star",
                    line=dict(width=1, color="white")),
        text=["MVP"], textposition="top left",
        textfont=dict(size=10, color=COLORS["green"]),
        name="Min. Variance Portfolio",
        hoverinfo="skip",
    ))

    if result_row is not None:
        wstr = " | ".join(
            f"{asset_names[i]}: {result_row[w_cols[i]] * 100:.1f}%"
            for i in range(n)
        )
        fig.add_trace(go.Scatter(
            x=[result_row["sd"]], y=[result_row["ret"]],
            mode="markers+text",
            marker=dict(size=18, color="#FF8C00", symbol="star",
                        line=dict(width=2, color="white")),
            text=["Result"], textposition="top right",
            textfont=dict(size=11, color="#FF8C00"),
            name="Solver Result",
            customdata=[[wstr, result_row["sharpe"]]],
            hovertemplate=(
                "<b>Solver Result</b><br>"
                "%{customdata[0]}<br>"
                "Exp. Return: %{y:.2f}%<br>"
                "Std. Dev.: %{x:.2f}%<br>"
                "Sharpe: %{customdata[1]:.3f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(_base_layout(
        title="Portfolio Solver — Result on Efficient Frontier",
        xaxis_title="Portfolio Std. Dev. (%)",
        yaxis_title="Portfolio Exp. Return (%)",
        subtitle="Gold star = solver result",
    ))
    return fig


def chart_n_summary_table(frontier_df, asset_names, mvp, max_sr):
    """
    Plotly table with key N-asset portfolios: 100% each asset, MVP, Max Sharpe.

    Returns
    -------
    go.Figure
    """
    n      = len(asset_names)
    w_cols = [f"w_{i+1}" for i in range(n)]

    def wstr(row):
        return " / ".join(
            f"{asset_names[i]}: {row[w_cols[i]] * 100:.1f}%"
            for i in range(n)
        )

    rows = []
    for i, name in enumerate(asset_names):
        col   = w_cols[i]
        ep_df = frontier_df[(frontier_df[col] > 0.97) &
                            (frontier_df[col] < 1.03)]
        if not ep_df.empty:
            row = ep_df.iloc[(ep_df[col] - 1.0).abs().argsort().iloc[0]]
            rows.append({
                "Portfolio":    f"100% {name}",
                "Weights":      wstr(row),
                "Exp. Return":  f"{row['ret']:.2f}%",
                "Std. Dev.":    f"{row['sd']:.2f}%",
                "Sharpe Ratio": f"{row['sharpe']:.3f}",
            })

    rows.append({
        "Portfolio":    "⭐ Min. Variance Portfolio (MVP)",
        "Weights":      wstr(mvp),
        "Exp. Return":  f"{mvp['ret']:.2f}%",
        "Std. Dev.":    f"{mvp['sd']:.2f}%",
        "Sharpe Ratio": f"{mvp['sharpe']:.3f}",
    })
    rows.append({
        "Portfolio":    "⭐ Max. Sharpe Portfolio (MSP)",
        "Weights":      wstr(max_sr),
        "Exp. Return":  f"{max_sr['ret']:.2f}%",
        "Std. Dev.":    f"{max_sr['sd']:.2f}%",
        "Sharpe Ratio": f"{max_sr['sharpe']:.3f}",
    })

    df     = pd.DataFrame(rows)
    n_cols = len(df.columns)
    fill   = [
        "#FFF3CD" if "⭐" in str(r.get("Portfolio", "")) else "#F8F9FA"
        for _, r in df.iterrows()
    ]

    fig = go.Figure(data=[go.Table(
        columnwidth=[200, 350, 110, 100, 110],
        header=dict(
            values=[f"<b>{c}</b>" for c in df.columns],
            fill_color=COLORS["navy"],
            font=dict(color="white", size=12),
            align="center", height=36,
        ),
        cells=dict(
            values=[df[c].tolist() for c in df.columns],
            fill_color=[fill] * n_cols,
            font=dict(color=COLORS["gray"], size=11),
            align=["left", "left"] + ["center"] * (n_cols - 2),
            height=32,
        ),
    )])
    fig.update_layout(
        title=dict(text="<b>Portfolio Summary — Key Allocation Points</b>",
                   font=dict(size=14, color=COLORS["navy"]),
                   x=0, xanchor="left"),
        margin=dict(t=60, b=20, l=0, r=0),
        paper_bgcolor=COLORS["white"],
    )
    return fig
