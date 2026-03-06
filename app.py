"""
app.py
------
Main Streamlit application — wires sidebar, calculations, and charts together.

FIN 511 - Investments I: Module 1, Lesson 1-5
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from calculations import (
    build_frontier, build_cal, build_rho_frontiers,
    find_mvp, find_max_sharpe, find_max_return,
    efficient_frontier_region, benchmark_stats,
    frontier_summary_table, cal_summary_table,
    rho_mvp_table, cal_equation_str, rho_mvp_sd,
)
from charts import (
    chart_frontier_all,
    chart_frontier_efficient_dominated,
    chart_frontier_short_A1,
    chart_frontier_long_A1,
    chart_cal_all,
    chart_cal_all_long,
    chart_cal_long_no_leverage,
    chart_cal_long_with_leverage,
    chart_rho_effect,
    chart_rho_mvp_table,
    chart_frontier_summary_table,
    chart_cal_summary_table,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Portfolio Frontier & CAL Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1F4E79;
        margin-bottom: 0.1rem;
    }
    .main-subtitle {
        font-size: 0.9rem;
        color: #595959;
        margin-bottom: 1.2rem;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #F2F7FC;
        border: 1px solid #BDD7EE;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Section headers in sidebar */
    .sidebar-section {
        font-size: 0.78rem;
        font-weight: 700;
        color: #1F4E79;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
        padding-bottom: 4px;
        border-bottom: 2px solid #BDD7EE;
    }

    /* Optimal portfolio card */
    .opt-card {
        background: #F0F7F0;
        border: 1px solid #C6EFCE;
        border-left: 4px solid #1E6B3A;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .opt-card-title {
        font-weight: 700;
        color: #1E6B3A;
        font-size: 0.88rem;
        margin-bottom: 6px;
    }
    .opt-card-row {
        display: flex;
        justify-content: space-between;
        color: #333;
        margin-bottom: 2px;
    }
    .opt-card-value {
        font-weight: 600;
        color: #1E6B3A;
        font-family: monospace;
    }

    /* Efficient frontier region card */
    .eff-region-card {
        background: #EBF3FB;
        border: 1px solid #BDD7EE;
        border-left: 4px solid #2E75B6;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.85rem;
    }
    .eff-region-title {
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }

    /* Dividers */
    .section-divider {
        border: none;
        border-top: 1px solid #E0E0E0;
        margin: 1.2rem 0;
    }

    /* Tab content top padding */
    .tab-content { padding-top: 0.5rem; }

    /* Info/warning boxes */
    .info-box {
        background: #EBF3FB;
        border-left: 4px solid #2E75B6;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #1F4E79;
        margin-bottom: 1rem;
    }
    .warn-box {
        background: #FFF8E7;
        border-left: 4px solid #E8A020;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #7B4F00;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — initialise defaults on first run
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = dict(
    f_r1=8.0, f_sd1=25.0,
    f_r2=15.0, f_sd2=50.0,
    f_rho=0.4, f_rf=3.0,
    c_r_risky=8.0, c_sd_risky=25.0, c_rf=3.0,
    allow_short=False,
)

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def apply_preset(preset):
    for k, v in preset.items():
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown("## ⚙️ Controls")

    # ── Short-selling toggle ───────────────────────────────────────────────
    allow_short = st.checkbox(
        "☐ Allow Short-Selling",
        value=st.session_state.allow_short,
        help="Enables short-selling (w < 0 or w > 1). Affects Tab 1 and Tab 2.",
        key="allow_short",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Presets ────────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>Quick Presets</div>",
                unsafe_allow_html=True)

    preset_cols = st.columns(2)
    with preset_cols[0]:
        if st.button("Default", use_container_width=True):
            apply_preset(DEFAULTS)
            st.rerun()
        if st.button("ρ = 0", use_container_width=True):
            apply_preset({**DEFAULTS, "f_rho": 0.0})
            st.rerun()
    with preset_cols[1]:
        if st.button("Assignment 1\n(ρ = −0.8)", use_container_width=True):
            apply_preset({**DEFAULTS, "f_rho": -0.8})
            st.rerun()
        if st.button("ρ = −1", use_container_width=True):
            apply_preset({**DEFAULTS, "f_rho": -1.0})
            st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Portfolio Frontier parameters ──────────────────────────────────────
    st.markdown("<div class='sidebar-section'>📊 Portfolio Frontier</div>",
                unsafe_allow_html=True)

    st.markdown("**Asset 1**")
    f_r1 = st.slider("Exp. Return (%)",
                     0.0, 25.0, st.session_state.f_r1, 0.5,
                     key="f_r1")
    f_sd1 = st.slider("Std. Dev. (%)",
                      5.0, 60.0, st.session_state.f_sd1, 1.0,
                      key="f_sd1")

    st.markdown("**Asset 2**")
    f_r2 = st.slider("Exp. Return (%) ",
                     0.0, 30.0, st.session_state.f_r2, 0.5,
                     key="f_r2")

    # ── Validation: Asset 2 sd must be > Asset 1 sd ────────────────────────
    sd2_min = f_sd1 + 1.0
    sd2_val = max(st.session_state.f_sd2, sd2_min)

    if st.session_state.f_sd2 < sd2_min:
        st.info(
            f"ℹ️ Asset 2 Std. Dev. adjusted to {sd2_val:.0f}% "
            f"— must exceed Asset 1 Std. Dev. ({f_sd1:.0f}%)."
        )

    f_sd2 = st.slider(
        "Std. Dev. (%)  ",
        min_value=sd2_min,
        max_value=80.0,
        value=sd2_val,
        step=1.0,
        key="f_sd2",
        help="Asset 2 must always be riskier than Asset 1.",
    )

    st.markdown("**Correlation & Risk-Free Rate**")
    f_rho = st.slider("Correlation (ρ)",
                      -1.0, 1.0, st.session_state.f_rho, 0.1,
                      key="f_rho")
    f_rf  = st.slider("Risk-Free Rate (%)",
                      0.0, 10.0, st.session_state.f_rf, 0.5,
                      key="f_rf",
                      help="Used for Sharpe ratio calculation only.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── CAL parameters ─────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>📈 Capital Allocation Line</div>",
                unsafe_allow_html=True)

    st.markdown("**Risky Asset**")
    c_r_risky  = st.slider("Exp. Return (%)",
                           0.0, 25.0, st.session_state.c_r_risky, 0.5,
                           key="c_r_risky")
    c_sd_risky = st.slider("Std. Dev. (%)",
                           1.0, 60.0, st.session_state.c_sd_risky, 1.0,
                           key="c_sd_risky")

    st.markdown("**Risk-Free Asset**")
    c_rf = st.slider("Risk-Free Rate (%)",
                     0.0, 10.0, st.session_state.c_rf, 0.5,
                     key="c_rf",
                     help="T-Bill rate — zero std. dev. by definition.")
    st.caption("📌 Risk-free asset: zero std. dev., zero correlation with risky asset.")


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE — run all calculations with current params
# ══════════════════════════════════════════════════════════════════════════════

# Frontier
frontier_df   = build_frontier(f_r1, f_sd1, f_r2, f_sd2, f_rho, f_rf)
mvp           = find_mvp(frontier_df)
max_sr        = find_max_sharpe(frontier_df, long_only=True)
max_ret_lo    = find_max_return(frontier_df, long_only=True)
max_ret_lev   = find_max_return(frontier_df, long_only=False)
eff_df, eff_summary = efficient_frontier_region(frontier_df)
benchmarks    = benchmark_stats(f_r1, f_sd1, f_r2, f_sd2, f_rho, f_rf)
f_summary_tbl = frontier_summary_table(
    frontier_df, f_r1, f_sd1, f_r2, f_sd2, f_rho, f_rf,
    allow_short=allow_short,
)

# CAL
cal_df        = build_cal(c_r_risky, c_sd_risky, c_rf)
c_summary_tbl = cal_summary_table(c_r_risky, c_sd_risky, c_rf,
                                   allow_short=allow_short)
cal_eq_str    = cal_equation_str(c_r_risky, c_sd_risky, c_rf)
cal_sharpe    = ((c_r_risky - c_rf) / c_sd_risky
                 if c_sd_risky > 0 else 0.0)

# Correlation
rho_frontiers = build_rho_frontiers(f_r1, f_sd1, f_r2, f_sd2, f_rf)
rho_mvp_df    = rho_mvp_table(f_r1, f_sd1, f_r2, f_sd2, f_rf,
                               current_rho=f_rho)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div class='main-header'>📊 Portfolio Frontier & CAL Explorer</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='main-subtitle'>"
    "FIN 511 · Module 1 · Lesson 1-5 — Adjust sliders in the sidebar to update all charts live"
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊  Portfolio Frontier",
    "📈  Capital Allocation Line",
    "🔗  Correlation Effect",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — PORTFOLIO FRONTIER
# ────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── Short-selling message ────────────────────────────────────────────────
    if not allow_short:
        st.markdown(
            "<div class='info-box'>"
            "ℹ️ <b>Short-selling is currently disabled.</b> "
            "Only portfolios with Asset 1 Weight: 0%→100% and Asset 2 Weight: 0%→100% are shown. "
            "This reflects the real-world constraint most investors face in 401k plans and "
            "standard brokerage accounts. Enable short-selling in the sidebar to see the full "
            "frontier including short-selling and leveraged allocations."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='warn-box'>"
            "⚠️ <b>Short-selling is enabled.</b> "
            "Charts now show all allocations including: "
            "Short Asset 1 / Long Asset 2 (Asset 1 Weight &lt; 0%) and "
            "Long Asset 1 / Short Asset 2 (Asset 1 Weight &gt; 100%). "
            "Note: Short-selling requires a margin account and involves borrowing costs "
            "not reflected here (per Prof. Weisbenner, Lesson 1-2.5)."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── METRICS ROW 1 — Benchmark portfolios ────────────────────────────────
    st.markdown("#### Benchmark Portfolios")

    def benchmark_card(title, ret, sd, sharpe, border_color="#2E75B6"):
        """Render a benchmark portfolio as a compact HTML card."""
        return (
            f'<div style="background:#F8FBFF;border:1px solid #BDD7EE;'
            f'border-top:4px solid {border_color};border-radius:8px;'
            f'padding:14px 18px;flex:1;">'
            f'<div style="font-weight:700;font-size:0.92rem;color:#1F4E79;'
            f'margin-bottom:12px;">{title}</div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
            f'<tr>'
            f'<td style="color:#595959;padding:3px 0;">Exp. Return</td>'
            f'<td style="text-align:right;font-weight:700;font-family:monospace;color:#1F4E79;">{ret:.2f}%</td>'
            f'</tr>'
            f'<tr>'
            f'<td style="color:#595959;padding:3px 0;">Std. Dev.</td>'
            f'<td style="text-align:right;font-weight:700;font-family:monospace;color:#1F4E79;">{sd:.2f}%</td>'
            f'</tr>'
            f'<tr>'
            f'<td style="color:#595959;padding:3px 0;">Sharpe Ratio</td>'
            f'<td style="text-align:right;font-weight:700;font-family:monospace;color:#1F4E79;">{sharpe:.3f}</td>'
            f'</tr>'
            f'</table>'
            f'</div>'
        )

    c1 = benchmark_card("100% Asset 1",
                        benchmarks['asset1']['ret'],
                        benchmarks['asset1']['sd'],
                        benchmarks['asset1']['sharpe'],
                        border_color="#1F4E79")
    c2 = benchmark_card("100% Asset 2",
                        benchmarks['asset2']['ret'],
                        benchmarks['asset2']['sd'],
                        benchmarks['asset2']['sharpe'],
                        border_color="#E8A020")
    c3 = benchmark_card("Equal Weight (50/50)",
                        benchmarks['equal']['ret'],
                        benchmarks['equal']['sd'],
                        benchmarks['equal']['sharpe'],
                        border_color="#1E6B3A")

    st.markdown(
        f'<div style="display:flex;gap:16px;margin-bottom:8px;">{c1}{c2}{c3}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── METRICS ROW 2 — Optimal portfolio cards ──────────────────────────────
    st.markdown("#### Optimal Portfolios")

    def opt_card(title, row, extra_note=""):
        """Render a styled optimal portfolio card."""
        note_html = (f"<div style='margin-top:6px;font-size:0.78rem;color:#595959'>{extra_note}</div>"
                     if extra_note else "")
        html = (
            f"<div class='opt-card'>"
            f"<div class='opt-card-title'>{title}</div>"
            f"<div class='opt-card-row'><span>Asset 1 Weight</span><span class='opt-card-value'>{row['w_A1']*100:.1f}%</span></div>"
            f"<div class='opt-card-row'><span>Asset 2 Weight</span><span class='opt-card-value'>{row['w_A2']*100:.1f}%</span></div>"
            f"<div class='opt-card-row'><span>Exp. Return</span><span class='opt-card-value'>{row['ret']:.2f}%</span></div>"
            f"<div class='opt-card-row'><span>Std. Dev.</span><span class='opt-card-value'>{row['sd']:.2f}%</span></div>"
            f"<div class='opt-card-row'><span>Sharpe Ratio</span><span class='opt-card-value'>{row['sharpe']:.3f}</span></div>"
            f"{note_html}"
            f"</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    n_cards = 4 if allow_short else 3
    card_cols = st.columns(n_cards)

    with card_cols[0]:
        opt_card("⭐ Min. Variance Portfolio", mvp,
                 "Lowest achievable Std. Dev.")
    with card_cols[1]:
        opt_card("⭐ Max. Sharpe Portfolio", max_sr,
                 "Highest risk-adjusted return (long-only)")
    with card_cols[2]:
        opt_card("⭐ Max. Return (Long Only)", max_ret_lo,
                 "Asset 2 Weight = 100%")
    if allow_short:
        with card_cols[3]:
            opt_card("⭐ Max. Return (Leveraged)", max_ret_lev,
                     "Asset 2 Weight = 200% — requires short-selling")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── METRICS ROW 3 — Efficient frontier region ────────────────────────────
    st.markdown("#### 📊 Efficient Frontier Region")
    st.caption(
        "Long-only portfolios above the Min. Variance Portfolio "
        "(Asset 1 Weight: 0%→100%, Exp. Return ≥ MVP Exp. Return) — "
        "consistent with Prof. Weisbenner's Lesson 1-5."
    )

    if eff_summary:
        # Row 1: weight ranges + peak sharpe (3 columns — less crowded)
        e1, e2, e3 = st.columns(3)
        e1.metric(
            "Asset 1 Weight Range",
            f"{eff_summary['w_A1_range'][0]*100:.0f}%",
            f"→ {eff_summary['w_A1_range'][1]*100:.0f}%",
            help="From MVP down to 100% Asset 2",
        )
        e2.metric(
            "Asset 2 Weight Range",
            f"{eff_summary['w_A2_range'][0]*100:.0f}%",
            f"→ {eff_summary['w_A2_range'][1]*100:.0f}%",
            help="From MVP up to 100% Asset 2",
        )
        e3.metric(
            "Peak Sharpe Ratio",
            f"{eff_summary['peak_sharpe']:.3f}",
            help=(f"At Asset 1 Weight = {eff_summary['peak_w_A1']*100:.0f}%, "
                  f"Asset 2 Weight = {eff_summary['peak_w_A2']*100:.0f}%"),
        )

        # Row 2: std dev range + return range + portfolio count
        e4, e5, e6 = st.columns(3)
        e4.metric(
            "Std. Dev. Range",
            f"{eff_summary['sd_range'][0]:.2f}%",
            f"→ {eff_summary['sd_range'][1]:.2f}%",
            help="From MVP (lowest) to 100% Asset 2 (highest)",
        )
        e5.metric(
            "Exp. Return Range",
            f"{eff_summary['ret_range'][0]:.2f}%",
            f"→ {eff_summary['ret_range'][1]:.2f}%",
            help="From MVP return up to 100% Asset 2 return",
        )
        e6.metric(
            "Portfolios in Region",
            f"{eff_summary['n_portfolios']}",
            help="Number of long-only portfolio combinations above the MVP",
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── CHARTS ───────────────────────────────────────────────────────────────
    st.markdown("#### Charts")

    if allow_short:
        # 2x2 grid — all 4 charts
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            st.plotly_chart(
                chart_frontier_all(frontier_df, f_r1, f_sd1, f_r2, f_sd2, mvp),
                use_container_width=True, key="f_all"
            )
        with row1_c2:
            st.plotly_chart(
                chart_frontier_efficient_dominated(
                    frontier_df, f_r1, f_sd1, f_r2, f_sd2, mvp),
                use_container_width=True, key="f_effdom"
            )
        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            st.plotly_chart(
                chart_frontier_short_A1(frontier_df, f_r1, f_sd1, f_r2, f_sd2),
                use_container_width=True, key="f_shortA1"
            )
        with row2_c2:
            st.plotly_chart(
                chart_frontier_long_A1(frontier_df, f_r1, f_sd1, f_r2, f_sd2),
                use_container_width=True, key="f_longA1"
            )
    else:
        # Single chart — efficient vs dominated only
        st.plotly_chart(
            chart_frontier_efficient_dominated(
                frontier_df, f_r1, f_sd1, f_r2, f_sd2, mvp),
            use_container_width=True, key="f_effdom_only"
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    st.markdown("#### Portfolio Summary Table")
    st.plotly_chart(
        chart_frontier_summary_table(f_summary_tbl),
        use_container_width=True, key="f_summary_tbl"
    )


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — CAPITAL ALLOCATION LINE
# ────────────────────────────────────────────────────────────────────────────
with tab2:

    # ── Short-selling message ────────────────────────────────────────────────
    if not allow_short:
        st.markdown(
            "<div class='info-box'>"
            "ℹ️ <b>Short-selling disabled.</b> "
            "CAL shown for long positions only (Risky Asset Weight ≥ 0%). "
            "The 'All Allocations' chart — which includes Risky Asset Weight &lt; 0% "
            "(shorting the risky asset to put more than 100% into T-Bills) — is hidden."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='warn-box'>"
            "⚠️ <b>Short-selling enabled.</b> "
            "'All Allocations' chart now includes Risky Asset Weight &lt; 0%: "
            "shorting the risky asset to invest more than 100% in T-Bills. "
            "Note: Sharpe ratio is negative in this region — "
            "you earn less than the risk-free rate."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── METRICS ──────────────────────────────────────────────────────────────
    st.markdown("#### CAL Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sharpe Ratio",    f"{cal_sharpe:.3f}")
    m2.metric("Risk-Free Rate",  f"{c_rf:.1f}%",
              help="Intercept of the CAL — 100% T-Bills")
    m3.metric("100% Risky",
              f"E[R]={c_r_risky:.1f}%  σ={c_sd_risky:.1f}%")
    m4.metric("w=2 (Leverage)",
              f"E[R]={2*c_r_risky - c_rf:.1f}%  σ={2*c_sd_risky:.1f}%",
              help="Borrow 100% of portfolio value at rf to double risky exposure")
    m5.metric("CAL Equation", cal_eq_str)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── CHARTS ───────────────────────────────────────────────────────────────
    st.markdown("#### Charts")

    if allow_short:
        # Chart 1: All Allocations (short-selling ON only)
        st.plotly_chart(
            chart_cal_all(cal_df, c_r_risky, c_sd_risky, c_rf),
            use_container_width=True, key="c_all"
        )

    # Chart 2: All Long (always visible)
    st.plotly_chart(
        chart_cal_all_long(cal_df, c_r_risky, c_sd_risky, c_rf),
        use_container_width=True, key="c_all_long"
    )

    # Charts 3 & 4 side by side
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.plotly_chart(
            chart_cal_long_no_leverage(cal_df, c_r_risky, c_sd_risky, c_rf),
            use_container_width=True, key="c_no_lev"
        )
    with c_col2:
        st.plotly_chart(
            chart_cal_long_with_leverage(cal_df, c_r_risky, c_sd_risky, c_rf),
            use_container_width=True, key="c_lev"
        )

    # Chart 5: Equation summary
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### CAL Equation Summary")

    eq_col1, eq_col2 = st.columns([1, 1])
    with eq_col1:
        st.markdown(f"""
        <div class='eff-region-card'>
            <div class='eff-region-title'>📐 Capital Allocation Line Equation</div>
            <p style='font-family:monospace; font-size:1.05rem; color:#1F4E79; font-weight:700;'>
                {cal_eq_str}
            </p>
            <div style='font-size:0.83rem; color:#595959;'>
                <b>Intercept:</b> {c_rf:.1f}% &nbsp;(100% in T-Bills, zero risk)<br>
                <b>Slope:</b> {cal_sharpe:.3f} &nbsp;(Sharpe Ratio — reward per unit of risk)<br>
                <b>Interpretation:</b> For every 1% increase in Portfolio Std. Dev.,
                Portfolio Exp. Return increases by {cal_sharpe:.3f}%.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with eq_col2:
        st.markdown(f"""
        <div class='eff-region-card'>
            <div class='eff-region-title'>📌 Key Portfolio Points</div>
            <div style='font-size:0.83rem;'>
                <div class='opt-card-row'>
                    <span>100% Risk-Free (w=0)</span>
                    <span class='opt-card-value'>E[R]={c_rf:.1f}%, σ=0%</span>
                </div>
                <div class='opt-card-row'>
                    <span>w = 0.5</span>
                    <span class='opt-card-value'>
                        E[R]={0.5*c_r_risky + 0.5*c_rf:.2f}%,
                        σ={0.5*c_sd_risky:.2f}%
                    </span>
                </div>
                <div class='opt-card-row'>
                    <span>100% Risky (w=1)</span>
                    <span class='opt-card-value'>E[R]={c_r_risky:.1f}%, σ={c_sd_risky:.1f}%</span>
                </div>
                <div class='opt-card-row'>
                    <span>w = 1.5 (leverage)</span>
                    <span class='opt-card-value'>
                        E[R]={1.5*c_r_risky - 0.5*c_rf:.2f}%,
                        σ={1.5*c_sd_risky:.2f}%
                    </span>
                </div>
                <div class='opt-card-row'>
                    <span>w = 2 (leverage)</span>
                    <span class='opt-card-value'>
                        E[R]={2*c_r_risky - c_rf:.2f}%,
                        σ={2*c_sd_risky:.2f}%
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    st.markdown("#### CAL Summary Table")
    st.plotly_chart(
        chart_cal_summary_table(c_summary_tbl),
        use_container_width=True, key="c_summary_tbl"
    )


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — CORRELATION EFFECT
# ────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown(
        "<div class='info-box'>"
        "ℹ️ Correlation frontiers are always displayed as long-only "
        "(Asset 1 & Asset 2 Weights: 0% → 100%) regardless of the short-selling setting. "
        "This is consistent with how Prof. Weisbenner presents correlation effects in Lesson 1-5."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── METRICS ──────────────────────────────────────────────────────────────
    st.markdown("#### Correlation Metrics")

    mvp_sd_neg1 = rho_mvp_sd(f_sd1, f_sd2, -1.0)
    mvp_sd_zero = rho_mvp_sd(f_sd1, f_sd2,  0.0)
    mvp_sd_pos1 = rho_mvp_sd(f_sd1, f_sd2,  1.0)
    mvp_sd_curr = rho_mvp_sd(f_sd1, f_sd2,  f_rho)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Current ρ",              f"{f_rho:.1f}")
    r2.metric("MVP Std. Dev. at ρ=−1",  f"{mvp_sd_neg1:.2f}%",
              help="Lowest achievable risk — perfect negative correlation")
    r3.metric("MVP Std. Dev. at ρ=0",   f"{mvp_sd_zero:.2f}%",
              help="Risk at zero correlation")
    r4.metric("MVP Std. Dev. at ρ=+1",  f"{mvp_sd_pos1:.2f}%",
              help="No diversification benefit — assets move in lockstep")

    # Diversification benefit vs ρ=0
    benefit = mvp_sd_zero - mvp_sd_curr
    st.metric(
        f"Diversification Benefit vs ρ=0 (current ρ={f_rho:.1f})",
        f"σ reduced by {benefit:.2f}%",
        help="How much the current correlation reduces MVP Std. Dev. vs zero correlation",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── CHARTS ───────────────────────────────────────────────────────────────
    st.markdown("#### Effect of Correlation on the Efficient Frontier")
    st.plotly_chart(
        chart_rho_effect(rho_frontiers, f_rho, f_r1, f_sd1, f_r2, f_sd2),
        use_container_width=True, key="rho_chart"
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("#### Min. Variance Portfolio — Comparison Across Correlations")
    st.plotly_chart(
        chart_rho_mvp_table(rho_mvp_df),
        use_container_width=True, key="rho_mvp_tbl"
    )

    st.caption(
        "💡 Assignment 1: Set ρ = −0.8 using the sidebar slider. "
        "Notice how MVP Std. Dev. drops dramatically — "
        "lower correlation means more diversification benefit."
    )
