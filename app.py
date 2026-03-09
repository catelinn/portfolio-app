"""
app.py
------
Main Streamlit application — wires sidebar, calculations, and charts together.

FIN 511 - Investments I: Module 1, Lesson 1-5
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

from calculations import (
    build_frontier, build_cal, build_rho_frontiers,
    find_mvp, find_max_sharpe, find_max_return,
    efficient_frontier_region, benchmark_stats,
    frontier_summary_table, cal_summary_table,
    rho_mvp_table, rho_msp_table, cal_equation_str, rho_mvp_sd,
    solve_portfolio,
    # N-asset
    build_n_frontier, n_find_mvp, n_find_max_sharpe,
    build_n_kappa_frontiers, n_solve_portfolio,
    validate_corr_matrix, nearest_psd_corr, parse_n_csv,
    n_portfolio_stats,
)
from charts import (
    chart_frontier_all,
    chart_frontier_long_only,
    chart_frontier_short_A1,
    chart_frontier_long_A1,
    chart_cal_all,
    chart_cal_all_long,
    chart_cal_long_no_leverage,
    chart_cal_long_with_leverage,
    chart_rho_effect,
    chart_rho_mvp_table,
    chart_rho_msp_table,
    chart_frontier_summary_table,
    chart_cal_summary_table,
    chart_frontier_with_solver,
    # N-asset
    chart_n_frontier, chart_n_weights_bar, chart_n_heatmap,
    chart_n_kappa_effect, chart_n_kappa_mvp_table,
    chart_n_solver, chart_n_summary_table,
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

    /* Reduce metric value font size app-wide */
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
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

    /* Parameter banner above expanders */
    .param-banner {
        background-color: #1F4E79;
        color: white;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 14px;
        border-radius: 6px 6px 0 0;
        margin-bottom: -1px;
        letter-spacing: 0.3px;
    }


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
    # Solver defaults
    sol_objective="Expected Return",
    sol_goal="Maximize",
    sol_constraint="Long Only",
    sol_target=10.0,
    sol_result_display="Both",
)

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── N-Asset defaults ───────────────────────────────────────────────────────────
_N_ASSET_DEFS = [
    ("Asset A",  8.0, 15.0),
    ("Asset B", 12.0, 25.0),
    ("Asset C",  6.0, 10.0),
    ("Asset D", 10.0, 20.0),
    ("Asset E", 14.0, 30.0),
    ("Asset F",  9.0, 18.0),
    ("Asset G", 11.0, 22.0),
    ("Asset H",  7.0, 12.0),
]

N_DEFAULTS: dict = {
    "n_n_assets":          3,
    "n_rf":                3.0,
    "n_kappa":             1.0,
    "n_csv_active":        False,
    "n_include_rf":        False,
    "n_sol_objective":     "Expected Return",
    "n_sol_goal":          "Maximize",
    "n_sol_target":        10.0,
    "n_sol_efficient_only": True,
}
for _i, (_name, _ret, _sd) in enumerate(_N_ASSET_DEFS):
    N_DEFAULTS[f"n_name_{_i}"] = _name
    N_DEFAULTS[f"n_ret_{_i}"]  = _ret
    N_DEFAULTS[f"n_sd_{_i}"]   = _sd
for _i in range(8):
    for _j in range(_i + 1, 8):
        N_DEFAULTS[f"n_corr_{_i}_{_j}"] = 0.3

for key, val in N_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def apply_preset(preset):
    """Store preset values under pending_ keys — sliders pick them up on next rerun."""
    for k, v in preset.items():
        st.session_state[f"pending_{k}"] = v


def _val(key):
    """Return pending value if a preset was just applied, else session_state value."""
    pending = st.session_state.pop(f"pending_{key}", None)
    if pending is not None:
        st.session_state[key] = pending
    return st.session_state[key]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — minimal, just short-selling toggle
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.caption("Parameters are in each tab — adjust them directly there.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    allow_short = st.checkbox(
        "☐ Allow Short-Selling",
        value=st.session_state.allow_short,
        help="Enables short-selling (w < 0 or w > 1). Affects Tab 1 and Tab 2.",
        key="allow_short",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Quick Presets</div>",
                unsafe_allow_html=True)
    preset_cols = st.columns(3)
    with preset_cols[0]:
        if st.button("Portfolio Frontier Baseline", use_container_width=True):
            apply_preset(DEFAULTS)
            st.rerun()
    with preset_cols[1]:
        if st.button("ρ = 1", use_container_width=True):
            apply_preset({**DEFAULTS, "f_rho": 1.0})
            st.rerun()
    with preset_cols[2]:
        if st.button("ρ = −1", use_container_width=True):
            apply_preset({**DEFAULTS, "f_rho": -1.0})
            st.rerun()
    cal_preset_cols = st.columns(2)
    with cal_preset_cols[0]:
        if st.button("CAL Baseline (Large Stock)", use_container_width=True):
            apply_preset({"c_rf": 3.0, "c_r_risky": 8.0, "c_sd_risky": 25.0})
            st.rerun()
    with cal_preset_cols[1]:
        if st.button("CAL Baseline (Small Stock)", use_container_width=True):
            apply_preset({"c_rf": 3.0, "c_r_risky": 15.0, "c_sd_risky": 50.0})
            st.rerun()

# ── Always read params from session state ─────────────────────────────────────
# Each tab renders its own sliders inside an expander which write to session_state.
# These variables are populated after the tab sliders run (Streamlit top-down order),
# so we initialise them from session_state here as fallback defaults.
f_r1       = st.session_state.f_r1
f_sd1      = st.session_state.f_sd1
f_r2       = st.session_state.f_r2
f_sd2      = st.session_state.f_sd2
f_rho      = st.session_state.f_rho
f_rf       = st.session_state.f_rf
c_r_risky  = st.session_state.c_r_risky
c_sd_risky = st.session_state.c_sd_risky
c_rf       = st.session_state.c_rf


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE HELPERS — called inside each tab after its sliders run
# ══════════════════════════════════════════════════════════════════════════════

def compute_frontier():
    """Recompute frontier data from current session state."""
    r1, sd1 = st.session_state.f_r1, st.session_state.f_sd1
    r2, sd2 = st.session_state.f_r2, st.session_state.f_sd2
    rho, rf = st.session_state.f_rho, st.session_state.f_rf
    short   = st.session_state.allow_short
    df      = build_frontier(r1, sd1, r2, sd2, rho, rf)
    return dict(
        frontier_df   = df,
        mvp           = find_mvp(df),
        max_sr        = find_max_sharpe(df, long_only=True),
        max_ret_lo    = find_max_return(df, long_only=True),
        max_ret_lev   = find_max_return(df, long_only=False),
        eff_summary   = efficient_frontier_region(df, allow_short=short)[1],
        benchmarks    = benchmark_stats(r1, sd1, r2, sd2, rho, rf),
        f_summary_tbl = frontier_summary_table(df, r1, sd1, r2, sd2, rho, rf, allow_short=short),
        f_r1=r1, f_sd1=sd1, f_r2=r2, f_sd2=sd2, f_rho=rho, f_rf=rf,
    )

def compute_cal():
    """Recompute CAL data from current session state."""
    r, sd, rf = st.session_state.c_r_risky, st.session_state.c_sd_risky, st.session_state.c_rf
    short     = st.session_state.allow_short
    return dict(
        cal_df        = build_cal(r, sd, rf),
        c_summary_tbl = cal_summary_table(r, sd, rf, allow_short=short),
        cal_eq_str    = cal_equation_str(r, sd, rf),
        cal_sharpe    = (r - rf) / sd if sd > 0 else 0.0,
        c_r_risky=r, c_sd_risky=sd, c_rf=rf,
    )

def compute_rho():
    """Recompute correlation data from current session state."""
    r1, sd1 = st.session_state.f_r1, st.session_state.f_sd1
    r2, sd2 = st.session_state.f_r2, st.session_state.f_sd2
    rho, rf = st.session_state.f_rho, st.session_state.f_rf
    short   = st.session_state.allow_short
    ref_rhos = [-0.8, -0.4, 0.0, 0.4, 0.8]
    rho_list = ref_rhos if any(abs(rho - r) < 1e-9 for r in ref_rhos) \
               else sorted(ref_rhos + [round(rho, 2)])
    rho_frontiers = build_rho_frontiers(r1, sd1, r2, sd2, rf, rho_list=rho_list, allow_short=short)
    # Build per-frontier MVP and MSP point dicts for chart markers
    mvp_points = {}
    msp_points = {}
    for _rho, _df in rho_frontiers.items():
        mvp_points[_rho] = find_mvp(_df)
        msp_points[_rho] = find_max_sharpe(_df, long_only=not short)
    return dict(
        rho_frontiers = rho_frontiers,
        rho_mvp_df    = rho_mvp_table(r1, sd1, r2, sd2, rf, current_rho=rho, allow_short=short),
        rho_msp_df    = rho_msp_table(r1, sd1, r2, sd2, rf, current_rho=rho, allow_short=short),
        mvp_points    = mvp_points,
        msp_points    = msp_points,
        f_r1=r1, f_sd1=sd1, f_r2=r2, f_sd2=sd2, f_rho=rho, f_rf=rf,
    )


# ── N-Asset helpers ────────────────────────────────────────────────────────────

def _n_get_params():
    """Read current N-asset parameters from session state."""
    if st.session_state.get("n_csv_active", False):
        names = st.session_state.n_csv_names
        mu    = np.array(st.session_state.n_csv_mu,   dtype=float)
        sd    = np.array(st.session_state.n_csv_sd,   dtype=float)
        corr  = np.array(st.session_state.n_csv_corr, dtype=float)
    else:
        n     = int(st.session_state.n_n_assets)
        names = [st.session_state.get(f"n_name_{i}", f"Asset {chr(65+i)}")
                 for i in range(n)]
        mu    = np.array([st.session_state.get(f"n_ret_{i}", 10.0)
                          for i in range(n)], dtype=float)
        sd    = np.array([st.session_state.get(f"n_sd_{i}",  20.0)
                          for i in range(n)], dtype=float)
        corr  = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = float(st.session_state.get(f"n_corr_{i}_{j}", 0.3))
                corr[i, j] = v
                corr[j, i] = v
    cov = np.diag(sd) @ corr @ np.diag(sd)
    rf  = float(st.session_state.get("n_rf", 3.0))
    if st.session_state.get("n_include_rf", False):
        n_orig = len(names)
        new_corr = np.eye(n_orig + 1)
        new_corr[:n_orig, :n_orig] = corr
        names = names + ["Risk-Free"]
        mu   = np.append(mu, rf)
        sd   = np.append(sd, 0.0)
        corr = new_corr
        cov  = np.diag(sd) @ corr @ np.diag(sd)
    return names, mu, sd, corr, cov, rf


@st.cache_data(show_spinner="Computing N-asset efficient frontier …")
def _cached_n_frontier(mu_t, cov_t, rf, allow_short):
    """Cache-friendly wrapper around build_n_frontier (uses tuples as keys)."""
    mu  = np.array(mu_t,  dtype=float)
    cov = np.array(cov_t, dtype=float)
    df  = build_n_frontier(mu, cov, rf, allow_short=allow_short, n_points=150)
    if df.empty:
        return None, None, None
    mvp    = n_find_mvp(df)
    max_sr = n_find_max_sharpe(df)
    return df, mvp, max_sr


@st.cache_data(show_spinner="Computing correlation-effect frontiers …")
def _cached_n_kappa(mu_t, sd_t, corr_t, rf, allow_short):
    """Cache-friendly wrapper around build_n_kappa_frontiers."""
    mu   = np.array(mu_t,   dtype=float)
    sd   = np.array(sd_t,   dtype=float)
    corr = np.array(corr_t, dtype=float)
    kf   = build_n_kappa_frontiers(mu, sd, corr, rf,
                                   allow_short=allow_short, n_points=80)
    mvp_pts = {k: n_find_mvp(df) for k, df in kf.items() if not df.empty}
    return kf, mvp_pts


def _n_template_csv(n, names=None):
    """Generate a CSV template string for N assets."""
    if names is None:
        names = [f"Asset {chr(65+i)}" for i in range(n)]
    rows = []
    for i, name in enumerate(names):
        row = {"name": name, "return_pct": 10.0, "sd_pct": 20.0}
        for j, n2 in enumerate(names):
            row[f"corr_{n2}"] = 1.0 if i == j else 0.3
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


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

tab_cal, tab_two, tab_n = st.tabs([
    "📈  Capital Allocation Line",
    "📊  Two Risky Assets",
    "🧮  N-Asset Portfolio",
])


# ────────────────────────────────────────────────────────────────────────────
# OUTER TAB — CAPITAL ALLOCATION LINE
# ────────────────────────────────────────────────────────────────────────────
with tab_cal:

    # ── PARAMETER EXPANDER ───────────────────────────────────────────────────
    st.markdown("<div class='param-banner'>⚙️ Parameters — Capital Allocation Line</div>", unsafe_allow_html=True)
    with st.expander("⚙️ Parameters — Capital Allocation Line", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Risky Asset**")
            c_r_risky  = st.slider("Exp. Return (%)", 0.0, 25.0, _val("c_r_risky"),  0.5, key="c_r_risky")
            c_sd_risky = st.slider("Std. Dev. (%)",   1.0, 60.0, _val("c_sd_risky"), 1.0, key="c_sd_risky")
        with cc2:
            st.markdown("**Risk-Free Asset**")
            c_rf = st.slider("Risk-Free Rate (%)", 0.0, 10.0, _val("c_rf"), 0.5, key="c_rf",
                             help="T-Bill rate — zero std. dev. by definition.")
            st.caption("📌 Zero std. dev., zero correlation with risky asset.")

    # ── Compute ──────────────────────────────────────────────────────────────
    _c = compute_cal()
    cal_df        = _c["cal_df"]
    c_summary_tbl = _c["c_summary_tbl"]
    cal_eq_str    = _c["cal_eq_str"]
    cal_sharpe    = _c["cal_sharpe"]
    c_r_risky     = _c["c_r_risky"]
    c_sd_risky    = _c["c_sd_risky"]
    c_rf          = _c["c_rf"]

    # ── Short-selling message ────────────────────────────────────────────────
    if not allow_short:
        st.markdown(
            "<div class='info-box'>"
            "ℹ️ <b>Short-selling disabled.</b> "
            "CAL shown for long positions only (Risky Asset Weight ≥ 0%). "
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
    st.caption(
        "Metrics shown are for long allocations only (Risky Asset Weight: 0% → 200%). "
        "Short-selling region (w < 0) is visible in the 'All Allocations' chart but excluded from the metric cards."
    )

    m1, m2 = st.columns(2)
    m1.metric("Sharpe Ratio", f"{cal_sharpe:.3f}",
              help="Slope of the CAL — reward per unit of risk taken")
    m2.metric("Risk-Free Rate", f"{c_rf:.2f}%",
              help="Intercept of the CAL — Exp. Return when 100% in T-Bills, σ = 0%")

    st.metric("CAL Equation",
              f"E[R] = {c_rf:.1f}% + {cal_sharpe:.3f} × σ")

    def cal_point_card(title, ret, sd, border_color="#2E75B6"):
        return (
            f'<div style="background:#F2F7FC;border:1px solid #BDD7EE;'
            f'border-top:4px solid {border_color};border-radius:8px;'
            f'padding:14px 18px;flex:1;">'
            f'<div style="font-size:0.82rem;color:#595959;margin-bottom:6px;">{title}</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#1F4E79;font-family:monospace;'
            f'margin-bottom:4px;">{ret}</div>'
            f'<div style="font-size:0.85rem;color:#595959;">σ = {sd}</div>'
            f'</div>'
        )

    cp1 = cal_point_card("w = 0 &nbsp;(100% Risk-Free)",
                         f"{c_rf:.2f}%", "0.00%", border_color="#595959")
    cp2 = cal_point_card("w = 1 &nbsp;(100% Risky)",
                         f"{c_r_risky:.2f}%", f"{c_sd_risky:.2f}%", border_color="#2E75B6")
    cp3 = cal_point_card("w = 1.5 &nbsp;(1.5× Leverage)",
                         f"{1.5*c_r_risky - 0.5*c_rf:.2f}%", f"{1.5*c_sd_risky:.2f}%", border_color="#E8A020")
    cp4 = cal_point_card("w = 2 &nbsp;(2× Leverage)",
                         f"{2*c_r_risky - c_rf:.2f}%", f"{2*c_sd_risky:.2f}%", border_color="#C00000")

    st.markdown(
        f'<div style="display:flex;gap:16px;margin-bottom:8px;">{cp1}{cp2}{cp3}{cp4}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── CHARTS ───────────────────────────────────────────────────────────────
    st.markdown("#### Charts")

    if allow_short:
        st.plotly_chart(
            chart_cal_all(cal_df, c_r_risky, c_sd_risky, c_rf),
            use_container_width=True, key="c_all"
        )

    st.plotly_chart(
        chart_cal_all_long(cal_df, c_r_risky, c_sd_risky, c_rf),
        use_container_width=True, key="c_all_long"
    )

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
                    <span>50% Risky (w=0.5)</span>
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
                    <span>150% Risky Asset (w=1.5) — Leverage</span>
                    <span class='opt-card-value'>
                        E[R]={1.5*c_r_risky - 0.5*c_rf:.2f}%,
                        σ={1.5*c_sd_risky:.2f}%
                    </span>
                </div>
                <div class='opt-card-row'>
                    <span>200% Risky Asset (w=2) — Leverage</span>
                    <span class='opt-card-value'>
                        E[R]={2*c_r_risky - c_rf:.2f}%,
                        σ={2*c_sd_risky:.2f}%
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### CAL Summary Table")
    st.plotly_chart(
        chart_cal_summary_table(c_summary_tbl),
        use_container_width=True, key="c_summary_tbl"
    )



# ────────────────────────────────────────────────────────────────────────────
# OUTER TAB — TWO RISKY ASSETS
# ────────────────────────────────────────────────────────────────────────────
with tab_two:

    # ── SHARED PARAMETER EXPANDER ─────────────────────────────────────────────
    st.markdown("<div class='param-banner'>⚙️ Parameters — Two Risky Assets &nbsp;·&nbsp; shared across all sub-tabs</div>", unsafe_allow_html=True)
    with st.expander("⚙️ Parameters — Two Risky Assets", expanded=False):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown("**Asset 1**")
            f_r1  = st.slider("Exp. Return (%)",  0.0, 25.0, _val("f_r1"),  0.5, key="f_r1")
            f_sd1 = st.slider("Std. Dev. (%)",    5.0, 60.0, _val("f_sd1"), 1.0, key="f_sd1")
        with pc2:
            st.markdown("**Asset 2 (more volatile)**")
            f_r2    = st.slider("Exp. Return (%) ", 0.0, 30.0, _val("f_r2"), 0.5, key="f_r2")
            sd2_min = f_sd1 + 1.0
            sd2_val = max(_val("f_sd2"), sd2_min)
            if _val("f_sd2") < sd2_min:
                st.info(f"ℹ️ Asset 2 Std. Dev. adjusted to {sd2_val:.0f}% — must exceed Asset 1 ({f_sd1:.0f}%).")
            f_sd2 = st.slider("Std. Dev. (%)  ", sd2_min, 80.0, sd2_val, 1.0, key="f_sd2",
                              help="Asset 2 must always be riskier than Asset 1.")
        with pc3:
            st.markdown("**Correlation & Risk-Free Rate**")
            f_rho = st.slider("Correlation (ρ)",   -1.0, 1.0,  _val("f_rho"), 0.1, key="f_rho")
            f_rf  = st.slider("Risk-Free Rate (%)", 0.0, 10.0, _val("f_rf"),  0.5, key="f_rf",
                              help="Used for Sharpe ratio calculation only.")

    # ── Compute frontier once (shared by all sub-tabs) ────────────────────────
    _f = compute_frontier()
    frontier_df   = _f["frontier_df"]
    mvp           = _f["mvp"]
    max_sr        = _f["max_sr"]
    max_ret_lo    = _f["max_ret_lo"]
    max_ret_lev   = _f["max_ret_lev"]
    eff_summary   = _f["eff_summary"]
    benchmarks    = _f["benchmarks"]
    f_summary_tbl = _f["f_summary_tbl"]
    f_r1, f_sd1   = _f["f_r1"], _f["f_sd1"]
    f_r2, f_sd2   = _f["f_r2"], _f["f_sd2"]
    f_rho, f_rf   = _f["f_rho"], _f["f_rf"]

    # ── INNER SUB-TABS ────────────────────────────────────────────────────────
    _SUB_OPTS = ["📊  Portfolio Frontier", "🔗  Correlation Effect", "🎯  Portfolio Solver"]
    _sub_tab = st.segmented_control(
        "Section",
        _SUB_OPTS,
        default=_SUB_OPTS[0],
        key="two_risky_sub_tab",
        label_visibility="collapsed",
    )

    # ── INNER SUB-TAB 1: Portfolio Frontier ──────────────────────────────────
    if _sub_tab == _SUB_OPTS[0]:

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

        # ── METRICS ROW 1 — Benchmark portfolios ─────────────────────────────
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
        c2 = benchmark_card("100% Asset 2 (more risky)",
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

        # ── METRICS ROW 2 — Optimal portfolio cards ───────────────────────────
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

        # ── METRICS ROW 3 — Efficient frontier region ─────────────────────────
        st.markdown("#### 📊 Efficient Frontier Region")
        st.caption(
            "Long-only portfolios above the Min. Variance Portfolio "
            "(Asset 1 Weight: 0%→100%, Exp. Return ≥ MVP Exp. Return) — "
            "consistent with Prof. Weisbenner's Lesson 1-5."
        )

        if eff_summary:
            mvp_row = eff_summary['mvp_row']
            hi_row  = eff_summary['hi_ret_row']

            mvp_w_a1 = mvp_row["w_A1"] * 100
            mvp_w_a2 = mvp_row["w_A2"] * 100
            mvp_sd   = mvp_row["sd"]
            mvp_ret  = mvp_row["ret"]

            hi_w_a1  = hi_row["w_A1"] * 100
            hi_w_a2  = hi_row["w_A2"] * 100
            hi_sd    = hi_row["sd"]
            hi_ret   = hi_row["ret"]

            endpoint = f"{hi_w_a1:.0f}% A1 / {hi_w_a2:.0f}% A2"
            w_a1_dir = "increases" if hi_w_a1 > mvp_w_a1 else "decreases"
            w_a2_dir = "increases" if hi_w_a2 > mvp_w_a2 else "decreases"
            sd_dir   = "increases" if hi_sd > mvp_sd else "decreases"

            pk_a1    = eff_summary['peak_w_A1'] * 100
            pk_a2    = eff_summary['peak_w_A2'] * 100

            e1, e2, e3 = st.columns(3)
            e1.metric(
                "Asset 1 Weight Range",
                f"{mvp_w_a1:.0f}%",
                f"→ {hi_w_a1:.0f}%",
                help=(f"Asset 1 weight at MVP = {mvp_w_a1:.0f}%, "
                      f"{w_a1_dir} to {hi_w_a1:.0f}% at the high-return endpoint ({endpoint})"),
            )
            e2.metric(
                "Asset 2 Weight Range",
                f"{mvp_w_a2:.0f}%",
                f"→ {hi_w_a2:.0f}%",
                help=(f"Asset 2 weight at MVP = {mvp_w_a2:.0f}%, "
                      f"{w_a2_dir} to {hi_w_a2:.0f}% at the high-return endpoint ({endpoint})"),
            )
            e3.metric(
                "Peak Sharpe Ratio",
                f"{eff_summary['peak_sharpe']:.3f}",
                help=(f"Highest Sharpe ratio in the efficient frontier region — "
                      f"at Asset 1 Weight = {pk_a1:.0f}%, Asset 2 Weight = {pk_a2:.0f}%"),
            )

            e4, e5, e6 = st.columns(3)
            e4.metric(
                "Std. Dev. Range",
                f"{eff_summary['sd_range'][0]:.2f}%",
                f"→ {eff_summary['sd_range'][1]:.2f}%",
                help=(f"Std. Dev. at MVP = {mvp_sd:.2f}%, "
                      f"{sd_dir} to {hi_sd:.2f}% at the high-return endpoint ({endpoint})"),
            )
            e5.metric(
                "Exp. Return Range",
                f"{eff_summary['ret_range'][0]:.2f}%",
                f"→ {eff_summary['ret_range'][1]:.2f}%",
                help=(f"Exp. Return at MVP = {mvp_ret:.2f}%, "
                      f"increases to {hi_ret:.2f}% at the high-return endpoint ({endpoint})"),
            )
            e6.metric(
                "Portfolios in Region",
                f"{eff_summary['n_portfolios']}",
                help=(f"Number of portfolios with Exp. Return ≥ MVP return ({mvp_ret:.2f}%) — "
                      f"Asset 1 weight from {mvp_w_a1:.0f}% (MVP) to {hi_w_a1:.0f}% ({endpoint})"),
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── CHARTS ───────────────────────────────────────────────────────────
        st.markdown("#### Charts")

        if allow_short:
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                st.plotly_chart(
                    chart_frontier_all(frontier_df, f_r1, f_sd1, f_r2, f_sd2, mvp, max_sr, max_ret_lo, max_ret_lev, allow_short=allow_short),
                    use_container_width=True, key="f_all"
                )
            with row1_c2:
                st.plotly_chart(
                    chart_frontier_long_only(frontier_df, f_r1, f_sd1, f_r2, f_sd2, mvp, max_sr, max_ret_lo),
                    use_container_width=True, key="f_effdom"
                )
            row2_c1, row2_c2 = st.columns(2)
            with row2_c1:
                st.plotly_chart(
                    chart_frontier_short_A1(frontier_df, f_r1, f_sd1, f_r2, f_sd2, max_sr, max_ret_lo, max_ret_lev, allow_short=allow_short),
                    use_container_width=True, key="f_shortA1"
                )
            with row2_c2:
                st.plotly_chart(
                    chart_frontier_long_A1(frontier_df, f_r1, f_sd1, f_r2, f_sd2, max_sr, max_ret_lo, max_ret_lev, allow_short=allow_short),
                    use_container_width=True, key="f_longA1"
                )
        else:
            st.plotly_chart(
                chart_frontier_long_only(frontier_df, f_r1, f_sd1, f_r2, f_sd2, mvp, max_sr, max_ret_lo),
                use_container_width=True, key="f_effdom_only"
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── SUMMARY TABLE ─────────────────────────────────────────────────────
        st.markdown("#### Portfolio Summary Table")
        st.plotly_chart(
            chart_frontier_summary_table(f_summary_tbl),
            use_container_width=True, key="f_summary_tbl"
        )

    # ── INNER SUB-TAB 2: Correlation Effect ──────────────────────────────────
    elif _sub_tab == _SUB_OPTS[1]:

        _r = compute_rho()
        rho_frontiers = _r["rho_frontiers"]
        rho_mvp_df    = _r["rho_mvp_df"]
        rho_msp_df    = _r["rho_msp_df"]
        mvp_points    = _r["mvp_points"]
        msp_points    = _r["msp_points"]
        f_r1, f_sd1   = _r["f_r1"], _r["f_sd1"]
        f_r2, f_sd2   = _r["f_r2"], _r["f_sd2"]
        f_rho, f_rf   = _r["f_rho"], _r["f_rf"]

        if not allow_short:
            st.markdown(
                "<div class='info-box'>"
                "ℹ️ Correlation frontiers are displayed as long-only "
                "(Asset 1 & Asset 2 Weights: 0% → 100%). "
                "Enable short-selling in the sidebar to see how unconstrained MVP weights change across correlations."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='warn-box'>"
                "⚠️ <b>Short-selling is enabled.</b> "
                "Frontiers now extend to Asset 1 Weight: −100% → +200% | Asset 2 Weight: 200% → -100%, revealing the full diversification curve. "
                "MVP weights are unconstrained — they may fall outside [0%, 100%]"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── METRICS ──────────────────────────────────────────────────────────
        st.markdown("#### Current ρ (Correlation)")

        mvp_sd_neg1 = rho_mvp_sd(f_sd1, f_sd2, -1.0, allow_short=allow_short)
        mvp_sd_zero = rho_mvp_sd(f_sd1, f_sd2,  0.0, allow_short=allow_short)
        mvp_sd_pos1 = rho_mvp_sd(f_sd1, f_sd2,  1.0, allow_short=allow_short)
        mvp_sd_curr = rho_mvp_sd(f_sd1, f_sd2,  f_rho, allow_short=allow_short)

        st.metric("Current ρ", f"{f_rho:.1f}")

        st.markdown("#### MVP Std. Deviation across Different Correlation")
        r1c, r2c, r3c, r4c = st.columns(4)
        r1c.metric(f"MVP Std. Dev. at ρ={f_rho:.1f}", f"{mvp_sd_curr:.2f}%",
                   help="MVP std dev at the current correlation")
        r2c.metric("MVP Std. Dev. at ρ=−1", f"{mvp_sd_neg1:.2f}%",
                   help="Lowest achievable risk — perfect negative correlation")
        r3c.metric("MVP Std. Dev. at ρ=0",  f"{mvp_sd_zero:.2f}%",
                   help="Risk at zero correlation")
        r4c.metric("MVP Std. Dev. at ρ=+1", f"{mvp_sd_pos1:.2f}%",
                   help="No diversification benefit — assets move in lockstep")

        benefit  = mvp_sd_zero - mvp_sd_curr
        abs_diff = abs(benefit)
        if abs_diff < 0.005:
            benefit_label = f"σ unchanged vs ρ=0  (both {mvp_sd_curr:.2f}%)"
            benefit_help  = "Current ρ produces the same MVP std dev as ρ=0 given these asset parameters."
        elif benefit > 0:
            benefit_label = f"σ reduced by {abs_diff:.2f}%  ✅ (MVP σ: {mvp_sd_curr:.2f}% vs {mvp_sd_zero:.2f}% at ρ=0)"
            benefit_help  = f"At ρ={f_rho:.1f}, MVP std dev is {mvp_sd_curr:.2f}% — lower than {mvp_sd_zero:.2f}% at ρ=0. Current correlation improves diversification."
        else:
            benefit_label = f"⚠️ MVP σ increased by {abs_diff:.2f}% at current ρ"
            benefit_help  = f"At ρ={f_rho:.1f}, MVP std dev is {mvp_sd_curr:.2f}% — higher than {mvp_sd_zero:.2f}% at ρ=0. Current correlation reduces diversification benefit."
        st.metric(
            f"Diversification vs ρ=0 (current ρ={f_rho:.1f})",
            benefit_label,
            help=benefit_help,
        )

        st.markdown("#### Max Sharpe Ratio Portfolio at Current ρ")
        _msp_curr = next(
            (v for k, v in msp_points.items() if abs(k - f_rho) < 1e-9),
            list(msp_points.values())[-1]
        )
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric(f"MSP Asset 1 Weight at ρ={f_rho:.1f}",
                    f"{_msp_curr['w_A1'] * 100:.1f}%",
                    help="Weight in Asset 1 for the Max Sharpe Portfolio at the current correlation")
        _mc2.metric(f"MSP Std. Dev. at ρ={f_rho:.1f}",
                    f"{_msp_curr['sd']:.2f}%",
                    help="Portfolio std dev of the Max Sharpe Portfolio at the current correlation")
        _mc3.metric(f"MSP Exp. Return at ρ={f_rho:.1f}",
                    f"{_msp_curr['ret']:.2f}%",
                    help="Expected return of the Max Sharpe Portfolio at the current correlation")
        _mc4.metric(f"MSP Sharpe Ratio at ρ={f_rho:.1f}",
                    f"{_msp_curr['sharpe']:.3f}",
                    help="Sharpe ratio of the Max Sharpe Portfolio at the current correlation")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── CHARTS ───────────────────────────────────────────────────────────
        st.markdown("#### Effect of Correlation on the Efficient Frontier")
        st.plotly_chart(
            chart_rho_effect(rho_frontiers, f_rho, f_r1, f_sd1, f_r2, f_sd2,
                             mvp_points=mvp_points, msp_points=msp_points,
                             allow_short=allow_short),
            use_container_width=True, key="rho_chart"
        )
        _note1 = (
            "• Full frontier: Asset 1 Weight −100% → +200% (short-selling enabled — MVP weights unconstrained)"
            if allow_short
            else "• Long-only portfolios (Asset 1 & Asset 2 Weights: 0% → 100%)"
        )
        _note2 = (
            "• Asset markers (100% A1, 0% A2) and (0% A1, 100% A2) are ρ-invariant — "
            "all frontier curves share the same endpoints — because "
            f"σₚ = √(w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂); when w₁=0 → σₚ=σ₂, when w₂=0 → σₚ=σ₁"
        )
        _note3 = (
            "• 0-100% A1 \| 100-0% A2: Lower ρ → frontier bows further left → more diversification benefit. "
            "Reason: both w₁ and w₂ are positive, the covariance term (2w₁w₂ρσ₁σ₂) decreases with lower ρ"
        )
        _note4 = (
            "• Beyond 100% A2 (short A1, w₁ &lt; 0): correlation effect <b>reverses</b> — "
            "higher ρ reduces variance here, lower ρ increases it. "
            "Reason: w₁ &lt; 0 flips the sign of the covariance term (2w₁w₂ρσ₁σ₂)."
            if allow_short
            else ""
        )

        st.markdown(
            f"<small>{_note1}<br>"
            f"{_note2}<br>"
            f"{_note3}<br>"
            f"{_note4}</small>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown("#### MVP Std. Dev. Comparison Across Correlation")
        st.plotly_chart(
            chart_rho_mvp_table(rho_mvp_df),
            use_container_width=True, key="rho_mvp_tbl"
        )

        st.markdown("#### Max Sharpe Ratio Portfolio Comparison Across Correlation")
        st.plotly_chart(
            chart_rho_msp_table(rho_msp_df),
            use_container_width=True, key="rho_msp_tbl"
        )

    # ── INNER SUB-TAB 3: Portfolio Solver ────────────────────────────────────
    elif _sub_tab == _SUB_OPTS[2]:

        st.caption("Parameters are shared from the expander above.")
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown("#### 🎯 Solver Configuration")

        _constraint_options_lo = [
            "All Allocations",
            "Long Only",
        ]
        _constraint_options_all = [
            "All Allocations",
            "Long Only",
            "Long Asset 1 / Short Asset 2",
            "Short Asset 1 / Long Asset 2",
        ]
        _constraint_options = _constraint_options_all if allow_short else _constraint_options_lo

        if st.session_state.sol_constraint not in _constraint_options:
            st.session_state.sol_constraint = _constraint_options[0]

        _constraint_key_map = {
            "All Allocations":               "full",
            "Long Only":                     "long_only",
            "Long Asset 1 / Short Asset 2":  "long_A1",
            "Short Asset 1 / Long Asset 2":  "short_A1",
        }
        _objective_key_map = {
            "Expected Return": "ret",
            "Std. Dev.":       "sd",
            "Sharpe Ratio":    "sharpe",
        }
        _goal_key_map = {
            "Minimize":         "min",
            "Maximize":         "max",
            "Hit Target Value": "target",
        }

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sol_objective = st.radio(
                "Objective Metric",
                ["Expected Return", "Std. Dev.", "Sharpe Ratio"],
                index=["Expected Return", "Std. Dev.", "Sharpe Ratio"].index(
                    st.session_state.sol_objective
                ),
                key="sol_objective",
                help="Which portfolio metric to optimise or target.",
            )
        with sc2:
            sol_goal = st.radio(
                "Goal",
                ["Minimize", "Maximize", "Hit Target Value"],
                index=["Minimize", "Maximize", "Hit Target Value"].index(
                    st.session_state.sol_goal
                ),
                key="sol_goal",
                help="Find the minimum, maximum, or the allocation closest to a specific value.",
            )
        with sc3:
            sol_constraint = st.selectbox(
                "Constraint Region",
                _constraint_options,
                index=_constraint_options.index(st.session_state.sol_constraint),
                key="sol_constraint",
                help="Restrict the search to this region of the frontier.",
            )

        _result_display_opts = ["Both", "Efficient Region Only", "Dominated Only"]
        sol_result_display = st.radio(
            "Result Display",
            _result_display_opts,
            index=_result_display_opts.index(st.session_state.sol_result_display),
            horizontal=True,
            key="sol_result_display",
            help="Show results from the efficient region, dominated region, or both.",
        )

        sol_target = None
        if sol_goal == "Hit Target Value":
            _obj_col = _objective_key_map[sol_objective]
            _ckey    = _constraint_key_map[sol_constraint]
            _df_tmp  = frontier_df.copy()
            if _ckey == "long_only":
                _df_tmp = _df_tmp[_df_tmp["weight_region"] == "long_only"]
            elif _ckey == "long_A1":
                _df_tmp = _df_tmp[_df_tmp["weight_region"] == "long_A1"]
            elif _ckey == "short_A1":
                _df_tmp = _df_tmp[_df_tmp["weight_region"] == "short_A1"]
            if sol_result_display == "Efficient Region Only":
                _df_tmp = _df_tmp[_df_tmp["region"] == "efficient"]
            elif sol_result_display == "Dominated Only":
                _df_tmp = _df_tmp[_df_tmp["region"] == "dominated"]

            if not _df_tmp.empty:
                _v_min     = float(_df_tmp[_obj_col].min())
                _v_max     = float(_df_tmp[_obj_col].max())
                _v_default = max(_v_min, min(_v_max, float(st.session_state.sol_target)))
            else:
                _v_min, _v_max, _v_default = 0.0, 30.0, 10.0

            _unit = "%" if sol_objective in ("Expected Return", "Std. Dev.") else ""
            sol_target = st.number_input(
                f"Target {sol_objective}{' (%)' if _unit else ''}",
                min_value=round(_v_min - abs(_v_min) * 0.5, 2),
                max_value=round(_v_max + abs(_v_max) * 0.5, 2),
                value=round(_v_default, 2),
                step=0.01,
                format="%.2f",
                key="sol_target",
                help=(f"Feasible range in current constraint region: "
                      f"{_v_min:.2f}{_unit} → {_v_max:.2f}{_unit}"),
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── RUN SOLVER ────────────────────────────────────────────────────────
        _obj_key  = _objective_key_map[sol_objective]
        _goal_key = _goal_key_map[sol_goal]
        _con_key  = _constraint_key_map[sol_constraint]
        _goal_label = {
            "min":    f"Min {sol_objective}",
            "max":    f"Max {sol_objective}",
            "target": f"Target {sol_objective}",
        }[_goal_key]

        if sol_result_display == "Both":
            _solver_runs = [("efficient", "Efficient Region"), ("dominated", "Dominated Region")]
        elif sol_result_display == "Efficient Region Only":
            _solver_runs = [("efficient", "Efficient Region")]
        else:
            _solver_runs = [("dominated", "Dominated Region")]

        _solver_results = []
        for _rf, _rf_label in _solver_runs:
            _rr, _feas, _msg = solve_portfolio(
                frontier_df,
                objective     = _obj_key,
                goal          = _goal_key,
                constraint    = _con_key,
                target        = sol_target,
                result_filter = _rf,
            )
            _solver_results.append((_rr, _feas, _msg, _rf_label))

        # ── RESULT DISPLAY ────────────────────────────────────────────────────
        _result_heading = {
            "Efficient Region Only": "Result (Efficient Frontier)",
            "Dominated Only":        "Result (Denominated)",
            "Both":                  "Result",
        }[sol_result_display]
        st.markdown(f"#### {_result_heading}")

        _feasible_results = [r for r in _solver_results if r[1]]
        if not _feasible_results:
            st.error(f"⚠️ Infeasible: {_solver_results[0][2]}")
        else:
            _res_cols = st.columns(len(_feasible_results))
            for _ci, (_rr, _feas, _msg, _rf_label) in enumerate(_feasible_results):
                with _res_cols[_ci]:
                    st.markdown(
                        f"<div class='opt-card'>"
                        f"<div class='opt-card-title'>🎯 {_goal_label} — {_rf_label}</div>"
                        f"<div class='opt-card-row'><span>Asset 1 Weight</span>"
                        f"<span class='opt-card-value'>{_rr['w_A1']*100:.1f}%</span></div>"
                        f"<div class='opt-card-row'><span>Asset 2 Weight</span>"
                        f"<span class='opt-card-value'>{_rr['w_A2']*100:.1f}%</span></div>"
                        f"<div class='opt-card-row'><span>Exp. Return</span>"
                        f"<span class='opt-card-value'>{_rr['ret']:.2f}%</span></div>"
                        f"<div class='opt-card-row'><span>Std. Dev.</span>"
                        f"<span class='opt-card-value'>{_rr['sd']:.2f}%</span></div>"
                        f"<div class='opt-card-row'><span>Sharpe Ratio</span>"
                        f"<span class='opt-card-value'>{_rr['sharpe']:.3f}</span></div>"
                        f"<div style='margin-top:8px;font-size:0.78rem;color:#595959'>{_msg}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            _chart_rows = [
                (_rr, f"{_goal_label} — {_rl}")
                for _rr, _feas, _, _rl in _solver_results
                if _feas and _rr is not None
            ]
            st.markdown("#### Frontier Chart — Solver Result")
            st.plotly_chart(
                chart_frontier_with_solver(
                    frontier_df, _chart_rows, _con_key, mvp,
                    allow_short=allow_short,
                ),
                use_container_width=True, key="solver_chart",
            )


# ────────────────────────────────────────────────────────────────────────────
# OUTER TAB — N-ASSET PORTFOLIO
# ────────────────────────────────────────────────────────────────────────────
with tab_n:

    _N_SUB_OPTS = [
        "📋  Assets",
        "📉  Efficient Frontier",
        "🎯  Solver",
        "🔗  Correlation Effect",
    ]
    _n_sub = st.segmented_control(
        "N-Asset Section",
        _N_SUB_OPTS,
        default=_N_SUB_OPTS[0],
        key="n_sub_tab",
        label_visibility="collapsed",
    )

    # ════════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — ASSETS
    # ════════════════════════════════════════════════════════════════════════
    if _n_sub == _N_SUB_OPTS[0]:

        # ── CSV upload (always shown at top) ─────────────────────────────
        st.markdown("<div class='param-banner'>📂 CSV Upload (optional — supports any number of assets)</div>",
                    unsafe_allow_html=True)
        with st.expander("Upload Asset Parameters CSV", expanded=False):
            _csv_c1, _csv_c2 = st.columns([3, 1])
            with _csv_c1:
                _uploaded = st.file_uploader(
                    "CSV with columns: name, return_pct, sd_pct, corr_<name1>, corr_<name2>, …",
                    type=["csv"], key="n_csv_upload",
                )
            with _csv_c2:
                st.markdown("**Template**")
                _n_cur = int(st.session_state.n_n_assets)
                _tpl   = _n_template_csv(_n_cur)
                st.download_button(
                    "⬇ Download Template",
                    data=_tpl,
                    file_name="n_asset_template.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            if _uploaded is not None:
                import io
                _raw_df = pd.read_csv(io.StringIO(_uploaded.getvalue().decode("utf-8")))
                _csv_names, _csv_mu, _csv_sd, _csv_corr, _csv_errs = parse_n_csv(_raw_df)
                if _csv_errs and _csv_names is None:
                    for _e in _csv_errs:
                        st.error(_e)
                else:
                    if _csv_errs:
                        for _e in _csv_errs:
                            st.warning(_e)
                    st.session_state.n_csv_names  = _csv_names
                    st.session_state.n_csv_mu     = _csv_mu.tolist()
                    st.session_state.n_csv_sd     = _csv_sd.tolist()
                    st.session_state.n_csv_corr   = _csv_corr.tolist()
                    st.session_state.n_csv_active = True
                    st.success(f"✅ Loaded {len(_csv_names)} assets from CSV.")

            if st.session_state.get("n_csv_active", False):
                if st.button("✖ Clear CSV — switch to manual entry"):
                    st.session_state.n_csv_active = False
                    st.rerun()

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Manual entry ──────────────────────────────────────────────────
        if st.session_state.get("n_csv_active", False):
            _csv_names_disp = st.session_state.n_csv_names
            st.markdown(
                f"<div class='info-box'>"
                f"📂 <b>CSV mode active — {len(_csv_names_disp)} assets loaded.</b> "
                "Switch to manual entry using the button above."
                "</div>",
                unsafe_allow_html=True,
            )
            _nc, _rf_col = st.columns([3, 1])
            with _rf_col:
                st.number_input("Risk-Free Rate (%)", 0.0, 15.0,
                                float(st.session_state.n_rf), 0.5, key="n_rf")
                st.checkbox(
                    "Risk Free in Portfolio?",
                    value=st.session_state.get("n_include_rf", False),
                    key="n_include_rf",
                    help="Adds the risk-free asset (return = RF rate, std. dev. = 0%, zero correlation with all risky assets) to the portfolio optimization.",
                )
            # Preview
            _prev_df = pd.DataFrame({
                "Asset":          _csv_names_disp,
                "Exp. Return (%)": st.session_state.n_csv_mu,
                "Std. Dev. (%)":   st.session_state.n_csv_sd,
            })
            st.dataframe(_prev_df, use_container_width=True, hide_index=True)
            st.plotly_chart(
                chart_n_heatmap(st.session_state.n_csv_corr, _csv_names_disp),
                use_container_width=True, key="n_corr_heatmap_csv",
            )

        else:
            # Apply pending asset-count update BEFORE the slider renders
            if "_n_n_assets_next" in st.session_state:
                st.session_state["n_n_assets"] = st.session_state.pop("_n_n_assets_next")

            # Number of assets + RF rate
            _na_c1, _na_c2 = st.columns([3, 1])
            with _na_c1:
                _n_assets = st.slider(
                    "Number of Assets", 2, 8,
                    int(st.session_state.n_n_assets), 1,
                    key="n_n_assets",
                )
            with _na_c2:
                st.number_input(
                    "Risk-Free Rate (%)", 0.0, 15.0,
                    float(st.session_state.n_rf), 0.5, key="n_rf",
                )
                st.checkbox(
                    "Risk Free in Portfolio?",
                    value=st.session_state.get("n_include_rf", False),
                    key="n_include_rf",
                    help="Adds the risk-free asset (return = RF rate, std. dev. = 0%, zero correlation with all risky assets) to the portfolio optimization.",
                )

            _n = int(st.session_state.n_n_assets)

            # Asset parameter inputs (up to 4 columns per row)
            st.markdown("#### Asset Parameters")
            _CPR   = min(_n, 4)
            _acols = st.columns(_CPR)
            for _i in range(_n):
                with _acols[_i % _CPR]:
                    _letter = chr(65 + _i)
                    st.checkbox(
                        f"Delete asset {_letter}",
                        value=False,
                        key=f"n_del_{_i}",
                    )
                    st.text_input(
                        f"Name {_letter}",
                        value=st.session_state.get(f"n_name_{_i}", f"Asset {_letter}"),
                        key=f"n_name_{_i}",
                    )
                    st.number_input(
                        f"Return (%) {_letter}", 0.0, 50.0,
                        float(st.session_state.get(f"n_ret_{_i}", 10.0)),
                        0.5, key=f"n_ret_{_i}",
                    )
                    st.number_input(
                        f"Std. Dev. (%) {_letter}", 0.0, 80.0,
                        float(st.session_state.get(f"n_sd_{_i}", 20.0)),
                        1.0, key=f"n_sd_{_i}",
                    )

            # --- Delete checked assets ---
            _del_indices = [_i for _i in range(_n) if st.session_state.get(f"n_del_{_i}", False)]
            _keep_count  = _n - len(_del_indices)
            _del_disabled = _keep_count < 2
            if st.button(
                f"Remove {len(_del_indices)} checked asset(s)",
                disabled=(_del_indices == [] or _del_disabled),
                help="At least 2 assets must remain." if _del_disabled else None,
            ):
                _keep = [_i for _i in range(_n) if _i not in _del_indices]
                # Snapshot values for kept assets before modifying session state
                _snap = {
                    _new: {
                        "name": st.session_state.get(f"n_name_{_old}", f"Asset {chr(65+_old)}"),
                        "ret":  float(st.session_state.get(f"n_ret_{_old}", 10.0)),
                        "sd":   float(st.session_state.get(f"n_sd_{_old}", 20.0)),
                    }
                    for _new, _old in enumerate(_keep)
                }
                _corr_snap = {}
                for _ni, _oi in enumerate(_keep):
                    for _nj, _oj in enumerate(_keep):
                        if _ni < _nj:
                            _lo, _hi = min(_oi, _oj), max(_oi, _oj)
                            _corr_snap[(_ni, _nj)] = float(
                                st.session_state.get(f"n_corr_{_lo}_{_hi}", 0.3)
                            )
                # Clear ALL old asset keys up to _n
                for _i in range(_n):
                    for _k in (f"n_name_{_i}", f"n_ret_{_i}", f"n_sd_{_i}", f"n_del_{_i}"):
                        st.session_state.pop(_k, None)
                for _i in range(_n):
                    for _j in range(_i + 1, _n):
                        st.session_state.pop(f"n_corr_{_i}_{_j}", None)
                # Write compacted values
                for _new, _vals in _snap.items():
                    st.session_state[f"n_name_{_new}"] = _vals["name"]
                    st.session_state[f"n_ret_{_new}"]  = _vals["ret"]
                    st.session_state[f"n_sd_{_new}"]   = _vals["sd"]
                for (_ni, _nj), _cv in _corr_snap.items():
                    st.session_state[f"n_corr_{_ni}_{_nj}"] = _cv
                st.session_state["_n_n_assets_next"] = len(_keep)
                st.rerun()

            # Correlation matrix editor
            st.markdown("#### Correlation Matrix")
            st.caption(
                "Edit upper-triangle correlations (above the diagonal) — "
                "the lower triangle mirrors each value automatically on commit. "
                "Diagonal cells are locked at 1.0 (an asset is always perfectly "
                "correlated with itself)."
            )

            _cur_names = [
                st.session_state.get(f"n_name_{i}", f"Asset {chr(65+i)}")
                for i in range(_n)
            ]

            # Build matrix DataFrame from session state
            _corr_init = {}
            for _j in range(_n):
                _col_data = []
                for _i in range(_n):
                    if _i == _j:
                        _col_data.append(1.0)
                    elif _i < _j:
                        _col_data.append(
                            float(st.session_state.get(f"n_corr_{_i}_{_j}", 0.3))
                        )
                    else:
                        _col_data.append(
                            float(st.session_state.get(f"n_corr_{_j}_{_i}", 0.3))
                        )
                _corr_init[_cur_names[_j]] = _col_data

            _corr_df_in = pd.DataFrame(_corr_init, index=_cur_names)

            # AgGrid: row-label column + correlation columns
            _ag_df = _corr_df_in.reset_index().rename(columns={"index": " "})

            _gb = GridOptionsBuilder.from_dataframe(_ag_df)
            _gb.configure_column(
                " ", editable=False, pinned="left", width=110,
                cellStyle={"backgroundColor": "#EEEEEE", "fontWeight": "bold"},
            )
            # JS array of asset names used across multiple JsCode snippets
            _col_names_js = "[" + ", ".join(f'"{n}"' for n in _cur_names) + "]"

            for _j, _col in enumerate(_cur_names):
                # Only cells strictly above the diagonal are editable
                _edit_js = JsCode(
                    f"function(p) {{ return p.node.rowIndex < {_j}; }}"
                )
                _style_js = JsCode(
                    f"function(p) {{"
                    f"  var r = p.node.rowIndex;"
                    f"  if (r === {_j}) return {{'backgroundColor':'#DEDEDE','color':'#444444','fontWeight':'bold'}};"
                    f"  if (r  >  {_j}) return {{'backgroundColor':'#F4F4F4','color':'#888888'}};"
                    f"  return {{'cursor':'pointer'}};"
                    f"}}"
                )
                _parser_js = JsCode(
                    "function(p) {"
                    " var v = parseFloat(p.newValue);"
                    " if (isNaN(v)) return p.oldValue;"
                    " return Math.max(-1, Math.min(1, Math.round(v * 1000) / 1000));"
                    "}"
                )
                _fmt_js = JsCode(
                    "function(p) { return (p.value != null) ? p.value.toFixed(2) : ''; }"
                )
                # valueGetter: diagonal → 1.0; lower triangle → live mirror of
                # the corresponding upper-triangle cell; upper triangle → stored value.
                _getter_js = JsCode(
                    f"function(p) {{"
                    f"  var r = p.node.rowIndex;"
                    f"  var cols = {_col_names_js};"
                    f"  if (r === {_j}) return 1.0;"
                    f"  if (r < {_j}) return p.data[cols[{_j}]];"
                    f"  var mirror = p.api.getDisplayedRowAtIndex({_j});"
                    f"  return mirror ? mirror.data[cols[r]] : p.data[cols[{_j}]];"
                    f"}}"
                )
                _gb.configure_column(
                    _col,
                    editable=_edit_js,
                    cellStyle=_style_js,
                    valueGetter=_getter_js,
                    valueParser=_parser_js,
                    valueFormatter=_fmt_js,
                    type=["numericColumn"],
                    width=80,
                )

            # After editing an upper-triangle cell, force-refresh the mirror
            # lower-triangle cell so its valueGetter re-evaluates immediately.
            _on_change_js = JsCode(
                f"function(params) {{"
                f"  var cols = {_col_names_js};"
                f"  var j = cols.indexOf(params.column.getColDef().field);"
                f"  var r = params.node.rowIndex;"
                f"  if (j < 0 || r >= j) return;"
                f"  var mirrorNode = params.api.getDisplayedRowAtIndex(j);"
                f"  if (mirrorNode) {{"
                f"    params.api.refreshCells({{ rowNodes: [mirrorNode], columns: [cols[r]], force: true }});"
                f"  }}"
                f"}}"
            )
            _gb.configure_grid_options(
                stopEditingWhenCellsLoseFocus=True,
                singleClickEdit=True,
                onCellValueChanged=_on_change_js,
            )

            _ag_resp = AgGrid(
                _ag_df,
                gridOptions=_gb.build(),
                update_mode=GridUpdateMode.VALUE_CHANGED,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                height=max(200, (_n + 2) * 42),
                key=f"n_corr_aggrid_{_n}",
            )

            # Read upper triangle only; build symmetric matrix
            _ret = _ag_resp["data"][_cur_names].values.astype(float)
            _arr = np.zeros((_n, _n))
            for _i in range(_n):
                for _j in range(_i + 1, _n):
                    _v = float(np.clip(_ret[_i, _j], -1.0, 1.0))
                    _arr[_i, _j] = _v
                    _arr[_j, _i] = _v
            np.fill_diagonal(_arr, 1.0)

            # Persist off-diagonal back to session state
            for _i in range(_n):
                for _j in range(_i + 1, _n):
                    st.session_state[f"n_corr_{_i}_{_j}"] = round(
                        float(_arr[_i, _j]), 3
                    )

            _is_valid, _corr_errs = validate_corr_matrix(_arr)
            if not _is_valid:
                for _e in _corr_errs:
                    st.warning(f"⚠️ {_e}")
                _arr = nearest_psd_corr(_arr)
                st.info(
                    "ℹ️ Correlation matrix adjusted to the nearest valid "
                    "positive semi-definite matrix."
                )

            # Live heatmap preview
            st.plotly_chart(
                chart_n_heatmap(_arr, _cur_names),
                use_container_width=True, key="n_corr_heatmap_manual",
            )

    # ════════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — EFFICIENT FRONTIER
    # ════════════════════════════════════════════════════════════════════════
    elif _n_sub == _N_SUB_OPTS[1]:

        _n_names, _n_mu, _n_sd, _n_corr, _n_cov, _n_rf = _n_get_params()
        _short = st.session_state.allow_short

        # Cache keys (convert arrays to tuples for hashing)
        _mu_t  = tuple(_n_mu.tolist())
        _cov_t = tuple(map(tuple, _n_cov.tolist()))

        _n_frontier_df, _n_mvp, _n_max_sr = _cached_n_frontier(
            _mu_t, _cov_t, _n_rf, _short
        )

        if _n_frontier_df is None or _n_frontier_df.empty:
            st.error(
                "❌ Could not compute the efficient frontier for the current parameters. "
                "Try adjusting asset parameters or the correlation matrix."
            )
        else:
            # ── Metric cards ──────────────────────────────────────────────
            st.markdown("#### Key Portfolio Statistics")
            _mk_cols = st.columns(3)
            with _mk_cols[0]:
                st.metric("MVP — Std. Dev.", f"{_n_mvp['sd']:.2f}%")
                st.metric("MVP — Exp. Return", f"{_n_mvp['ret']:.2f}%")
                st.metric("MVP — Sharpe", f"{_n_mvp['sharpe']:.3f}")
            with _mk_cols[1]:
                st.metric("Max Sharpe — Std. Dev.", f"{_n_max_sr['sd']:.2f}%")
                st.metric("Max Sharpe — Exp. Return", f"{_n_max_sr['ret']:.2f}%")
                st.metric("Max Sharpe — Sharpe", f"{_n_max_sr['sharpe']:.3f}")
            with _mk_cols[2]:
                st.metric("Assets", str(len(_n_names)))
                st.metric("Risk-Free Rate", f"{_n_rf:.1f}%")
                st.metric("Short-Selling",
                          "Enabled" if _short else "Disabled")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Frontier chart ────────────────────────────────────────────
            st.markdown("#### Efficient Frontier")
            st.plotly_chart(
                chart_n_frontier(_n_frontier_df, _n_names, _n_mvp, _n_max_sr, _n_mu, _n_sd, rf=_n_rf),
                use_container_width=True, key="n_frontier_chart",
            )

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Weight allocation bar chart ───────────────────────────────
            st.markdown("#### Weight Allocations — Key Portfolios")
            _port_list = [("MVP", _n_mvp), ("Max Sharpe", _n_max_sr)]
            st.plotly_chart(
                chart_n_weights_bar(_port_list, _n_names),
                use_container_width=True, key="n_weights_bar",
            )

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Summary table ─────────────────────────────────────────────
            st.markdown("#### Summary Table")
            st.plotly_chart(
                chart_n_summary_table(_n_frontier_df, _n_names, _n_mvp, _n_max_sr),
                use_container_width=True, key="n_summary_table",
            )

    # ════════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — SOLVER
    # ════════════════════════════════════════════════════════════════════════
    elif _n_sub == _N_SUB_OPTS[2]:

        _n_names, _n_mu, _n_sd, _n_corr, _n_cov, _n_rf = _n_get_params()
        _short = st.session_state.allow_short
        _mu_t  = tuple(_n_mu.tolist())
        _cov_t = tuple(map(tuple, _n_cov.tolist()))

        _n_frontier_df, _n_mvp, _n_max_sr = _cached_n_frontier(
            _mu_t, _cov_t, _n_rf, _short
        )

        if _n_frontier_df is None or _n_frontier_df.empty:
            st.error("❌ No frontier available. Check the Assets tab.")
        else:
            # ── Solver controls ───────────────────────────────────────────
            st.markdown("<div class='param-banner'>🎯 Solver Settings</div>",
                        unsafe_allow_html=True)
            with st.expander("🎯 Solver Settings", expanded=True):
                _sc1, _sc2, _sc3 = st.columns(3)
                with _sc1:
                    _sol_obj = st.selectbox(
                        "Objective",
                        ["Expected Return", "Std. Dev.", "Sharpe Ratio"],
                        index=["Expected Return", "Std. Dev.", "Sharpe Ratio"].index(
                            st.session_state.n_sol_objective
                        ),
                        key="n_sol_objective",
                    )
                with _sc2:
                    _sol_goal = st.selectbox(
                        "Goal",
                        ["Maximize", "Minimize", "Target"],
                        index=["Maximize", "Minimize", "Target"].index(
                            st.session_state.n_sol_goal
                        ),
                        key="n_sol_goal",
                    )
                with _sc3:
                    _sol_target = st.number_input(
                        "Target Value",
                        value=float(st.session_state.n_sol_target),
                        step=0.5,
                        key="n_sol_target",
                        disabled=(st.session_state.n_sol_goal != "Target"),
                    )

                _sol_eff_only = st.checkbox(
                    "Restrict search to efficient portfolios only",
                    value=bool(st.session_state.n_sol_efficient_only),
                    key="n_sol_efficient_only",
                )

            # Map UI labels to internal keys
            _obj_map  = {
                "Expected Return": "ret",
                "Std. Dev.":       "sd",
                "Sharpe Ratio":    "sharpe",
            }
            _goal_map = {"Maximize": "max", "Minimize": "min", "Target": "target"}
            _obj_key  = _obj_map[st.session_state.n_sol_objective]
            _goal_key = _goal_map[st.session_state.n_sol_goal]
            _target   = (float(st.session_state.n_sol_target)
                         if _goal_key == "target" else None)

            _sol_row, _feasible, _sol_msg = n_solve_portfolio(
                _n_frontier_df,
                objective=_obj_key,
                goal=_goal_key,
                target=_target,
                efficient_only=bool(st.session_state.n_sol_efficient_only),
            )

            # ── Result ────────────────────────────────────────────────────
            if _feasible and _sol_row is not None:
                st.success(_sol_msg)
                _r_cols = st.columns(3)
                with _r_cols[0]:
                    st.metric("Exp. Return", f"{_sol_row['ret']:.2f}%")
                with _r_cols[1]:
                    st.metric("Std. Dev.", f"{_sol_row['sd']:.2f}%")
                with _r_cols[2]:
                    st.metric("Sharpe Ratio", f"{_sol_row['sharpe']:.3f}")

                # Weight cards
                st.markdown("**Optimal Weights**")
                _wc = st.columns(len(_n_names))
                for _i, _nm in enumerate(_n_names):
                    with _wc[_i]:
                        _wval = _sol_row[f"w_{_i+1}"] * 100
                        st.metric(_nm, f"{_wval:.1f}%")

                # Weight bar chart
                st.plotly_chart(
                    chart_n_weights_bar([("Optimal", _sol_row)], _n_names),
                    use_container_width=True, key="n_solver_weights",
                )
            else:
                st.warning(_sol_msg)
                _sol_row = None

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Frontier chart with result ─────────────────────────────────
            st.markdown("#### Frontier Chart — Solver Result")
            st.plotly_chart(
                chart_n_solver(_n_frontier_df, _sol_row, _n_mvp, _n_names),
                use_container_width=True, key="n_solver_chart",
            )

    # ════════════════════════════════════════════════════════════════════════
    # SUB-TAB 4 — CORRELATION EFFECT
    # ════════════════════════════════════════════════════════════════════════
    elif _n_sub == _N_SUB_OPTS[3]:

        _n_names, _n_mu, _n_sd, _n_corr, _n_cov, _n_rf = _n_get_params()
        _short = st.session_state.allow_short

        # ── Correlation heatmap ───────────────────────────────────────────
        st.markdown("#### Current Correlation Matrix")
        st.plotly_chart(
            chart_n_heatmap(_n_corr, _n_names),
            use_container_width=True, key="n_kappa_heatmap",
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Kappa slider ──────────────────────────────────────────────────
        st.markdown("#### Correlation Scalar (κ) Effect")
        st.markdown(
            "<div class='info-box'>"
            "ℹ️ <b>κ scales all pairwise correlations.</b> "
            "At κ = 0 all assets are uncorrelated (diagonal covariance). "
            "At κ = 1 the full correlation matrix above is used. "
            "Watch how diversification benefits shrink as κ increases."
            "</div>",
            unsafe_allow_html=True,
        )
        _kappa = st.slider(
            "Correlation Scalar κ",
            0.0, 1.0,
            float(st.session_state.n_kappa),
            0.05,
            key="n_kappa",
            help="κ = 0 → uncorrelated  |  κ = 1 → full correlation",
        )

        # Cache keys
        _mu_t   = tuple(_n_mu.tolist())
        _sd_t   = tuple(_n_sd.tolist())
        _corr_t = tuple(map(tuple, _n_corr.tolist()))

        _kappa_frs, _kappa_mvps = _cached_n_kappa(
            _mu_t, _sd_t, _corr_t, _n_rf, _short
        )

        # Diversification benefit metrics
        _mvp_full = _kappa_mvps.get(1.0)
        _mvp_zero = _kappa_mvps.get(0.0)
        if _mvp_full is not None and _mvp_zero is not None:
            _div_benefit = _mvp_zero["sd"] - _mvp_full["sd"]
            _div_pct     = (
                (_mvp_zero["sd"] - _mvp_full["sd"]) / _mvp_zero["sd"] * 100
                if _mvp_zero["sd"] > 0 else 0
            )
            _mb_cols = st.columns(3)
            with _mb_cols[0]:
                st.metric("MVP Std. Dev. (κ=0, uncorrelated)",
                          f"{_mvp_zero['sd']:.2f}%")
            with _mb_cols[1]:
                st.metric("MVP Std. Dev. (κ=1, full correlation)",
                          f"{_mvp_full['sd']:.2f}%")
            with _mb_cols[2]:
                st.metric(
                    "Diversification Benefit",
                    f"{_div_benefit:.2f}%",
                    delta=f"{_div_pct:.1f}% reduction",
                    delta_color="inverse",
                )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Overlaid frontiers chart ──────────────────────────────────────
        st.markdown("#### Efficient Frontiers Across κ Values")
        st.plotly_chart(
            chart_n_kappa_effect(_kappa_frs, _kappa, _kappa_mvps),
            use_container_width=True, key="n_kappa_chart",
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── MVP comparison table ──────────────────────────────────────────
        st.markdown("#### MVP Comparison Across κ Values")
        st.plotly_chart(
            chart_n_kappa_mvp_table(_kappa_frs, _kappa_mvps, _kappa),
            use_container_width=True, key="n_kappa_mvp_table",
        )
