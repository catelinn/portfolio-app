"""
calculations.py
---------------
Pure portfolio math functions — no UI, no Streamlit, no Plotly.
All functions take parameters and return DataFrames or dicts.

FIN 511 - Investments I: Module 1, Lesson 1-5
"""

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────
W_MIN  = -1.0   # minimum weight scan (short-selling floor)
W_MAX  =  2.0   # maximum weight scan (leverage ceiling)
W_STEP =  0.01  # step size for brute-force scan


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORE MATH
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_stats(w, r1, sd1, r2, sd2, rho, rf):
    """
    Calculate expected return, std deviation, and Sharpe ratio
    for a portfolio with weight w in Asset 1 and (1-w) in Asset 2.

    Parameters
    ----------
    w   : float  weight in Asset 1 (e.g. 0.94 = 94%)
    r1  : float  Asset 1 expected return (%)
    sd1 : float  Asset 1 std deviation (%)
    r2  : float  Asset 2 expected return (%)
    sd2 : float  Asset 2 std deviation (%)
    rho : float  correlation between Asset 1 and Asset 2 (-1 to +1)
    rf  : float  risk-free rate (%) — used for Sharpe ratio only

    Returns
    -------
    (ret, sd, sharpe) : tuple of floats
    """
    ret      = w * r1 + (1 - w) * r2
    variance = (w**2 * sd1**2
                + (1 - w)**2 * sd2**2
                + 2 * w * (1 - w) * rho * sd1 * sd2)
    sd       = np.sqrt(max(variance, 0.0))
    sharpe   = (ret - rf) / sd if sd > 1e-10 else 0.0
    return ret, sd, sharpe


def cal_stats(w, r_risky, sd_risky, rf):
    """
    Calculate expected return, std deviation, and Sharpe ratio
    for a Capital Allocation Line portfolio.

    Risk-free asset has zero variance and zero correlation with
    the risky asset — so variance formula simplifies to:
        σp = |w| × σ_risky

    Parameters
    ----------
    w        : float  weight in risky asset (>1 = leverage, <0 = short)
    r_risky  : float  risky asset expected return (%)
    sd_risky : float  risky asset std deviation (%)
    rf       : float  risk-free rate (%)

    Returns
    -------
    (ret, sd, sharpe) : tuple of floats
    """
    ret    = w * r_risky + (1 - w) * rf
    sd     = abs(w) * sd_risky
    sharpe = (r_risky - rf) / sd_risky if sd_risky > 1e-10 else 0.0
    return ret, sd, sharpe


def mvp_weight(sd1, sd2, rho):
    """
    Analytical formula for the weight in Asset 1 that minimises
    portfolio variance (Minimum Variance Portfolio).

        w* = (σ₂² - ρσ₁σ₂) / (σ₁² + σ₂² - 2ρσ₁σ₂)

    Parameters
    ----------
    sd1 : float  Asset 1 std deviation (%)
    sd2 : float  Asset 2 std deviation (%)
    rho : float  correlation

    Returns
    -------
    w_star : float  optimal weight in Asset 1
    """
    numerator   = sd2**2 - rho * sd1 * sd2
    denominator = sd1**2 + sd2**2 - 2 * rho * sd1 * sd2
    if abs(denominator) < 1e-10:
        return 0.5  # fallback: equal weight
    return numerator / denominator


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATAFRAME BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_frontier(r1, sd1, r2, sd2, rho, rf,
                   w_min=W_MIN, w_max=W_MAX, w_step=W_STEP):
    """
    Build the full portfolio frontier by scanning all weights
    from w_min to w_max in steps of w_step.

    Assigns each row a 'region' label:
        - 'efficient'  : long-only (0 ≤ w ≤ 1), ret ≥ MVP ret
        - 'dominated'  : long-only (0 ≤ w ≤ 1), ret < MVP ret
        - 'short_A1'   : w < 0  (short Asset 1, long Asset 2)
        - 'long_A1'    : w > 1  (long Asset 1, short Asset 2)

    Parameters
    ----------
    r1, sd1 : float  Asset 1 expected return and std deviation (%)
    r2, sd2 : float  Asset 2 expected return and std deviation (%)
    rho     : float  correlation
    rf      : float  risk-free rate (%) — for Sharpe ratio
    w_min, w_max, w_step : float  weight scan range and step

    Returns
    -------
    df : pd.DataFrame
        Columns: w_A1, w_A2, ret, sd, sharpe, region
    """
    weights = np.arange(w_min, w_max + w_step / 2, w_step)
    rows    = []

    for w in weights:
        ret, sd, sharpe = portfolio_stats(w, r1, sd1, r2, sd2, rho, rf)
        rows.append({
            "w_A1":   round(w, 4),
            "w_A2":   round(1 - w, 4),
            "ret":    round(ret, 4),
            "sd":     round(sd, 4),
            "sharpe": round(sharpe, 4),
        })

    df = pd.DataFrame(rows)

    # Find MVP return threshold for region labelling
    mvp_idx = df["sd"].idxmin()
    mvp_ret = df.loc[mvp_idx, "ret"]

    # Assign region — efficient/dominated based on return vs MVP, for ALL weights.
    # weight_region handles the short_A1/long_only/long_A1 distinction separately.
    def assign_region(row):
        if row["ret"] >= mvp_ret:
            return "efficient"
        else:
            return "dominated"

    df["region"] = df.apply(assign_region, axis=1)

    # Assign weight_region for filtering in efficient_frontier_region
    def assign_weight_region(row):
        w = row["w_A1"]
        if w < 0:
            return "short_A1"
        elif w > 1:
            return "long_A1"
        else:
            return "long_only"

    df["weight_region"] = df.apply(assign_weight_region, axis=1)

    # chart_region: strict marker ownership — boundaries (w=0, w=1) belong to chart2 only.
    # chart3 = strictly w < 0; chart4 = strictly w > 1.
    def assign_chart_region(row):
        w = row["w_A1"]
        if w < 0:
            return "chart3"
        elif w > 1:
            return "chart4"
        else:
            return "chart2"

    df["chart_region"] = df.apply(assign_chart_region, axis=1)
    return df


def build_cal(r_risky, sd_risky, rf,
              w_min=W_MIN, w_max=W_MAX, w_step=W_STEP):
    """
    Build the Capital Allocation Line by scanning weights
    from w_min to w_max.

    Assigns each row a 'region' label:
        - 'short'       : w < 0  (short risky, over-invest in RF)
        - 'long_no_lev' : 0 ≤ w ≤ 1
        - 'long_lev'    : w > 1  (leverage — borrow at rf)

    Parameters
    ----------
    r_risky  : float  risky asset expected return (%)
    sd_risky : float  risky asset std deviation (%)
    rf       : float  risk-free rate (%)

    Returns
    -------
    df : pd.DataFrame
        Columns: w_risky, w_rf, ret, sd, sharpe, region
    """
    weights = np.arange(w_min, w_max + w_step / 2, w_step)
    rows    = []

    for w in weights:
        ret, sd, sharpe = cal_stats(w, r_risky, sd_risky, rf)
        if w < 0:
            region = "short"
        elif w <= 1:
            region = "long_no_lev"
        else:
            region = "long_lev"

        rows.append({
            "w_risky": round(w, 4),
            "w_rf":    round(1 - w, 4),
            "ret":     round(ret, 4),
            "sd":      round(sd, 4),
            "sharpe":  round(sharpe, 4),
            "region":  region,
        })

    return pd.DataFrame(rows)


def build_rho_frontiers(r1, sd1, r2, sd2, rf,
                        rho_list=None, w_step=0.02):
    """
    Build long-only frontiers for multiple correlation values.
    Used in Tab 3 — Correlation Effect.

    Always long-only (w: 0 → 1) per Prof. Weisbenner's lecture.

    Parameters
    ----------
    r1, sd1 : float  Asset 1 params
    r2, sd2 : float  Asset 2 params
    rf      : float  risk-free rate
    rho_list: list   correlations to compute (default: [-0.8,-0.4,0,0.4,0.8])
    w_step  : float  step size (coarser = faster for comparison chart)

    Returns
    -------
    dict : { rho_value : pd.DataFrame }
        Each DataFrame has columns: w_A1, w_A2, ret, sd, sharpe
    """
    if rho_list is None:
        rho_list = [-0.8, -0.4, 0.0, 0.4, 0.8]

    weights = np.arange(0.0, 1.0 + w_step / 2, w_step)
    result  = {}

    for rho in rho_list:
        rows = []
        for w in weights:
            ret, sd, sharpe = portfolio_stats(w, r1, sd1, r2, sd2, rho, rf)
            rows.append({
                "w_A1":   round(w, 4),
                "w_A2":   round(1 - w, 4),
                "ret":    round(ret, 4),
                "sd":     round(sd, 4),
                "sharpe": round(sharpe, 4),
            })
        result[rho] = pd.DataFrame(rows)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. KEY PORTFOLIO FINDERS
# ══════════════════════════════════════════════════════════════════════════════

def find_mvp(frontier_df):
    """
    Find the Minimum Variance Portfolio — row with lowest Std. Dev.

    Parameters
    ----------
    frontier_df : pd.DataFrame  output of build_frontier()

    Returns
    -------
    pd.Series  — single row from frontier_df
    """
    idx = frontier_df["sd"].idxmin()
    return frontier_df.loc[idx]


def find_max_sharpe(frontier_df, long_only=True):
    """
    Find the portfolio with the highest Sharpe Ratio.

    Parameters
    ----------
    frontier_df : pd.DataFrame  output of build_frontier()
    long_only   : bool  if True, restrict to w_A1 in [0, 1]

    Returns
    -------
    pd.Series  — single row from frontier_df
    """
    df = frontier_df.copy()
    if long_only:
        df = df[(df["w_A1"] >= 0) & (df["w_A1"] <= 1)]
    idx = df["sharpe"].idxmax()
    return frontier_df.loc[idx]


def find_max_return(frontier_df, long_only=True):
    """
    Find the portfolio with the highest Expected Return.

    Parameters
    ----------
    frontier_df : pd.DataFrame  output of build_frontier()
    long_only   : bool
        True  → w_A1 in [0, 1]   (max return = 100% Asset 2)
        False → w_A1 in [-1, 2]  (max return = 200% Asset 2)

    Returns
    -------
    pd.Series  — single row from frontier_df
    """
    df = frontier_df.copy()
    if long_only:
        df = df[(df["w_A1"] >= 0) & (df["w_A1"] <= 1)]
    idx = df["ret"].idxmax()
    return frontier_df.loc[idx]


def efficient_frontier_region(frontier_df, allow_short=False):
    """
    Extract the efficient frontier region:
    long-only portfolios at or above MVP return.

    Parameters
    ----------
    frontier_df : pd.DataFrame  output of build_frontier()
    allow_short : bool          if False, filter to weight_region == 'long_only'

    Returns
    -------
    pd.DataFrame  — subset where region == 'efficient'
    dict          — summary stats for the region
    """
    eff_df = frontier_df[frontier_df["region"] == "efficient"].copy()

    if not allow_short:
        eff_df = eff_df[eff_df["weight_region"] == "long_only"]

    if eff_df.empty:
        return eff_df, {}

    peak_sr_row = eff_df.loc[eff_df["sharpe"].idxmax()]
    mvp_row     = eff_df.loc[eff_df["ret"].idxmin()]
    hi_ret_row  = eff_df.loc[eff_df["ret"].idxmax()]

    summary = {
        "w_A1_range":   (eff_df["w_A1"].max(), eff_df["w_A1"].min()),
        "w_A2_range":   (eff_df["w_A2"].min(), eff_df["w_A2"].max()),
        "sd_range":     (eff_df["sd"].min(),    eff_df["sd"].max()),
        "ret_range":    (eff_df["ret"].min(),   eff_df["ret"].max()),
        "peak_sharpe":  peak_sr_row["sharpe"],
        "peak_w_A1":    peak_sr_row["w_A1"],
        "peak_w_A2":    peak_sr_row["w_A2"],
        "n_portfolios": len(eff_df),
        "mvp_row":      mvp_row,
        "hi_ret_row":   hi_ret_row,
    }

    return eff_df, summary


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY TABLE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def frontier_summary_table(frontier_df, r1, sd1, r2, sd2, rho, rf,
                           allow_short=False):
    """
    Build the key-points summary table shown at the bottom of Tab 1.

    Always includes: 100% A1, MVP, Max Sharpe, Equal Weight, 100% A2,
                     Max Return (Long Only)
    When allow_short=True, also includes: Max Return (Leveraged)

    Parameters
    ----------
    frontier_df : pd.DataFrame  output of build_frontier()
    allow_short : bool

    Returns
    -------
    pd.DataFrame with display-ready columns
    """
    mvp      = find_mvp(frontier_df)
    max_sr   = find_max_sharpe(frontier_df, long_only=True)
    max_ret_lo = find_max_return(frontier_df, long_only=True)

    def make_row(label, w_a1, ret, sd, sharpe):
        return {
            "Portfolio":      label,
            "Asset 1 Weight": f"{w_a1 * 100:.1f}%",
            "Asset 2 Weight": f"{(1 - w_a1) * 100:.1f}%",
            "Exp. Return":    f"{ret:.2f}%",
            "Std. Dev.":      f"{sd:.2f}%",
            "Sharpe Ratio":   f"{sharpe:.3f}",
        }

    # Equal weight stats
    eq_ret, eq_sd, eq_sr = portfolio_stats(0.5, r1, sd1, r2, sd2, rho, rf)

    rows = [
        make_row("100% Asset 1",             1.0,           r1,            sd1,           (r1 - rf) / sd1 if sd1 > 0 else 0),
        make_row("⭐ Min. Variance Portfolio", mvp["w_A1"],   mvp["ret"],    mvp["sd"],     mvp["sharpe"]),
        make_row("⭐ Max. Sharpe Portfolio",   max_sr["w_A1"], max_sr["ret"], max_sr["sd"], max_sr["sharpe"]),
        make_row("Equal Weight (50/50)",      0.5,           eq_ret,        eq_sd,         eq_sr),
        make_row("100% Asset 2",             0.0,           r2,            sd2,           (r2 - rf) / sd2 if sd2 > 0 else 0),
        make_row("⭐ Max. Return (Long Only)", max_ret_lo["w_A1"], max_ret_lo["ret"], max_ret_lo["sd"], max_ret_lo["sharpe"]),
    ]

    if allow_short:
        max_ret_lev = find_max_return(frontier_df, long_only=False)
        rows.append(make_row(
            "⭐ Max. Return (Leveraged)",
            max_ret_lev["w_A1"], max_ret_lev["ret"],
            max_ret_lev["sd"],   max_ret_lev["sharpe"]
        ))

    return pd.DataFrame(rows)


def cal_summary_table(r_risky, sd_risky, rf, allow_short=False):
    """
    Build the key-points summary table shown at the bottom of Tab 2.

    Always includes: w=0, w=0.5, w=1, w=1.5, w=2
    When allow_short=True, also includes: w=−1

    Parameters
    ----------
    r_risky  : float
    sd_risky : float
    rf       : float
    allow_short : bool

    Returns
    -------
    pd.DataFrame with display-ready columns
    """
    def make_row(label, w):
        ret, sd, sharpe = cal_stats(w, r_risky, sd_risky, rf)
        return {
            "Portfolio":          label,
            "Risky Asset Weight": f"{w * 100:.1f}%",
            "Risk-Free Weight":   f"{(1 - w) * 100:.1f}%",
            "Exp. Return":        f"{ret:.2f}%",
            "Std. Dev.":          f"{sd:.2f}%",
            "Sharpe Ratio":       f"{sharpe:.3f}" if sd > 0 else "—",
            "Region":             ("Short" if w < 0
                                   else "Long No Leverage" if w <= 1
                                   else "Long With Leverage"),
        }

    rows = []
    if allow_short:
        rows.append(make_row("200% Risk-Free, -100% Risky (Short-Selling) (w=−1)",   -1.0))
    rows += [
        make_row("100% Risk-Free (w=0)",                  0.0),
        make_row("50% Risky (w=0.5)",                      0.5),
        make_row("100% Risky Asset (w=1)",                  1.0),
        make_row("150% Risky Asset (w=1.5) — Leverage",    1.5),
        make_row("200% Risky Asset (w=2) — Leverage",      2.0),
    ]

    return pd.DataFrame(rows)


def rho_mvp_table(r1, sd1, r2, sd2, rf,
                  rho_list=None, current_rho=0.4):
    """
    Build the MVP comparison table across correlation values for Tab 3.

    Parameters
    ----------
    current_rho : float  highlights the current ρ row

    Returns
    -------
    pd.DataFrame with display-ready columns + 'is_current' flag
    """
    if rho_list is None:
        rho_list = [-0.8, -0.4, 0.0, 0.4, 0.8]

    rows = []
    for rho in rho_list:
        w_star      = mvp_weight(sd1, sd2, rho)
        w_star      = max(0.0, min(1.0, w_star))   # clamp to long-only
        ret, sd, sr = portfolio_stats(w_star, r1, sd1, r2, sd2, rho, rf)
        rows.append({
            "Correlation (ρ)":  rho,
            "Asset 1 Weight":   f"{w_star * 100:.1f}%",
            "Asset 2 Weight":   f"{(1 - w_star) * 100:.1f}%",
            "Std. Dev.":        f"{sd:.2f}%",
            "Exp. Return":      f"{ret:.2f}%",
            "Sharpe Ratio":     f"{sr:.3f}",
            "is_current":       abs(rho - current_rho) < 1e-9,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5. METRICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_stats(r1, sd1, r2, sd2, rho, rf):
    """
    Calculate stats for the three benchmark portfolios shown in Tab 1 Row 1:
    100% Asset 1, 100% Asset 2, and Equal Weight (50/50).

    Returns
    -------
    dict with keys: 'asset1', 'asset2', 'equal'
    Each value is a dict: {ret, sd, sharpe}
    """
    eq_ret, eq_sd, eq_sr = portfolio_stats(0.5, r1, sd1, r2, sd2, rho, rf)

    return {
        "asset1": {
            "ret":    r1,
            "sd":     sd1,
            "sharpe": round((r1 - rf) / sd1, 3) if sd1 > 0 else 0.0,
        },
        "asset2": {
            "ret":    r2,
            "sd":     sd2,
            "sharpe": round((r2 - rf) / sd2, 3) if sd2 > 0 else 0.0,
        },
        "equal": {
            "ret":    round(eq_ret, 4),
            "sd":     round(eq_sd,  4),
            "sharpe": round(eq_sr,  3),
        },
    }


def cal_equation_str(r_risky, sd_risky, rf):
    """
    Return a human-readable CAL equation string.
    e.g. "Exp. Return = 3.0% + 0.200 × Std. Dev."
    """
    sr = (r_risky - rf) / sd_risky if sd_risky > 0 else 0
    return f"Exp. Return = {rf:.1f}% + {sr:.3f} × Std. Dev."


def rho_mvp_sd(sd1, sd2, rho):
    """
    Return the MVP Std. Dev. for a given correlation.
    Used in Tab 3 top metrics.
    """
    w = mvp_weight(sd1, sd2, rho)
    w = max(0.0, min(1.0, w))
    variance = (w**2 * sd1**2
                + (1 - w)**2 * sd2**2
                + 2 * w * (1 - w) * rho * sd1 * sd2)
    return round(np.sqrt(max(variance, 0.0)), 4)
