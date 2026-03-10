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
                        rho_list=None, w_step=0.02, allow_short=False):
    """
    Build frontiers for multiple correlation values.
    Used in Tab 2 — Correlation Effect.

    Long-only (w: 0 → 1) when allow_short=False per Prof. Weisbenner's lecture.
    Full range (w: -1 → 2) when allow_short=True.

    Parameters
    ----------
    r1, sd1 : float  Asset 1 params
    r2, sd2 : float  Asset 2 params
    rf      : float  risk-free rate
    rho_list: list   correlations to compute (default: [-0.8,-0.4,0,0.4,0.8])
    w_step  : float  step size (coarser = faster for comparison chart)
    allow_short : bool  if True, scan full weight range including short-selling

    Returns
    -------
    dict : { rho_value : pd.DataFrame }
        Each DataFrame has columns: w_A1, w_A2, ret, sd, sharpe
    """
    if rho_list is None:
        rho_list = [-0.8, -0.4, 0.0, 0.4, 0.8]

    if allow_short:
        weights = np.arange(W_MIN, W_MAX + w_step / 2, w_step)
    else:
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
        df = pd.DataFrame(rows)
        _mvp_ret = df.loc[df["sd"].idxmin(), "ret"]
        df["region"] = df["ret"].apply(lambda r: "efficient" if r >= _mvp_ret else "dominated")
        result[rho] = df

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
            "Asset 1 Weight": round(w_a1 * 100, 1),
            "Asset 2 Weight": round((1 - w_a1) * 100, 1),
            "Exp. Return":    round(ret, 2),
            "Std. Dev.":      round(sd, 2),
            "Sharpe Ratio":   round(sharpe, 3),
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
            "Risky Asset Weight": round(w * 100, 1),
            "Risk-Free Weight":   round((1 - w) * 100, 1),
            "Exp. Return":        round(ret, 2),
            "Std. Dev.":          round(sd, 2),
            "Sharpe Ratio":       round(sharpe, 3) if sd > 0 else None,
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
                  rho_list=None, current_rho=0.4, allow_short=False):
    """
    Build the MVP comparison table across correlation values for Tab 2.

    Parameters
    ----------
    current_rho : float  highlights the current ρ row
    allow_short : bool   if True, MVP weight is unconstrained (can be <0 or >1)

    Returns
    -------
    pd.DataFrame with display-ready columns + 'is_current' flag
    """
    if rho_list is None:
        rho_list = [-0.8, -0.4, 0.0, 0.4, 0.8]

    # Insert current_rho if it isn't already in the list
    if not any(abs(r - current_rho) < 1e-9 for r in rho_list):
        rho_list = sorted(rho_list + [current_rho])

    rows = []
    for rho in rho_list:
        w_star = mvp_weight(sd1, sd2, rho)
        if not allow_short:
            w_star = max(0.0, min(1.0, w_star))   # clamp to long-only
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


def rho_msp_table(r1, sd1, r2, sd2, rf,
                  rho_list=None, current_rho=0.4, allow_short=False):
    """
    Build the Max Sharpe Portfolio comparison table across correlation values
    for Tab 2, mirroring rho_mvp_table.

    Parameters
    ----------
    current_rho : float  highlights the current ρ row
    allow_short : bool   if True, search is unconstrained

    Returns
    -------
    pd.DataFrame with display-ready columns + 'is_current' flag
    """
    if rho_list is None:
        rho_list = [-0.8, -0.4, 0.0, 0.4, 0.8]

    if not any(abs(r - current_rho) < 1e-9 for r in rho_list):
        rho_list = sorted(rho_list + [current_rho])

    rows = []
    for rho in rho_list:
        df = build_frontier(r1, sd1, r2, sd2, rho, rf)
        msp = find_max_sharpe(df, long_only=not allow_short)
        w_star = msp["w_A1"]
        ret, sd, sr = portfolio_stats(w_star, r1, sd1, r2, sd2, rho, rf)
        rows.append({
            "Correlation (ρ)": rho,
            "Asset 1 Weight":  f"{w_star * 100:.1f}%",
            "Asset 2 Weight":  f"{(1 - w_star) * 100:.1f}%",
            "Std. Dev.":       f"{sd:.2f}%",
            "Exp. Return":     f"{ret:.2f}%",
            "Sharpe Ratio":    f"{sr:.3f}",
            "is_current":      abs(rho - current_rho) < 1e-9,
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


def rho_mvp_sd(sd1, sd2, rho, allow_short=False):
    """
    Return the MVP Std. Dev. for a given correlation.
    Used in Tab 2 top metrics.

    Parameters
    ----------
    allow_short : bool  if True, MVP weight is unconstrained (can be <0 or >1)
    """
    w = mvp_weight(sd1, sd2, rho)
    if not allow_short:
        w = max(0.0, min(1.0, w))
    variance = (w**2 * sd1**2
                + (1 - w)**2 * sd2**2
                + 2 * w * (1 - w) * rho * sd1 * sd2)
    return round(np.sqrt(max(variance, 0.0)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# 6. PORTFOLIO SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def solve_portfolio(frontier_df, objective, goal, constraint, target=None,
                    result_filter=None):
    """
    Find the portfolio that minimises, maximises, or hits a target value
    of a given metric within a specified constraint region.

    Parameters
    ----------
    frontier_df   : pd.DataFrame  output of build_frontier()
    objective     : str  'ret' | 'sd' | 'sharpe'
    goal          : str  'min' | 'max' | 'target'
    constraint    : str  'full' | 'long_only' | 'long_A1' | 'short_A1'
    target        : float or None  used only when goal == 'target'
    result_filter : str or None  'efficient' | 'dominated' — further restrict
                    the search to a sub-region of the frontier

    Returns
    -------
    result_row : pd.Series or None
    feasible   : bool
    message    : str
    """
    df = frontier_df.copy()

    if constraint == "full":
        pass
    elif constraint == "long_only":
        df = df[df["weight_region"] == "long_only"]
    elif constraint == "long_A1":
        df = df[df["weight_region"] == "long_A1"]
    elif constraint == "short_A1":
        df = df[df["weight_region"] == "short_A1"]
    else:
        return None, False, f"Unknown constraint: {constraint}"

    if result_filter == "efficient":
        df = df[df["region"] == "efficient"]
    elif result_filter == "dominated":
        df = df[df["region"] == "dominated"]

    if df.empty:
        return None, False, (
            "No portfolios available in the selected constraint region. "
            "Try enabling short-selling or choosing a different region."
        )

    obj_labels = {"ret": "Exp. Return", "sd": "Std. Dev.", "sharpe": "Sharpe Ratio"}
    obj_label  = obj_labels.get(objective, objective)

    if goal == "min":
        idx        = df[objective].idxmin()
        result_row = frontier_df.loc[idx]
        actual     = result_row[objective]
        message    = f"Minimum {obj_label} = {actual:.4f} within the selected region."

    elif goal == "max":
        idx        = df[objective].idxmax()
        result_row = frontier_df.loc[idx]
        actual     = result_row[objective]
        message    = f"Maximum {obj_label} = {actual:.4f} within the selected region."

    elif goal == "target":
        if target is None:
            return None, False, "A target value is required."
        diff       = (df[objective] - target).abs()
        idx        = diff.idxmin()
        result_row = frontier_df.loc[idx]
        actual     = result_row[objective]
        delta      = abs(actual - target)
        range_min  = df[objective].min()
        range_max  = df[objective].max()
        if target < range_min or target > range_max:
            message = (
                f"Target {obj_label} = {target:.4f} is outside the constraint region "
                f"range [{range_min:.4f}, {range_max:.4f}]. "
                f"Nearest feasible value: {actual:.4f}  (Δ = {delta:.4f})."
            )
        else:
            message = (
                f"Closest portfolio to {obj_label} = {target:.4f}. "
                f"Actual: {actual:.4f}  (Δ = {delta:.4f})."
            )
    else:
        return None, False, f"Unknown goal: {goal}"

    return result_row, True, message


# ══════════════════════════════════════════════════════════════════════════════
# 7. N-ASSET PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

def n_portfolio_stats(w, mu, cov, rf):
    """
    Portfolio statistics for an N-asset portfolio.

    Parameters
    ----------
    w   : array-like, shape (N,)   portfolio weights (sum = 1)
    mu  : array-like, shape (N,)   expected returns (%)
    cov : array-like, shape (N, N) covariance matrix (same % units)
    rf  : float                    risk-free rate (%)

    Returns
    -------
    (ret, sd, sharpe) : tuple of floats
    """
    w   = np.asarray(w,   dtype=float)
    mu  = np.asarray(mu,  dtype=float)
    cov = np.asarray(cov, dtype=float)
    ret    = float(w @ mu)
    var    = float(w @ cov @ w)
    sd     = np.sqrt(max(var, 0.0))
    sharpe = (ret - rf) / sd if sd > 1e-10 else 0.0
    return ret, sd, sharpe


def build_n_frontier(mu, cov, rf, allow_short=False, n_points=150):
    """
    Build the N-asset efficient frontier via SLSQP quadratic programming.

    Sweeps target returns from MVP return to max feasible return,
    minimising portfolio variance at each level.

    Parameters
    ----------
    mu          : array-like, shape (N,)   expected returns (%)
    cov         : array-like, shape (N, N) covariance matrix (same % units)
    rf          : float                    risk-free rate (%)
    allow_short : bool                     if True, weights in [−1, 2]
    n_points    : int                      frontier resolution

    Returns
    -------
    pd.DataFrame
        Columns: ret, sd, sharpe, w_1 … w_N, region ('efficient')
        Returns empty DataFrame if optimisation fails entirely.
    """
    from scipy.optimize import minimize

    mu  = np.asarray(mu,  dtype=float)
    cov = np.asarray(cov, dtype=float)
    n   = len(mu)

    lb     = -1.0 if allow_short else 0.0
    ub     =  2.0 if allow_short else 1.0
    bounds = [(lb, ub)] * n

    sum1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def var_fn(w):   return float(w @ cov @ w)
    def var_grad(w): return 2.0 * (cov @ w)

    w0 = np.ones(n) / n

    # ── 1. Find MVP ────────────────────────────────────────────────────────
    res_mvp = minimize(var_fn, w0, jac=var_grad, method="SLSQP",
                       bounds=bounds, constraints=[sum1],
                       options={"ftol": 1e-14, "maxiter": 2000})
    w_mvp   = res_mvp.x if res_mvp.success else w0
    mvp_ret = float(w_mvp @ mu)

    # ── 2. Return range ────────────────────────────────────────────────────
    if allow_short:
        max_ret = float(mu.max()) * 1.5 + float(mu.min()) * (-0.5)
        max_ret = min(max_ret, float(mu.max()) * 2.0)
    else:
        max_ret = float(mu.max())

    target_rets = np.linspace(mvp_ret, max_ret, n_points)

    # ── 3. Sweep helper ────────────────────────────────────────────────────
    def _sweep(targets, region):
        rows   = []
        w_prev = w_mvp.copy()
        for target_ret in targets:
            ret_con = {"type": "eq",
                       "fun": lambda w, r=target_ret: float(w @ mu) - r}
            res = minimize(var_fn, w_prev, jac=var_grad, method="SLSQP",
                           bounds=bounds, constraints=[sum1, ret_con],
                           options={"ftol": 1e-10, "maxiter": 500})
            if not res.success:
                res = minimize(var_fn, w0, jac=var_grad, method="SLSQP",
                               bounds=bounds, constraints=[sum1, ret_con],
                               options={"ftol": 1e-10, "maxiter": 500})
            if res.success:
                ws = res.x
                ret_v, sd_v, sr_v = n_portfolio_stats(ws, mu, cov, rf)
                row = {"ret": round(ret_v, 4), "sd": round(sd_v, 4),
                       "sharpe": round(sr_v, 4), "region": region}
                for i, wi in enumerate(ws):
                    row[f"w_{i+1}"] = round(float(wi), 4)
                rows.append(row)
                w_prev = res.x
            else:
                w_prev = w0
        return rows

    rows_eff = _sweep(target_rets, "efficient")

    # ── 4. Sweep dominated frontier (min return → MVP) ─────────────────────
    if allow_short:
        min_ret = mvp_ret - (max_ret - mvp_ret)
    else:
        min_ret = float(mu.min())

    rows_dom = []
    if min_ret < mvp_ret - 1e-6:
        dom_targets = np.linspace(min_ret, mvp_ret, n_points)
        rows_dom = _sweep(dom_targets, "dominated")

    rows = rows_eff + rows_dom
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ret", "sd"]).reset_index(drop=True)
    return df


def n_find_mvp(frontier_df):
    """Find the Minimum Variance Portfolio row from an N-asset frontier DataFrame."""
    return frontier_df.loc[frontier_df["sd"].idxmin()]


def n_find_max_sharpe(frontier_df):
    """Find the Max Sharpe Portfolio row from an N-asset frontier DataFrame."""
    return frontier_df.loc[frontier_df["sharpe"].idxmax()]


def build_n_kappa_frontiers(mu, sd_arr, corr_matrix, rf,
                             kappa_list=None, allow_short=False, n_points=80):
    """
    Build N-asset frontiers for multiple correlation scalar values κ.

    At κ=0 all assets are uncorrelated (diagonal covariance).
    At κ=1 the full user-supplied correlation matrix is used.

        cov_κ = diag(σ) @ [κ·(ρ − I) + I] @ diag(σ)

    Parameters
    ----------
    mu          : array-like, shape (N,)
    sd_arr      : array-like, shape (N,)
    corr_matrix : array-like, shape (N, N)
    rf          : float
    kappa_list  : list of floats  (default: 0.0, 0.25, 0.5, 0.75, 1.0)
    allow_short : bool
    n_points    : int  (coarser than full frontier for speed)

    Returns
    -------
    dict : { κ_value : pd.DataFrame }
    """
    if kappa_list is None:
        kappa_list = [0.0, 0.25, 0.5, 0.75, 1.0]

    mu   = np.asarray(mu,   dtype=float)
    sd   = np.asarray(sd_arr, dtype=float)
    corr = np.asarray(corr_matrix, dtype=float)
    I    = np.eye(len(mu))
    D    = np.diag(sd)

    result = {}
    for kappa in kappa_list:
        corr_k = kappa * (corr - I) + I
        cov_k  = D @ corr_k @ D
        result[kappa] = build_n_frontier(mu, cov_k, rf,
                                         allow_short=allow_short,
                                         n_points=n_points)
    return result


def n_solve_portfolio(frontier_df, objective, goal, target=None,
                      efficient_only=True):
    """
    Find the best N-asset portfolio from a pre-computed frontier DataFrame.

    Parameters
    ----------
    frontier_df   : pd.DataFrame  output of build_n_frontier()
    objective     : str  'ret' | 'sd' | 'sharpe'
    goal          : str  'min' | 'max' | 'target'
    target        : float or None  used when goal == 'target'
    efficient_only: bool  if True, restrict to region == 'efficient'

    Returns
    -------
    result_row : pd.Series or None
    feasible   : bool
    message    : str
    """
    df = frontier_df[frontier_df["region"] == "efficient"].copy() \
         if efficient_only else frontier_df.copy()

    if df.empty:
        return None, False, "No portfolios available."

    obj_labels = {"ret": "Exp. Return", "sd": "Std. Dev.", "sharpe": "Sharpe Ratio"}
    label = obj_labels.get(objective, objective)

    if goal == "min":
        idx = df[objective].idxmin()
        row = frontier_df.loc[idx]
        msg = f"Minimum {label} = {row[objective]:.4f} on the efficient frontier."
    elif goal == "max":
        idx = df[objective].idxmax()
        row = frontier_df.loc[idx]
        msg = f"Maximum {label} = {row[objective]:.4f} on the efficient frontier."
    elif goal == "target":
        if target is None:
            return None, False, "A target value is required."
        diff  = (df[objective] - target).abs()
        idx   = diff.idxmin()
        row   = frontier_df.loc[idx]
        delta = abs(row[objective] - target)
        rng   = (df[objective].min(), df[objective].max())
        if target < rng[0] or target > rng[1]:
            msg = (f"Target {label} = {target:.4f} is outside the frontier range "
                   f"[{rng[0]:.4f}, {rng[1]:.4f}]. "
                   f"Nearest feasible: {row[objective]:.4f}  (Δ = {delta:.4f}).")
        else:
            msg = (f"Closest portfolio to {label} = {target:.4f}. "
                   f"Actual: {row[objective]:.4f}  (Δ = {delta:.4f}).")
    else:
        return None, False, f"Unknown goal: {goal}"

    return row, True, msg


def validate_corr_matrix(corr_matrix):
    """
    Validate a correlation matrix.

    Returns
    -------
    (is_valid : bool, errors : list of str)
    """
    corr   = np.asarray(corr_matrix, dtype=float)
    errors = []

    if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
        errors.append("Diagonal elements must equal 1.")
    if not np.allclose(corr, corr.T, atol=1e-6):
        errors.append("Matrix must be symmetric.")
    if np.any(corr < -1.0 - 1e-6) or np.any(corr > 1.0 + 1e-6):
        errors.append("All correlations must be in [−1, 1].")
    eigvals = np.linalg.eigvalsh(corr)
    if np.any(eigvals < -1e-6):
        errors.append(
            f"Matrix is not positive semi-definite "
            f"(min eigenvalue = {eigvals.min():.4f})."
        )
    return len(errors) == 0, errors


def nearest_psd_corr(corr_matrix):
    """
    Project a symmetric matrix onto the nearest positive semi-definite
    correlation matrix by clipping negative eigenvalues and re-normalising.
    """
    corr = np.asarray(corr_matrix, dtype=float)
    corr = (corr + corr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.0)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_psd))
    d = np.where(d < 1e-12, 1.0, d)
    corr_psd = corr_psd / np.outer(d, d)
    np.fill_diagonal(corr_psd, 1.0)
    return np.clip(corr_psd, -1.0, 1.0)


def parse_n_csv(uploaded_df):
    """
    Parse an uploaded CSV DataFrame into N-asset parameters.

    Expected format::

        name,return_pct,sd_pct,corr_Asset A,corr_Asset B,...
        Asset A,8.0,15.0,1.00,0.30,0.20
        ...

    Returns
    -------
    asset_names : list of str | None on error
    mu          : np.ndarray shape (N,)
    sd_arr      : np.ndarray shape (N,)
    corr_matrix : np.ndarray shape (N, N)
    errors      : list of str  (empty = OK)
    """
    required = ["name", "return_pct", "sd_pct"]
    missing  = [c for c in required if c not in uploaded_df.columns]
    if missing:
        return None, None, None, None, [f"Missing required columns: {missing}"]

    try:
        asset_names = uploaded_df["name"].astype(str).tolist()
        mu          = uploaded_df["return_pct"].astype(float).values
        sd_arr      = uploaded_df["sd_pct"].astype(float).values
    except Exception as exc:
        return None, None, None, None, [f"Error parsing asset parameters: {exc}"]

    n = len(asset_names)
    if n < 2:
        return None, None, None, None, ["CSV must contain at least 2 assets."]
    if n > 20:
        return None, None, None, None, ["CSV may not contain more than 20 assets."]

    errors = []
    corr_cols = [c for c in uploaded_df.columns if c.startswith("corr_")]
    if len(corr_cols) == n:
        try:
            corr_matrix = uploaded_df[corr_cols].astype(float).values
        except Exception as exc:
            errors.append(f"Error parsing correlation columns: {exc}")
            corr_matrix = np.full((n, n), 0.3)
            np.fill_diagonal(corr_matrix, 1.0)
    else:
        corr_matrix = np.full((n, n), 0.3)
        np.fill_diagonal(corr_matrix, 1.0)
        if corr_cols:
            errors.append(
                f"Expected {n} corr_ columns, found {len(corr_cols)}. "
                "Using default ρ = 0.3."
            )

    # Enforce symmetry, diagonal = 1, clip to [-1, 1]
    corr_matrix = (corr_matrix + corr_matrix.T) / 2.0
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    is_valid, corr_errors = validate_corr_matrix(corr_matrix)
    if not is_valid:
        errors.extend(corr_errors)
        corr_matrix = nearest_psd_corr(corr_matrix)
        errors.append("Correlation matrix adjusted to nearest valid PSD matrix.")

    return asset_names, mu, sd_arr, corr_matrix, errors
