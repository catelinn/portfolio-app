# Portfolio Frontier & CAL Explorer
## Design & Requirements Document

**Course:** FIN 511 — Investments I: Fundamentals of Performance Evaluation  
**Module:** 1 — Lesson 1-5: Portfolio Choice in General Settings  
**Version:** 2.5 | March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Application Structure](#2-application-structure)
3. [Sidebar Design](#3-sidebar-design)
4. [Tab 1 — Portfolio Frontier](#4-tab-1--portfolio-frontier)
5. [Tab 2 — Capital Allocation Line](#5-tab-2--capital-allocation-line)
6. [Tab 3 — Correlation Effect](#6-tab-3--correlation-effect)
7. [Display Terminology](#7-display-terminology)
8. [Core Calculations](#8-core-calculations)
9. [Future Enhancements](#9-future-enhancements)
10. [Change Log](#10-change-log)

---

## 1. Project Overview

### 1.1 Purpose

Interactive web application for exploring portfolio theory concepts from FIN 511 Module 1.  
Allows real-time adjustment of asset parameters to see how they affect the efficient frontier and Capital Allocation Lines.

### 1.2 Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| UI Framework | Streamlit | Interactive web app — sliders, tabs, layout. No JavaScript needed |
| Charts | Plotly | Interactive charts with hover tooltips, zoom, and pan |
| Math | NumPy | Portfolio calculations — variance, std dev, Sharpe ratio |
| Data | Pandas | DataFrames for frontier and CAL data |
| Language | Python | All code — calculations, charts, and UI |
| Deployment | Streamlit Community Cloud + GitHub | Free hosting, shareable public URL |

### 1.3 File Structure

```
portfolio_app/
├── app.py              # Main entry point, Streamlit UI, tab layout
├── calculations.py     # Pure math functions, no UI dependencies
├── charts.py           # Plotly figure builders, one function per chart
├── requirements.txt    # Dependencies for Streamlit Cloud deployment
└── DESIGN.md           # This file
```

### 1.4 Deployment

- Code stored in a **public GitHub repository** (MIT license)
- **Streamlit Community Cloud** connected to the GitHub repo
- Every save to GitHub auto-redeploys the app within ~30 seconds
- Shareable URL format: `https://yourname-portfolio-app.streamlit.app`
- Browser editor: `github.dev/yourname/portfolio-app` (no local install needed)

---

## 2. Application Structure

### 2.1 Overall Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Header: App title + subtitle                               │
├──────────────┬──────────────────────────────────────────────┤
│              │  Tab 1: Portfolio Frontier                   │
│   SIDEBAR    │    ▶ ⚙️ Parameters (expander, collapsed)     │
│  (minimal)   │    metrics / charts / summary table          │
│              │                                              │
│              │  Tab 2: Capital Allocation Line              │
│              │    ▶ ⚙️ Parameters (expander, collapsed)     │
│              │    metrics / charts / summary table          │
│              │                                              │
│              │  Tab 3: Correlation Effect                   │
│              │    ▶ ⚙️ Parameters (expander, collapsed)     │
│              │    metrics / charts                          │
└──────────────┴──────────────────────────────────────────────┘
```

### 2.2 Tab Structure

| Tab | Content | Uses Parameters From |
|---|---|---|
| Tab 1 — Portfolio Frontier | 4 charts + metrics rows + summary table | Frontier sidebar section |
| Tab 2 — Capital Allocation Line | 5 charts + metrics row + summary table | CAL sidebar section |
| Tab 3 — Correlation Effect | 1 multi-line chart + MVP comparison table | Frontier sidebar section (ρ slider) |

### 2.3 Data Flow

```
User moves slider / toggles checkbox
         ↓
app.py reads new params from st.sidebar
         ↓
calculations.py recomputes all DataFrames
         ↓
charts.py builds new Plotly figures
         ↓
Streamlit re-renders only changed components
```

---

## 3. Sidebar Design

### 3.1 Layout

The sidebar is **minimal** — it holds only global controls. All parameter sliders live inside each tab in a collapsible expander.

```
SIDEBAR
│
├── ☐ Allow Short-Selling   (checkbox, default: OFF)
│   Affects Tab 1 and Tab 2 charts and metrics
│
└── QUICK PRESETS
    ├── [Baseline]   [ρ = 1]   [ρ = −1]
```

| Button | Sets |
|---|---|
| Baseline | All parameters reset to defaults |
| ρ = 1 | All defaults + Correlation = +1.0 |
| ρ = −1 | All defaults + Correlation = −1.0 |

### 3.2 Parameter Expanders — Inside Each Tab

Each tab has a collapsible expander at the top for its own parameters. Expanders are **collapsed by default** so charts are immediately visible.

#### Tab 1 — Portfolio Frontier expander (3 columns)

```
▶ ⚙️ Parameters — Portfolio Frontier

  Asset 1               Asset 2 (more risky)  Correlation & RF
  ─────────────         ─────────────         ─────────────────
  Exp. Return (%)       Exp. Return (%)        Correlation (ρ)
  slider: 0–25%         slider: 0–30%          slider: -1 to +1
  default 8%            default 15%            default 0.4

  Std. Dev. (%)         Std. Dev. (%)          Risk-Free Rate (%)
  slider: 5–60%         slider: sd1+1–80%      slider: 0–10%
  default 25%           default 50%            default 3%
```

#### Tab 2 — CAL expander (2 columns)

```
▶ ⚙️ Parameters — Capital Allocation Line

  Risky Asset           Risk-Free Asset
  ─────────────         ─────────────────
  Exp. Return (%)       Risk-Free Rate (%)
  slider: 0–25%         slider: 0–10%
  default 8%            default 3%

  Std. Dev. (%)         Note: zero std. dev.,
  slider: 1–60%         zero correlation
  default 25%           with risky asset
```

#### Tab 3 — Correlation Effect expander (3 columns)

```
▶ ⚙️ Parameters — Correlation Effect

  Asset 1               Asset 2               Correlation & RF
  (same sliders as Tab 1 — shared via session state)
```

> **Note:** Frontier and CAL parameters are independent — changing one does not affect the other. Tab 3 shares the same parameters as Tab 1.

### 3.3 Parameter Validation — Asset 2 Std. Deviation

| Rule | Behaviour | Implementation |
|---|---|---|
| Asset 2 Std. Dev. ≥ Asset 1 Std. Dev. + 1% | Slider minimum dynamically set to `sd1 + 1` | `min_value = sd1 + 1` in `st.slider()` |
| If sd2 < sd1 + 1 after sd1 is raised | sd2 snaps up to sd1 + 1 automatically | `value = max(sd2, sd1 + 1)` |
| Notice shown when snap occurs | `st.info()` message displayed | `if sd2_prev < sd1 + 1: st.info(...)` |

**Notice message text:**
> ℹ️ Asset 2 Std. Dev. has been adjusted upward to maintain the constraint that Asset 2 must be riskier than Asset 1.

### 3.4 Session State Keys

| Key | Type | Purpose |
|---|---|---|
| `sd2` | float | Remembers Asset 2 Std. Dev. across rerenders for snap validation |
| `frontier_rf` | float | Risk-free rate for Frontier tab (Sharpe calculation only) |
| `cal_rf` | float | Risk-free rate for CAL tab |
| `allow_short` | bool | Short-selling checkbox state |

---

## 4. Tab 1 — Portfolio Frontier

### 4.1 Metrics — Top of Tab

#### Row 1 — Benchmark Portfolios

Three side-by-side **HTML cards** (flexbox layout, colour-coded border-top):

| Card | Border Colour | Asset 1 Weight | Asset 2 Weight | Metrics Shown |
|---|---|---|---|---|
| Asset 1 Only | Navy | 100% | 0% | Exp. Return, Std. Dev., Sharpe Ratio |
| Asset 2 Only (more risky) | Amber | 0% | 100% | Exp. Return, Std. Dev., Sharpe Ratio |
| Equal Weight | Green | 50% | 50% | Exp. Return, Std. Dev., Sharpe Ratio |

Each card uses a label/value table layout — values never truncate regardless of screen width.

#### Row 2 — Optimal Portfolio Cards (styled like MVP card)

| Card | Condition | Shown When |
|---|---|---|
| ⭐ Min. Variance Portfolio | Lowest achievable Std. Dev. | Always |
| ⭐ Max. Sharpe Portfolio | Highest Sharpe Ratio (long-only) | Always |
| ⭐ Max. Return (Long Only) | Asset 2 Weight = 100% | Always |
| ⭐ Max. Return (Leveraged) | Asset 2 Weight = 200%, Asset 1 Weight = −100% | Short-selling ON only |

Each card displays: **Asset 1 Weight, Asset 2 Weight, Exp. Return, Std. Dev., Sharpe Ratio**

#### Row 3 — Efficient Frontier Region Summary

**Definition:** Long-only portfolios above MVP (Asset 1 Weight: 0%→100%, Exp. Return ≥ MVP Exp. Return)

> Based on Prof. Weisbenner's lecture — efficient frontier is always presented as long-only in Module 1.

Displayed as **2 rows of 3 `st.metric()` cards** to avoid truncation:

**Row 3a** (3 columns):

| Metric | Example | Notes |
|---|---|---|
| Asset 1 Weight Range | Start: 94% / Delta: → 0% | Main value = start, delta = end |
| Asset 2 Weight Range | Start: 6% / Delta: → 100% | Main value = start, delta = end |
| Peak Sharpe Ratio | 0.272 | With hover tooltip showing weights |

**Row 3b** (3 columns):

| Metric | Example | Notes |
|---|---|---|
| Std. Dev. Range | Start: 24.85% / Delta: → 50.00% | Main value = MVP σ, delta = max σ |
| Exp. Return Range | Start: 8.50% / Delta: → 15.00% | Main value = MVP ret, delta = max ret |
| Portfolios in Region | 95 | Count of long-only portfolios above MVP |

### 4.2 Charts

| Chart | Asset 1 Weight Range | Asset 2 Weight Range | Short-Selling Required | Visible When |
|---|---|---|---|---|
| All Allocations | −100% → +200% | +200% → −100% | Yes | Short-selling ON only |
| Efficient vs Dominated | 0% → 100% | 0% → 100% | No | Always |
| Short A1 / Long A2 | −100% → 0% | 100% → 200% | Yes | Short-selling ON only |
| Long A1 / Short A2 | 100% → 200% | −100% → 0% | Yes | Short-selling ON only |

### 4.3 Short-Selling Checkbox Behaviour

#### Short-Selling OFF (default)

> ℹ️ **Short-selling is currently disabled.** Only portfolios with Asset 1 Weight: 0%→100% and Asset 2 Weight: 0%→100% are shown. This reflects the real-world constraint most investors face in 401k plans and standard brokerage accounts. Enable short-selling above to see the full frontier including leveraged allocations.

- Charts shown: **Long Only** (1 chart)
- Metrics Row 2: **3 cards** — Min. Variance Portfolio, Max. Sharpe, Max. Return (Long Only)

#### Short-Selling ON

> ⚠️ **Short-selling is enabled.** Charts now show all allocations including: Short A1 / Long A2 (Asset 1 Weight < 0%) and Long A1 / Short A2 (Asset 1 Weight > 100%). Note: Short-selling requires a margin account and involves borrowing costs not reflected here. This simple example ignores margin and collateral requirements (per Prof. Weisbenner, Lesson 1-2.5).

- Charts shown: **All 4 charts**
- Metrics Row 2: **4 cards** — Min. Variance Portfolio, Max. Sharpe, Max. Return (Long Only), Max. Return (Leveraged)

### 4.4 Hover Tooltip

```
Asset 1 Weight:      94.0%
Asset 2 Weight:       6.0%
────────────────────────────
Exp. Return:          8.50%
Std. Dev.:           24.85%
Sharpe Ratio:         0.228
```

### 4.5 Summary Table — Bottom of Tab

Key allocation points: 100% A1, Min. Variance Portfolio, Max. Sharpe, Equal Weight (50/50), 100% A2, Max. Return (Long Only), Max. Return (Leveraged — if short-selling ON)

| Column | Label in Table | Format |
|---|---|---|
| `w_A1` | Asset 1 Weight | XX.X% |
| `w_A2` | Asset 2 Weight | XX.X% |
| `ret` | Exp. Return | XX.XX% |
| `sd` | Std. Dev. | XX.XX% |
| `sharpe` | Sharpe Ratio | X.XXX |
| `region` | Region | Text label |

---

## 5. Tab 2 — Capital Allocation Line

### 5.1 Metrics — Top of Tab

Displayed as **2 rows** to avoid truncation:

**Row 1** — 3 columns (CAL summary):

| Metric | Formula | Example |
|---|---|---|
| Sharpe Ratio | (Exp. Return − rf) / Std. Dev. | 0.200 |
| Risk-Free Rate | rf (slider value) | 3.00% |
| CAL Equation | E[R] = rf + SR × σ | E[R] = 3.0% + 0.200 × σ |

**Row 2** — 4 columns (key portfolio points, main value = Exp. Return, delta = σ):

| Metric | Exp. Return | Std. Dev. |
|---|---|---|
| w = 0 (100% Risk-Free) | rf = 3.00% | σ = 0.00% |
| w = 1 (100% Risky) | 8.00% | σ = 25.00% |
| w = 1.5 (Leverage) | 10.50% | σ = 37.50% |
| w = 2 (Max Leverage) | 13.00% | σ = 50.00% |

### 5.2 Charts — Order and Ranges

| # | Chart Title | Risky Asset Weight | Risk-Free Weight | Short-Selling Required | Visible When |
|---|---|---|---|---|---|
| 1 | All Allocations | −100% → +200% | +200% → −100% | Yes | Short-selling ON only |
| 2 | All Long | 0% → +200% | +100% → −100% | No | Always |
| 3 | Long Without Leverage | 0% → +100% | +100% → 0% | No | Always |
| 4 | Long With Leverage | +100% → +200% | 0% → −100% | No | Always |
| 5 | Equation Summary | — | — | No | Always |

### 5.3 Short-Selling Checkbox Behaviour

#### Short-Selling OFF

> ℹ️ **Short-selling disabled.** CAL shown for long positions only (Risky Asset Weight ≥ 0%). The "All Allocations" chart (which includes Risky Asset Weight < 0%, i.e. shorting the risky asset to invest more in T-Bills) is hidden.

- Charts 2–5 shown (All Allocations hidden)

#### Short-Selling ON

> ⚠️ **Short-selling enabled.** "All Allocations" chart now includes Risky Asset Weight < 0%: shorting the risky asset to put more than 100% into T-Bills. Note: negative Sharpe in this region — you earn less than the risk-free rate.

- All 5 charts shown

### 5.4 Hover Tooltip

```
Risky Asset Weight:   70.0%
Risk-Free Weight:     30.0%
────────────────────────────
Exp. Return:           6.50%
Std. Dev.:            17.50%
Sharpe Ratio:          0.200

── Leverage region (Risky Asset Weight > 100%): ──
Risky Asset Weight:  150.0%
Risk-Free Weight:    -50.0%    ← negative = borrowing at rf
────────────────────────────
Exp. Return:          10.50%
Std. Dev.:            37.50%
Sharpe Ratio:          0.200
```

### 5.5 Summary Table — Bottom of Tab

Key w points: w=−1 (if short ON), w=0, w=0.5, w=1, w=1.5, w=2

| Column | Label in Table | Format |
|---|---|---|
| `w_risky` | Risky Asset Weight | XX.X% |
| `w_rf` | Risk-Free Weight | XX.X% |
| `ret` | Exp. Return | XX.XX% |
| `sd` | Std. Dev. | XX.XX% |
| `sharpe` | Sharpe Ratio | X.XXX |
| `region` | Region | Text label |

---

## 6. Tab 3 — Correlation Effect

### 6.1 Metrics — Top of Tab

| Metric | Description |
|---|---|
| Current ρ | Value of the correlation slider |
| Min. Variance Portfolio Std. Dev. at ρ=−1 | Lowest achievable risk with perfect negative correlation |
| Min. Variance Portfolio Std. Dev. at ρ=0 | Risk at zero correlation |
| Min. Variance Portfolio Std. Dev. at ρ=+1 | Risk at perfect positive correlation (= weighted avg of σ) |

### 6.2 Charts

#### Chart 1 — Overlaid Frontiers

- Five efficient frontiers plotted simultaneously for ρ = −0.8, −0.4, 0, +0.4, +0.8
- Current ρ frontier shown with **thicker line, full opacity**
- Other ρ frontiers shown with **dashed lines, reduced opacity**
- Always long-only (Asset 1 & Asset 2 Weights: 0%→100%) regardless of short-selling checkbox

> ℹ️ Correlation frontiers always displayed as long-only regardless of the short-selling setting. This is consistent with how Prof. Weisbenner presents correlation effects in Lesson 1-5.

#### Chart 2 — MVP Comparison Table

| ρ Value | Asset 1 Weight | Asset 2 Weight | Std. Dev. | Exp. Return |
|---|---|---|---|---|
| −0.8 | XX.X% | XX.X% | XX.XX% | XX.XX% |
| −0.4 | XX.X% | XX.X% | XX.XX% | XX.XX% |
| 0.0 | XX.X% | XX.X% | XX.XX% | XX.XX% |
| +0.4 ← current | XX.X% | XX.X% | XX.XX% | XX.XX% |
| +0.8 | XX.X% | XX.X% | XX.XX% | XX.XX% |

Current ρ row highlighted. Lower ρ → lower MVP Std. Dev. → more diversification benefit.

---

## 7. Display Terminology

### 7.1 Rules

- **Drop "Portfolio" prefix** when context is already clear (inside cards, tables, tooltips)
- **Keep "Portfolio" prefix** on chart axis labels and standalone messages
- Asset-level slider labels use `Exp. Return` and `Std. Dev.` without "Portfolio"

### 7.2 Full Reference

| Internal Variable | Short Label (cards / tooltips) | Full Label (chart axes) |
|---|---|---|
| `w_A1` | Asset 1 Weight | Asset 1 Weight (%) |
| `w_A2` | Asset 2 Weight | Asset 2 Weight (%) |
| `w_risky` | Risky Asset Weight | Risky Asset Weight (%) |
| `w_rf` | Risk-Free Weight | Risk-Free Weight (%) |
| `ret` | Exp. Return | Portfolio Exp. Return (%) |
| `sd` | Std. Dev. | Portfolio Std. Dev. (%) |
| `sharpe` | Sharpe Ratio | Sharpe Ratio |
| `rho` / `ρ` | Correlation (ρ) | Correlation (ρ) |
| `mvp` | Min. Variance Portfolio | — |
| `r1` | Asset 1 Exp. Return | — |
| `r2` | Asset 2 Exp. Return | — |
| `sd1` | Asset 1 Std. Dev. | — |
| `sd2` | Asset 2 Std. Dev. | — |
| `rf` | Risk-Free Rate | — |

---

## 8. Core Calculations

### 8.1 calculations.py — Functions

| Function | Inputs | Returns | Used In |
|---|---|---|---|
| `portfolio_stats(w, r1, sd1, r2, sd2, rho, rf)` | weight + asset params | `(Exp. Return, Std. Dev., Sharpe)` | frontier builder |
| `build_frontier(params)` | all frontier params | DataFrame: `w, ret, sd, sharpe, region` | Tab 1 |
| `build_cal(r_risky, sd_risky, rf)` | CAL params | DataFrame: `w_risky, w_rf, ret, sd, sharpe, region` | Tab 2 |
| `build_rho_frontiers(params, rho_list)` | frontier params + list of ρ values | `dict {ρ: DataFrame}` | Tab 3 |
| `find_mvp(frontier_df)` | frontier DataFrame | Row with minimum Std. Dev. | Metrics |
| `find_max_sharpe(frontier_df)` | frontier DataFrame | Row with maximum Sharpe Ratio | Metrics |

### 8.2 Key Formulas

| Formula | Expression |
|---|---|
| Portfolio Exp. Return | `E[Rp] = w × r1 + (1−w) × r2` |
| Portfolio Variance | `σ²p = w²σ₁² + (1−w)²σ₂² + 2w(1−w)ρσ₁σ₂` |
| Portfolio Std. Dev. | `σp = √(σ²p)` |
| Sharpe Ratio | `SR = (E[Rp] − rf) / σp` |
| CAL Exp. Return | `E[Rp] = rf + SR × σp` |
| CAL Std. Dev. | `σp = |w| × σ_risky` |
| MVP Weight (Asset 1) | `w* = (σ₂² − ρσ₁σ₂) / (σ₁² + σ₂² − 2ρσ₁σ₂)` |

### 8.3 DataFrame Columns

#### `frontier_df`

| Column | Type | Description |
|---|---|---|
| `w_A1` | float | Weight in Asset 1 (e.g. 0.94 = 94%) |
| `w_A2` | float | Weight in Asset 2 (= 1 − w_A1) |
| `ret` | float | Portfolio Exp. Return (%) |
| `sd` | float | Portfolio Std. Dev. (%) |
| `sharpe` | float | Sharpe Ratio |
| `region` | str | `"efficient"`, `"dominated"`, `"short_A1"`, `"long_A1"` |

#### `cal_df`

| Column | Type | Description |
|---|---|---|
| `w_risky` | float | Weight in risky asset |
| `w_rf` | float | Weight in risk-free asset (= 1 − w_risky) |
| `ret` | float | Portfolio Exp. Return (%) |
| `sd` | float | Portfolio Std. Dev. (%) |
| `sharpe` | float | Sharpe Ratio |
| `region` | str | `"short"`, `"long_no_lev"`, `"long_lev"` |

---

## 9. Future Enhancements

Items to consider for future versions:

| # | Enhancement | Complexity | Notes |
|---|---|---|---|
| 1 | Add 3rd asset to frontier | Medium | Extends to N-asset efficient frontier — needed for Assignment 2 (gold, international) |
| 2 | Export charts as PNG / PDF | Low | Plotly supports download buttons natively |
| 3 | Save / load parameter presets | Low | Store in JSON or Streamlit session state |
| 4 | Show tangency portfolio on CAL | Low | Where CAL is tangent to efficient frontier — foundation for CAPM in Module 2 |
| 5 | Expense ratio comparison (dominated fund) | Low | Real-world application from Lesson 1-5.4 |
| 6 | Module 2 CAPM extension | High | Beta, Security Market Line, alpha — after Module 2 lectures |
| 7 | Animate frontier as ρ slider moves | Medium | Real-time frontier shifting with correlation changes |
| 8 | Mobile responsive layout | Medium | Streamlit mobile layout adjustments |

---

## 10. Change Log

| Version | Date | Change |
|---|---|---|
| 1.0 | March 2026 | Initial design document — all requirements locked before coding begins |
| 1.1 | March 2026 | Layout fixes: Benchmark cards → HTML flexbox; EF Region → 2×3 metric rows; CAL metrics → 2 rows. Fixed HTML rendering bug (indented multi-line strings treated as code blocks by Streamlit Markdown parser) |
| 1.2 | March 2026 | UI restructure: moved all parameter sliders from sidebar into per-tab collapsible expanders. Sidebar now holds only short-selling toggle and quick presets. Each tab computes from session state after its own sliders run |
| 1.3 | March 2026 | Presets: renamed "Default"→"Baseline", removed "ρ=0" and "Assignment 1", added "ρ=1"; labels: "Asset 2"→"Asset 2 (more risky)" in expander and benchmark card; Chart 2 renamed "Long Only"; Chart 4 title removed "— Dominated"; all 4 charts now always show red dashed dominated line; Charts 3 & 4 show dominated annotation dynamically based on r1 vs r2 |
| 1.4 | March 2026 | Charts 1, 3, 4: replaced region-colored lines with efficient+dominated only (dominated defined by MVP position, not long/short composition); added 200% A1 and 200% A2 open-diamond markers to Charts 1, 3, 4; removed separate orange/purple short-selling region lines; renamed chart function to chart_frontier_long_only |
| 1.5 | March 2026 | Chart 1: restored 3-segment coloring (long only/short A1/long A1) all solid lines; Charts 2/3/4: filter frontier by weight_region before plotting efficient+dominated; Charts 3 & 4: pass filtered df to extreme markers so only in-range 200% markers appear; legend moved outside plot area to prevent overlap |
| 1.6 | March 2026 | Aligned dominated annotation across Charts 2, 3, 4 to data-driven check (rows exist in filtered df) instead of r1 vs r2 comparison; added dominated annotation to Chart 2 |
| 1.7 | March 2026 | Renamed all markers to dual-weight labels e.g. "(100% A1, 0% A2)"; fixed missing boundary markers in Charts 3 & 4 by widening filter to ±0.02 around w_A1=0 and w_A1=1; all marker helpers fully data-driven with no hardcoded r1/sd1/r2/sd2 params |
| 1.8 | March 2026 | Added Max Sharpe, Max Return (Long Only), and Max Return (Leveraged) markers to all 4 frontier charts via _add_key_portfolio_markers helper; leveraged marker conditional on allow_short |
| 1.9 | March 2026 | Verified and documented that all frontier chart markers and lines are fully data-driven; _add_key_portfolio_markers checks w_A1 against filtered df before plotting; CAL chart hardcoding confirmed intentional |
| 2.0 | March 2026 | Added chart_region column to build_frontier() for strict marker ownership (boundaries w=0/w=1 belong to chart2 only); _add_key_portfolio_markers uses chart_region lookup instead of magic number ranges; weight_region retained for line coloring and boundary-widened df filtering |
| 2.1 | March 2026 | Replaced all hardcoded ? tooltip texts in Efficient Frontier Region metrics with dynamic values derived from eff_summary and current parameters — direction words (increases/decreases), actual MVP values, and which asset has the higher return are all computed at runtime |
| 2.2 | March 2026 | Fixed Efficient Frontier Region tooltip endpoint labels — removed hardcoded "100% Asset 1/2" assumption; all 6 metric tooltips now derive endpoint from actual w_A1_range/w_A2_range end values (e.g. "200% A1 / -100% A2" when short-selling is on) |
| 2.3 | March 2026 | Replaced eff_summary tuple-based tooltip helpers with direct frontier_df row lookups — mvp row for start point, max_ret_lev (short-selling on) or max_ret_lo (long-only) for high-return endpoint; correct for all parameter combinations without any tuple ordering assumptions |
| 2.4 | March 2026 | Added mvp_row (ret.idxmin()) and hi_ret_row (ret.idxmax()) to eff_summary in calculations.py; EFR card display and tooltips now sourced entirely from these rows — Asset 1/2 Weight Range always shows MVP value → high-return endpoint value regardless of which asset has higher return |
| 2.5 | March 2026 | Added allow_short parameter to efficient_frontier_region(); filters to weight_region == "long_only" when False, uses full frontier df when True — EFR card now correctly reflects the active weight range in both modes; build_frontier() column assignments unchanged |

> **How to update:** Add a new row to this table whenever a design decision changes, noting what changed and why. Commit the updated `DESIGN.md` in the same pull request as the code change.

---

*Portfolio Frontier & CAL Explorer — Design & Requirements Document v1.0*
