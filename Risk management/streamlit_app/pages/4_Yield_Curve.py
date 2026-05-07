"""Yield Curve · Bundesbank Svensson with live β-shift visualisation."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import (apply_theme, hero, eyebrow, insight, footer,
                              COLORS)
from components.sidebar import render_sidebar
from components.data_loader import baseline_yield_curve, load_data_layer

st.set_page_config(page_title="Yield curve · Cross-cutting", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Bundesbank Yield Curve",
    eyebrow="Cross-cutting · Risk-free rate engine",
    deck="The Svensson (1994) zero-coupon curve drives every risk channel — "
         "Merton's discount factor, the Δr_10y macro shock, and the "
         "duration-based mark-to-market on the sovereign book. Move the β "
         "sliders in the sidebar to visualise the resulting curve shift.",
)

base = baseline_yield_curve()
if base.empty:
    st.error("Svensson cache missing.")
    st.stop()

from components.backend_path import setup
setup()
from svensson import historical_curve, shift_curve, curve_grid, zero_rate, params_from_row  # type: ignore

data = load_data_layer()
sven_df = data["svensson"]
brent_df = data["brent"]

base_params = historical_curve(sven_df.index[-1], sven_df)
shifted = shift_curve(
    base_params,
    dlevel=config["d_b0"], dslope=config["d_b1"],
    dcurv1=config["d_b2"], dcurv2=config["d_b3"],
)
mats = np.arange(0.25, 30.25, 0.25)
stressed = curve_grid(shifted, maturities=mats)

# =====================================================================
# Key-maturity KPI strip
# =====================================================================
eyebrow("Key-maturity rates · baseline vs stressed")

key_mats = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
kpi_cols = st.columns(len(key_mats), gap="small")
for col, m in zip(kpi_cols, key_mats):
    r_base = float(zero_rate(m, base_params, as_decimal=False))
    r_str  = float(zero_rate(m, shifted, as_decimal=False))
    delta_bp = (r_str - r_base) * 100   # rates already in pct → diff in pp → ×100 = bp
    label = f"{int(m*12)}m" if m < 1 else f"{int(m)}y"
    if abs(delta_bp) > 1e-2:
        col.metric(f"{label} rate",
                   f"{r_str:.3f}%",
                   f"{delta_bp:+.0f} bp")
    else:
        col.metric(f"{label} rate",
                   f"{r_base:.3f}%",
                   "baseline", delta_color="off")

st.caption(
    f"Bundesbank Svensson zero-coupon curve · last update "
    f"**{sven_df.index[-1].date()}** · Lookback "
    f"{len(sven_df):,} trading days "
    f"({sven_df.index.min().date()} → {sven_df.index.max().date()})"
)

st.divider()

# =====================================================================
# Curve plot — baseline + stressed + key historical reference
# =====================================================================
eyebrow("Curve topology · baseline / stressed / historical context")

# Historical reference curves (12m and 24m ago, if available)
ref_dates = []
today = sven_df.index[-1]
for months_back, label in [(12, "12m ago"), (36, "3y ago")]:
    target = today - pd.DateOffset(months=months_back)
    nearest_idx = sven_df.index.get_indexer([target], method="nearest")[0]
    if nearest_idx >= 0 and nearest_idx < len(sven_df):
        ref_dates.append((sven_df.index[nearest_idx], label))

fig = go.Figure()
# Historical references (lighter)
hist_colors = [COLORS["stone"], "#B0B0B0"]
for (d, lbl), c in zip(ref_dates, hist_colors):
    p = historical_curve(d, sven_df, method="ffill")
    rates = curve_grid(p, maturities=mats)
    fig.add_trace(go.Scatter(
        x=rates.index, y=rates.values,
        name=f"{lbl} ({d.date()})",
        line=dict(color=c, width=1.2, dash="dot"),
        opacity=0.7,
    ))

# Baseline today
fig.add_trace(go.Scatter(
    x=base["maturity"], y=base["rate_pct"],
    name=f"Baseline ({sven_df.index[-1].date()})",
    line=dict(color=COLORS["navy"], width=2.8),
))
# Stressed
shifted_at_zero = (config["d_b0"] == 0 and config["d_b1"] == 0
                   and config["d_b2"] == 0 and config["d_b3"] == 0)
if not shifted_at_zero:
    fig.add_trace(go.Scatter(
        x=stressed.index, y=stressed.values,
        name="Stressed (live β)",
        line=dict(color=COLORS["crimson"], width=2.5, dash="dash"),
    ))

fig.add_hline(y=0, line_dash="dot", line_color=COLORS["hairline"], line_width=1)
fig.update_layout(
    title=None,
    xaxis_title="Maturity [years]",
    yaxis_title="Zero rate [%]",
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# Factor returns time series (Brent + Δr_10y daily, 252-day window)
# =====================================================================
st.divider()
eyebrow("Macro factors · 252-day rolling history")

# Build the factor return series the same way factor_model does
from svensson import zero_rate as sv_zr  # type: ignore

# Δr_10y daily diff
rates_10y = pd.Series(
    [sv_zr(10.0, params_from_row(r), as_decimal=False)
     for _, r in sven_df.iterrows()],
    index=sven_df.index,
    name="r_10y_pct",
)
d_r_10y = rates_10y.diff().rename("d_r_10y_pp")

# Brent log-returns
if brent_df is not None and "Return_log" in brent_df.columns:
    r_brent = brent_df["Return_log"].rename("r_brent")
else:
    r_brent = pd.Series(dtype=float)

# Align last 252 trading days (use d_r_10y as base index)
factor_df = pd.concat({"r_brent": r_brent, "d_r_10y_pp": d_r_10y},
                      axis=1).dropna().tail(252)

if not factor_df.empty:
    fac_l, fac_r = st.columns(2, gap="medium")

    with fac_l:
        # Cumulative Brent
        cum_brent = factor_df["r_brent"].cumsum()
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=cum_brent.index, y=(np.exp(cum_brent) - 1) * 100,
            name="Cumulative Brent return",
            line=dict(color=COLORS["bright_blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(34,81,255,0.12)",
        ))
        fig_b.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
        fig_b.update_layout(
            title="Brent crude · cumulative return (252-day)",
            yaxis_title="Cumulative return [%]",
            height=320, showlegend=False,
        )
        st.plotly_chart(fig_b, use_container_width=True)

    with fac_r:
        # 10y rate level
        r10_window = rates_10y.tail(252)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=r10_window.index, y=r10_window.values,
            name="10y zero rate",
            line=dict(color=COLORS["mid_blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(3,75,111,0.10)",
        ))
        fig_r.update_layout(
            title="10-year zero rate · level (252-day)",
            yaxis_title="Rate [%]",
            height=320, showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Empirical statistics
    cov = factor_df.cov().to_numpy()
    corr = factor_df.corr().to_numpy()
    std_brent_yr = float(factor_df["r_brent"].std() * np.sqrt(252) * 100)
    std_dr_yr    = float(factor_df["d_r_10y_pp"].std() * np.sqrt(252))
    rho_emp      = float(corr[0, 1])

    eyebrow("Empirical factor statistics · 252-day window")
    s1, s2, s3, s4 = st.columns(4, gap="small")
    s1.metric("σ(Brent) annualised", f"{std_brent_yr:.1f}%",
              "log-return", delta_color="off")
    s2.metric("σ(Δr_10y) annualised", f"{std_dr_yr:.2f} pp",
              "daily diff", delta_color="off")
    s3.metric("ρ̂(Brent, Δr_10y)", f"{rho_emp:+.3f}",
              "empirical correlation", delta_color="off")
    s4.metric("Observations", f"{len(factor_df)}",
              "trading days", delta_color="off")

    if rho_emp > 0.05:
        insight(
            f"Brent and 10y rates show <strong>positive empirical "
            f"correlation ({rho_emp:+.2f})</strong> — consistent with the "
            f"inflation-channel narrative: oil-price spikes coincide with "
            f"hawkish rate moves. This is the fingerprint underlying the "
            f"Mahalanobis route in the macro→M mapping."
        )
    elif rho_emp < -0.05:
        insight(
            f"Brent and 10y rates show <strong>negative empirical "
            f"correlation ({rho_emp:+.2f})</strong> — likely reflecting "
            f"a regime where oil weakness coincides with risk-off rate cuts."
        )

st.divider()

# =====================================================================
# Svensson parameters table
# =====================================================================
eyebrow("Svensson parameters · baseline vs stressed")
params_table = pd.DataFrame({
    "Parameter": ["β₀ (level)", "β₁ (slope)", "β₂ (curvature 1)",
                  "β₃ (curvature 2)", "τ₁", "τ₂"],
    "Baseline": [base_params.beta0, base_params.beta1,
                 base_params.beta2, base_params.beta3,
                 base_params.tau1, base_params.tau2],
    "Stressed": [shifted.beta0, shifted.beta1, shifted.beta2,
                 shifted.beta3, shifted.tau1, shifted.tau2],
    "Δ":        [shifted.beta0 - base_params.beta0,
                 shifted.beta1 - base_params.beta1,
                 shifted.beta2 - base_params.beta2,
                 shifted.beta3 - base_params.beta3,
                 0.0, 0.0],
})
params_table["Baseline"] = params_table["Baseline"].round(4)
params_table["Stressed"] = params_table["Stressed"].round(4)
params_table["Δ"]        = params_table["Δ"].round(4)
st.dataframe(params_table, use_container_width=True, hide_index=True)

footer(
    "Validated against the Bundesbank's published curve to 2.66e-14% "
    "max absolute error · methodology in MODEL_ASSUMPTIONS.md §6 and §9."
)
