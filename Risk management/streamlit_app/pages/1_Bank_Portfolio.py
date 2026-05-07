"""Bank Portfolio · Vasicek/ASRF view of EBA-anchored European banks."""
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
                              COLORS, PALETTE_DISCRETE)
from components.sidebar import render_sidebar
from components.data_loader import load_data_layer
from components.methodology import render_loan_methodology
from components.backend_path import setup
setup()

from eba_loader import load_eba_universe                        # type: ignore
from macro_factor import (                                       # type: ignore
    anchor_from_eba, hybrid_mapping, factor_stats,
)
from vasicek import conditional_pd, asset_correlation            # type: ignore
from config import KAPPA_DOWNTURN_LGD                             # type: ignore

st.set_page_config(page_title="Bank portfolio · Vasicek", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Bank Portfolio View",
    eyebrow="Tier 2 · Regulatory · Vasicek/ASRF",
    deck="Top-10 European banks by IRB exposure under macroeconomic stress. "
         "Basel III IRB capital formulas applied segment-by-segment; the macro "
         "shock is mapped onto the Vasicek systematic factor M via an EBA "
         "stress-test anchor.",
)

# === Load universe + map macro shock to M =============================
@st.cache_data(ttl=24*3600, show_spinner="Loading EBA Transparency exercise …")
def _load_universe(top_n: int):
    return load_eba_universe(vintage="2025", top_n=top_n, prefer_real=True)

with st.sidebar:
    with st.expander("Universe configuration", expanded=False):
        top_n = st.slider("Top-N banks (by EAD)", 5, 30, 10, 1,
                          key="vasicek_top_n",
                          help="Selection from the EBA Transparency Exercise.")

universe = _load_universe(top_n)
is_real = universe.source.lower().startswith("eba transparency")


# Empirical factor covariance from cached Brent + Bundesbank-Δr_10y (252-day window).
# Falls back to a synthetic 2x2 only if the data layer is missing.
@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_factor_cov():
    data = load_data_layer()
    if data["brent"] is None or data["svensson"] is None:
        return None
    stats = factor_stats(data["brent"], data["svensson"], lookback=252)
    return {
        "sigma": stats["sigma"],
        "corr":  stats["corr"],
        "n_obs": stats["n_obs"],
        "cols":  stats["cols"],
    }


_FACTOR_STATS = _load_factor_cov()
if _FACTOR_STATS is None:
    cov_factors = np.array([[4e-4, 2e-5], [2e-5, 1e-4]])
    cov_source = "synthetic fallback (cache missing)"
    n_cov_obs = 0
    empirical_corr = float("nan")
else:
    cov_factors = _FACTOR_STATS["sigma"]
    cov_source = f"empirical · {_FACTOR_STATS['n_obs']} trading days"
    n_cov_obs = _FACTOR_STATS["n_obs"]
    empirical_corr = float(_FACTOR_STATS["corr"][0, 1])

# Vasicek/IRB standard horizon = 1 year (Basel III default)
H_DAYS = 252

anchor = anchor_from_eba("2025")
mapping = hybrid_mapping(
    delta_brent_log=config["d_brent"],
    delta_rate_10y_pp=config["d_r_10y_pp"] * 100,   # decimal -> pp
    anchor=anchor,
    cov_factors=cov_factors,
    horizon_days=H_DAYS,
)

# === Mapping diagnostics strip ========================================
eyebrow("Macro shock mapped to Vasicek systematic factor M")
m1, m2, m3, m4, m5 = st.columns(5, gap="small")
m1.metric("ΔBrent (log)", f"{config['d_brent']:+.2f}",
          f"{(np.exp(config['d_brent'])-1)*100:+.0f}%", delta_color="off")
m2.metric("Δr_10y", f"{config['d_r_10y_pp']*100:+.0f} bp",
          f"{config['d_r_10y_pp']*100:+.2f} pp", delta_color="off")
m3.metric("M (anchor)", f"{mapping['m_anchor']:+.2f}",
          "EBA 2025 calibration", delta_color="off")
if n_cov_obs > 0:
    m4_caption = f"Mahalanobis · ρ̂ = {empirical_corr:+.2f}"
else:
    m4_caption = "Mahalanobis (synthetic Σ)"
m4.metric("M (data)", f"{mapping['m_data']:+.2f}",
          m4_caption, delta_color="off")
m5.metric("M (hybrid)", f"{mapping['m_hybrid']:+.2f}",
          "primary", delta_color="off")

m_used = mapping["m_hybrid"]

source_tag = "real EBA Transparency 2025 data" if is_real else "synthetic anchor data"
if abs(m_used) < 1e-3:
    insight(
        f"<strong>No macro shock applied.</strong> Metrics below reflect the "
        f"regulatory baseline derived from {source_tag}. Move the sliders in "
        f"the sidebar to see the full transmission chain "
        f"<strong>Macro → (PD, LGD) → IRB Capital → ΔRWA</strong>."
    )
else:
    direction = "adverse" if m_used < 0 else "benign"
    lgd_uplift_pct = KAPPA_DOWNTURN_LGD * abs(min(m_used, 0)) * 100
    insight(
        f"Mapped to <strong>M = {m_used:+.2f}</strong> ({direction}) on top of "
        f"{source_tag}. Two risk channels are stressed: "
        f"<strong>PD</strong> via Vasicek conditional-PD "
        f"P(default | M) = N((N⁻¹(PD) − √ρ · M) / √(1−ρ)), and "
        f"<strong>LGD</strong> via downturn-LGD with κ = {KAPPA_DOWNTURN_LGD:.2f} "
        f"(LGD lifted by ≈ {lgd_uplift_pct:.0f}% of base under adverse shock)."
    )

st.divider()

# === Methodology disclosure boxes (Critique 2 + 3) ====================
render_loan_methodology(kappa_lgd=KAPPA_DOWNTURN_LGD)

st.divider()

# === Compute baseline + stressed metrics across all banks =============
baseline_rows = []
stressed_rows = []
for bank_name, portfolio in universe.banks.items():
    base_kpi = portfolio.portfolio_kpis(confidence=0.999)
    base_kpi["Bank"] = bank_name
    baseline_rows.append(base_kpi)

    if abs(m_used) > 1e-9:
        s_kpi = portfolio.stressed_kpis(z_factor=m_used, confidence=0.999)
        s_kpi["Bank"] = bank_name
        stressed_rows.append(s_kpi)

baseline_df = pd.DataFrame(baseline_rows).set_index("Bank")
stressed_df = (pd.DataFrame(stressed_rows).set_index("Bank")
               if stressed_rows else None)

# === Aggregate KPIs ===================================================
eyebrow(f"EU top-{universe.n_banks} aggregate")
total_ead = baseline_df["total_ead"].sum()
total_el  = baseline_df["el_eur"].sum()
total_ul  = baseline_df["ul_eur"].sum()
total_rwa = baseline_df["rwa"].sum()

if stressed_df is not None:
    s_total_el  = stressed_df["el_eur"].sum()
    s_total_rwa = stressed_df["rwa"].sum()
    delta_el  = s_total_el - total_el
    delta_rwa = s_total_rwa - total_rwa
else:
    s_total_el = total_el
    s_total_rwa = total_rwa
    delta_el  = 0.0
    delta_rwa = 0.0

a1, a2, a3, a4, a5, a6 = st.columns(6, gap="small")
a1.metric("Σ EAD", f"€{total_ead/1e9:.0f} bn",
          f"{universe.n_banks} banks", delta_color="off")
a2.metric("EL baseline", f"€{total_el/1e9:.1f} bn",
          f"{total_el/total_ead*100:.2f}% of EAD", delta_color="off")
a3.metric("EL stressed", f"€{s_total_el/1e9:.1f} bn",
          f"{delta_el/1e9:+.1f} bn")
a4.metric("UL (Basel capital)", f"€{total_ul/1e9:.0f} bn",
          f"{total_ul/total_ead*100:.2f}% of EAD", delta_color="off")
a5.metric("RWA baseline", f"€{total_rwa/1e9:.0f} bn",
          f"density {total_rwa/total_ead*100:.0f}%", delta_color="off")
a6.metric("RWA stressed", f"€{s_total_rwa/1e9:.0f} bn",
          f"{delta_rwa/1e9:+.0f} bn")

st.divider()

# === Bank league table ================================================
eyebrow(f"League table · top-{universe.n_banks} EU banks")

table_rows = []
for bank_name in baseline_df.index:
    row = {
        "Bank":         bank_name,
        "EAD bn":       baseline_df.loc[bank_name, "total_ead"] / 1e9,
        "EL base bn":   baseline_df.loc[bank_name, "el_eur"] / 1e9,
        "EL stress bn": (stressed_df.loc[bank_name, "el_eur"] / 1e9
                         if stressed_df is not None else
                         baseline_df.loc[bank_name, "el_eur"] / 1e9),
        "RWA base bn":  baseline_df.loc[bank_name, "rwa"] / 1e9,
        "RWA stress bn": (stressed_df.loc[bank_name, "rwa"] / 1e9
                          if stressed_df is not None else
                          baseline_df.loc[bank_name, "rwa"] / 1e9),
        "RWA dens":     baseline_df.loc[bank_name, "rwa_density"],
    }
    row["Δ EL bn"]  = row["EL stress bn"] - row["EL base bn"]
    row["Δ RWA bn"] = row["RWA stress bn"] - row["RWA base bn"]
    table_rows.append(row)

league = pd.DataFrame(table_rows).set_index("Bank")
league = league.sort_values("EAD bn", ascending=False)
display_league = league.copy()
display_league["EAD bn"]        = display_league["EAD bn"].round(0).astype(int)
display_league["EL base bn"]    = display_league["EL base bn"].round(2)
display_league["EL stress bn"]  = display_league["EL stress bn"].round(2)
display_league["RWA base bn"]   = display_league["RWA base bn"].round(0).astype(int)
display_league["RWA stress bn"] = display_league["RWA stress bn"].round(0).astype(int)
display_league["RWA dens"]      = (display_league["RWA dens"] * 100).round(1).astype(str) + "%"
display_league["Δ EL bn"]       = display_league["Δ EL bn"].round(2)
display_league["Δ RWA bn"]      = display_league["Δ RWA bn"].round(0).astype(int)

st.dataframe(display_league, use_container_width=True, height=420)

st.divider()

# === Bank-by-bank EL bar (baseline vs stressed) =======================
eyebrow("Expected loss by bank · baseline vs. stressed")

ranked = league.sort_values("EL stress bn", ascending=True)
fig = go.Figure()
fig.add_trace(go.Bar(
    y=ranked.index, x=ranked["EL base bn"],
    name="Baseline",
    orientation="h",
    marker_color=COLORS["mid_blue"],
    text=[f"{v:.1f}" for v in ranked["EL base bn"]],
    textposition="outside",
    textfont=dict(size=10, color=COLORS["stone"]),
))
fig.add_trace(go.Bar(
    y=ranked.index, x=ranked["EL stress bn"],
    name="Stressed",
    orientation="h",
    marker_color=COLORS["crimson"],
    text=[f"{v:.1f}" for v in ranked["EL stress bn"]],
    textposition="outside",
    textfont=dict(size=10, color=COLORS["navy"]),
))
fig.update_layout(
    title=None,
    xaxis_title="Expected loss [bn EUR]",
    height=460,
    barmode="group",
    bargap=0.25,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# === Capital Bridge · Wirkungskette pro Bank ==========================
eyebrow("Wirkungskette · Capital Bridge per bank")
st.caption(
    "End-to-end transmission chain  Macro shock → conditional PD + downturn "
    "LGD → IRB Capital. Sequential activation: first the PD shift (LGD held "
    "at base), then the LGD shift (PD already stressed). The two contributions "
    "sum exactly to ΔK by construction."
)

bridge_bank = st.selectbox(
    "Bank",
    list(universe.banks.keys()),
    index=0,
    format_func=lambda n: f"{n}  (€{universe.banks[n].total_ead/1e9:.0f} bn EAD)",
    label_visibility="collapsed",
    key="bridge_bank",
)

bridge_portfolio = universe.banks[bridge_bank]
if abs(m_used) > 1e-9:
    bridge = bridge_portfolio.capital_bridge(
        z_factor=m_used,
        kappa_lgd=KAPPA_DOWNTURN_LGD,
        confidence=0.999,
    )

    def _bridge_chart(metric_label: str, base: float, dpd: float, dlgd: float,
                      stress: float, color_axis: str = "Capital requirement K") -> go.Figure:
        """Build a 4-bar waterfall for a given decomposition triplet."""
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=[f"{metric_label} base",
               "+ Δ from PD shift",
               "+ Δ from LGD shift",
               f"{metric_label} stress"],
            text=[f"€{base/1e9:.2f} bn",
                  f"€{dpd/1e9:+.2f} bn",
                  f"€{dlgd/1e9:+.2f} bn",
                  f"€{stress/1e9:.2f} bn"],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["navy"]),
            y=[base/1e9, dpd/1e9, dlgd/1e9, stress/1e9],
            connector={"line": {"color": COLORS["hairline"], "width": 1}},
            increasing={"marker": {"color": COLORS["crimson"]}},
            decreasing={"marker": {"color": COLORS["teal"]}},
            totals     ={"marker": {"color": COLORS["navy"]}},
        ))
        wf.update_layout(
            title=f"{bridge_bank} · {metric_label} decomposition",
            yaxis_title=f"{color_axis} [bn EUR]",
            height=380,
            showlegend=False,
        )
        return wf

    def _share_caption(metric_label: str, total: float, dpd: float, dlgd: float):
        """Render share-of-stress caption if there is a meaningful change."""
        if abs(total) < 1.0:
            return
        pd_share = dpd / total * 100
        lgd_share = dlgd / total * 100
        st.caption(
            f"Of the total Δ{metric_label} = €{total/1e9:+.1f} bn, the "
            f"PD channel contributes **{pd_share:.0f}%** and the LGD "
            f"channel **{lgd_share:.0f}%**."
        )

    tab_K, tab_RWA, tab_EL = st.tabs([
        "Capital (K)",
        "RWA",
        "Expected Loss (EL)",
    ])

    with tab_K:
        bcol_l, bcol_r = st.columns([3, 2], gap="medium")
        with bcol_l:
            st.plotly_chart(
                _bridge_chart("K", bridge["K_base"], bridge["delta_K_pd"],
                              bridge["delta_K_lgd"], bridge["K_stress"],
                              "Capital requirement K"),
                use_container_width=True,
            )
        with bcol_r:
            st.markdown(r"""
**Formula** &nbsp;
$K = (L_\alpha - \text{PD}\!\cdot\!\text{LGD}) \cdot \text{MA}(M_{\text{eff}})$
&nbsp;·&nbsp; $L_\alpha$ ist das ASRF-99.9%-Loss-Quantil aus Vasicek 2002.

**Reading the bridge.** Stage 1 lifts only PD via Conditional-PD;
LGD bleibt auf Baseline. Stage 2 hebt zusätzlich LGD via Downturn-LGD-
Funktion. Beide Beiträge sind exakt additiv zu ΔK.
""")
            _share_caption("K", bridge["delta_K"],
                           bridge["delta_K_pd"], bridge["delta_K_lgd"])

    with tab_RWA:
        bcol_l, bcol_r = st.columns([3, 2], gap="medium")
        with bcol_l:
            st.plotly_chart(
                _bridge_chart("RWA", bridge["rwa_base"], bridge["delta_rwa_pd"],
                              bridge["delta_rwa_lgd"], bridge["rwa_stress"],
                              "Risk-weighted assets"),
                use_container_width=True,
            )
        with bcol_r:
            st.markdown(r"""
**Formula** &nbsp; $\text{RWA} = K \cdot 12{.}5 \cdot \text{EAD}$

**Reading.** RWA folgt strukturell der gleichen Decomposition wie K
(EAD ist konstant pro Segment unter dem V1-Modell). Diese Größe ist
relevant für die regulatorische Eigenkapitalquote — sie steht im
**Nenner der CET1-Ratio**.
""")
            _share_caption("RWA", bridge["delta_rwa"],
                           bridge["delta_rwa_pd"], bridge["delta_rwa_lgd"])

    with tab_EL:
        bcol_l, bcol_r = st.columns([3, 2], gap="medium")
        with bcol_l:
            st.plotly_chart(
                _bridge_chart("EL", bridge["el_base"], bridge["delta_el_pd"],
                              bridge["delta_el_lgd"], bridge["el_stress"],
                              "Expected loss"),
                use_container_width=True,
            )
        with bcol_r:
            st.markdown(r"""
**Formula** &nbsp; $\text{EL} = \text{PD} \cdot \text{LGD} \cdot \text{EAD}$
&nbsp;·&nbsp; ΔEL exakt zerlegbar in (siehe Methodology box):

$$\Delta\text{EL} = (\text{PD}^*\!-\!\text{PD})\,\text{LGD}\,\text{EAD}
+ \text{PD}^*\,(\text{LGD}^*\!-\!\text{LGD})\,\text{EAD}$$

**Reading.** EL ist der Erwartungswert (P&L-Vorsorge). Die Capital-
Charge im Tab links hingegen ist die **Unexpected-Loss-Komponente**
oberhalb von EL — beide werden gemeinsam für die Risiko-Bilanz
gebraucht.
""")
            _share_caption("EL", bridge["delta_el"],
                           bridge["delta_el_pd"], bridge["delta_el_lgd"])

    # Per-segment breakdown (collapsible — useful for auditors/validators)
    with st.expander("Segment-level decomposition", expanded=False):
        seg = bridge["per_segment"].copy()
        disp = pd.DataFrame({
            "Segment":      seg["name"],
            "EAD bn":       (seg["ead"] / 1e9).round(1),
            "PD base":      (seg["pd_base"] * 100).map(lambda v: f"{v:.2f}%"),
            "PD stress":    (seg["pd_stress"] * 100).map(lambda v: f"{v:.2f}%"),
            "LGD base":     (seg["lgd_base"] * 100).map(lambda v: f"{v:.0f}%"),
            "LGD stress":   (seg["lgd_stress"] * 100).map(lambda v: f"{v:.0f}%"),
            "K base m":     (seg["K_base"] / 1e6).round(0).astype(int),
            "ΔK PD m":      (seg["dK_pd"] / 1e6).round(0).astype(int),
            "ΔK LGD m":     (seg["dK_lgd"] / 1e6).round(0).astype(int),
            "K stress m":   (seg["K_stress"] / 1e6).round(0).astype(int),
        })
        st.dataframe(disp, use_container_width=True, hide_index=True,
                     height=260)
else:
    st.info("Apply a macro shock in the sidebar to view the capital "
            "bridge decomposition (3 tabs: Capital · RWA · Expected Loss).")

st.divider()

# === Per-bank drilldown ==============================================
eyebrow("Bank drilldown · exposure-class breakdown")
sel_bank = st.selectbox(
    "Bank",
    list(universe.banks.keys()),
    index=0,
    label_visibility="collapsed",
    key="vasicek_drill_bank",
)

drill_portfolio = universe.banks[sel_bank]
drill_base = drill_portfolio.baseline_metrics(confidence=0.999)
if abs(m_used) > 1e-9:
    drill_stress = drill_portfolio.stressed_metrics(z_factor=m_used,
                                                    confidence=0.999)
else:
    drill_stress = None

col_a, col_b = st.columns([3, 2], gap="medium")

with col_a:
    eyebrow(f"{sel_bank} · segment metrics")
    cols_show = ["name", "exposure_class", "ead", "pd", "rho",
                 "el_eur", "ul_eur", "rwa"]
    seg_disp = drill_base[cols_show].copy()
    seg_disp["ead"]    = (seg_disp["ead"] / 1e9).round(1)
    seg_disp["pd"]     = (seg_disp["pd"] * 100).round(2).astype(str) + "%"
    seg_disp["rho"]    = (seg_disp["rho"] * 100).round(1).astype(str) + "%"
    seg_disp["el_eur"] = (seg_disp["el_eur"] / 1e6).round(0).astype(int)
    seg_disp["ul_eur"] = (seg_disp["ul_eur"] / 1e9).round(2)
    seg_disp["rwa"]    = (seg_disp["rwa"] / 1e9).round(1)
    seg_disp.columns = ["Segment", "Class", "EAD bn", "PD", "ρ",
                        "EL m", "UL bn", "RWA bn"]
    st.dataframe(seg_disp, use_container_width=True, hide_index=True,
                 height=260)

    if drill_stress is not None:
        eyebrow(f"{sel_bank} · stressed conditional PDs")
        stress_disp = drill_stress[["name", "pd_baseline", "pd_stressed",
                                    "delta_pd", "el_eur", "rwa"]].copy()
        stress_disp["pd_baseline"] = (stress_disp["pd_baseline"] * 100).round(2).astype(str) + "%"
        stress_disp["pd_stressed"] = (stress_disp["pd_stressed"] * 100).round(2).astype(str) + "%"
        stress_disp["delta_pd"]    = (stress_disp["delta_pd"] * 100).round(2).astype(str) + " pp"
        stress_disp["el_eur"]      = (stress_disp["el_eur"] / 1e6).round(0).astype(int)
        stress_disp["rwa"]         = (stress_disp["rwa"] / 1e9).round(1)
        stress_disp.columns = ["Segment", "PD base", "PD stress",
                               "Δ PD", "EL stress m", "RWA stress bn"]
        st.dataframe(stress_disp, use_container_width=True, hide_index=True,
                     height=260)

with col_b:
    eyebrow(f"{sel_bank} · EAD composition")
    ead_share = drill_base["ead"] / drill_base["ead"].sum()
    fig_pie = go.Figure(go.Pie(
        labels=drill_base["name"],
        values=drill_base["ead"] / 1e9,
        hole=0.55,
        marker=dict(colors=PALETTE_DISCRETE,
                    line=dict(color=COLORS["white"], width=2)),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(family="Inter", size=10, color=COLORS["navy"]),
    ))
    fig_pie.update_layout(
        title=None,
        showlegend=False,
        height=380,
        margin=dict(l=10, r=10, t=20, b=20),
        annotations=[dict(
            text=f"€{drill_portfolio.total_ead/1e9:.0f} bn",
            font=dict(family="Source Serif Pro", size=18, color=COLORS["navy"]),
            showarrow=False,
        )],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# === Conditional PD curve =============================================
eyebrow("Conditional-PD response curve · how a segment reacts to M")

curve_seg = st.selectbox(
    "Segment",
    list(drill_base["name"]),
    index=0,
    key="vasicek_curve_seg",
    label_visibility="collapsed",
)
seg_row = drill_base[drill_base["name"] == curve_seg].iloc[0]
seg_pd  = float(seg_row["pd"])
seg_rho = float(seg_row["rho"])

z_grid = np.linspace(-3.5, 3.5, 141)
pd_curve = np.array([float(conditional_pd(seg_pd, seg_rho, z)) for z in z_grid])

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(
    x=z_grid, y=pd_curve * 100,
    mode="lines", name="PD(M)",
    line=dict(color=COLORS["navy"], width=2.5),
))
fig_curve.add_hline(y=seg_pd * 100, line_dash="dot",
                    line_color=COLORS["mid_blue"],
                    annotation_text=f"Baseline PD {seg_pd*100:.2f}%",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color=COLORS["mid_blue"]))
fig_curve.add_vline(x=m_used, line_dash="dash",
                    line_color=COLORS["crimson"],
                    annotation_text=f"M = {m_used:+.2f}",
                    annotation_position="top",
                    annotation_font=dict(size=10, color=COLORS["crimson"]))
stressed_pd = float(conditional_pd(seg_pd, seg_rho, m_used))
fig_curve.add_trace(go.Scatter(
    x=[m_used], y=[stressed_pd * 100],
    mode="markers", showlegend=False,
    marker=dict(color=COLORS["crimson"], size=11,
                line=dict(color=COLORS["white"], width=2)),
))
fig_curve.update_layout(
    title=f"{sel_bank} · {curve_seg} · PD(M)",
    xaxis_title="Systematic factor M",
    yaxis_title="Conditional PD [%]",
    height=380,
    showlegend=False,
)
st.plotly_chart(fig_curve, use_container_width=True)

footer(
    f"Source: {universe.source} · Anchor: {anchor.label} "
    f"(z = {anchor.z_factor:+.2f}) · Factor Σ: {cov_source} · "
    f"LGD per Basel F-IRB defaults · "
    f"PD = observed default ratio (Item 2520512 ÷ Item 2520502)"
)
