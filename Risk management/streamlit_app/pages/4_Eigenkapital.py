"""Capital Adequacy · CET1-Ratio under three-channel stress.

Architektur (siehe Annahmen-Page):
  Numerator (CET1):
    CET1_stress = CET1_base
                  − ΔEL_loan_book           (Loan-Book Provisions-Hit)
                  + Δ_sovereign_MtM_signed    (Sovereign FVOCI/AfS via OCI)
                  + Δ_tb_pnl_signed           (Trading-Book P&L change)

  Denominator (Total RWA):
    RWA_stress = RWA_base
                 + ΔRWA_credit_loan_book   (from Vasicek capital_bridge)
                 + ΔRWA_market_TB          (FRTB-style RWA uplift)
                 + RWA_operational_base    (unchanged)

  CET1-Ratio = CET1 / Total-RWA — Basel-III Pillar-1-Threshold = 4.5%,
  Pillar-2 + CCB ≈ 7-10.5%.
"""
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
from components.data_loader import load_data_layer
from components.backend_path import setup
setup()

from config import (KAPPA_DOWNTURN_LGD, EBA_RAW_DIR)                # type: ignore
from eba_loader import (                                            # type: ignore
    load_eba_universe, load_bank_directory,
    parse_capital_overview, parse_sovereign_csv,
    sovereign_maturity_ladder, rate_shock_pnl,
    trading_book_stress, cet1_ratio_bridge,
)
from macro_factor import (anchor_from_eba, hybrid_mapping,           # type: ignore
                          factor_stats)


# Basel-III thresholds for visualisation
PILLAR1_MIN_CET1   = 0.045   # 4.5% — Pillar 1 minimum CET1 ratio
CCB_BUFFER         = 0.025   # 2.5% Capital Conservation Buffer
SII_BUFFER         = 0.010   # 1.0% Systemically Important Institution buffer (avg)
TARGET_CET1_RATIO  = PILLAR1_MIN_CET1 + CCB_BUFFER  # 7.0% basic guidance
SREP_TARGET        = TARGET_CET1_RATIO + SII_BUFFER  # 8.0% typical SREP target


st.set_page_config(page_title="Eigenkapital · CET1-Wirkung", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Eigenkapital-Wirkung · CET1-Quote",
    eyebrow="Tab 4 · 3-Channel CET1 stress · regulatorische Headline",
    deck="Derselbe Macro-Schock treibt drei Risiko-Channels simultan: "
         "Loan-Book Provisions (PD + LGD via Vasicek), Sovereign "
         "Mark-to-Market (Duration), und Trading-Book (Market-RWA + P&L). "
         "Alle drei wirken auf den Zähler und Nenner der CET1-Quote — "
         "der regulatorischen Headline für Kapitaladäquanz. Inklusive "
         "Threshold-Analyse (welche Banken fallen unter 4.5% / 7% / 8%) "
         "und CET1-Sensitivity-Curve über M-Scan.",
)

# === Load data + macro shock =========================================
@st.cache_data(ttl=24*3600, show_spinner="Loading EBA capital + RWA data …")
def _load_capital(top_n: int):
    cap = parse_capital_overview(EBA_RAW_DIR / "tr_oth.csv", period=202506)
    bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
    return cap, bank_dir


@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_universe(top_n: int):
    return load_eba_universe(vintage="2025", top_n=top_n, prefer_real=True)


@st.cache_data(ttl=24*3600, show_spinner="Loading sovereign maturity ladder …")
def _load_sov_pnl():
    sov_raw = parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv", period=202506)
    return sovereign_maturity_ladder(sov_raw, period=202506)


@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_factor_cov():
    data = load_data_layer()
    if data["brent"] is None or data["svensson"] is None:
        return None
    return factor_stats(data["brent"], data["svensson"], lookback=252)


with st.sidebar:
    with st.expander("Universe configuration", expanded=False):
        top_n = st.slider("Top-N banks (by EAD)", 5, 30, 10, 1,
                          key="capadeq_top_n")

cap_df, bank_dir = _load_capital(top_n)
universe = _load_universe(top_n)
sov_mat = _load_sov_pnl()
fac_stats = _load_factor_cov()

# Macro shock → M
cov_factors = fac_stats["sigma"] if fac_stats else np.array([[4e-4, 2e-5], [2e-5, 1e-4]])
anchor = anchor_from_eba("2025")
mapping = hybrid_mapping(
    delta_brent_log=config["d_brent"],
    delta_rate_10y_pp=config["d_r_10y_pp"],   # already in pp
    anchor=anchor,
    cov_factors=cov_factors,
    horizon_days=252,
)
# Honour Single-Factor mode (user drives M directly)
if config.get("m_direct") is not None:
    m_used = float(config["m_direct"])
    mapping["m_hybrid"] = m_used
else:
    m_used = mapping["m_hybrid"]
delta_r_pp = config["d_r_10y_pp"]   # already in pp

# Compute the three channels per bank in the universe -----------------
# Channel 1 — loan book bridge (from the universe BankPortfolio objects)
loan_bridges: dict[str, dict] = {}
name_to_lei: dict[str, str] = {}
for bank_name, portfolio in universe.banks.items():
    matches = bank_dir[bank_dir["bank_name"] == bank_name]
    if len(matches) == 0:
        continue
    lei = matches["lei"].iloc[0]
    name_to_lei[bank_name] = lei
    if abs(m_used) > 1e-9:
        loan_bridges[lei] = portfolio.capital_bridge(
            z_factor=m_used, kappa_lgd=KAPPA_DOWNTURN_LGD, confidence=0.999,
            rho_multiplier=float(config.get("rho_mult", 1.0)),
            lgd_calibration=float(config.get("lgd_calibration", 1.0)),
            stress_exponent=float(config.get("stress_exponent", 1.0)),
            cap_mode=str(config.get("cap_mode", "hard")),
        )

# Channel 2 — sovereign rate-shock P&L per bank
sov_pnl_df = (rate_shock_pnl(sov_mat, delta_r_pp=delta_r_pp)
              if abs(delta_r_pp) > 1e-3 else
              pd.DataFrame(columns=["LEI_Code", "delta_pnl_eur"]))
sov_pnl_lookup: dict[str, float] = dict(
    zip(sov_pnl_df.get("LEI_Code", []), sov_pnl_df.get("delta_pnl_eur", []))
)

# Channel 3 — Trading Book stress
tb_stress_df = trading_book_stress(cap_df, m_factor=m_used)

# Restrict capital_df to universe banks only
universe_leis = list(name_to_lei.values())
cap_universe = cap_df[cap_df["LEI_Code"].isin(universe_leis)].copy()
tb_stress_universe = tb_stress_df[tb_stress_df["LEI_Code"].isin(universe_leis)].copy()

# Compute the three-channel CET1 bridge -------------------------------
bridge_df = cet1_ratio_bridge(
    cap_universe, loan_bridges, sov_pnl_lookup, tb_stress_universe,
)
bridge_df = bridge_df.merge(bank_dir[["lei", "bank_name"]],
                            left_on="LEI_Code", right_on="lei", how="left")


# === Aggregate KPI strip =============================================
eyebrow(f"EU top-{universe.n_banks} aggregate · CET1 adequacy")

n_banks = len(bridge_df)
cet1_total_base   = bridge_df["cet1_base"].sum()
cet1_total_stress = bridge_df["cet1_stress"].sum()
rwa_total_base    = bridge_df["rwa_total_base"].sum()
rwa_total_stress  = bridge_df["rwa_total_stress"].sum()
ratio_base   = cet1_total_base / rwa_total_base   if rwa_total_base > 0 else 0
ratio_stress = cet1_total_stress / rwa_total_stress if rwa_total_stress > 0 else 0

# Banks below 4.5% Pillar 1 under stress
breaches = bridge_df[bridge_df["cet1_ratio_stress"] < PILLAR1_MIN_CET1]

a1, a2, a3, a4, a5 = st.columns(5, gap="small")
a1.metric("Σ CET1 base", f"€{cet1_total_base/1e9:.0f} bn",
          f"{n_banks} banks", delta_color="off")
a2.metric("Σ Total RWA base", f"€{rwa_total_base/1e9:.0f} bn",
          f"density {rwa_total_base/sum(p.total_ead for p in universe.banks.values())*100:.0f}%",
          delta_color="off")
a3.metric("Aggregate CET1 ratio · base",
          f"{ratio_base*100:.2f}%",
          f"vs SREP target {SREP_TARGET*100:.1f}%",
          delta_color="off")
if abs(m_used) > 1e-9:
    a4.metric("Aggregate CET1 ratio · stress",
              f"{ratio_stress*100:.2f}%",
              f"{(ratio_stress-ratio_base)*100:+.2f} pp")
    a5.metric("Banks below 4.5% Pillar 1",
              f"{len(breaches)} / {n_banks}",
              "post-stress" if len(breaches) > 0 else "all banks adequate",
              delta_color="off")
else:
    a4.metric("Aggregate CET1 ratio · stress", "—",
              "no shock applied", delta_color="off")
    a5.metric("Banks below 4.5% Pillar 1", "—",
              "no shock applied", delta_color="off")

# === Insight box =====================================================
if abs(m_used) < 1e-3:
    insight(
        "<strong>No macro shock applied.</strong> Move the sliders in "
        "the sidebar to see how the three transmission channels — "
        "loan-book provisions, sovereign mark-to-market, and trading-"
        "book P&amp;L + market-RWA — combine into the regulatory "
        "CET1 Ratio."
    )
else:
    direction = "adverse" if m_used < 0 else "benign"
    insight(
        f"At <strong>M = {m_used:+.2f}</strong> ({direction}), the "
        f"aggregate CET1 ratio moves from <strong>{ratio_base*100:.2f}%</strong> "
        f"to <strong>{ratio_stress*100:.2f}%</strong> "
        f"(<strong>{(ratio_stress-ratio_base)*100:+.2f} pp</strong>). "
        f"The headline number masks bank-specific divergence — see "
        f"the league table and per-bank waterfall below."
    )

# Methodology disclaimer
st.caption(
    "**Methodik-Hinweis.** Loan-Book ΔRWA wird via Vasicek-IRB-99.9%-"
    "Quantile auf den bedingten PD/LGD berechnet — entspricht regulatorischer "
    "Pillar-1-Capital-under-Stress (Basel III), **nicht** EBA-Stress-Test-"
    "realisierten P&L-Pfaden über 3 Jahre. Erwartung: ΔRatio in dieser "
    "Sicht ist *konservativer* (höher) als in EBA-Methodology, weil Tail-"
    "Quantile auf bereits gestresste Parameter angewandt werden. Siehe "
    "Annahmen-Page Approximation A-06."
)

st.divider()

# === League table ====================================================
eyebrow(f"League table · CET1 ratio movement (top-{n_banks} EU banks)")

if abs(m_used) > 1e-9:
    league = bridge_df.sort_values("cet1_ratio_stress").copy()

    def _flag(row):
        if pd.isna(row["cet1_ratio_stress"]):
            return "—"
        if row["cet1_ratio_stress"] < PILLAR1_MIN_CET1:
            return "● breach P1"
        if row["cet1_ratio_stress"] < TARGET_CET1_RATIO:
            return "◐ below CCB"
        if row["cet1_ratio_stress"] < SREP_TARGET:
            return "○ below SREP"
        return "✓ adequate"

    league["Status"] = league.apply(_flag, axis=1)
    display_l = pd.DataFrame({
        "Bank":             league["bank_name"],
        "CET1 base bn":     (league["cet1_base"] / 1e9).round(1),
        "CET1 stress bn":   (league["cet1_stress"] / 1e9).round(1),
        "RWA base bn":      (league["rwa_total_base"] / 1e9).round(0).astype(int),
        "RWA stress bn":    (league["rwa_total_stress"] / 1e9).round(0).astype(int),
        "Ratio base %":     (league["cet1_ratio_base"] * 100).round(2),
        "Ratio stress %":   (league["cet1_ratio_stress"] * 100).round(2),
        "Δ Ratio pp":       league["delta_cet1_ratio_pp"].round(2),
        "Status":           league["Status"],
    })
    st.dataframe(display_l, use_container_width=True, hide_index=True,
                 height=420)
    st.caption(
        f"Status: ✓ ≥ SREP ({SREP_TARGET*100:.1f}%) · "
        f"○ < SREP · ◐ < CCB ({TARGET_CET1_RATIO*100:.1f}%) · "
        f"● < Pillar 1 minimum ({PILLAR1_MIN_CET1*100:.1f}%)"
    )
else:
    display_base = pd.DataFrame({
        "Bank":           bridge_df["bank_name"],
        "CET1 bn":        (bridge_df["cet1_base"] / 1e9).round(1),
        "RWA bn":         (bridge_df["rwa_total_base"] / 1e9).round(0).astype(int),
        "RWA-Credit bn":  (cap_universe.set_index("LEI_Code")
                           ["rwa_credit_eur"].reindex(bridge_df["LEI_Code"])
                           .values / 1e9).round(0).astype(int),
        "RWA-Market bn":  (cap_universe.set_index("LEI_Code")
                           ["rwa_market_eur"].reindex(bridge_df["LEI_Code"])
                           .values / 1e9).round(0).astype(int),
        "RWA-Op bn":      (cap_universe.set_index("LEI_Code")
                           ["rwa_operational_eur"].reindex(bridge_df["LEI_Code"])
                           .values / 1e9).round(0).astype(int),
        "Ratio %":        (bridge_df["cet1_ratio_base"] * 100).round(2),
    }).sort_values("CET1 bn", ascending=False)
    st.dataframe(display_base, use_container_width=True, hide_index=True,
                 height=420)

st.divider()

# === Per-bank CET1 ratio waterfall ====================================
eyebrow("CET1 ratio waterfall · per bank")

bank_options = sorted(bridge_df["bank_name"].dropna().tolist())
sel_bank_name = st.selectbox(
    "Bank",
    bank_options,
    index=0,
    format_func=lambda n: (
        f"{n}  (CET1 ratio base "
        f"{bridge_df[bridge_df['bank_name']==n]['cet1_ratio_base'].iloc[0]*100:.2f}%)"
    ),
    label_visibility="collapsed",
    key="cet1_drill_bank",
)
sel_row = bridge_df[bridge_df["bank_name"] == sel_bank_name].iloc[0]

if abs(m_used) > 1e-9:
    # Build a compact waterfall: Base ratio -> Numerator effects -> Denominator effect -> Stress ratio
    base_ratio = sel_row["cet1_ratio_base"]
    rwa_b = sel_row["rwa_total_base"]
    rwa_s = sel_row["rwa_total_stress"]

    # Decompose ratio change into numerator and denominator effects
    # (additive in pp — exact decomposition is multiplicative but pp is intuitive)
    cet1_b = sel_row["cet1_base"]
    cet1_s = sel_row["cet1_stress"]
    # Stage 1: only numerator changes (RWA stays at base)
    ratio_after_num = cet1_s / rwa_b if rwa_b > 0 else 0
    # Stage 2: also denominator changes
    ratio_after_den = cet1_s / rwa_s if rwa_s > 0 else 0

    d_loan_pp   = (sel_row["delta_cet1_loan"] / rwa_b) * 100
    d_sov_pp    = (sel_row["delta_cet1_sovereign"] / rwa_b) * 100
    d_tb_pp     = (sel_row["delta_cet1_tb"] / rwa_b) * 100
    # Aggregate denominator effect (after numerator already shifted)
    d_rwa_pp    = (ratio_after_den - ratio_after_num) * 100

    wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=["CET1 Ratio base",
           "− Loan-Book Δ (Provisions)",
           "+ Sovereign Δ (OCI)",
           "+ Trading Book Δ (P&L)",
           "Δ from RWA expansion",
           "CET1 Ratio stress"],
        text=[f"{base_ratio*100:.2f}%",
              f"{d_loan_pp:+.2f} pp",
              f"{d_sov_pp:+.2f} pp",
              f"{d_tb_pp:+.2f} pp",
              f"{d_rwa_pp:+.2f} pp",
              f"{sel_row['cet1_ratio_stress']*100:.2f}%"],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["navy"]),
        y=[base_ratio*100,
           d_loan_pp, d_sov_pp, d_tb_pp, d_rwa_pp,
           sel_row["cet1_ratio_stress"]*100],
        connector={"line": {"color": COLORS["hairline"], "width": 1}},
        increasing={"marker": {"color": COLORS["teal"]}},
        decreasing={"marker": {"color": COLORS["crimson"]}},
        totals     ={"marker": {"color": COLORS["navy"]}},
    ))
    # Threshold lines
    wf.add_hline(y=PILLAR1_MIN_CET1*100, line_dash="solid",
                 line_color=COLORS["crimson"], line_width=1,
                 annotation_text=f"Pillar 1 minimum {PILLAR1_MIN_CET1*100:.1f}%",
                 annotation_position="top right",
                 annotation_font=dict(size=10, color=COLORS["crimson"]))
    wf.add_hline(y=TARGET_CET1_RATIO*100, line_dash="dot",
                 line_color=COLORS["amber"], line_width=1,
                 annotation_text=f"P1 + CCB {TARGET_CET1_RATIO*100:.1f}%",
                 annotation_position="top right",
                 annotation_font=dict(size=10, color=COLORS["amber"]))
    wf.add_hline(y=SREP_TARGET*100, line_dash="dot",
                 line_color=COLORS["mid_blue"], line_width=1,
                 annotation_text=f"SREP target {SREP_TARGET*100:.1f}%",
                 annotation_position="top right",
                 annotation_font=dict(size=10, color=COLORS["mid_blue"]))

    wf.update_layout(
        title=f"{sel_bank_name} · CET1 Ratio decomposition",
        yaxis_title="CET1 Ratio [%]",
        height=460,
        showlegend=False,
    )
    st.plotly_chart(wf, use_container_width=True)

    # Numerical breakdown
    detail_l, detail_r = st.columns(2, gap="medium")

    with detail_l:
        eyebrow("Numerator (CET1 Capital) · in EUR")
        num_table = pd.DataFrame([
            ("CET1 base",                   sel_row["cet1_base"]/1e9),
            ("Loan-Book Provisions impact", sel_row["delta_cet1_loan"]/1e9),
            ("Sovereign MtM (via OCI)",     sel_row["delta_cet1_sovereign"]/1e9),
            ("Trading-Book P&L change",     sel_row["delta_cet1_tb"]/1e9),
            ("CET1 stress",                 sel_row["cet1_stress"]/1e9),
        ], columns=["Channel", "EUR bn"])
        num_table["EUR bn"] = num_table["EUR bn"].map(lambda v: f"{v:+.2f}")
        st.dataframe(num_table, use_container_width=True, hide_index=True,
                     height=210)

    with detail_r:
        eyebrow("Denominator (Total RWA) · in EUR")
        den_table = pd.DataFrame([
            ("RWA base (Credit + Market + Op)", sel_row["rwa_total_base"]/1e9),
            ("ΔRWA Credit (Vasicek IRB stress)", sel_row["delta_rwa_credit"]/1e9),
            ("ΔRWA Market (FRTB-style uplift)", sel_row["delta_rwa_market"]/1e9),
            ("RWA Operational (unchanged)",     0.0),
            ("RWA stress",                       sel_row["rwa_total_stress"]/1e9),
        ], columns=["Component", "EUR bn"])
        den_table["EUR bn"] = den_table["EUR bn"].map(lambda v: f"{v:+.2f}" if abs(v) < 50 else f"{v:+.0f}")
        st.dataframe(den_table, use_container_width=True, hide_index=True,
                     height=210)

    # Final ratio outcome
    delta_pp = sel_row["delta_cet1_ratio_pp"]
    breach_msg = ""
    if sel_row["cet1_ratio_stress"] < PILLAR1_MIN_CET1:
        breach_msg = (f" — **breach** of the {PILLAR1_MIN_CET1*100:.1f}% Pillar 1 "
                      "minimum under stress")
    elif sel_row["cet1_ratio_stress"] < TARGET_CET1_RATIO:
        breach_msg = (f" — **below** the {TARGET_CET1_RATIO*100:.1f}% Pillar 1 + CCB "
                      "guidance")
    st.markdown(
        f"**Outcome.** CET1 ratio moves from "
        f"<strong>{sel_row['cet1_ratio_base']*100:.2f}%</strong> "
        f"to <strong>{sel_row['cet1_ratio_stress']*100:.2f}%</strong> "
        f"({delta_pp:+.2f} pp){breach_msg}.",
        unsafe_allow_html=True,
    )
else:
    st.info("Apply a macro shock in the sidebar to view the CET1 "
            "ratio waterfall.")

st.divider()

# =====================================================================
# NEW · Threshold-Analyse (welche Banken fallen unter Regulatorik?)
# =====================================================================
eyebrow("Threshold-Analyse · regulatorische CET1-Mindestquoten")

st.caption(
    "Alle Banken im Aggregat, sortiert nach **Post-Stress-CET1-Quote**. "
    "Farb-Code zeigt Threshold-Breaches: "
    "<span style='color:#A52F4D;font-weight:600;'>● rot = unter 4.5% (Pillar-1-Minimum)</span> · "
    "<span style='color:#C9A227;font-weight:600;'>● gelb = unter 7.0% (inkl. CCB)</span> · "
    "<span style='color:#034B6F;font-weight:600;'>● blau = unter 8.0% (typischer SREP)</span> · "
    "<span style='color:#00A9A5;font-weight:600;'>● grün = über 8.0% (komfortabel)</span>",
    unsafe_allow_html=True,
)


def _classify_threshold(ratio: float) -> tuple[str, str]:
    """Return (label, hex-color) based on CET1 ratio post-stress."""
    if ratio < PILLAR1_MIN_CET1:
        return "Pillar-1 Breach (< 4.5%)", "#A52F4D"
    elif ratio < TARGET_CET1_RATIO:
        return "CCB Breach (< 7.0%)", "#C9A227"
    elif ratio < SREP_TARGET:
        return "SREP Breach (< 8.0%)", "#034B6F"
    else:
        return "Komfortabel (> 8.0%)", "#00A9A5"


# Build threshold table
thr_rows = []
for _, row in bridge_df.iterrows():
    label, color = _classify_threshold(row["cet1_ratio_stress"])
    thr_rows.append({
        "Bank":            row["bank_name"],
        "CET1 vorher":     f"{row['cet1_ratio_base']*100:.2f}%",
        "CET1 nachher":    f"{row['cet1_ratio_stress']*100:.2f}%",
        "Δ pp":            f"{row['delta_cet1_ratio_pp']:+.2f}",
        "CET1 Capital bn": round(row["cet1_base"]/1e9, 1),
        "RWA bn":          round(row["rwa_total_base"]/1e9, 1),
        "Status":          label,
        "_color":          color,
    })
thr_df = pd.DataFrame(thr_rows).sort_values(
    "CET1 nachher",
    key=lambda s: s.str.rstrip("%").astype(float),
)

# Threshold summary KPIs (count per category)
status_counts = thr_df["Status"].value_counts()
ts1, ts2, ts3, ts4 = st.columns(4, gap="small")
ts1.metric("Pillar-1 Breach (< 4.5%)",
           int(status_counts.get("Pillar-1 Breach (< 4.5%)", 0)),
           "Banken unter regulatorischem Minimum",
           delta_color="off")
ts2.metric("CCB Breach (< 7.0%)",
           int(status_counts.get("CCB Breach (< 7.0%)", 0)),
           "Capital-Conservation-Buffer verletzt",
           delta_color="off")
ts3.metric("SREP Breach (< 8.0%)",
           int(status_counts.get("SREP Breach (< 8.0%)", 0)),
           "typischer Supervisor-Target verletzt",
           delta_color="off")
ts4.metric("Komfortabel (> 8.0%)",
           int(status_counts.get("Komfortabel (> 8.0%)", 0)),
           "über SREP-Target", delta_color="off")

# Display the threshold table (drop helper color column)
disp_thr = thr_df.drop(columns="_color")
st.dataframe(disp_thr, use_container_width=True, hide_index=True,
             height=420)

st.divider()

# =====================================================================
# NEW · CET1-Sensitivity-Curve · scan M und plotte ΔCET1-Quote
# =====================================================================
eyebrow("CET1-Sensitivity-Curve · Systemfaktor-Scan M ∈ [−3, +3]")

st.caption(
    "Stress-Pfad: bei welchem M würde der Aggregat unter 4.5% / 7% / 8% "
    "fallen? Aggregat = aktuelle Bank-Auswahl. Berechnung pro M-Punkt "
    "via Vasicek-Capital-Bridge (Loan-Channel) + EBA-Anker-Kopplung "
    "(Sovereign-MtM + Trading-Book)."
)


@st.cache_data(ttl=24*3600, show_spinner="Computing CET1 sensitivity curve …")
def _sensitivity_curve(
    selected_bank_names: tuple, kappa_lgd: float, rho_mult: float,
    cet1_base_total: float, rwa_base_total: float,
    sov_pnl_per_pp: float, tb_pnl_per_m: float, tb_rwa_per_m: float,
    lgd_cal: float = 1.0, stress_exp: float = 1.0, cap_mode: str = "hard",
):
    """Scan M and compute aggregate CET1 ratio at each grid point."""
    # Aggregate portfolio (subset of universe)
    sub_portfolio = sum(
        (universe.banks[n].segments for n in selected_bank_names),
        [],
    )
    from vasicek import BankPortfolio                                  # type: ignore
    agg = BankPortfolio("scan")
    agg.segments = sub_portfolio

    m_grid = np.linspace(-3.0, 3.0, 31)
    ratios = []
    for m in m_grid:
        if abs(m) < 1e-6:
            ratios.append(cet1_base_total / rwa_base_total)
            continue
        br = agg.capital_bridge(z_factor=float(m), kappa_lgd=kappa_lgd,
                                confidence=0.999, rho_multiplier=rho_mult,
                                lgd_calibration=lgd_cal,
                                stress_exponent=stress_exp,
                                cap_mode=cap_mode)
        # Coupled (Brent, Δr) from EBA anchor
        anchor_z = anchor.z_factor
        if abs(anchor_z) > 1e-9:
            scale = -m / abs(anchor_z) * np.sign(-anchor_z)
            d_r_m  = anchor.rate_10y_pp * scale   # pp
        else:
            d_r_m = 0.0
        # CET1 numerator: subtract loan ΔEL + add sov MtM + add TB P&L
        sov_dp  = sov_pnl_per_pp * d_r_m
        tb_pnl  = tb_pnl_per_m * (-m if m < 0 else 0.0)  # only adverse hits
        cet1_m  = cet1_base_total - br["delta_el"] + sov_dp + tb_pnl
        # RWA denominator: add loan ΔRWA + TB-RWA-uplift
        tb_drwa = tb_rwa_per_m * (-m if m < 0 else 0.0)
        rwa_m   = rwa_base_total + br["delta_rwa"] + tb_drwa
        ratios.append(cet1_m / rwa_m if rwa_m > 0 else 0.0)
    return m_grid, np.array(ratios)


# Pre-compute static coupling coefficients for fast scan
_anchor_r_pp = anchor.rate_10y_pp
_sov_pnl_per_pp = float(sov_pnl_df["delta_pnl_eur"].sum()) / max(delta_r_pp, 1e-9) \
                    if abs(delta_r_pp) > 1e-9 else 0.0
# If user has zero current shock, fall back to direct computation
if abs(_sov_pnl_per_pp) < 1.0:
    # Compute sov P&L per pp directly via a unit shock
    _unit_sov = rate_shock_pnl(
        sovereign_maturity_ladder(parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv",
                                                       period=202506), period=202506),
        delta_r_pp=1.0)
    _unit_sov = _unit_sov[_unit_sov["LEI_Code"].isin(universe_leis)]
    _sov_pnl_per_pp = float(_unit_sov["delta_pnl_eur"].sum())

# TB sensitivities: from existing trading_book_stress at current M, normalised
if abs(m_used) > 1e-9:
    _tb_pnl_per_m = float(tb_stress_universe["delta_tb_pnl_eur"].sum()) / abs(m_used) \
                      if m_used < 0 else 0.0
    _tb_rwa_per_m = float(tb_stress_universe["delta_rwa_market_eur"].sum()) / abs(m_used) \
                      if m_used < 0 else 0.0
else:
    # Compute unit-shock TB stress
    _unit_tb = trading_book_stress(cap_universe, m_factor=-1.0)
    _tb_pnl_per_m = float(_unit_tb["delta_tb_pnl_eur"].sum())
    _tb_rwa_per_m = float(_unit_tb["delta_rwa_market_eur"].sum())

_cet1_base_total = float(bridge_df["cet1_base"].sum())
_rwa_base_total  = float(bridge_df["rwa_total_base"].sum())

m_grid_sens, ratio_sens = _sensitivity_curve(
    tuple(selected_banks) if "selected_banks" in dir() else tuple(universe.banks.keys()),
    KAPPA_DOWNTURN_LGD, float(config.get("rho_mult", 1.0)),
    _cet1_base_total, _rwa_base_total,
    _sov_pnl_per_pp, _tb_pnl_per_m, _tb_rwa_per_m,
    lgd_cal=float(config.get("lgd_calibration", 1.0)),
    stress_exp=float(config.get("stress_exponent", 1.0)),
    cap_mode=str(config.get("cap_mode", "hard")),
)

# Plot
fig_sens = go.Figure()
fig_sens.add_trace(go.Scatter(
    x=m_grid_sens, y=ratio_sens * 100,
    mode="lines+markers",
    line=dict(color=COLORS["navy"], width=2.8),
    marker=dict(size=6, color=COLORS["navy"]),
    name="CET1 ratio (Aggregat)",
))
# Threshold lines
fig_sens.add_hline(y=4.5, line_dash="dot", line_color="#A52F4D",
                   annotation_text="4.5% · Pillar 1", annotation_position="right",
                   annotation_font=dict(size=10, color="#A52F4D"))
fig_sens.add_hline(y=7.0, line_dash="dot", line_color="#C9A227",
                   annotation_text="7.0% · inkl. CCB", annotation_position="right",
                   annotation_font=dict(size=10, color="#C9A227"))
fig_sens.add_hline(y=8.0, line_dash="dot", line_color="#034B6F",
                   annotation_text="8.0% · SREP", annotation_position="right",
                   annotation_font=dict(size=10, color="#034B6F"))
# Vertical line at current M
fig_sens.add_vline(x=m_used, line_dash="dash", line_color=COLORS["crimson"],
                   annotation_text=f"aktuell M = {m_used:+.2f}",
                   annotation_position="top",
                   annotation_font=dict(size=10, color=COLORS["crimson"]))
# Baseline marker at M=0
ratio_at_zero = ratio_sens[np.argmin(np.abs(m_grid_sens))] * 100
fig_sens.add_annotation(x=0, y=ratio_at_zero,
                         text=f"Baseline {ratio_at_zero:.2f}%",
                         xshift=10, yshift=8,
                         showarrow=False,
                         font=dict(size=10, color=COLORS["stone"]))
fig_sens.update_layout(
    title=f"CET1-Quote in Abhängigkeit von M · Aggregat über "
          f"{universe.n_banks} Banken",
    xaxis_title="Systemfaktor M (adverse ⟵ ⟶ benigne)",
    yaxis_title="CET1-Quote [%]",
    height=440, showlegend=False,
)
st.plotly_chart(fig_sens, use_container_width=True)

# Find breaking points
def _find_breach(ratio_arr, m_grid_arr, thr_pct):
    """Find the M-value at which ratio first drops below threshold."""
    below = ratio_arr * 100 < thr_pct
    if not below.any():
        return None
    # Find first index where below=True moving from M=0 leftward
    zero_idx = int(np.argmin(np.abs(m_grid_arr)))
    for i in range(zero_idx, -1, -1):
        if below[i]:
            return float(m_grid_arr[i])
    return None


bp_45 = _find_breach(ratio_sens, m_grid_sens, 4.5)
bp_70 = _find_breach(ratio_sens, m_grid_sens, 7.0)
bp_80 = _find_breach(ratio_sens, m_grid_sens, 8.0)

bp_l, bp_r = st.columns([1, 1], gap="medium")
with bp_l:
    st.markdown("**Breaking-Points** · M-Wert bei dem die Aggregat-CET1-Quote unter die Schwelle fällt:")
    bp_table = pd.DataFrame([
        ("4.5% (Pillar-1)", f"M = {bp_45:+.2f}σ" if bp_45 is not None else "nicht erreicht im Scan-Bereich"),
        ("7.0% (inkl. CCB)", f"M = {bp_70:+.2f}σ" if bp_70 is not None else "nicht erreicht im Scan-Bereich"),
        ("8.0% (SREP)",      f"M = {bp_80:+.2f}σ" if bp_80 is not None else "nicht erreicht im Scan-Bereich"),
    ], columns=["Threshold", "Breaking-Point M"])
    st.dataframe(bp_table, use_container_width=True, hide_index=True, height=140)

with bp_r:
    insight(
        f"<strong>Lesart.</strong> Bei M = −2.5 entspricht ungefähr der "
        f"EBA-2025-Adverse-Härte. Falls die Sensitivity-Kurve dort bereits unter "
        f"7% liegt, würde die Aggregat-Bank den Capital-Conservation-Buffer "
        f"unter dem EBA-Stress reißen. Aktuelles Aggregat: "
        f"<strong>{ratio_at_zero:.2f}% Baseline</strong>."
    )

st.divider()

# === Methodology footer =============================================
with st.expander("Methodology · CET1 three-channel architecture",
                 expanded=False):
    st.markdown(r"""
**Numerator (CET1 capital):**

$$\text{CET1}^{\text{stress}} = \text{CET1}^{\text{base}}
- \Delta\text{EL}_{\text{loan}}
+ \Delta\text{MtM}_{\text{sov}}
+ \Delta\text{P\&L}_{\text{TB}}$$

- $\Delta\text{EL}_{\text{loan}}$ — Vasicek-Capital-Bridge ΔEL pro Bank
  (PD-Channel + LGD-Channel via downturn-LGD), reduziert CET1 als
  zusätzliche Provisions-Aufwendung.
- $\Delta\text{MtM}_{\text{sov}}$ — signiert; bei Rate-Up negativ über alle
  Maturity-Buckets, fließt für FVOCI-Bestände durch OCI direkt zur CET1.
  V1 modelliert die gesamte Sovereign-Exposure-Maturity-Ladder als FVOCI-
  ähnlich (konservativ — überschätzt den OCI-Channel).
- $\Delta\text{P\&L}_{\text{TB}}$ — Trading-Book-Earnings-Haircut bei
  adversem Stress (κ_PnL = 0.50/2.5 ≈ 0.20 pro |M|).

**Denominator (Total RWA):**

$$\text{RWA}^{\text{stress}} = \text{RWA}_{\text{cr}}^{\text{base}} + \Delta\text{RWA}_{\text{cr}}
+ \text{RWA}_{\text{mr}}^{\text{base}} + \Delta\text{RWA}_{\text{mr}}
+ \text{RWA}_{\text{op}}^{\text{base}}$$

- $\Delta\text{RWA}_{\text{cr}}$ — Vasicek-Capital-Bridge ΔRWA (= K · 12.5 · EAD,
  K via IRB-Formel mit gestressten PD/LGD).
- $\Delta\text{RWA}_{\text{mr}}$ — FRTB-style Multiplier auf Market-RWA
  (κ_RWA = 0.30/2.5 ≈ 0.12 pro |M|), stellt VaR/SVaR-Multiplier-Anstieg ab.
- Operational-RWA bleibt konstant (out-of-stress in V1).

**Was nicht modelliert wird:**
- Banking-Book-Securities-FVOCI-Channel separat von Sovereign (Limitation:
  EBA-Public-Disclosure unterscheidet Asset-Kategorie, aber nicht
  Instrument-Typ within Kategorie)
- Equity-Holdings (kleine Position, unter 3% bei Top-EU-Banks)
- Operational-Risk-Stress (Pillar 2)
- Sovereign-Spread-Risiko separat von Rate-Shift
- IFRS-9-Lifetime-EL-Migration (Stage 2 / 3)

**Threshold-Linien im Waterfall:**
- $\textcolor{#A52F4D}{4{,}5\%}$ — Pillar-1-CET1-Minimum (CRR Art. 92)
- $\textcolor{#C9A227}{7{,}0\%}$ — Pillar 1 + Capital Conservation Buffer
- $\textcolor{#034B6F}{8{,}0\%}$ — typischer SREP-Target (mit SII-Aufschlag)
""")

footer(
    f"Three-channel CET1 architecture · Vasicek IRB + Sovereign Duration "
    f"+ Trading Book FRTB-style · M = {m_used:+.2f} · "
    f"κ_LGD = {KAPPA_DOWNTURN_LGD:.2f} · See Annahmen page for full "
    f"approximations inventory."
)
