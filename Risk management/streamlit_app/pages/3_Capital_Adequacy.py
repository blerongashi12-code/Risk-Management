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


st.set_page_config(page_title="Capital Adequacy · CET1", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Capital Adequacy · CET1 Ratio",
    eyebrow="Tier 2 · Three-channel CET1 stress",
    deck="The same Macro shock drives three risk channels simultaneously: "
         "Loan-Book Provisions (PD + LGD via Vasicek), Sovereign Mark-to-"
         "Market (Duration), and Trading Book (Market-RWA + P&L). All three "
         "feed the CET1-Ratio numerator and denominator — the regulatory "
         "headline of capital adequacy.",
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
    delta_rate_10y_pp=config["d_r_10y_pp"] * 100,
    anchor=anchor,
    cov_factors=cov_factors,
    horizon_days=252,
)
m_used = mapping["m_hybrid"]
delta_r_pp = config["d_r_10y_pp"] * 100

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
