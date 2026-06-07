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

from components.theme import (tab_breadcrumb, apply_theme, hero, eyebrow, insight, footer,
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
from macro_factor import (anchor_from_eba,                          # type: ignore
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
    eyebrow="Tab 4 · 3-Channel CET1 stress · 2-Faktor-Modell · 10 IRB-Banken",
    deck="Die zwei Macro-Faktoren (ΔBrent + Δr_10y) treiben drei "
         "Risiko-Channels simultan: Loan-Book-Provisions (sektor-"
         "differenzierte ΔPD und ΔLGD pro Exposure-Klasse), Sovereign-"
         "Mark-to-Market (Modified-Duration · Δr_10y), und Trading-"
         "Book (Market-RWA + P&L). Alle drei wirken auf Zähler "
         "(CET1) und Nenner (RWA) der regulatorischen Headline-"
         "Kennzahl. Daten: bank-spezifische Pillar-3 EU-CR6 (31.12.2024) "
         "für PDs/LGDs (CRR Art. 180-konform, 10/10 Banken Pillar-3-"
         "verifiziert). Inklusive Threshold-Analyse "
         "(Pillar-1 4.5% / CCB 7% / SREP 8%) und Bank-Drilldown.",
)

tab_breadcrumb(4)
# === Load data + macro shock =========================================
@st.cache_data(ttl=24*3600, show_spinner="Loading EBA capital + RWA data …")
def _load_capital(top_n: int):
    cap = parse_capital_overview(EBA_RAW_DIR / "tr_oth.csv", period=202506)
    bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
    return cap, bank_dir


@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_universe(top_n: int):
    # Top-10-Filter via pillar3_bank_pd_lgd.csv (bank-spezifische Pillar-3 31.12.2024)
    from eba_pd_loader import filter_universe_to_top10                # type: ignore
    u = load_eba_universe(vintage="2025", top_n=None, prefer_real=True)
    return filter_universe_to_top10(u)


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
        st.caption(
            "Feste Datenbasis: **10 kuratierte EU-IRB-Banken** mit "
            "bank-spezifischen Pillar-3-PDs/LGDs (31.12.2024, "
            "`pillar3_bank_pd_lgd.csv`). Keine variable Top-N-Auswahl — "
            "alle Tabs nutzen dasselbe 10-Banken-Universe."
        )

top_n = 10  # konstant: das Universe ist auf die kuratierten 10 Banken fixiert
cap_df, bank_dir = _load_capital(top_n)
universe = _load_universe(top_n)
sov_mat = _load_sov_pnl()
fac_stats = _load_factor_cov()

# === 2-Faktor-Stress: Bridge direkt mit dem 2-Faktor-Modell rechnen ====
# Statt Vasicek-M-Mapping: direkte Anwendung der sektor-differenzierten
# β-Sensitivitäten aus two_factor_stress (Brent + Δr_10y separat).
# WICHTIG: Wir nutzen die 2-Faktor-Bridge ``capital_bridge_2factor``, die
# baseline und stressed PD/LGD direkt aus den β-Sensitivitäten ableitet.
# Die alte ``capital_bridge(z_factor=0)`` führte über ``conditional_pd(s.pd,
# rho, 0)`` eine implizite Vasicek-Median-Transformation durch, welche die
# PD um typisch 30–50 % reduzierte und so EL falsch herabsetzte (siehe
# Diskussions-Bug 2026-05-29).
from two_factor_stress import (apply_stress_to_universe,             # type: ignore
                                reset_universe_to_baseline,
                                capital_bridge_2factor)

_d_brent      = float(config["d_brent"])
_d_r_10y_pp   = float(config["d_r_10y_pp"])
_sens_overrides = config.get("sensitivity_overrides")

# Universe sicher in Baseline-Zustand (für Re-Renders aus Cache)
reset_universe_to_baseline(universe)
m_used = 0.0
_is_stressed = (abs(_d_brent) > 1e-6) or (abs(_d_r_10y_pp) > 1e-6)
delta_r_pp = _d_r_10y_pp

# Legacy-Werte für nachgelagerte UI-Caches
cov_factors = fac_stats["sigma"] if fac_stats else np.array([[4e-4, 2e-5], [2e-5, 1e-4]])
anchor = anchor_from_eba("2025")
mapping = {"m_hybrid": 0.0, "m_anchor": 0.0, "m_data": 0.0}

# Compute the three channels per bank in the universe -----------------
# Channel 1 — loan book bridge via 2-Faktor-Modell (KEIN conditional_pd-Shift)
loan_bridges: dict[str, dict] = {}
name_to_lei: dict[str, str] = {}
for bank_name, portfolio in universe.banks.items():
    matches = bank_dir[bank_dir["bank_name"] == bank_name]
    if len(matches) == 0:
        continue
    lei = matches["lei"].iloc[0]
    name_to_lei[bank_name] = lei
    if _is_stressed:
        loan_bridges[lei] = capital_bridge_2factor(
            portfolio, _d_brent, _d_r_10y_pp,
            confidence=0.999,
            rho_multiplier=float(config.get("rho_mult", 1.0)),
            lgd_calibration=float(config.get("lgd_calibration", 1.0)),
            override_betas=_sens_overrides,
        )

# Stress jetzt auch in-place auf das Universum applizieren, damit
# nachgelagerte Funktionen (cet1_ratio_bridge etc.) konsistent mit
# gestressten Segmenten arbeiten.
if _is_stressed:
    apply_stress_to_universe(universe, _d_brent, _d_r_10y_pp,
                             override_betas=_sens_overrides)

# Channel 2 — sovereign rate-shock P&L per bank
sov_pnl_df = (rate_shock_pnl(sov_mat, delta_r_pp=delta_r_pp)
              if abs(delta_r_pp) > 1e-3 else
              pd.DataFrame(columns=["LEI_Code", "delta_pnl_eur"]))
sov_pnl_lookup: dict[str, float] = dict(
    zip(sov_pnl_df.get("LEI_Code", []), sov_pnl_df.get("delta_pnl_eur", []))
)

# Channel 3 — Trading Book stress
tb_stress_df = trading_book_stress(cap_df, m_factor=0.0)

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
if _is_stressed:
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
if not _is_stressed:
    insight(
        "<strong>Kein Macro-Schock aktiv.</strong> Bewege die zwei "
        "Slider in der Sidebar (ΔBrent + Δr_10y), um zu sehen, wie die "
        "drei Transmissions-Kanäle — Loan-Book-Provisions (sektor-"
        "differenzierte ΔPD und ΔLGD), Sovereign-Mark-to-Market "
        "(Duration · Δr), Trading-Book-P&amp;L + Market-RWA — zur "
        "regulatorischen CET1-Quote zusammenwirken."
    )
else:
    insight(
        f"Unter dem 2-Faktor-Schock ΔBrent = <strong>"
        f"{_d_brent:+.2f}</strong> und Δr_10y = <strong>"
        f"{_d_r_10y_pp:+.2f} pp</strong> bewegt sich die aggregierte "
        f"CET1-Quote von <strong>{ratio_base*100:.2f}%</strong> "
        f"auf <strong>{ratio_stress*100:.2f}%</strong> "
        f"(<strong>{(ratio_stress-ratio_base)*100:+.2f} pp</strong>). "
        f"Die System-Headline verbirgt bank-spezifische "
        f"Differenzierung — siehe Liga-Tabelle und Waterfall unten."
    )

# Methodology disclaimer
st.caption(
    "**Methodik-Hinweis.** Loan-Book ΔRWA wird über die Basel-III-"
    "IRB-Capital-Formel auf das 99.9%-Quantil berechnet — mit den "
    "**bereits 2-Faktor-gestressten PD/LGD-Werten** als Inputs. Das "
    "entspricht regulatorischer Pillar-1-Capital-under-Stress (BCBS "
    "2017), nicht EBA-Stress-Test-realisierten P&L-Pfaden über 3 Jahre. "
    "Erwartung: ΔRatio in dieser Sicht ist konservativer (höher) als in "
    "EBA-Methodology, weil Tail-Quantile auf bereits gestresste "
    "Parameter angewandt werden. Siehe MODEL_ASSUMPTIONS.md, A-04/A-05."
)

st.divider()

# === League table ====================================================
eyebrow(f"League table · CET1 ratio movement (top-{n_banks} EU banks)")

if _is_stressed:
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

if _is_stressed:
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
# CET1-Sensitivity-Curve · 2-Faktor-Scan über Δr_10y
# =====================================================================
eyebrow("Sensitivitäts-Analyse · wie reagiert die CET1-Quote auf "
        "den Zinsschock?")

st.markdown(
    '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
    'border-left:2px solid #051C2C;padding:0.95rem 1.2rem;'
    'border-radius:4px;margin:0.4rem 0 1.0rem 0;color:#051C2C;'
    'font-size:0.88rem;line-height:1.65;">'
    '<strong>Was diese Kurve zeigt:</strong> Wir scannen den 10y-Zins-Schock '
    'von −1 bis +5 Prozentpunkten und rechnen für jeden Punkt das volle '
    '2-Faktor-Modell durch (alle drei Stress-Kanäle, alle 10 Banken '
    'aggregiert). Der Brent-Schock bleibt auf dem aktuellen Sidebar-Wert. '
    '<br><br>'
    '<strong>Wozu?</strong> Du siehst auf einen Blick: ab welchem '
    'Δr-Schock fällt die CET1-Quote unter die regulatorischen Schwellen '
    '4.5 % (Pillar 1), 7.0 % (inkl. Kapitalerhaltungs-Puffer CCB) und '
    '8.0 % (typische SREP-Zielmarke)? Das sind die sogenannten '
    '<em>Breaking Points</em> — bei welchem Stress-Niveau die Bank '
    'erstmals in Aufsichts-Reichweite kommt.'
    '</div>',
    unsafe_allow_html=True,
)


# --- 2-Faktor-Sensitivity-Scan: vary Δr, hold ΔBrent fixed ---------
@st.cache_data(ttl=24*3600, show_spinner="Berechne 2-Faktor-Sensitivity-Curve …")
def _two_factor_sensitivity(
    selected_bank_names: tuple,
    d_brent_fixed: float,
    cet1_base_total: float,
    rwa_base_total: float,
    sov_pnl_per_pp: float,
):
    """Scan Δr_10y while applying 2-factor stress with fixed ΔBrent.

    Returns dr_grid, cet1_ratios (in decimals).
    """
    from two_factor_stress import (apply_stress_to_universe,           # type: ignore
                                    reset_universe_to_baseline)
    from vasicek import BankPortfolio                                   # type: ignore

    dr_grid = np.linspace(-1.0, 5.0, 25)  # −100bp bis +500bp
    ratios = []

    # 1) Baseline-KPIs auf ungestresstem Universum (einmalig)
    reset_universe_to_baseline(universe)
    if selected_bank_names:
        try:
            agg_base = BankPortfolio("scan_base")
            for n in selected_bank_names:
                if n in universe.banks:
                    agg_base.segments.extend(universe.banks[n].segments)
        except Exception:
            agg_base = universe.aggregated_portfolio("scan_base")
    else:
        agg_base = universe.aggregated_portfolio("scan_base")
    kpi_base = agg_base.portfolio_kpis(confidence=0.999)
    el_base_total  = float(kpi_base["el_eur"])
    rwa_base_loan  = float(kpi_base["rwa"])

    # 2) Pro Scan-Punkt: Stress applizieren, KPIs frisch erfassen,
    #    danach wieder reset_universe_to_baseline.
    for d_r in dr_grid:
        apply_stress_to_universe(universe, d_brent_fixed, float(d_r))

        if selected_bank_names:
            try:
                agg = BankPortfolio("scan")
                for n in selected_bank_names:
                    if n in universe.banks:
                        agg.segments.extend(universe.banks[n].segments)
            except Exception:
                agg = universe.aggregated_portfolio("scan")
        else:
            agg = universe.aggregated_portfolio("scan")

        kpi_stress = agg.portfolio_kpis(confidence=0.999)
        delta_el  = float(kpi_stress["el_eur"]) - el_base_total
        delta_rwa = float(kpi_stress["rwa"])    - rwa_base_loan

        # Sovereign-MtM and Trading-Book scaling (linear approx.)
        sov_dp = sov_pnl_per_pp * float(d_r)

        # Reset für nächste Iteration
        reset_universe_to_baseline(universe)

        cet1_scan = cet1_base_total - delta_el + sov_dp
        rwa_scan  = rwa_base_total  + delta_rwa
        ratios.append(cet1_scan / rwa_scan if rwa_scan > 0 else 0.0)

    return dr_grid, np.array(ratios)


# Sovereign sensitivity per pp (recomputed at unit shock)
_sov_pnl_per_pp = 0.0
try:
    _unit_sov = rate_shock_pnl(sov_mat, delta_r_pp=1.0)
    _unit_sov = _unit_sov[_unit_sov["LEI_Code"].isin(universe_leis)]
    _sov_pnl_per_pp = float(_unit_sov["delta_pnl_eur"].sum())
except Exception:
    pass

_cet1_base_total = float(bridge_df["cet1_base"].sum())
_rwa_base_total  = float(bridge_df["rwa_total_base"].sum())
_selected_names = (tuple(selected_banks)
                   if "selected_banks" in dir()
                   else tuple(universe.banks.keys()))

dr_grid_sens, ratio_sens = _two_factor_sensitivity(
    _selected_names,
    _d_brent,
    _cet1_base_total, _rwa_base_total,
    _sov_pnl_per_pp,
)

# Plot
fig_sens = go.Figure()
fig_sens.add_trace(go.Scatter(
    x=dr_grid_sens, y=ratio_sens * 100,
    mode="lines+markers",
    line=dict(color=COLORS["navy"], width=2.8),
    marker=dict(size=6, color=COLORS["navy"]),
    name="CET1-Quote (Aggregat)",
    hovertemplate="Δr_10y = %{x:+.2f} pp<br>CET1 = %{y:.2f} %<extra></extra>",
))
# Schwellen-Linien
fig_sens.add_hline(y=4.5, line_dash="dot", line_color="#A52F4D",
                   annotation_text="4.5 % · Pillar 1", annotation_position="right",
                   annotation_font=dict(size=10, color="#A52F4D"))
fig_sens.add_hline(y=7.0, line_dash="dot", line_color="#C9A227",
                   annotation_text="7.0 % · inkl. CCB", annotation_position="right",
                   annotation_font=dict(size=10, color="#C9A227"))
fig_sens.add_hline(y=8.0, line_dash="dot", line_color="#034B6F",
                   annotation_text="8.0 % · SREP", annotation_position="right",
                   annotation_font=dict(size=10, color="#034B6F"))
# Aktueller Δr-Slider-Wert markieren
fig_sens.add_vline(x=_d_r_10y_pp, line_dash="dash",
                    line_color=COLORS["crimson"], line_width=1.5,
                    annotation_text=f"Aktueller Δr = {_d_r_10y_pp:+.2f} pp",
                    annotation_position="top",
                    annotation_font=dict(size=10, color=COLORS["crimson"]))
# Baseline-Marker bei Δr = 0
zero_idx = int(np.argmin(np.abs(dr_grid_sens)))
ratio_at_zero = ratio_sens[zero_idx] * 100
fig_sens.add_annotation(x=0, y=ratio_at_zero,
                         text=f"Baseline {ratio_at_zero:.2f} %",
                         xshift=10, yshift=10,
                         showarrow=False,
                         font=dict(size=10, color=COLORS["stone"]))
fig_sens.update_layout(
    title=(f"CET1-Quote als Funktion von Δr_10y · Aggregat über "
           f"{universe.n_banks} Banken · ΔBrent fix bei {_d_brent:+.2f}"),
    xaxis_title="Δr_10y · 10-Jahres-Zinsverschiebung (Prozentpunkte)",
    yaxis_title="CET1-Quote [%]",
    height=440, showlegend=False,
)
st.plotly_chart(fig_sens, use_container_width=True)

# Breaking-Points pro Schwelle
def _find_breach_dr(ratio_arr, dr_grid_arr, thr_pct):
    """Finde Δr-Wert, bei dem CET1 erstmals unter Threshold fällt
    (suchend vom Baseline Δr=0 nach oben)."""
    below = ratio_arr * 100 < thr_pct
    if not below.any():
        return None
    zero_idx = int(np.argmin(np.abs(dr_grid_arr)))
    for i in range(zero_idx, len(dr_grid_arr)):
        if below[i]:
            return float(dr_grid_arr[i])
    return None


bp_45 = _find_breach_dr(ratio_sens, dr_grid_sens, 4.5)
bp_70 = _find_breach_dr(ratio_sens, dr_grid_sens, 7.0)
bp_80 = _find_breach_dr(ratio_sens, dr_grid_sens, 8.0)

bp_l, bp_r = st.columns([1, 1], gap="medium")
with bp_l:
    st.markdown(
        "**Breaking Points** — bei welchem Δr_10y-Schock die CET1-Quote "
        "erstmals unter die jeweilige Aufsichtsschwelle fällt:"
    )
    def _fmt_bp(x):
        if x is None:
            return "nicht erreicht im Scan-Bereich (−1 bis +5 pp)"
        return f"Δr = {x:+.2f} pp  ({x*100:+.0f} bp)"
    bp_table = pd.DataFrame([
        ("4.5 % · Pillar 1",    _fmt_bp(bp_45)),
        ("7.0 % · inkl. CCB",   _fmt_bp(bp_70)),
        ("8.0 % · SREP-Ziel",   _fmt_bp(bp_80)),
    ], columns=["Aufsichtsschwelle", "Breaking Point Δr_10y"])
    st.dataframe(bp_table, use_container_width=True, hide_index=True,
                 height=140)

with bp_r:
    insight(
        f"<strong>Wie ist die Kurve zu lesen?</strong> Bei Δr = 0 "
        f"(keine Zinsschock-Komponente, Baseline) liegt die Aggregat-"
        f"CET1-Quote bei <strong>{ratio_at_zero:.2f} %</strong>. "
        f"Steigt Δr, fällt die Quote — drei Effekte überlagern sich: "
        f"(1) höhere Loan-PDs/LGDs erhöhen ΔRWA und ΔEL, "
        f"(2) Sovereign-Bonds verlieren MtM-Wert, "
        f"(3) Trading-Book wird über FRTB-Multiplier belastet. Der "
        f"EBA-Adverse-Stress-Test 2025 unterstellt typisch +200 bp — "
        f"bei diesem Δr siehst du am Kurvenverlauf, wie nah die "
        f"Aggregat-Bank an den Schwellen liegt."
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
    f"+ Trading Book FRTB-style · ΔBrent = {_d_brent:+.2f}, Δr = {_d_r_10y_pp:+.2f} pp · "
    f"κ_LGD = {KAPPA_DOWNTURN_LGD:.2f} · See Annahmen page for full "
    f"approximations inventory."
)
