"""Capital Adequacy · CET1-Ratio under two-channel stress.

Architektur (siehe Annahmen-Page):
  Numerator (CET1):
    CET1_stress = CET1_base
                  − ΔEL_loan_book           (Loan-Book Provisions-Hit)
                  + Δ_sovereign_MtM_signed    (Sovereign FVOCI/AfS via OCI)

  Denominator (Total RWA):
    RWA_stress = RWA_base
                 + ΔRWA_credit_loan_book   (from Vasicek capital_bridge)
                 (Market- + Operational-RWA unverändert in V1)

  Der frühere Trading-Book-Kanal wurde entfernt (kleine Handelsbücher,
  keine belastbare FRTB-Kalibrierung aus EBA-Aggregaten).

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

from components.theme import (tab_breadcrumb, apply_theme, eyebrow, insight, footer,
                              COLORS)
from components.sidebar import render_sidebar
from components.data_loader import load_data_layer
from components.backend_path import setup
setup()

from config import (KAPPA_DOWNTURN_LGD, EBA_RAW_DIR)                # type: ignore
from eba_loader import (                                            # type: ignore
    load_eba_universe, load_bank_directory,
    parse_capital_overview, parse_sovereign_csv,
    sovereign_by_accounting_class, sovereign_cet1_pnl_lookup,
    cet1_ratio_bridge,
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

st.markdown(
    '<div style="margin:0.35rem 0 0.75rem 0;padding:0.95rem 1.1rem;'
    'background:#FFFFFF;border:1px solid #E6E6E6;border-left:4px solid #051C2C;'
    'border-top:4px solid #2251FF;color:#051C2C;">'
    '<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;'
    'text-transform:uppercase;color:#6E6E6E;margin-bottom:0.25rem;">'
    'Tab 4 · Eigenkapital-Stressmodell · CET1-Bridge</div>'
    '<div style="font-size:1.28rem;font-weight:800;line-height:1.2;'
    'font-family:Inter,-apple-system,Segoe UI,sans-serif;'
    'letter-spacing:0.005em;margin-bottom:0.35rem;">'
    'Eigenkapital: CET1-Quote, RWA-Effekt und regulatorische Schwellen</div>'
    '<div style="font-size:0.90rem;line-height:1.55;color:#3A4A57;'
    'max-width:1080px;">'
    'Dieser Tab führt Kreditbuch und Marktbuch in einer Kapitalrechnung '
    'zusammen. Der Kreditbuch-Stress reduziert CET1 über zusätzliche '
    'Expected-Loss-Vorsorge und erhöht Risk-Weighted Assets; der Sovereign-'
    'Zinskanal reduziert CET1 über marktwertgeführte IFRS-9-Bestände. '
    'Die resultierende CET1-Quote wird gegen Pillar 1, Kapitalerhaltungs-'
    'puffer und typische SREP-Zielmarke gelesen.'
    '</div>'
    '<div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-top:0.65rem;">'
    '<span style="font-size:0.70rem;font-weight:700;color:#034B6F;'
    'background:#EEF6FA;border:1px solid #D8E6ED;padding:0.25rem 0.45rem;">'
    'Kreditbuch EL/RWA</span>'
    '<span style="font-size:0.70rem;font-weight:700;color:#034B6F;'
    'background:#EEF6FA;border:1px solid #D8E6ED;padding:0.25rem 0.45rem;">'
    'Sovereign MtM</span>'
    '<span style="font-size:0.70rem;font-weight:700;color:#034B6F;'
    'background:#EEF6FA;border:1px solid #D8E6ED;padding:0.25rem 0.45rem;">'
    'CET1 / RWA</span>'
    '<span style="font-size:0.70rem;font-weight:700;color:#051C2C;'
    'background:#F4F4F4;border:1px solid #E0E0E0;padding:0.25rem 0.45rem;">'
    'Schwellen</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
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


@st.cache_data(ttl=24*3600, show_spinner="Loading sovereign IFRS-9 split …")
def _load_sov_acct():
    # Bank-individuell GEMELDETER IFRS-9-Split (EBA tr_sov Items
    # 2520812-2520815) — Datenbasis des Sovereign-Kanals. Nur die zum
    # Marktwert gefuehrten Klassen (HfT/FVTPL/FVOCI) wirken auf CET1;
    # AC bleibt zu Buchwert. Ersetzt die fruehere V1-Vereinfachung
    # "gesamte Maturity-Ladder FVOCI-aehnlich" (100%-Durchleitung).
    sov_raw = parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv", period=202506)
    return sovereign_by_accounting_class(sov_raw, period=202506)


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
sov_acct = _load_sov_acct()
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

# Compute the two channels per bank in the universe -------------------
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

# Channel 2 — CET1-wirksamer Sovereign-MtM pro Bank: nur die zum
# Marktwert gefuehrten IFRS-9-Klassen (HfT/FVTPL/FVOCI) aus dem
# bank-individuell gemeldeten EBA-Split; AC bleibt zu Buchwert.
sov_pnl_lookup: dict[str, float] = (
    sovereign_cet1_pnl_lookup(sov_acct, delta_r_pp=delta_r_pp)
    if abs(delta_r_pp) > 1e-3 else {}
)

# Restrict capital_df to universe banks only
universe_leis = list(name_to_lei.values())
cap_universe = cap_df[cap_df["LEI_Code"].isin(universe_leis)].copy()

# Compute the two-channel CET1 bridge (Loan-Book + Sovereign) ---------
# Trading-Book-Kanal in V1 entfernt — siehe cet1_ratio_bridge-Docstring.
bridge_df = cet1_ratio_bridge(
    cap_universe, loan_bridges, sov_pnl_lookup,
)
bridge_df = bridge_df.merge(bank_dir[["lei", "bank_name"]],
                            left_on="LEI_Code", right_on="lei", how="left")


# === Aggregate KPI strip =============================================
eyebrow("Tab-Übersicht · was in der Eigenkapital-Bridge gezeigt wird")

st.markdown(
    '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));'
    'gap:0.75rem;margin:0.25rem 0 1.0rem 0;">'
    '<div style="background:#FFFFFF;border:1px solid #E1E6EA;'
    'border-top:3px solid #034B6F;border-radius:6px;padding:0.85rem;">'
    '<div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;'
    'letter-spacing:0.08em;text-transform:uppercase;">Input · Kreditbuch</div>'
    '<div style="font-weight:760;color:#051C2C;margin-top:0.2rem;">EL und Kredit-RWA</div>'
    '<div style="font-size:0.78rem;color:#536774;line-height:1.45;'
    'margin-top:0.25rem;">Gestresste PD/LGD erhöhen Expected Loss und Kredit-RWA.</div>'
    '</div>'
    '<div style="background:#FFFFFF;border:1px solid #E1E6EA;'
    'border-top:3px solid #2251FF;border-radius:6px;padding:0.85rem;">'
    '<div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;'
    'letter-spacing:0.08em;text-transform:uppercase;">Input · Marktbuch</div>'
    '<div style="font-weight:760;color:#051C2C;margin-top:0.2rem;">Sovereign MtM</div>'
    '<div style="font-size:0.78rem;color:#536774;line-height:1.45;'
    'margin-top:0.25rem;">Zinsschock erzeugt MtM-Verlust auf CET1-wirksame IFRS-9-Bestände.</div>'
    '</div>'
    '<div style="background:#FFFFFF;border:1px solid #E1E6EA;'
    'border-top:3px solid #051C2C;border-radius:6px;padding:0.85rem;">'
    '<div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;'
    'letter-spacing:0.08em;text-transform:uppercase;">Analyse · Quote</div>'
    '<div style="font-weight:760;color:#051C2C;margin-top:0.2rem;">CET1 / RWA</div>'
    '<div style="font-size:0.78rem;color:#536774;line-height:1.45;'
    'margin-top:0.25rem;">Zähler- und Nennereffekte werden zur CET1-Quote verdichtet.</div>'
    '</div>'
    '<div style="background:#FFFFFF;border:1px solid #E1E6EA;'
    'border-top:3px solid #6E6E6E;border-radius:6px;padding:0.85rem;">'
    '<div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;'
    'letter-spacing:0.08em;text-transform:uppercase;">Output · Aufsicht</div>'
    '<div style="font-weight:760;color:#051C2C;margin-top:0.2rem;">Schwellen & Drilldown</div>'
    '<div style="font-size:0.78rem;color:#536774;line-height:1.45;'
    'margin-top:0.25rem;">Systemansicht, Bank-Bridge und regulatorische Zonen.</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="background:#F4F4F4;border-left:3px solid #034B6F;'
    'padding:0.7rem 0.9rem;margin:0.2rem 0 1.0rem 0;color:#051C2C;'
    'font-size:0.86rem;line-height:1.55;">'
    '<strong>Formel.</strong> CET1-Quote = CET1-Kapital / RWA. '
    'Im Stress sinkt der Zähler durch Kreditvorsorge und Sovereign-MtM; '
    'der Nenner steigt durch zusätzliche Kredit-RWA. Genau diese beiden '
    'Effekte werden unten für System und Einzelbank getrennt gezeigt.'
    '</div>',
    unsafe_allow_html=True,
)

eyebrow(f"System Snapshot · CET1-Ausstattung der {universe.n_banks} Banken")

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
a1.metric("Σ CET1 vor Stress", f"€{cet1_total_base/1e9:.0f} bn",
          f"{n_banks} Banken", delta_color="off")
a2.metric("Σ RWA vor Stress", f"€{rwa_total_base/1e9:.0f} bn",
          f"RWA-Dichte {rwa_total_base/sum(p.total_ead for p in universe.banks.values())*100:.0f}%",
          delta_color="off")
a3.metric("Quote vorher",
          f"{ratio_base*100:.2f}%",
          f"SREP-Ziel {SREP_TARGET*100:.1f}%",
          delta_color="off")
if _is_stressed:
    a4.metric("Quote nachher",
              f"{ratio_stress*100:.2f}%",
              f"{(ratio_stress-ratio_base)*100:+.2f} pp")
    a5.metric("unter 4,5%",
              f"{len(breaches)} / {n_banks}",
              "nach Stress" if len(breaches) > 0 else "alle über Minimum",
              delta_color="off")
else:
    a4.metric("Quote nachher", "—",
              "kein Schock aktiv", delta_color="off")
    a5.metric("unter 4,5%", "—",
              "kein Schock aktiv", delta_color="off")

# === Insight box =====================================================
if not _is_stressed:
    insight(
        "<strong>Kein Makro-Schock aktiv.</strong> Bewege die zwei "
        "Slider in der Sidebar (ΔBrent + Δr_10y), um zu sehen, wie die "
        "zwei Transmissions-Kanäle — Kreditbuch-Vorsorge (sektor-"
        "differenzierte ΔPD und ΔLGD) und Sovereign-Mark-to-Market "
        "(Duration · Δr) — zur regulatorischen CET1-Quote zusammenwirken."
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
    "**Methodik-Hinweis.** Kredit-RWA werden mit der Basel-III-IRB-Formel "
    "auf Basis der bereits gestressten PD/LGD-Werte berechnet. Das ist "
    "eine regulatorische Kapital-under-Stress-Sicht und kein dreijähriger "
    "EBA-P&L-Pfad. Details: MODEL_ASSUMPTIONS.md, A-04/A-05."
)

st.divider()

# === League table ====================================================
eyebrow(f"Bankenübersicht · CET1-Bewegung im Stress ({n_banks} Banken)")

if _is_stressed:
    league = bridge_df.sort_values("cet1_ratio_stress").copy()

    def _flag(row):
        if pd.isna(row["cet1_ratio_stress"]):
            return "—"
        if row["cet1_ratio_stress"] < PILLAR1_MIN_CET1:
            return "Pillar 1 verletzt"
        if row["cet1_ratio_stress"] < TARGET_CET1_RATIO:
            return "unter Kapitalpuffer"
        if row["cet1_ratio_stress"] < SREP_TARGET:
            return "unter SREP-Ziel"
        return "komfortabel"

    league["Status"] = league.apply(_flag, axis=1)
    display_l = pd.DataFrame({
        "Bank":             league["bank_name"],
        "CET1 vorher bn":   (league["cet1_base"] / 1e9).round(1),
        "CET1 nachher bn":  (league["cet1_stress"] / 1e9).round(1),
        "RWA vorher bn":    (league["rwa_total_base"] / 1e9).round(0).astype(int),
        "RWA nachher bn":   (league["rwa_total_stress"] / 1e9).round(0).astype(int),
        "Quote vorher %":   (league["cet1_ratio_base"] * 100).round(2),
        "Quote nachher %":  (league["cet1_ratio_stress"] * 100).round(2),
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
eyebrow("Bank-Drilldown · CET1-Bridge pro Bank")

bank_options = sorted(bridge_df["bank_name"].dropna().tolist())
sel_bank_name = st.selectbox(
    "Bank",
    bank_options,
    index=0,
    format_func=lambda n: (
        f"{n}  (CET1-Quote vorher "
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
    # Aggregate denominator effect (after numerator already shifted)
    d_rwa_pp    = (ratio_after_den - ratio_after_num) * 100

    wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["CET1-Quote vorher",
           "Kreditbuch · Vorsorge",
           "Sovereign · MtM",
           "RWA-Ausweitung",
           "CET1-Quote nachher"],
        text=[f"{base_ratio*100:.2f}%",
              f"{d_loan_pp:+.2f} pp",
              f"{d_sov_pp:+.2f} pp",
              f"{d_rwa_pp:+.2f} pp",
              f"{sel_row['cet1_ratio_stress']*100:.2f}%"],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["navy"]),
        y=[base_ratio*100,
           d_loan_pp, d_sov_pp, d_rwa_pp,
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
        title=f"{sel_bank_name} · Zerlegung der CET1-Quote",
        yaxis_title="CET1-Quote [%]",
        height=460,
        showlegend=False,
    )
    st.plotly_chart(wf, use_container_width=True)

    # Numerical breakdown
    detail_l, detail_r = st.columns(2, gap="medium")

    with detail_l:
        eyebrow("Zähler · CET1-Kapital")
        num_table = pd.DataFrame([
            ("CET1 vor Stress",             sel_row["cet1_base"]/1e9),
            ("Kreditbuch-Vorsorge",         sel_row["delta_cet1_loan"]/1e9),
            ("Sovereign MtM",               sel_row["delta_cet1_sovereign"]/1e9),
            ("CET1 nach Stress",            sel_row["cet1_stress"]/1e9),
        ], columns=["Baustein", "EUR bn"])
        num_table["EUR bn"] = num_table["EUR bn"].map(lambda v: f"{v:+.2f}")
        st.dataframe(num_table, use_container_width=True, hide_index=True,
                     height=210)

    with detail_r:
        eyebrow("Nenner · Risk-Weighted Assets")
        den_table = pd.DataFrame([
            ("RWA vor Stress",                sel_row["rwa_total_base"]/1e9),
            ("Δ Kredit-RWA",                  sel_row["delta_rwa_credit"]/1e9),
            ("Markt-/Op-RWA unverändert",     0.0),
            ("RWA nach Stress",               sel_row["rwa_total_stress"]/1e9),
        ], columns=["Baustein", "EUR bn"])
        den_table["EUR bn"] = den_table["EUR bn"].map(lambda v: f"{v:+.2f}" if abs(v) < 50 else f"{v:+.0f}")
        st.dataframe(den_table, use_container_width=True, hide_index=True,
                     height=210)

    # Final ratio outcome
    delta_pp = sel_row["delta_cet1_ratio_pp"]
    breach_msg = ""
    if sel_row["cet1_ratio_stress"] < PILLAR1_MIN_CET1:
        breach_msg = (f" — unter dem Pillar-1-Minimum von "
                      f"{PILLAR1_MIN_CET1*100:.1f}%")
    elif sel_row["cet1_ratio_stress"] < TARGET_CET1_RATIO:
        breach_msg = (f" — unter Pillar 1 plus Kapitalerhaltungspuffer "
                      f"({TARGET_CET1_RATIO*100:.1f}%)")
    st.markdown(
        f"**Ergebnis.** Die CET1-Quote bewegt sich von "
        f"<strong>{sel_row['cet1_ratio_base']*100:.2f}%</strong> "
        f"auf <strong>{sel_row['cet1_ratio_stress']*100:.2f}%</strong> "
        f"({delta_pp:+.2f} pp){breach_msg}.",
        unsafe_allow_html=True,
    )
else:
    st.info("Setze einen Makro-Schock in der Sidebar, um die CET1-Bridge zu sehen.")

st.divider()

# =====================================================================
# NEW · Threshold-Analyse (welche Banken fallen unter Regulatorik?)
# =====================================================================
eyebrow("Threshold-Analyse · regulatorische CET1-Mindestquoten")

st.markdown(
    '<div style="background:#F4F4F4;border-left:3px solid #051C2C;'
    'padding:0.7rem 0.9rem;margin:0.2rem 0 0.85rem 0;color:#051C2C;'
    'font-size:0.86rem;line-height:1.55;">'
    '<strong>Wie liest man die Tabelle?</strong> Die Banken sind nach '
    'CET1-Quote nach Stress sortiert. Der Status ordnet jede Bank einer '
    'regulatorischen Zone zu: unter Pillar 1, unter Kapitalpuffer, unter '
    'typischer SREP-Zielmarke oder komfortabel oberhalb von 8,0%.'
    '</div>',
    unsafe_allow_html=True,
)


def _classify_threshold(ratio: float) -> tuple[str, str]:
    """Return (label, hex-color) based on CET1 ratio post-stress."""
    if ratio < PILLAR1_MIN_CET1:
        return "Pillar 1 verletzt (< 4,5%)", COLORS["crimson"]
    elif ratio < TARGET_CET1_RATIO:
        return "unter Kapitalpuffer (< 7,0%)", COLORS["amber"]
    elif ratio < SREP_TARGET:
        return "unter SREP-Ziel (< 8,0%)", COLORS["mid_blue"]
    else:
        return "komfortabel (> 8,0%)", COLORS["stone"]


# Build threshold table
thr_rows = []
for _, row in bridge_df.iterrows():
    label, color = _classify_threshold(row["cet1_ratio_stress"])
    thr_rows.append({
        "Bank":            row["bank_name"],
        "CET1 vorher":     f"{row['cet1_ratio_base']*100:.2f}%",
        "CET1 nachher":    f"{row['cet1_ratio_stress']*100:.2f}%",
        "Δ pp":            f"{row['delta_cet1_ratio_pp']:+.2f}",
        "CET1-Kapital bn": round(row["cet1_base"]/1e9, 1),
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
ts1.metric("Pillar 1 verletzt",
           int(status_counts.get("Pillar 1 verletzt (< 4,5%)", 0)),
           "< 4,5%",
           delta_color="off")
ts2.metric("unter Kapitalpuffer",
           int(status_counts.get("unter Kapitalpuffer (< 7,0%)", 0)),
           "< 7,0%",
           delta_color="off")
ts3.metric("unter SREP-Ziel",
           int(status_counts.get("unter SREP-Ziel (< 8,0%)", 0)),
           "< 8,0%",
           delta_color="off")
ts4.metric("komfortabel",
           int(status_counts.get("komfortabel (> 8,0%)", 0)),
           "> 8,0%", delta_color="off")

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
    '<div style="background:#F4F4F4;border-left:3px solid #034B6F;'
    'padding:0.75rem 0.9rem;margin:0.25rem 0 1.0rem 0;color:#051C2C;'
    'font-size:0.86rem;line-height:1.55;">'
    '<strong>Was zeigt die Kurve?</strong> Der Zinsschock wird von -1 bis '
    '+5 Prozentpunkten gescannt; der Ölpreisschock bleibt auf dem aktuellen '
    'Sidebar-Wert. Die Linie zeigt, wie nah die aggregierte CET1-Quote an '
    'die Aufsichtsschwellen rückt.'
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

        # Sovereign-MtM scaling (linear approx. in Δr)
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
    _unit_sov = sovereign_cet1_pnl_lookup(sov_acct, delta_r_pp=1.0)
    _sov_pnl_per_pp = float(sum(
        v for k, v in _unit_sov.items() if k in set(universe_leis)
    ))
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
        "**Kritische Zinsschocks** — bei welchem Δr_10y-Schock die CET1-Quote "
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
    ], columns=["Aufsichtsschwelle", "kritischer Δr_10y"])
    st.dataframe(bp_table, use_container_width=True, hide_index=True,
                 height=140)

with bp_r:
    insight(
        f"<strong>Wie ist die Kurve zu lesen?</strong> Bei Δr = 0 "
        f"(keine Zinsschock-Komponente, Baseline) liegt die Aggregat-"
        f"CET1-Quote bei <strong>{ratio_at_zero:.2f} %</strong>. "
        f"Steigt Δr, fällt die Quote — zwei Effekte überlagern sich: "
        f"(1) höhere Loan-PDs/LGDs erhöhen ΔRWA und ΔEL, "
        f"(2) Sovereign-Bonds verlieren MtM-Wert (OCI). Der "
        f"EBA-Adverse-Stress-Test 2025 unterstellt typisch +200 bp — "
        f"bei diesem Δr siehst du am Kurvenverlauf, wie nah die "
        f"Aggregat-Bank an den Schwellen liegt."
    )

st.divider()

# === Methodology footer =============================================
with st.expander("Methodik · CET1-Zwei-Kanal-Architektur",
                 expanded=False):
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:3px solid #051C2C;padding:0.85rem 1.0rem;'
        'color:#051C2C;font-size:0.88rem;line-height:1.6;">'
        '<strong>Ökonomische Logik.</strong> Die CET1-Quote ist '
        '<code>CET1-Kapital / RWA</code>. Der Makro-Schock trifft diese '
        'Quote über zwei Bilanzkanäle: Das Kreditbuch erhöht Vorsorge und '
        'Kredit-RWA; das Sovereign-Buch erzeugt marktwertgeführte Verluste '
        'auf IFRS-9-Beständen, die über P&L oder OCI in CET1 sichtbar werden.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
| Baustein | Modellierte Wirkung | Quelle / Datenbasis |
|---|---|---|
| **CET1-Zähler** | `CET1_stress = CET1_base - ΔEL_loan + ΔMtM_sovereign` | CET1 aus EBA `tr_oth.csv`; Kreditbuch aus Pillar-3 PD/LGD; Sovereign-IFRS-9-Split aus EBA `tr_sov.csv` |
| **Kreditbuch** | Gestresste PD/LGD erhöhen Expected Loss und Kredit-RWA | Basel-IRB / CRR Art. 153; bank-spezifische Pillar-3 EU-CR6-Werte |
| **Sovereign MtM** | `ΔMtM = - Duration × Δr × Exposure`; CET1-wirksam nur HfT, FVTPL und FVOCI | EBA `tr_sov.csv` Items 2520812-2520815; Duration-Approximation siehe Marktbuch |
| **RWA-Nenner** | `RWA_stress = RWA_base + ΔRWA_credit` | EBA `tr_oth.csv`; Kredit-RWA aus IRB-Capital-Charge |
""",
    )

    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));'
        'gap:0.65rem;margin:0.85rem 0;">'
        '<div style="background:#F4F4F4;border-left:3px solid #051C2C;'
        'padding:0.65rem 0.75rem;">'
        '<div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;'
        'letter-spacing:0.08em;text-transform:uppercase;">4,5%</div>'
        '<div style="font-weight:760;">Pillar-1-Minimum</div>'
        '<div style="font-size:0.78rem;color:#536774;line-height:1.4;">'
        'Harte CET1-Mindestanforderung nach CRR Art. 92.</div></div>'
        '<div style="background:#F4F4F4;border-left:3px solid #034B6F;'
        'padding:0.65rem 0.75rem;">'
        '<div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;'
        'letter-spacing:0.08em;text-transform:uppercase;">7,0%</div>'
        '<div style="font-weight:760;">Pillar 1 + Kapitalpuffer</div>'
        '<div style="font-size:0.78rem;color:#536774;line-height:1.4;">'
        'Pillar 1 plus 2,5% Capital Conservation Buffer.</div></div>'
        '<div style="background:#F4F4F4;border-left:3px solid #6E6E6E;'
        'padding:0.65rem 0.75rem;">'
        '<div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;'
        'letter-spacing:0.08em;text-transform:uppercase;">8,0%</div>'
        '<div style="font-weight:760;">typische SREP-Zielmarke</div>'
        '<div style="font-size:0.78rem;color:#536774;line-height:1.4;">'
        'Orientierungswert inklusive zusätzlichem SII-/SREP-Puffer.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**Bewusste Modellgrenzen in V1.** Kein separater Trading-Book-Kanal,
keine Hedging-Rekonstruktion, keine Sovereign-Spread-Komponente neben dem
Zinsshift, keine IFRS-9-Stage-Migration und kein erzwungener Verkauf von
Amortised-Cost-Beständen. Diese Abgrenzung hält die CET1-Bridge eng bei
den öffentlich verfügbaren EBA- und Pillar-3-Daten.
"""
    )

footer(
    f"Zwei-Kanal-CET1-Architektur · Vasicek IRB (CRR Art. 153) + Sovereign "
    f"Duration (Tuckman/Serrat 2012) · ΔBrent = {_d_brent:+.2f}, "
    f"Δr = {_d_r_10y_pp:+.2f} pp · κ_LGD = {KAPPA_DOWNTURN_LGD:.2f} · "
    f"Annahmen-Inventar auf der Validierungs-Seite."
)
