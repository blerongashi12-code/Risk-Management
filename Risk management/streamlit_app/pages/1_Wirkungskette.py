"""Wirkungskette · End-to-end Transmission Chain (Macro Shock → CET1).

Diese Page ist der konzeptionelle Entry-Point der App: sie macht die komplette
Pipeline auf einer einzigen Seite sichtbar, mit Formel + Visualisierung +
Insight pro Stufe.

Stufen:
  1. Input        — Echo der Sidebar-Slider (ΔBrent, Δr_10y)
  2. Macro → M    — 2-Route-Mapping (Anchor + Mahalanobis), 2D-Scatter
                    historischer Realisationen vs. aktueller Schock
  3. M → PD       — Vasicek Conditional-PD pro Exposure-Class
  4. M → LGD      — Downturn-LGD (kinked, nur für M < 0)
  5. PD+LGD → K   — IRB-Capital-Bridge (PD-Shift, dann LGD-Shift)
  6. Channels     — 3-Channel-CET1-Wirkung (Loan + Sovereign + Trading Book)

Jede Stufe liefert genau eine Aussage: "was passiert hier methodisch, was
sieht man im aktuellen Schock". Dadurch wird die Wirkungskette nicht durch
mehrere Tabs verteilt — sie steht einmal komplett auf einer Page.
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
                              COLORS, PALETTE_DISCRETE)
from components.sidebar import render_sidebar
from components.data_loader import load_data_layer
from components.backend_path import setup
setup()

from eba_loader import (load_eba_universe,                          # type: ignore
                         parse_capital_overview,
                         parse_sovereign_csv, sovereign_maturity_ladder,
                         rate_shock_pnl, trading_book_stress,
                         cet1_ratio_bridge, load_bank_directory)
from macro_factor import (anchor_from_eba, hybrid_mapping,           # type: ignore
                           factor_returns, factor_stats)
from vasicek import conditional_pd, downturn_lgd                    # type: ignore
from config import KAPPA_DOWNTURN_LGD, EBA_RAW_DIR                   # type: ignore


st.set_page_config(page_title="Wirkungskette · Home", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Wirkungskette · Macro Shock to CET1 Impact",
    eyebrow="Tab 1 · End-to-end transmission chain · Executive view",
    deck="Die komplette Pipeline auf einer Seite. Vom (ΔBrent, Δr)-Schock "
         "über den Vasicek-Systemfaktor M zur Conditional-PD, Downturn-LGD, "
         "IRB-Capital-Charge und 3-Channel-CET1-Impact. Jede Stufe mit "
         "Formel, einem Chart und einer Aussage zum aktuellen Schock.",
)

# Coverage banner + ρ-Override-Warning werden später eingefügt (nach Universe-Load)

# =====================================================================
# Live macro shock + cached factor stats
# =====================================================================
# NOTE on units: config["d_r_10y_pp"] is in PERCENTAGE POINTS (pp).
# Earlier versions of this file multiplied by 100 — that was a unit bug
# that produced M ≈ −237 under the EBA-Adverse preset. Now passes pp through.
delta_brent_log = float(config["d_brent"])
delta_r_pp      = float(config["d_r_10y_pp"])         # already in pp
m_direct        = config.get("m_direct")              # None in multi-factor mode
rho_mult        = float(config.get("rho_mult", 1.0))  # ρ-Multiplikator
lgd_cal         = float(config.get("lgd_calibration", 1.0))  # V2: A-IRB-Proxy
stress_exp      = float(config.get("stress_exponent", 1.0))  # V2: Tail-Risk
cap_mode        = str(config.get("cap_mode", "hard"))        # V2: Logit-Cap


@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_factor_layer():
    data = load_data_layer()
    if data["brent"] is None or data["svensson"] is None:
        return None, None, None
    fr = factor_returns(data["brent"], data["svensson"], maturity=10.0).dropna()
    stats = factor_stats(data["brent"], data["svensson"], lookback=252)
    return data, fr, stats


@st.cache_data(ttl=24*3600, show_spinner="Loading EBA Transparency exercise …")
def _load_full_universe():
    """Loads the COMPLETE EBA universe (all reporting banks)."""
    # top_n=200 effectively means 'all' — EBA 2025 publishes ~120 banks
    return load_eba_universe(vintage="2025", top_n=200, prefer_real=True)


_data, _fr, _stats = _load_factor_layer()
universe_full = _load_full_universe()
N_TOTAL = universe_full.n_banks

# Pre-compute EAD per bank for ranking + filter UI
_bank_ead_pairs = sorted(
    ((name, p.total_ead) for name, p in universe_full.banks.items()),
    key=lambda x: -x[1],
)
_all_bank_names = [n for n, _ in _bank_ead_pairs]
_total_ead_full = sum(e for _, e in _bank_ead_pairs)


with st.sidebar:
    with st.expander("Aggregat · Bank-Auswahl", expanded=False):
        st.caption(
            f"Datenbasis: **{N_TOTAL} Banken** aus der EBA Transparency "
            f"Exercise 2025 (Stichtag Juni 2025). Σ EAD = "
            f"€{_total_ead_full/1e12:.1f} Bio."
        )
        agg_mode = st.radio(
            "Modus",
            ["Alle Banken", "Top-N nach EAD", "Eigene Auswahl"],
            index=0, key="tx_agg_mode",
            help="Standard: alle Banken aggregiert. Optional einschränken auf "
                 "Top-N nach Exposure (EAD) oder freie Bank-Auswahl.",
        )
        if agg_mode == "Top-N nach EAD":
            top_n_pick = st.slider(
                "Anzahl Banken (Top-N nach EAD)",
                5, N_TOTAL, min(10, N_TOTAL), 1, key="tx_top_n",
            )
            selected_banks = _all_bank_names[:top_n_pick]
        elif agg_mode == "Eigene Auswahl":
            selected_banks = st.multiselect(
                "Banken auswählen", _all_bank_names,
                default=_all_bank_names[:10], key="tx_custom_pick",
            )
            if not selected_banks:
                st.warning("Mindestens 1 Bank wählen — verwende Top-10 als Fallback.")
                selected_banks = _all_bank_names[:10]
        else:  # "Alle Banken"
            selected_banks = _all_bank_names

# Build the filtered universe (lightweight view over the full one)
from eba_loader import EbaUniverse  # type: ignore
universe = EbaUniverse(
    banks={n: universe_full.banks[n] for n in selected_banks},
    adverse_anchor=universe_full.adverse_anchor,
    source=universe_full.source,
)
_selected_ead = sum(universe.banks[n].total_ead for n in selected_banks)
_ead_share = (_selected_ead / _total_ead_full * 100) if _total_ead_full > 0 else 0.0


# =====================================================================
# Coverage banner + ρ-Override warning (rendered AFTER universe is loaded)
# =====================================================================
_coverage_color = COLORS["mid_blue"]
st.markdown(
    f'<div style="background:#F4F4F4;border-left:4px solid {_coverage_color};'
    f'padding:0.95rem 1.1rem;border-radius:6px;margin-bottom:0.8rem;'
    f'color:#051C2C;font-size:0.92rem;line-height:1.6;">'
    f'<div style="font-weight:600;font-size:1.0rem;margin-bottom:0.4rem;'
    f'color:{_coverage_color};">Datenbasis-Coverage · '
    f'67 IRB-Banken (EU-Bankensystem)</div>'
    f'<table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
    f'<tr><td style="padding:0.3rem 0.6rem 0.3rem 0;width:170px;color:#6E6E6E;">EBA-Reporting-Universe</td>'
    f'<td><strong>120 Banken</strong> aus 26 Ländern · '
    f'Stichtag Juni 2025 · Quelle: EBA Transparency Exercise 2025</td></tr>'
    f'<tr><td style="padding:0.3rem 0.6rem 0.3rem 0;color:#6E6E6E;">Aktive Aggregation</td>'
    f'<td><strong>{universe.n_banks} von {N_TOTAL} IRB-Banken</strong> · '
    f'Σ EAD = €{_selected_ead/1e12:.2f} Bio. '
    f'({_ead_share:.0f}% der IRB-EAD von €{_total_ead_full/1e12:.1f} Bio.) · '
    f'Bank-Auswahl in der Sidebar konfigurierbar</td></tr>'
    f'<tr><td style="padding:0.3rem 0.6rem 0.3rem 0;color:#6E6E6E;vertical-align:top;">Modell-Filter</td>'
    f'<td>Nur Banken mit <strong>IRB-Approach</strong> (Basel III Internal '
    f'Ratings Based) sind im Aggregat enthalten. Das sind 67 von 120 EBA-'
    f'Banken — typischerweise die größeren Institute, deren bank-interne '
    f'Rating-Systeme aufsichtsrechtlich abgenommen wurden.<br/>'
    f'<strong>Warum diese Filterung?</strong> Das Vasicek-/ASRF-Modell '
    f'(Single-Factor Credit Risk Model, Basel III §272–284) setzt PD '
    f'und LGD pro Schuldner voraus — diese Inputs liefern nur IRB-Banken '
    f'über die EBA-Disclosure. Die 53 STA-Banken (Standardized Approach) '
    f'rechnen mit aufsichtsrechtlich vorgegebenen Risikogewichten und '
    f'liefern keine PD/LGD-Granularität — sie können in diesem Modell '
    f'nicht aggregiert werden. Hinsichtlich Bilanzsumme repräsentieren '
    f'die 67 IRB-Banken aber <strong>~85% des EU-Bankensystems</strong>.</td></tr>'
    f'</table>'
    f'</div>',
    unsafe_allow_html=True,
)

if abs(rho_mult - 1.0) > 0.01:
    _rho_color = COLORS["crimson"] if rho_mult > 1.0 else COLORS["teal"]
    _rho_dir   = "höher" if rho_mult > 1.0 else "niedriger"
    st.markdown(
        f'<div style="background:#FAECE7;border-left:4px solid {_rho_color};'
        f'padding:0.6rem 1.0rem;border-radius:6px;margin-bottom:1.0rem;'
        f'color:#051C2C;font-size:0.88rem;">'
        f'<strong>⚠ ρ-Override aktiv:</strong> Multiplikator '
        f'<strong>{rho_mult:.2f}×</strong> auf Basel-formelmäßige '
        f'Asset-Korrelation. Alle PD- und Capital-Werte unten zeigen das '
        f'angepasste Modell ({(abs(rho_mult-1.0))*100:.0f}% {_rho_dir} '
        f'als Basel-Standard) — kein regulatorisches Ergebnis, sondern '
        f'Sensitivity-Analyse.'
        f'</div>',
        unsafe_allow_html=True,
    )

if _stats is None:
    cov_factors = np.array([[4e-4, 2e-5], [2e-5, 1e-4]])
    cov_source  = "synthetic fallback (cache missing)"
    n_obs       = 0
    corr_hat    = float("nan")
else:
    cov_factors = _stats["sigma"]
    cov_source  = f"empirical · {_stats['n_obs']} trading days"
    n_obs       = _stats["n_obs"]
    corr_hat    = float(_stats["corr"][0, 1])


anchor = anchor_from_eba("2025")
mapping = hybrid_mapping(
    delta_brent_log=delta_brent_log,
    delta_rate_10y_pp=delta_r_pp,
    anchor=anchor,
    cov_factors=cov_factors,
    horizon_days=252,
)
# In Single-Factor-Mode treibt der User M direkt → übernehme den Wert
# unverändert. In Multi-Factor-Mode wird M aus (Brent, Δr) gemappt.
if m_direct is not None:
    m_used = float(m_direct)
    # Override the anchor/data routes for display consistency
    mapping["m_anchor"] = m_used
    mapping["m_data"]   = m_used
    mapping["m_hybrid"] = m_used
    mapping["consistency"] = 0.0
else:
    m_used = mapping["m_hybrid"]


# =====================================================================
# Glossary · Abkürzungsverzeichnis (sticky reference for all stages)
# =====================================================================
with st.expander("Abkürzungsverzeichnis · Glossar der Fachbegriffe "
                 "(jederzeit aufklappbar)", expanded=False):
    st.markdown("""
| Abk. | Langform (Englisch) | Bedeutung (Deutsch) |
|---|---|---|
| **PD** | Probability of Default | Ausfallwahrscheinlichkeit eines Schuldners auf 1-Jahres-Sicht |
| **LGD** | Loss Given Default | Verlustquote bei Ausfall — Anteil der Forderung, der nicht zurückkommt |
| **EAD** | Exposure at Default | Risikoposition bei Ausfall — ausstehender Betrag im Default-Zeitpunkt |
| **EL** | Expected Loss | Erwarteter Verlust = PD · LGD · EAD (Provisionsbasis) |
| **UL** | Unexpected Loss | Unerwarteter Verlust = Quantil − EL (Eigenkapital-Basis) |
| **RWA** | Risk-Weighted Assets | Risikogewichtete Aktiva = K · 12.5 · EAD (Basel-III-Nenner) |
| **CET1** | Common Equity Tier 1 | Hartes Kernkapital — primärer Eigenkapital-Puffer |
| **M** | Systemic Factor | Vasicek-Systemfaktor (standardnormal, N(0,1)) — treibt alle PDs |
| **ρ (rho)** | Asset Correlation | Vermögens-Korrelation pro Exposure-Class (Basel-Formel) |
| **κ (kappa)** | Downturn-LGD Multiplier | Stress-Faktor für LGD-Uplift (κ = 0.30 EBA 2023) |
| **IRB** | Internal Ratings Based | Bankinternes Rating-Verfahren (Basel III, Foundation oder Advanced) |
| **ASRF** | Asymptotic Single Risk Factor | Vasicek-2002-Modell · 1-Faktor-Approximation des Kreditportfolios |
| **MA** | Maturity Adjustment | Laufzeit-Anpassung in der IRB-Capital-Formel |
| **F-IRB** | Foundation IRB | Bank schätzt nur PD selbst — LGD, EAD aus aufsichtsrechtlichen Defaults |
| **A-IRB** | Advanced IRB | Bank schätzt PD, LGD und EAD selbst |
| **ΔBrent** | Δ Brent log-return | Ölpreis-Schock als kumulativer log-Return über den Horizon |
| **Δr_10y** | Δ 10-year zero rate | Zins-Schock auf der 10-Jahres-Bundesbank-Svensson-Kurve, in Prozentpunkten |
| **β₀…β₃** | Svensson-Parameter | Level / Slope / Curvature1 / Curvature2 der Zinskurve |
| **bp / pp** | basis points / percentage points | 1 bp = 0.01 % · 1 pp = 1 % absolute |
| **FVOCI** | Fair Value through OCI | Buchhaltungsklasse mit ΔFV via OCI → CET1 (durchschlagend) |
| **FVTPL** | Fair Value through P&L | Buchhaltungsklasse mit ΔFV via Gewinn-/Verlust-Rechnung |
| **HfT** | Held for Trading | Handelsbestand — Fair Value, ΔFV via P&L → CET1 |
| **AC / HtM** | Amortised Cost / Held to Maturity | Buchwert-Bewertung — kein direkter CET1-Effekt unter Rate-Stress |
| **OCI** | Other Comprehensive Income | Sonstiges Ergebnis — fließt direkt ins CET1 |
| **TB / BB** | Trading Book / Banking Book | Handelsbuch (täglich MtM) / Anlagebuch (Buchwert + Provisioning) |
| **MtM / FV** | Mark-to-Market / Fair Value | Bewertung zu aktuellen Marktpreisen |
| **ABS / MBS** | Asset-/Mortgage-Backed Securities | Forderungs- und hypothekenbesicherte Wertpapiere (Verbriefungen) |
| **NPL** | Non-Performing Loans | Notleidende Kredite (Default-Status, Item EBA 2520512) |
| **HHI** | Herfindahl-Hirschman Index | Konzentrations-Index (Σ Marktanteile²) · 10000. < 1500 = niedrig konzentriert |
| **CCB** | Capital Conservation Buffer | Kapitalerhaltungspuffer (2.5 pp auf CET1-Minimum) |
| **CCF** | Credit Conversion Factor | Umwandlungsfaktor für Off-Balance-Positionen → EAD |
| **CRR** | Capital Requirements Regulation | EU-Aufsichtsverordnung (Art. 178/180/181 für PD/LGD-Definitionen) |
| **EBA** | European Banking Authority | EU-Bankenaufsicht · Quelle der Transparency-Exercise-Daten |
| **BCBS** | Basel Committee on Banking Supervision | Basler Ausschuss · Referenz für Vasicek/ASRF-IRB-Kalibrierung |
| **FRTB** | Fundamental Review of the Trading Book | Neues Markt-Risiko-Rahmenwerk (Basel III, Phase 2) |
""")

st.markdown(" ")  # vertical breathing room


# =====================================================================
# Prozess-Map · die komplette Pipeline als visueller Flowchart
# =====================================================================
eyebrow("Prozess-Map · Wirkungskette in einem Bild")
st.caption(
    "Sechs Stufen, sequenziell. Pro Stufe: Datenquelle (woher kommen die Inputs) "
    "und Prämisse (welche zentrale Annahme steckt drin). Live-Werte rechts."
)

# Pre-compute concise live values for the map
def _clean(v: float, eps: float = 5e-3) -> float:
    """Clamp near-zero so format strings don't show '-0.00' at baseline."""
    return 0.0 if abs(v) < eps else float(v)

_brent_pct  = (np.exp(delta_brent_log) - 1) * 100
_d_brent_c  = _clean(delta_brent_log)
_d_r_c      = _clean(delta_r_pp, eps=0.5)   # < 0.5 bp = effectively zero
_m_anc_c    = _clean(mapping['m_anchor'])
_m_dat_c    = _clean(mapping['m_data'])
_m_used_c   = _clean(m_used)
_map_stage1 = (f"ΔBrent = {_d_brent_c:+.2f} log (≈ {_brent_pct:+.0f}%)  ·  "
               f"Δr_10y = {_d_r_c:+.0f} bp")
_map_stage2 = (f"M (hybrid) = {_m_used_c:+.2f}    "
               f"[Anker: {_m_anc_c:+.2f}  ·  Daten: {_m_dat_c:+.2f}]")
if abs(m_used) < 1e-3:
    _live_state = "Baseline (kein Schock angelegt — Slider bewegen)"
elif m_used < 0:
    _live_state = f"Adverse: |M| = {abs(m_used):.2f}σ · "\
                  f"{abs(m_used/anchor.z_factor)*100:.0f}% des EBA-Adverse-Ankers"
else:
    _live_state = "Benign (positives M, kein PD/LGD-Stress)"

# Color tokens
C_NAVY  = COLORS["navy"]
C_MID   = COLORS["mid_blue"]
C_BLUE  = COLORS["bright_blue"]
C_CRIM  = COLORS["crimson"]
C_TEAL  = COLORS["teal"]
C_AMBER = COLORS["amber"]
C_STONE = COLORS["stone"]
C_HAIR  = COLORS["hairline"]

def _stage_html(num: str, accent: str, title: str, value: str,
                source: str, premise: str) -> str:
    """One stage box. Three columns: number · body · datasource + plain-language.

    Returns a SINGLE-LINE HTML string so Streamlit's markdown parser does
    not mistake indented lines for code blocks.

    Layout:
      [num] | title + formula/value | datasource (top) + "Was hier passiert" (bottom)
    """
    return (
        f'<div style="display:grid;grid-template-columns:64px 1fr 380px;'
        f'gap:1.0rem;padding:0.95rem 1.15rem;border:1px solid {C_HAIR};'
        f'border-left:4px solid {accent};border-radius:6px;'
        f'background:#FFFFFF;align-items:start;">'
        f'<div style="font-family:\'Source Serif Pro\',Georgia,serif;'
        f'font-size:2.0rem;color:{accent};font-weight:600;text-align:center;'
        f'line-height:1;padding-top:0.2rem;">{num}</div>'
        f'<div style="padding-top:0.1rem;">'
        f'<div style="font-weight:600;color:{C_NAVY};font-size:1.02rem;'
        f'margin-bottom:0.35rem;line-height:1.3;">{title}</div>'
        f'<div style="font-family:\'JetBrains Mono\',\'SF Mono\',Consolas,'
        f'monospace;color:{C_MID};font-size:0.86rem;letter-spacing:0.01em;'
        f'background:#F4F4F4;padding:0.35rem 0.55rem;border-radius:3px;'
        f'border-left:2px solid {accent};">{value}</div>'
        f'</div>'
        f'<div style="color:{C_STONE};font-size:0.82rem;line-height:1.55;'
        f'border-left:1px solid {C_HAIR};padding-left:0.95rem;">'
        f'<div style="margin-bottom:0.45rem;">'
        f'<strong style="color:{C_NAVY};">Datenbasis:</strong> {source}'
        f'</div>'
        f'<div>{premise}</div>'
        f'</div>'
        f'</div>'
    )

_arrow = (
    f'<div style="text-align:center;color:{C_HAIR};font-size:1.4rem;'
    f'line-height:1;margin:0.05rem 0;">▼</div>'
)

stages_html = "".join([
    _stage_html(
        "1", C_STONE,
        "Eingangs-Schock · was wird gestresst?",
        _map_stage1,
        "Brent-Ölpreis (Energie-Schock) + 10-Jahres-Zins (Geldpolitik-Schock). "
        "Beide Werte werden aus Marktdaten beobachtet.",
        "<strong>Was hier passiert:</strong> Du gibst die Schwere des Schocks "
        "vor (M-Slider in der Sidebar). Daraus werden ΔBrent und Δr_10y "
        "live abgeleitet — sie sind die zwei Markt-Schocks, die das Modell "
        "downstream durchrechnet.",
    ),
    _arrow,
    _stage_html(
        "2", C_MID,
        "Macro → Systemfaktor M · Übersetzung in Modell-Sprache",
        _map_stage2 + f"   ·   Zustand: {_live_state}",
        "Basel-III-Vasicek/ASRF-Modell · EBA-2025-Adverse-Anker (M = −2.50 = "
        "schwerer Stress) liefert den Kalibrierungs-Punkt.",
        "<strong>Was hier passiert:</strong> Der Markt-Schock wird auf <em>eine</em> "
        "Zahl reduziert — den Systemfaktor M. Das ist das gemeinsame Risiko-Signal, "
        "auf das alle Kredite gleichzeitig reagieren. M = 0 = Baseline, "
        "M = −2.5 = EBA-Adverse, M positiv = günstiges Umfeld.",
    ),
    _arrow,
    _stage_html(
        "3", C_BLUE,
        "M → Conditional PD · wie viel mehr Banken-Schuldner fallen aus?",
        "PD(M) = N( (N⁻¹(PD) − √ρ · M) / √(1 − ρ) )    "
        "[Vasicek 2002 · BCBS 2017]",
        "EBA Transparency 2025 Default-Quoten pro Exposure-Class · "
        f"<strong>Aggregat über {N_TOTAL} IRB-Banken</strong>.",
        "<strong>Was hier passiert:</strong> Bei adversem M steigen die "
        "Ausfallwahrscheinlichkeiten — wie stark, hängt von der Asset-Korrelation "
        "ρ ab (Corporate hat 12–24%, Mortgage 15%, Retail nur 3–16%). "
        "Höhere ρ → stärkere Reaktion.",
    ),
    _arrow,
    _stage_html(
        "4", C_AMBER,
        "M → Downturn-LGD · wie viel verlieren Banken pro Ausfall?",
        "LGD_stress = LGD_base · (1 + κ · max(−M, 0))    "
        f"[κ = {KAPPA_DOWNTURN_LGD:.2f}, EBA-Stresstest 2023]",
        "Basel-F-IRB-LGDs (45% / 20% / 65% / 45% je Klasse) · "
        f"<strong>Aggregat über {N_TOTAL} IRB-Banken</strong>.",
        "<strong>Was hier passiert:</strong> Im Stress sind nicht nur mehr "
        "Ausfälle, jeder Ausfall wird auch teurer (z.B. weil Sicherheiten "
        "weniger wert sind). Bei M = −2.5 wird LGD um 75% angehoben. "
        "Asymmetrisch — bei benignem M bleibt LGD auf Baseline.",
    ),
    _arrow,
    _stage_html(
        "5", C_CRIM,
        "PD + LGD → Capital · wie viel Eigenkapital braucht die Bank?",
        "ΔK = ΔK(PD-Shift) + ΔK(LGD-Shift)   ·   "
        "K = (L₀.₉₉₉ − PD·LGD) · MA(M_eff) · EAD",
        f"<strong>Aggregat über {N_TOTAL} IRB-Banken</strong> · "
        "Basel-III-IRB-Formel pro Segment.",
        "<strong>Was hier passiert:</strong> Die regulatorische "
        "Kapital-Anforderung K steigt sequenziell — erst durch höhere PDs "
        "(Stage 3), dann zusätzlich durch höhere LGD (Stage 4). Die zwei "
        "Beiträge sind exakt additiv (kein Doppel-Zählen).",
    ),
    _arrow,
    _stage_html(
        "6", C_NAVY,
        "Drei Channels → CET1-Quote · regulatorische Headline",
        "ΔCET1-Quote = ƒ(ΔEL_loan, ΔRWA_credit, ΔFV_sov, ΔP&L_TB, ΔRWA_market)",
        f"<strong>Aggregat über {N_TOTAL} IRB-Banken</strong> · "
        "Loan-Book (Vasicek) + Sovereign-Bonds (Duration-MtM) + "
        "Trading-Book (FRTB-Markt-RWA + P&L-Haircut).",
        "<strong>Was hier passiert:</strong> Drei Risiko-Kanäle wirken "
        "gleichzeitig auf CET1: Kredite (mehr Provisionen + höheres RWA), "
        "Staatsanleihen (Kurswertverluste bei Zinsanstieg), Handelsbuch "
        "(P&L-Verluste + höheres Markt-Risiko-RWA). Die Bridge unten zeigt "
        "jeden Effekt einzeln.",
    ),
])

st.markdown(
    f'<div style="display:flex;flex-direction:column;gap:0.05rem;'
    f'margin:0.5rem 0 1.5rem 0;">{stages_html}</div>',
    unsafe_allow_html=True,
)


# =====================================================================
# Methodische Begründung · warum EIN Faktor M und nicht zwei separat?
# =====================================================================
with st.expander("Methodische Begründung · was ist M, und warum EIN "
                 "Systemfaktor statt zwei getrennter Risikofaktoren?",
                 expanded=False):
    st.markdown(r"""
### Was ist M?

**M ist der Vasicek-Systemfaktor** — eine standardnormale Zufallsvariable
$M \sim \mathcal{N}(0, 1)$, die das **gemeinsame Risiko-Signal** für alle
Schuldner im Portfolio darstellt. Du kannst dir M als "Konjunktur-Stempel"
vorstellen:

- $M = 0$  → Baseline, durchschnittliche Wirtschaftslage, kein Stress
- $M < 0$ → adverse, Wirtschaft schwächelt, Ausfallraten steigen
  - $M = −1.0$ ≈ moderate Rezession
  - $M = −2.0$ ≈ schwere Rezession
  - $M = −2.5$ = exakt der EBA-2025-Adverse-Anker (Stress-Test-Szenario)
- $M > 0$ → benigne, Wirtschaft wächst, Ausfallraten sinken

### Was passiert in UNSEREM Modell, wenn du M änderst?

Bewegst du den **M-Slider** in der Sidebar, läuft folgende Pipeline live durch
(alle 5 Tabs aktualisieren sich gleichzeitig):

1. **M wird in Markt-Schocks rückübersetzt** — via EBA-Anker-Kopplung. Bei
   $M = −2.5$ ergibt das ΔBrent ≈ +0.47 log und Δr_10y ≈ +200 bp.
2. **Conditional PD pro Schuldner-Klasse** wird neu berechnet (Stage 3):
   $\text{PD}(M) = N\!\!\left(\frac{N^{-1}(\text{PD}) - \sqrt{\rho}M}{\sqrt{1-\rho}}\right)$.
3. **Downturn-LGD** wird angewendet (Stage 4): bei adversem M wird LGD um
   $\kappa \cdot |M|$-Multiplikator erhöht.
4. **Basel-IRB-Capital** wird mit den neuen (PD, LGD) berechnet (Stage 5).
5. **CET1-Quote** für jede der 67 IRB-Banken wird neu aggregiert (Stage 6
   + Tab 4 „Eigenkapital").
6. **Backtest** auf Tab 5 zeigt, wo historische Krisen-Episoden auf der
   M-Skala lagen.

Der **ρ-Multiplikator-Slider** (optional, in der Sidebar) skaliert
zusätzlich die Asset-Korrelation — höhere ρ = stärkere PD-Reaktion auf
denselben M.

### Warum EIN M statt zwei getrennte Faktoren?

Das Vasicek/ASRF-Modell, das Basel III IRB zugrundeliegt, ist
**per Konstruktion** ein **Ein-Faktor-Modell**. Diese fundamentale
Modellannahme wurde von Basel bewusst gewählt, weil sie **portfolio-
invariante** Kapital-Aufschläge ermöglicht (Gordy 2003).

**Warum nicht zwei separate Risikofaktoren $M_{\text{Brent}}$ und
$M_{\text{Rate}}$?**

Ein 2-Faktor-Modell wäre methodisch interessant, hätte aber drei
Probleme:

1. **Bricht mit Basel III IRB.** Die ASRF-Formel
   $K = (L_{0.999} - \text{PD}\cdot\text{LGD}) \cdot \text{MA}$ gilt
   **nur** unter dem Ein-Faktor-Modell. Für ein 2-Faktor-Modell wäre die
   geschlossene Form nicht mehr verfügbar — man müsste auf
   Monte-Carlo-Simulation umsteigen, was die regulatorische Vergleichbarkeit
   bricht.
2. **Identifikations-Problem.** Die ρ-Korrelationen pro Exposure-Class
   sind in Basel als $\rho(\text{PD})$ kalibriert (BCBS-IRB 2017) —
   alle gegen **einen** Faktor. Eine bank-spezifische 2-Faktor-Loading-
   Matrix wäre aus EBA-Public-Disclosure nicht schätzbar.
3. **Mehrwert gering.** Empirisch sind Brent und Δr_10y über 252-Tage-
   Horizons stark co-bewegend (oft ρ̂ > 0.3 in Stress-Phasen). Eine
   Faktor-Trennung würde mehr Komplexität schaffen, ohne wesentlich
   neue Information freizulegen.

**Was wir stattdessen tun (und transparent dokumentieren):**

Die zwei beobachtbaren Markt-Schocks $(\Delta\text{Brent},\Delta r_{10y})$
sind die **Inputs**. Sie werden via **Hybrid-Mapping** (Anchor +
Mahalanobis) auf den einen latenten Vasicek-Faktor $M$ projiziert. Damit
bleibt das Modell:
- **Basel-kompatibel** (single-factor ASRF)
- **multi-channel-transparent** (man sieht im Cockpit getrennt
  $\Delta\text{Brent}$ und $\Delta r_{10y}$)
- **regulatorisch validierbar** (M-Anker = EBA-2025-Adverse-Stress-Test
  Kalibrierung)

Die separat sichtbaren Channels in **Stufe 6** (Loan-Book vs. Sovereign
vs. Trading-Book) leisten die Trennung der Wirkungs-Kanäle **am
Output**, was fachlich der saubere Ort dafür ist.
""")

st.divider()


# =====================================================================
# Stage 1 · Echo der Eingangs-Schocks
# =====================================================================
eyebrow("Stufe 1 · Eingangs-Schock (Sidebar-Slider)")
in1, in2, in3 = st.columns([1, 1, 2], gap="small")
in1.metric("ΔBrent · Ölpreis-Schock (kumulativ, log-Return)",
           f"{delta_brent_log:+.2f}",
           f"entspricht ≈ {(np.exp(delta_brent_log)-1)*100:+.0f}% Preisänderung",
           delta_color="off")
in2.metric("Δr_10y · 10-Jahres-Zins-Schock",
           f"{delta_r_pp:+.0f} bp  ({delta_r_pp:+.2f} pp)",
           "implizit aus Δβ-Shifts via Svensson-Funktion",
           delta_color="off")
with in3:
    insight(
        "Zwei Markt-Schocks → ein Schock-Vektor <em>(ΔBrent, Δr_10y)</em>. "
        "Dieser Vektor wird in den folgenden Stufen sequenziell in PD-, "
        "LGD-, Capital- und CET1-Bewegung übersetzt. Reset auf der "
        "Sidebar setzt alles auf Baseline (kein Stress)."
    )

st.divider()


# =====================================================================
# Stage 2 · Macro → Vasicek Systemic Factor M
# =====================================================================
eyebrow("Stufe 2 · Macro → Vasicek Systemic Factor M")

st_col_l, st_col_r = st.columns([3, 2], gap="medium")

with st_col_l:
    # 2D scatter: historical daily Brent vs Δr, plus current shock + anchor + presets
    if _fr is not None and len(_fr) > 0:
        # Use 252-day rolling cumulative for comparability with slider semantics
        roll = _fr.copy()
        # rate column from factor_returns is `d_r_10y` (Δ daily); cumulative
        # over 252 trading days gives roughly a 1-year horizon shock in pp.
        rate_col = [c for c in roll.columns if c.startswith("d_r_")][0]
        roll["brent_252"] = roll["r_brent"].rolling(252).sum()
        roll["rate_252"]  = roll[rate_col].rolling(252).sum()
        roll = roll.dropna()
        recent = roll.tail(min(len(roll), 2500))

        fig_sc = go.Figure()
        # historical realizations (252-day cumulative)
        fig_sc.add_trace(go.Scatter(
            x=recent["brent_252"], y=recent["rate_252"],
            mode="markers",
            marker=dict(size=4, color=COLORS["hairline"],
                        line=dict(width=0)),
            name="252-Tage rollierend (hist.)",
            hovertemplate="ΔBrent_log=%{x:.2f}<br>Δr_10y_pp=%{y:.2f}<extra></extra>",
        ))
        # Quick-scenario presets
        presets = [
            ("Corona 2020",   -1.20, -0.65, COLORS["teal"]),
            ("Ukraine 2022",  +0.12, +0.67, COLORS["amber"]),
            ("Iran 2026",     +0.55, +0.50, COLORS["crimson"]),
            ("EBA 2025 adv.", +0.47, +2.00, COLORS["navy"]),
        ]
        for label, db, dr, col in presets:
            fig_sc.add_trace(go.Scatter(
                x=[db], y=[dr], mode="markers+text",
                marker=dict(size=10, color=col,
                            line=dict(width=1.5, color=COLORS["white"])),
                text=[label], textposition="top center",
                textfont=dict(size=9, color=col),
                name=label, showlegend=False,
                hovertemplate=f"{label}<br>ΔBrent=%{{x:.2f}}<br>Δr=%{{y:.2f}} pp<extra></extra>",
            ))
        # Current shock — large crimson dot
        fig_sc.add_trace(go.Scatter(
            x=[delta_brent_log], y=[delta_r_pp],
            mode="markers+text",
            marker=dict(size=16, color=COLORS["bright_blue"],
                        line=dict(width=2.5, color=COLORS["white"]),
                        symbol="star"),
            text=["aktueller Schock"], textposition="bottom center",
            textfont=dict(size=10, color=COLORS["bright_blue"], family="Inter"),
            name="aktueller Schock", showlegend=False,
            hovertemplate="ΔBrent=%{x:.2f}<br>Δr=%{y:.2f} pp<extra></extra>",
        ))
        fig_sc.update_layout(
            title="Schock-Vektor im (ΔBrent, Δr_10y)-Raum · "
                  "252-Tage-rollierend, historische Punktewolke + Szenarien",
            xaxis_title="ΔBrent — Brent-Öl kumulativer log-Return, 252 Handelstage",
            yaxis_title="Δr_10y — Veränderung 10-Jahres-Zins [Prozentpunkte], "
                        "252-Tage kumulativ",
            height=420, showlegend=False,
        )
        fig_sc.add_hline(y=0, line_dash="dot", line_color=COLORS["hairline"])
        fig_sc.add_vline(x=0, line_dash="dot", line_color=COLORS["hairline"])
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Brent/Svensson-Cache fehlt — historischer Punktewolken-Plot "
                "übersprungen. Aktueller Schock wird trotzdem auf M gemappt.")

with st_col_r:
    st.markdown(r"""
**Mapping-Methodik** (zwei Routen, beide hier sichtbar):

**Anchor-Route (primary):**
$$M_{\text{anchor}} = z_{\text{EBA}} \cdot \cos(\angle) \cdot \frac{\|\text{shock}\|}{\|\text{anchor}\|}$$

**Daten-Route (validation):**
$$M_{\text{data}} = -\sqrt{\text{shock}^\top \Sigma^{-1} \text{shock}} \cdot \cos(\angle_{\text{adv}})$$

**Hybrid:** Mittel beider Routen → robust gegen Anker-Mis-Spezifikation.
""")
    if m_direct is not None:
        # Single-Factor mode: user drives M directly
        m1, m2 = st.columns([1, 1], gap="small")
        m1.metric(
            "M · Direkt vom User gesetzt",
            f"{m_used:+.2f}σ",
            "Single-Factor-Modus — M ist der Eingangs-Regler",
            delta_color="off",
        )
        m2.metric(
            "Implizierte Schock-Kopplung (aus EBA-Anker)",
            f"ΔBrent {delta_brent_log:+.2f}  ·  Δr {delta_r_pp*100:+.0f} bp",
            f"Anker: M = {anchor.z_factor:.2f} → "
            f"({anchor.brent_log:+.2f}, {anchor.rate_10y_pp:+.1f} pp)",
            delta_color="off",
        )
    else:
        # Multi-Factor mode: (Brent, Δr) → M via two routes
        m1, m2, m3 = st.columns(3, gap="small")
        m1.metric("M · Anchor-Route (Primary)",
                  f"{mapping['m_anchor']:+.2f}",
                  "kalibriert via EBA 2025 Adverse Anchor",
                  delta_color="off")
        m2.metric("M · Daten-Route (Validation)",
                  f"{mapping['m_data']:+.2f}",
                  f"Mahalanobis · empirische Korrelation ρ̂ = {corr_hat:+.2f}",
                  delta_color="off")
        m3.metric("M · Hybrid (verwendet im Modell)",
                  f"{m_used:+.2f}",
                  "Mittel beider Routen → robust",
                  delta_color="off")

# Diagnostic / insight line for stage 2
if abs(m_used) < 1e-3:
    insight(
        "<strong>Kein Schock</strong> — Schock-Vektor liegt im Ursprung, "
        "also M ≈ 0. Alle Folgestufen bleiben auf Baseline. "
        "Sidebar-Slider bewegen → Pipeline aktiviert sich."
    )
else:
    direction = "adverse" if m_used < 0 else "benign"
    anchor_share = abs(m_used / anchor.z_factor) * 100 if anchor.z_factor != 0 else 0
    insight(
        f"<strong>M = {m_used:+.2f}</strong> ({direction}). Das ist "
        f"<strong>{anchor_share:.0f}%</strong> der Härte des "
        f"EBA-2025-Adverse-Ankers ({anchor.z_factor:+.2f}). "
        f"Konsistenz Anchor↔Data: |Δ| = {mapping['consistency']:.2f} σ — "
        f"je kleiner, desto belastbarer das Mapping. "
        f"Σ-Quelle: {cov_source}."
    )

st.divider()


# =====================================================================
# Stage 3 · M → Conditional PD pro Exposure-Class
# =====================================================================
eyebrow("Stufe 3 · M → Conditional PD pro Exposure-Class")
st.caption(
    f"Datenbasis · **Aggregat über {universe.n_banks} von {N_TOTAL} "
    f"IRB-Banken** · EAD-gewichtete Mittelung von PD und ρ pro Vasicek-"
    "Exposure-Class (Corporate / Mortgage / QRRE / Other Retail / Bank / "
    "Sovereign)."
)

# Aggregate the universe to one consolidated portfolio (top-N banks)
agg_pf = universe.aggregated_portfolio("EU Top-N Aggregate")
base_df = agg_pf.baseline_metrics(confidence=0.999, rho_multiplier=rho_mult,
                                     lgd_calibration=lgd_cal)

# Group segments by exposure_class, EAD-weighted PD/ρ
def _aggregate_by_class(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("exposure_class").apply(
        lambda x: pd.Series({
            "ead":     x["ead"].sum(),
            "pd_avg":  np.average(x["pd"], weights=x["ead"]) if x["ead"].sum() > 0 else 0.0,
            "lgd_avg": np.average(x["lgd"], weights=x["ead"]) if x["ead"].sum() > 0 else 0.0,
            "rho_avg": np.average(x["rho"], weights=x["ead"]) if x["ead"].sum() > 0 else 0.0,
            "el_eur":  x["el_eur"].sum(),
            "rwa":     x["rwa"].sum(),
        }),
        include_groups=False,
    )
    return g.reset_index().sort_values("ead", ascending=False)


class_base = _aggregate_by_class(base_df)
# Compute stressed PD/LGD per class using the class-aggregate ρ and PD
class_base["pd_stress"]  = [float(conditional_pd(pd_, rho, m_used))
                             for pd_, rho in zip(class_base["pd_avg"], class_base["rho_avg"])]
class_base["lgd_stress"] = [float(downturn_lgd(
                                lgd, m_factor=m_used, kappa=KAPPA_DOWNTURN_LGD,
                                exponent=stress_exp, cap_mode=cap_mode))
                             for lgd in class_base["lgd_avg"]]

pd_col_l, pd_col_r = st.columns([3, 2], gap="medium")

with pd_col_l:
    # Two grouped bars per class: PD_base vs PD_stress
    fig_pd = go.Figure()
    fig_pd.add_trace(go.Bar(
        x=class_base["exposure_class"], y=class_base["pd_avg"] * 100,
        name="PD base", marker_color=COLORS["mid_blue"],
        text=[f"{v*100:.2f}%" for v in class_base["pd_avg"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["navy"]),
    ))
    fig_pd.add_trace(go.Bar(
        x=class_base["exposure_class"], y=class_base["pd_stress"] * 100,
        name="PD stress", marker_color=COLORS["crimson"],
        text=[f"{v*100:.2f}%" for v in class_base["pd_stress"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["crimson"]),
    ))
    fig_pd.update_layout(
        title=f"Conditional PD = bedingte Ausfallwahrscheinlichkeit "
              f"pro Exposure-Class — Baseline vs. Stress @ M = {m_used:+.2f}",
        xaxis_title="Vasicek Exposure-Class (Basel III IRB)",
        yaxis_title="PD — Probability of Default [%]",
        height=380, barmode="group", bargap=0.20,
    )
    st.plotly_chart(fig_pd, use_container_width=True)

with pd_col_r:
    st.markdown(r"""
**Vasicek Conditional-PD** (BCBS-2001 IRB-Foundation):

$$\text{PD}(M) = N\!\left(\frac{N^{-1}(\text{PD}) - \sqrt{\rho}\,M}{\sqrt{1-\rho}}\right)$$

mit ρ-Funktion pro Class (Corporate: 12–24%, Mortgage 15%, QRRE 4%, Other-Retail 3–16%, SME-adjusted für Sales).

**Reading:** je größer ρ und je tiefer PD-Baseline → desto stärker hebelt ein adverse M die Stress-PD. Mortgage hat hohe ρ aber niedrige Baseline → kleine absolute Punkte. Corporate liegt in der Mitte. QRRE und Other-Retail haben niedrigere ρ → flachere Reaktion.
""")
    # Pick the segment with largest Δ pd in pp
    class_base["pd_delta_pp"] = (class_base["pd_stress"] - class_base["pd_avg"]) * 100
    top = class_base.sort_values("pd_delta_pp", ascending=False).iloc[0]
    if abs(m_used) > 1e-3:
        st.metric(
            f"Stärkste PD-Reaktion · Klasse: {top['exposure_class']}",
            f"+{top['pd_delta_pp']:.2f} pp (Prozentpunkte)",
            f"PD-Verschiebung: {top['pd_avg']*100:.2f}% → "
            f"{top['pd_stress']*100:.2f}%  ·  Asset-Korrelation ρ̄ = "
            f"{top['rho_avg']*100:.1f}%",
        )

st.divider()


# =====================================================================
# Stage 4 · M → Downturn LGD
# =====================================================================
eyebrow("Stufe 4 · M → Downturn LGD")
st.caption(
    f"Datenbasis · **Aggregat über {universe.n_banks} von {N_TOTAL} "
    f"IRB-Banken** · LGD aus Basel-F-IRB-Defaults (kein bank-spezifischer "
    f"A-IRB-Wert in EBA-Disclosure)."
)

lgd_col_l, lgd_col_r = st.columns([3, 2], gap="medium")

with lgd_col_l:
    # Step function showing the LGD multiplier (1 + κ * max(-M, 0)) over M ∈ [-3, +3]
    m_grid = np.linspace(-3.0, 3.0, 121)
    mult = 1.0 + KAPPA_DOWNTURN_LGD * np.maximum(-m_grid, 0.0)
    fig_lgd = go.Figure()
    fig_lgd.add_trace(go.Scatter(
        x=m_grid, y=mult,
        mode="lines",
        line=dict(color=COLORS["navy"], width=2.5),
        name="LGD-Multiplikator",
    ))
    fig_lgd.add_vline(x=m_used, line_dash="dash", line_color=COLORS["crimson"],
                      annotation_text=f"M = {m_used:+.2f}",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=COLORS["crimson"]))
    fig_lgd.add_vline(x=0, line_dash="dot", line_color=COLORS["hairline"])
    fig_lgd.add_hline(y=1.0, line_dash="dot", line_color=COLORS["hairline"])
    fig_lgd.update_layout(
        title=f"Downturn-LGD-Funktion · LGD-Multiplikator = 1 + κ · max(−M, 0), "
              f"mit κ = {KAPPA_DOWNTURN_LGD:.2f}  [EBA Stresstest 2023]",
        xaxis_title="Systemfaktor M (Vasicek, standardnormal N(0,1))",
        yaxis_title="LGD-Multiplikator (= LGD_stress / LGD_base)",
        height=340, showlegend=False,
    )
    st.plotly_chart(fig_lgd, use_container_width=True)

    # Class-level LGD base vs stress
    fig_lgd_class = go.Figure()
    fig_lgd_class.add_trace(go.Bar(
        x=class_base["exposure_class"], y=class_base["lgd_avg"] * 100,
        name="LGD base", marker_color=COLORS["mid_blue"],
        text=[f"{v*100:.0f}%" for v in class_base["lgd_avg"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["navy"]),
    ))
    fig_lgd_class.add_trace(go.Bar(
        x=class_base["exposure_class"], y=class_base["lgd_stress"] * 100,
        name="LGD stress", marker_color=COLORS["crimson"],
        text=[f"{v*100:.0f}%" for v in class_base["lgd_stress"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["crimson"]),
    ))
    fig_lgd_class.update_layout(
        title="LGD = Loss Given Default pro Exposure-Class · "
              "Baseline (Basel F-IRB) vs. Stress (Downturn-LGD)",
        xaxis_title="Vasicek Exposure-Class",
        yaxis_title="LGD — Verlustquote bei Ausfall [%]",
        height=320, barmode="group", bargap=0.20,
    )
    st.plotly_chart(fig_lgd_class, use_container_width=True)

with lgd_col_r:
    st.markdown(r"""
**Downturn-LGD-Funktion** (EBA Stresstest 2023, CRR Art. 181):

$$\text{LGD}_{\text{stress}} = \text{LGD}_{\text{base}} \cdot \bigl(1 + \kappa \cdot \max(-M, 0)\bigr)$$

gekappt bei 100%.

**Asymmetrisch:** Nur adverse M (negativ) hebt LGD. Benign-Schocks senken LGD **nicht** unter Baseline — vorsichtige regulatorische Konvention.
""")
    if m_used < 0:
        uplift = KAPPA_DOWNTURN_LGD * abs(m_used) * 100
        st.metric(
            "LGD-Uplift @ aktuellem Systemfaktor M",
            f"+{uplift:.0f}% (relativ zur Baseline)",
            f"Systemfaktor M = {m_used:+.2f}  ·  "
            f"Stress-Parameter κ = {KAPPA_DOWNTURN_LGD:.2f}",
        )
    else:
        st.metric("LGD-Uplift @ aktuellem Systemfaktor M",
                  "0% (keine Erhöhung)",
                  "M ≥ 0 → benigner Schock, asymmetrische Funktion",
                  delta_color="off")

st.divider()


# =====================================================================
# Stage 5 · PD + LGD → IRB Capital (Sequential Bridge)
# =====================================================================
eyebrow("Stufe 5 · PD + LGD → IRB Capital (sequentielle Bridge)")
st.caption(
    f"Datenbasis · **Aggregat über {universe.n_banks} von {N_TOTAL} "
    "IRB-Banken** · konsolidiert in ein virtuelles Aggregat-Portfolio "
    "(Summe aller Segmente). Capital-Bridge zeigt PD-Effekt + LGD-Effekt "
    "additiv."
)

if abs(m_used) > 1e-9:
    bridge = agg_pf.capital_bridge(
        z_factor=m_used,
        kappa_lgd=KAPPA_DOWNTURN_LGD,
        confidence=0.999,
        rho_multiplier=rho_mult,
        lgd_calibration=lgd_cal,
        stress_exponent=stress_exp,
        cap_mode=cap_mode,
    )

    cb_col_l, cb_col_r = st.columns([3, 2], gap="medium")

    with cb_col_l:
        # 4-bar waterfall: K_base → +ΔK_PD → +ΔK_LGD → K_stress (aggregate)
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["K Baseline<br>(Capital Requirement)",
               "+ Δ aus PD-Verschiebung<br>(LGD noch unverändert)",
               "+ Δ aus LGD-Verschiebung<br>(PD bereits gestresst)",
               "K Stress<br>(Capital Requirement)"],
            text=[f"€{bridge['K_base']/1e9:.1f} bn",
                  f"€{bridge['delta_K_pd']/1e9:+.1f} bn",
                  f"€{bridge['delta_K_lgd']/1e9:+.1f} bn",
                  f"€{bridge['K_stress']/1e9:.1f} bn"],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["navy"]),
            y=[bridge['K_base']/1e9,
               bridge['delta_K_pd']/1e9,
               bridge['delta_K_lgd']/1e9,
               bridge['K_stress']/1e9],
            connector={"line": {"color": COLORS["hairline"], "width": 1}},
            increasing={"marker": {"color": COLORS["crimson"]}},
            decreasing={"marker": {"color": COLORS["teal"]}},
            totals    ={"marker": {"color": COLORS["navy"]}},
        ))
        wf.update_layout(
            title=f"Aggregat-Bank (Top-{universe.n_banks} EU-Banken) · "
                  "Capital-Bridge sequentiell: PD-Effekt + LGD-Effekt",
            yaxis_title="K — IRB Capital Requirement [Mrd. EUR]",
            height=400, showlegend=False,
        )
        st.plotly_chart(wf, use_container_width=True)

    with cb_col_r:
        st.markdown(r"""
**IRB Capital Charge** (Basel III §272–284):

$$K = \bigl(\,L_{\alpha}(M) - \text{PD} \cdot \text{LGD}\,\bigr) \cdot \text{MA}(M_{\text{eff}})$$

mit $L_{\alpha}$ = ASRF-Loss-Quantil 99.9% und Maturity-Adjustment MA.

**Sequentielle Decomposition** (im Waterfall sichtbar):
1. Stufe 1: Nur PD↑, LGD unverändert → ΔK_PD
2. Stufe 2: PD bereits gestresst, jetzt zusätzlich LGD↑ → ΔK_LGD

Die zwei Beiträge sind **exakt additiv** zu ΔK — keine Cross-Terms.
""")
        total = bridge["delta_K"]
        if abs(total) > 1.0:
            pd_share  = bridge["delta_K_pd"]  / total * 100
            lgd_share = bridge["delta_K_lgd"] / total * 100
            st.metric(
                "Aufteilung ΔK · PD-Beitrag / LGD-Beitrag",
                f"{pd_share:.0f}%  /  {lgd_share:.0f}%",
                f"Gesamtveränderung Σ ΔK = €{total/1e9:+.1f} Mrd.",
            )
        st.caption(
            "RWA-Bridge analog: ΔRWA (Risk-Weighted Assets) = ΔK · 12.5 · EAD. "
            f"Aggregat ΔRWA = €{bridge['delta_rwa']/1e9:+.0f} Mrd. "
            "(Basel-III-Standardumrechnung: Faktor 12.5 = 1 / 8% Mindest-Kapitalquote)."
        )
else:
    st.info("Schock anwenden, um die Capital-Bridge zu aktivieren.")

st.divider()


# =====================================================================
# Stage 6 · Three Channels → CET1 Impact
# =====================================================================
eyebrow("Stufe 6 · Drei Channels → CET1-Quote (vorher/nachher)")
st.caption(
    f"Datenbasis · **Aggregat über {universe.n_banks} von {N_TOTAL} "
    "IRB-Banken** · drei Risiko-Kanäle zur CET1-Quote (Loan-Book + "
    "Sovereign-Bonds + Trading-Book). Klicke unten auf die einzelnen "
    "Bridge-Positionen für die jeweilige Berechnung."
)


@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_channel_data():
    cap = parse_capital_overview(EBA_RAW_DIR / "tr_oth.csv", period=202506)
    sov = parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv", period=202506)
    sov_mat = sovereign_maturity_ladder(sov, period=202506)
    bd  = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
    return cap, sov_mat, bd


if abs(m_used) > 1e-9:
    cap_over, sov_mat, bd = _load_channel_data()

    # Match top-N bank names to LEIs
    name_to_lei = {}
    loan_bridges_lei = {}
    for bank_name, portfolio in universe.banks.items():
        rows = bd[bd["bank_name"] == bank_name]
        if rows.empty:
            continue
        lei = rows["lei"].iloc[0]
        name_to_lei[bank_name] = lei
        loan_bridges_lei[lei] = portfolio.capital_bridge(
            z_factor=m_used, kappa_lgd=KAPPA_DOWNTURN_LGD, confidence=0.999,
            rho_multiplier=rho_mult,
            lgd_calibration=lgd_cal,
            stress_exponent=stress_exp,
            cap_mode=cap_mode,
        )

    sov_pnl_df = (rate_shock_pnl(sov_mat, delta_r_pp=delta_r_pp)
                  if abs(delta_r_pp) > 1e-5 else
                  pd.DataFrame(columns=["LEI_Code", "delta_pnl_eur"]))
    sov_lookup = dict(zip(sov_pnl_df.get("LEI_Code", []),
                          sov_pnl_df.get("delta_pnl_eur", [])))

    tb_stress = trading_book_stress(cap_over, m_factor=m_used)
    universe_leis = list(name_to_lei.values())
    cap_uni = cap_over[cap_over["LEI_Code"].isin(universe_leis)]
    tb_uni  = tb_stress[tb_stress["LEI_Code"].isin(universe_leis)]
    cet1_bridge = cet1_ratio_bridge(cap_uni, loan_bridges_lei,
                                     sov_lookup, tb_uni)

    # Aggregate channel contributions
    cet1_b = float(cet1_bridge["cet1_base"].sum())
    cet1_s = float(cet1_bridge["cet1_stress"].sum())
    rwa_b  = float(cet1_bridge["rwa_total_base"].sum())
    rwa_s  = float(cet1_bridge["rwa_total_stress"].sum())
    delta_cet1 = cet1_s - cet1_b
    delta_rwa  = rwa_s  - rwa_b

    # Decompose ΔCET1 into channels (loan-EL, sovereign-FV, TB-PnL)
    delta_loan_el = -float(sum(b["delta_el"] for b in loan_bridges_lei.values()))
    delta_sov_fv  = float(sum(sov_lookup.values())) if sov_lookup else 0.0
    delta_tb_pnl  = float(tb_uni["delta_tb_pnl_eur"].sum())
    delta_loan_rwa   = float(sum(b["delta_rwa"] for b in loan_bridges_lei.values()))
    delta_market_rwa = float(tb_uni["delta_rwa_market_eur"].sum())

    ratio_b = cet1_b / rwa_b if rwa_b > 0 else 0.0
    ratio_s = cet1_s / rwa_s if rwa_s > 0 else 0.0

    cc_col_l, cc_col_r = st.columns([3, 2], gap="medium")

    with cc_col_l:
        # Waterfall: CET1-Quote base → +channel contributions → CET1-Quote stress
        # Show in percentage-points
        def _pp_contrib(delta_cap: float, delta_rwa_ch: float) -> float:
            """Approximate impact on CET1 ratio from one channel."""
            if rwa_b <= 0:
                return 0.0
            # First-order: d(ratio) = dCET1/RWA - ratio*dRWA/RWA
            return (delta_cap / rwa_b - ratio_b * delta_rwa_ch / rwa_b) * 100

        ch_loan_el  = _pp_contrib(delta_loan_el, 0.0)
        ch_sov_fv   = _pp_contrib(delta_sov_fv, 0.0)
        ch_tb_pnl   = _pp_contrib(delta_tb_pnl, 0.0)
        ch_loan_rwa = _pp_contrib(0.0, delta_loan_rwa)
        ch_tb_rwa   = _pp_contrib(0.0, delta_market_rwa)

        # Residual to ensure waterfall ends exactly on stressed ratio
        explained = ch_loan_el + ch_sov_fv + ch_tb_pnl + ch_loan_rwa + ch_tb_rwa
        actual_change_pp = (ratio_s - ratio_b) * 100
        residual = actual_change_pp - explained

        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative",
                     "relative", "relative", "relative", "total"],
            x=["CET1-Quote<br>Baseline",
               "Loan-Book<br>ΔEL (Expected Loss)",
               "Loan-Book<br>ΔRWA (Vasicek)",
               "Sovereign-Bonds<br>ΔFV (Duration-MtM)",
               "Trading Book<br>ΔP&L (Haircut)",
               "Market-RWA<br>Uplift (FRTB)",
               "Second-Order-<br>Korrektur",
               "CET1-Quote<br>nach Stress"],
            text=[f"{ratio_b*100:.2f}%",
                  f"{ch_loan_el:+.2f} pp",
                  f"{ch_loan_rwa:+.2f} pp",
                  f"{ch_sov_fv:+.2f} pp",
                  f"{ch_tb_pnl:+.2f} pp",
                  f"{ch_tb_rwa:+.2f} pp",
                  f"{residual:+.2f} pp",
                  f"{ratio_s*100:.2f}%"],
            textposition="outside",
            textfont=dict(size=10, color=COLORS["navy"]),
            y=[ratio_b * 100, ch_loan_el, ch_loan_rwa, ch_sov_fv,
               ch_tb_pnl, ch_tb_rwa, residual, ratio_s * 100],
            connector={"line": {"color": COLORS["hairline"], "width": 1}},
            increasing={"marker": {"color": COLORS["teal"]}},
            decreasing={"marker": {"color": COLORS["crimson"]}},
            totals    ={"marker": {"color": COLORS["navy"]}},
        ))
        # Add regulatory threshold lines
        wf.add_hline(y=4.5, line_dash="dot", line_color=COLORS["crimson"],
                     annotation_text="4.5% CET1-Minimum (Basel-III Pillar 1)",
                     annotation_position="left",
                     annotation_font=dict(size=9, color=COLORS["crimson"]))
        wf.add_hline(y=7.0, line_dash="dot", line_color=COLORS["amber"],
                     annotation_text="7% CET1 inkl. Capital Conservation Buffer (CCB)",
                     annotation_position="left",
                     annotation_font=dict(size=9, color=COLORS["amber"]))
        wf.update_layout(
            title=f"CET1-Quote (Common Equity Tier 1) · 3-Channel-Decomposition · "
                  f"Aggregat Top-{universe.n_banks} EU-Banken",
            yaxis_title="CET1-Quote [%] = Hartes Kernkapital ÷ Risk-Weighted Assets",
            height=460, showlegend=False,
        )
        st.plotly_chart(wf, use_container_width=True)

    with cc_col_r:
        st.markdown(r"""
**CET1-Quote unter Stress** — 3 Channels:

1. **Loan-Book**:  ΔEL ↓ CET1-Kapital direkt, ΔRWA ↓ Quote über Nenner
2. **Sovereign**:  ΔFV durchläuft P&L (HfT/FVTPL) und OCI (FVOCI) → CET1
3. **Trading Book**:  TB-P&L direkt durch CET1, Market-RWA-Uplift via Nenner

**Regulatorische Thresholds** (gestrichelte Linien):
- 4.5% CET1-Minimum (Pillar 1)
- 7% CET1 inkl. CCB (Pillar 1 + Capital Conservation Buffer)
""")
        st.metric(
            "CET1-Quote · vorher → nachher",
            f"{ratio_b*100:.2f}%  →  {ratio_s*100:.2f}%",
            f"Veränderung: {(ratio_s-ratio_b)*100:+.2f} Prozentpunkte",
        )
        st.metric(
            "Σ CET1-Kapital · hartes Kernkapital",
            f"€{cet1_b/1e9:.0f} Mrd. → €{cet1_s/1e9:.0f} Mrd.",
            f"Veränderung: {(cet1_s-cet1_b)/1e9:+.1f} Mrd. EUR",
        )
        st.metric(
            "Σ Risk-Weighted Assets (RWA) · risikogewichtete Aktiva",
            f"€{rwa_b/1e9:.0f} Mrd. → €{rwa_s/1e9:.0f} Mrd.",
            f"Veränderung: {(rwa_s-rwa_b)/1e9:+.0f} Mrd. EUR",
        )

    # Channel-attribution narrative
    biggest_drop = min(
        [("Loan-Book EL", ch_loan_el),
         ("Loan-Book ΔRWA", ch_loan_rwa),
         ("Sovereign ΔFV", ch_sov_fv),
         ("Trading Book P&L", ch_tb_pnl),
         ("Market-RWA Uplift", ch_tb_rwa)],
        key=lambda x: x[1],
    )
    insight(
        f"<strong>Dominanter Channel:</strong> {biggest_drop[0]} "
        f"({biggest_drop[1]:+.2f} pp). "
        f"Loan-Book trägt {(ch_loan_el+ch_loan_rwa):+.2f} pp · "
        f"Sovereign-FV {ch_sov_fv:+.2f} pp · "
        f"Trading-Book {(ch_tb_pnl+ch_tb_rwa):+.2f} pp. "
        f"Full bank-by-bank-Decomposition auf der "
        f"<em>Capital Adequacy</em>-Page."
    )

    # =================================================================
    # Info-Ribbons · klickbare Erklärung pro Bridge-Position
    # =================================================================
    st.markdown(" ")
    st.markdown(
        '<div style="font-size:0.85rem;color:#034B6F;font-weight:600;'
        'letter-spacing:0.05em;text-transform:uppercase;'
        'margin:0.6rem 0 0.4rem 0;">'
        'Info-Ribbon · klicke auf eine Bridge-Position für Definition + Berechnung'
        '</div>',
        unsafe_allow_html=True,
    )

    info_cols = st.columns(7, gap="small")
    with info_cols[0]:
        with st.expander("ⓘ CET1 base", expanded=False):
            st.markdown(rf"""
**Definition.** Hartes Kernkapital (CET1) der Aggregat-Bank
vor jedem Schock.

**Berechnung.**
$$\text{{CET1}}^{{\text{{base}}}} = \sum_{{b \in 67}} \text{{CET1}}_b$$

**Aktueller Wert.** €{cet1_b/1e9:,.1f} Mrd. EUR
(Summe über {universe.n_banks} IRB-Banken).

**EBA-Datenquelle.** `tr_oth.csv` · Item 2520102
(Common Equity Tier 1 Capital, regulatorisch).
""")
    with info_cols[1]:
        with st.expander("ⓘ Loan-Book ΔEL", expanded=False):
            st.markdown(rf"""
**Definition.** Veränderung des Expected Loss im Loan-Book
durch höhere Conditional-PD und Downturn-LGD. Erhöht die
Provisions und reduziert direkt das CET1-Kapital.

**Berechnung pro Segment.**
$$\Delta \text{{EL}} = (\text{{PD}}(M) \cdot \text{{LGD}}(M) - \text{{PD}} \cdot \text{{LGD}}) \cdot \text{{EAD}}$$

dann aggregiert pro Bank, dann über alle 67 IRB-Banken.

**Aktueller Beitrag zur CET1-Quote.**
{ch_loan_el:+.2f} pp = {delta_loan_el/1e9:+.1f} Mrd. EUR
(ΔEL_loan ÷ RWA_base).

**Wirkung.** Reduziert CET1-Zähler.
""")
    with info_cols[2]:
        with st.expander("ⓘ Loan-Book ΔRWA", expanded=False):
            st.markdown(rf"""
**Definition.** Veränderung der Credit-RWA durch IRB-Capital-
Bridge — gestresste PD + LGD ergeben höhere Capital-Anforderung
K und damit höheres Risk-Weighted-Asset.

**Berechnung.**
$$\Delta \text{{RWA}}_{{\text{{cr}}}} = \Delta K \cdot 12.5 \cdot \text{{EAD}}$$

mit $K = (L_{{0.999}} - \text{{PD}} \cdot \text{{LGD}}) \cdot \text{{MA}}(M_{{\text{{eff}}}})$
und $L_{{0.999}}$ = ASRF-Loss-Quantil bei 99.9% Konfidenz.

**Aktueller Beitrag.**
{ch_loan_rwa:+.2f} pp = ΔRWA {delta_loan_rwa/1e9:+.0f} Mrd. EUR
(via Verdünnung im Nenner).

**Wirkung.** Erhöht RWA-Nenner → Quote sinkt.
""")
    with info_cols[3]:
        with st.expander("ⓘ Sovereign ΔFV", expanded=False):
            st.markdown(rf"""
**Definition.** Mark-to-Market-Verlust auf Sovereign-Anleihen
durch Zinsanstieg. Wirkt für FVOCI- und FVTPL-Bestände direkt
durch OCI / P&L auf CET1.

**Berechnung pro Bank.**
$$\Delta \text{{FV}} = - \sum_{{m \in \text{{buckets}}}} D_m \cdot \Delta y \cdot E_{{b,m}}$$

mit $D_m$ = Modified-Duration des Maturity-Buckets,
$\Delta y$ = aktueller Δr_10y in Dezimal.

**Aktueller Beitrag.**
{ch_sov_fv:+.2f} pp = {delta_sov_fv/1e9:+.1f} Mrd. EUR
ΔFV (über alle Buckets).

**Wirkung.** Reduziert CET1 wenn Zinsen steigen.
""")
    with info_cols[4]:
        with st.expander("ⓘ TB-P&L", expanded=False):
            st.markdown(rf"""
**Definition.** Trading-Book-P&L-Haircut bei adversem Stress.
Reduziert das Net-Income und damit direkt CET1.

**Berechnung.**
$$\Delta \text{{P\&L}}_{{\text{{TB}}}} = -\kappa_{{\text{{PnL}}}} \cdot |M| \cdot \text{{TB-P\&L}}_{{\text{{base}}}}$$

mit $\kappa_{{\text{{PnL}}}}$ = 0.50 / 2.5 = 0.20 pro |M|.

**Aktueller Beitrag.**
{ch_tb_pnl:+.2f} pp = {delta_tb_pnl/1e9:+.1f} Mrd. EUR
TB-P&L-Verlust.

**Wirkung.** Reduziert CET1-Zähler nur bei M < 0.
""")
    with info_cols[5]:
        with st.expander("ⓘ Market-RWA", expanded=False):
            st.markdown(rf"""
**Definition.** FRTB-style Multiplier auf das aggregate Market-
Risk-RWA bei adversem Stress (proxy für VaR/SVaR-Anstieg).

**Berechnung.**
$$\Delta \text{{RWA}}_{{\text{{mr}}}} = \kappa_{{\text{{RWA}}}} \cdot |M| \cdot \text{{RWA}}_{{\text{{mr}}}}^{{\text{{base}}}}$$

mit $\kappa_{{\text{{RWA}}}}$ = 0.30 / 2.5 = 0.12 pro |M|.

**Aktueller Beitrag.**
{ch_tb_rwa:+.2f} pp = ΔRWA {delta_market_rwa/1e9:+.0f} Mrd. EUR
(via Nenner).

**Wirkung.** Erhöht RWA-Nenner nur bei M < 0.
""")
    with info_cols[6]:
        with st.expander("ⓘ CET1 stress", expanded=False):
            st.markdown(rf"""
**Definition.** CET1-Quote nach Stress = aggregierte
Eigenkapital-Quote über alle drei Channels.

**Berechnung.**
$$\text{{Quote}}^{{\text{{stress}}}} = \frac{{\text{{CET1}}^{{\text{{base}}}} - \Delta\text{{EL}}_{{\text{{loan}}}} + \Delta\text{{FV}}_{{\text{{sov}}}} + \Delta\text{{P\&L}}_{{\text{{TB}}}}}}{{\text{{RWA}}_{{\text{{base}}}} + \Delta\text{{RWA}}_{{\text{{cr}}}} + \Delta\text{{RWA}}_{{\text{{mr}}}}}}$$

**Aktueller Wert.** {ratio_s*100:.2f}%
({(ratio_s-ratio_b)*100:+.2f} pp gegenüber {ratio_b*100:.2f}% Baseline).

**Regulatorische Schwellen.** 4.5% (Pillar 1) ·
7.0% (inkl. CCB) · 8.0% (typischer SREP-Target).
""")

else:
    st.info("Schock anwenden, um die 3-Channel-CET1-Bridge live zu sehen.")

st.divider()


# =====================================================================
# Compact summary table — entire chain at-a-glance
# =====================================================================
# =====================================================================
# Full bank-level data basis (transparency block)
# =====================================================================
with st.expander(
    f"Datenbasis · alle {N_TOTAL} EBA-Banken (aufklappen für vollständige Liste)",
    expanded=False,
):
    st.caption(
        f"Komplette Bank-Liste aus der EBA Transparency Exercise 2025, "
        f"sortiert nach EAD. Die Aggregat-Berechnungen oben aggregieren "
        f"die durch die Sidebar-Auswahl markierten {universe.n_banks} Banken."
    )

    sel_set = set(selected_banks)
    table_rows = []
    for rank, (name, ead) in enumerate(_bank_ead_pairs, start=1):
        portfolio = universe_full.banks[name]
        base = portfolio.portfolio_kpis(confidence=0.999)
        table_rows.append({
            "Rang":         rank,
            "Bank":         name,
            "EAD (Mrd. €)": round(ead / 1e9, 1),
            "EAD-Anteil":   f"{ead/_total_ead_full*100:.2f}%",
            "EL Baseline (Mrd. €)": round(base["el_eur"] / 1e9, 2),
            "RWA Baseline (Mrd. €)": round(base["rwa"] / 1e9, 1),
            "RWA-Dichte":   f"{base['rwa_density']*100:.0f}%",
            "In Aggregat?": "Ja" if name in sel_set else "—",
        })
    bank_table = pd.DataFrame(table_rows)
    st.dataframe(
        bank_table, use_container_width=True, hide_index=True,
        height=min(560, 38 + 32 * len(bank_table)),
    )
    st.caption(
        "EL = Expected Loss · RWA = Risk-Weighted Assets · "
        "RWA-Dichte = RWA ÷ EAD (effektives Risikogewicht). "
        "Auswahl-Modus in der Sidebar steuert, welche Banken ins Aggregat einfließen."
    )

st.divider()


eyebrow("Wirkungskette · Zusammenfassung in einer Tabelle")

summary_rows = [
    ("1. Eingangs-Schock",
     "ΔBrent (Ölpreis) + Δr_10y (10-Jahres-Zins)",
     f"ΔBrent = {delta_brent_log:+.2f} log  ·  Δr_10y = {delta_r_pp:+.0f} bp"),
    ("2. Macro → Systemfaktor M",
     "Hybrid-Mapping (Anchor-Route + Mahalanobis-Route)",
     f"Systemfaktor M = {m_used:+.2f}σ"),
    ("3. M → Conditional PD",
     "Vasicek Conditional-PD pro Exposure-Class (BCBS-2017-IRB)",
     (f"Stärkster Klassen-ΔPD: {class_base.sort_values('pd_delta_pp', ascending=False).iloc[0]['exposure_class']} "
      f"+{class_base['pd_delta_pp'].max():.2f} pp"
      if abs(m_used) > 1e-3 else "Baseline (kein Schock angelegt)")),
    ("4. M → Downturn LGD",
     "Asymmetrische LGD-Funktion (EBA-Stresstest 2023, κ = 0.30)",
     (f"LGD-Uplift = +{KAPPA_DOWNTURN_LGD*abs(min(m_used,0))*100:.0f}% relativ zur Baseline"
      if m_used < 0 else "0% (M ≥ 0 → keine Erhöhung)")),
    ("5. PD + LGD → IRB Capital K",
     "Sequentielle Capital-Bridge (PD-Effekt + LGD-Effekt, additiv)",
     (f"ΔK (Capital Requirement) = €{bridge['delta_K']/1e9:+.1f} Mrd."
      if abs(m_used) > 1e-9 else "Baseline")),
    ("6. Drei Channels → CET1-Quote",
     "Loan-Book ΔRWA/ΔEL + Sovereign ΔFV + Trading-Book P&L/RWA",
     (f"CET1-Quote: {ratio_b*100:.2f}% → {ratio_s*100:.2f}% "
      f"(Veränderung: {(ratio_s-ratio_b)*100:+.2f} Prozentpunkte)"
      if abs(m_used) > 1e-9 else "Baseline")),
]
summary_df = pd.DataFrame(summary_rows,
                          columns=["Stufe der Wirkungskette",
                                    "Methodik · Modell-Baustein",
                                    "Ergebnis unter aktuellem Schock"])
st.dataframe(summary_df, use_container_width=True, hide_index=True, height=260)


footer(
    f"Source: {universe.source} · Anchor: {anchor.label} "
    f"(z = {anchor.z_factor:+.2f}) · Factor Σ: {cov_source} · "
    f"Aggregat über Top-{universe.n_banks} EBA-Banken · κ_LGD = "
    f"{KAPPA_DOWNTURN_LGD:.2f}"
)
