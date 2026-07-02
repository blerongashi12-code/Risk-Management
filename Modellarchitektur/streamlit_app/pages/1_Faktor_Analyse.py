"""Faktor-Analyse & Transmission · empirische Validierung + Stress-Bridge.

Diese Page liefert zwei Dinge auf einer Seite:

  A. die <em>empirische Validierung</em> der zwei gewählten Macro-Faktoren —
     5-Jahres-Korrelations-Analyse Brent vs. Δr_10y (Kennzahlen-Karten + Diagnostik);
  B. die <em>sequentielle Stress-Bridge</em> ΔBrent + Δr_10y → ΔPD → ΔLGD →
     IRB-Capital → CET1, durchgerechnet auf dem Top-10-EU-IRB-Universe.

Stufen (jeweils Formel + Chart + Live-Aussage):
  1. Input               — Echo der Sidebar-Slider (ΔBrent, Δr_10y)
  2. ΔPD pro Klasse      — 2-Faktor-Sensitivitäten je Exposure-Class
  3. ΔLGD pro Klasse     — Downturn-LGD-Aufschlag aus dem 2-Faktor-Modell
  4. PD + LGD → Capital  — IRB-Capital-Bridge (BCBS 2017, weiterhin aktiv)
  5. Zwei Kanäle → CET1   — Loan-Book + Sovereign aggregiert
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
                              COLORS, PALETTE_DISCRETE)
from components.sidebar import render_sidebar
from components.data_loader import load_data_layer
from components.backend_path import setup
setup()

from eba_loader import (load_eba_universe,                          # type: ignore
                         parse_capital_overview,
                         parse_sovereign_csv, sovereign_by_accounting_class,
                         sovereign_cet1_pnl_lookup,
                         cet1_ratio_bridge, load_bank_directory)
from macro_factor import (anchor_from_eba,                          # type: ignore
                           factor_returns, factor_stats)
from config import EBA_RAW_DIR                                       # type: ignore


st.set_page_config(page_title="Faktor-Analyse & Transmission",
                   layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Faktor-Analyse & Transmission",
    eyebrow="Tab 1 · empirische Faktor-Validierung + sequentielle "
            "Stress-Bridge",
    deck="Zwei Bausteine auf einer Seite. Oben: die <strong>empirische "
         "Korrelations-Analyse</strong> (5 Jahre tägliche Daten) — "
         "sie belegt, dass Brent und der 10-Jahres-Zins zwei "
         "unabhängige Risiko-Faktoren sind und deshalb nicht zu einem "
         "Einzelfaktor aggregiert werden dürfen. Unten: die "
         "<strong>sequentielle Stress-Bridge</strong> ΔBrent + Δr_10y → "
         "ΔPD → ΔLGD → IRB-Capital → CET1, durchgerechnet auf dem "
         "Top-10-EU-IRB-Universe. Datenbasis: bank-spezifische Pillar-3-"
         "EU-CR6-Disclosures (10/10 Banken Pillar-3-verifiziert) am "
         "einheitlichen Stichtag 31.12.2024.",
)

tab_breadcrumb(1)
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


@st.cache_data(ttl=24*3600, show_spinner="Loading EBA Top-10 universe …")
def _load_full_universe():
    """Loads the Top-10 IRB-bank universe with regulatory PDs from
    the bank-specific Pillar-3 EU-CR6 disclosures at 31.12.2024
    (data/pillar3_bank_pd_lgd.csv).
    """
    from eba_pd_loader import filter_universe_to_top10                # type: ignore
    u = load_eba_universe(vintage="2025", top_n=None, prefer_real=True)
    return filter_universe_to_top10(u)


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
    with st.expander("Datenbasis", expanded=False):
        st.caption(
            f"Feste Datenbasis: **alle {N_TOTAL} kuratierten EU-IRB-Banken** "
            f"mit bank-spezifischen Pillar-3-PDs/LGDs (31.12.2024). "
            f"Σ EAD = €{_total_ead_full/1e12:.1f} Bio. Keine Top-N- oder "
            f"Einzelauswahl — alle Tabs nutzen dasselbe 10-Banken-Universe."
        )

# Feste Datenbasis: immer alle 10 kuratierten Banken (keine Teilauswahl).
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

# Coverage-Status aus dem Loader ziehen
try:
    from eba_pd_loader import coverage_report                  # type: ignore
    _pd_cov = coverage_report()
    _n_verified = len(_pd_cov.get("verified_banks", []))
    _verified_list = _pd_cov.get("verified_banks", [])
    _status_counts = _pd_cov.get("by_status", {})
    _direct_cells = int(_status_counts.get("pillar3_verified", 0))
    _eligible_cells = int(_pd_cov.get("n_irb_eligible", _direct_cells))
    _na_cells = int(_pd_cov.get("n_rows_total", 70)) - _eligible_cells
except Exception:
    _n_verified = 0
    _verified_list = []
    _direct_cells, _eligible_cells, _na_cells = 0, 0, 0

_verified_html = " · ".join(_verified_list) if _verified_list else "—"
# Compute pending banks (all banks where NO row is pillar3_verified)
try:
    from eba_pd_loader import load_pd_table                  # type: ignore
    _df_pd = load_pd_table()
    _verified_banks_set = set(_df_pd.loc[_df_pd["status"] == "pillar3_verified",
                                           "bank_name"].unique())
    _all_banks_set = set(_df_pd["bank_name"].unique())
    _pending_banks = sorted(_all_banks_set - _verified_banks_set)
except Exception:
    _pending_banks = []
_pending_html  = " · ".join(_pending_banks) if _pending_banks else "keine"

st.markdown(
    f'<div style="background:#F4F4F4;border-left:4px solid {_coverage_color};'
    f'padding:0.95rem 1.1rem;border-radius:6px;margin-bottom:0.8rem;'
    f'color:#051C2C;font-size:0.92rem;line-height:1.6;">'
    f'<div style="font-weight:600;font-size:1.0rem;margin-bottom:0.4rem;'
    f'color:{_coverage_color};">Datenbasis-Coverage · '
    f'10 EU-IRB-Top-Banken &nbsp;·&nbsp; PD/LGD-Quelle</div>'
    f'<table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'

    # --- Zeile 1: Ausgangs-Universe ---
    f'<tr><td style="padding:0.3rem 0.6rem 0.3rem 0;width:200px;color:#6E6E6E;'
    f'vertical-align:top;">Ausgangs-Universe</td>'
    f'<td><strong>120 Banken aus 26 Ländern</strong> · Stichtag Juni 2025 · '
    f'Quelle: <em>EBA Transparency Exercise 2025</em>.</td></tr>'

    # --- Zeile 2: Filter IRB ---
    f'<tr><td style="padding:0.3rem 0.6rem 0.3rem 0;color:#6E6E6E;'
    f'vertical-align:top;">Filter 1 · IRB-Approach</td>'
    f'<td><strong>67 von 120 Banken</strong> nutzen den Internal-Ratings-'
    f'Based-Ansatz (Basel III · CRR Art. 142 ff.) — nur diese liefern bank-'
    f'interne PD/LGD über die EBA-COREP-C-9.02-Templates.</td></tr>'

    # --- Zeile 3: Top-10 ---
    f'<tr><td style="padding:0.3rem 0.6rem 0.3rem 0;color:#6E6E6E;'
    f'vertical-align:top;">Filter 2 · Top-10</td>'
    f'<td><strong>Crédit Agricole · BNP Paribas · ING · Société Générale · '
    f'BPCE · Deutsche Bank · Crédit Mutuel · Santander · Rabobank · '
    f'UniCredit</strong> — Auswahl nach kombiniertem Score aus '
    f'Σ EAD + Klassen-Coverage. Repräsentieren €'
    f'{_total_ead_full/1e12:.1f} Bio. EAD ≈ 55 % der EU-IRB-EAD.</td></tr>'

    # --- Zeile 4: PD-Quelle pro Bank ---
    f'<tr><td style="padding:0.5rem 0.6rem 0.5rem 0;color:#6E6E6E;'
    f'vertical-align:top;">PD/LGD-Quelle pro Bank</td>'
    f'<td><strong style="color:#1E7A4E;">Pillar-3 verifiziert '
    f'({_n_verified}/10):</strong> {_verified_html}<br>'
    f'<span style="color:#6E6E6E;">→ Quelle: jeweilige Pillar-3-Reports der '
    f'Bank, EU-CR6-Tabelle (regulatorisch publiziert nach CRR Art. 431–455, '
    f'EBA ITS/2020/04). PDs sind die EAD-gewichteten 1-Jahres-Average-PD '
    f'pro IRB-Exposure-Klasse.</span><br><br>'
    f'<strong style="color:{("#A52F4D" if _pending_banks else "#1E7A4E")};">'
    f'Banken vollständig auf Country-Aggregat '
    f'({len(_pending_banks)}/10):</strong> {_pending_html}<br>'
    f'<span style="color:#6E6E6E;">→ {_direct_cells} von '
    f'{_eligible_cells} IRB-fähigen Bank-Klassen enthalten PD und LGD '
    f'direkt aus der bankeigenen EU-CR6-Tabelle. {_na_cells} Kombination '
    f'ist nicht anwendbar: Santander Sovereign läuft zu 100 % im '
    f'Standardansatz und wird aus dem IRB-Kreditbuch ausgeschlossen.'
    f'</span></td></tr>'

    # --- Zeile 5: Stichtag ---
    f'<tr><td style="padding:0.5rem 0.6rem 0.5rem 0;color:#6E6E6E;'
    f'vertical-align:top;">Stichtag · Baseline-PD</td>'
    f'<td><strong>31.12.2024 einheitlich</strong> für alle 10 Banken und '
    f'alle 70 Bank-Klassen-Kombinationen. Methodisch im '
    f'EBA-Stress-Test-2025-Standard '
    f'verankert: ein <em>uniformer „Heute"-Snapshot</em> ist der Ausgangs-'
    f'punkt, von dem aus der Macro-Schock (ΔBrent + Δr_10y) als '
    f'Forward-Stress simuliert wird. Vintage-Mix wäre methodisch '
    f'unzulässig.<br>'
    f'<span style="color:#6E6E6E;font-size:0.84rem;">'
    f'<strong>Backtest-Annahme (2020–2024):</strong> Die 31.12.2024-PDs '
    f'gelten als Baseline-Proxy für alle historischen Quartale. '
    f'A-IRB-PDs sind <em>Through-the-Cycle</em> (CRR Art. 180) — '
    f'glättet sich auf ±0,1–0,3 pp Quartalsdrift. Die Macro-Dynamik '
    f'des Backtests sitzt in den Brent + Δr-Zeitreihen + β-'
    f'Sensitivitäten, nicht im PD-Level.</span></td></tr>'

    # --- Zeile 6: Aktuelle Aggregation ---
    f'<tr><td style="padding:0.3rem 0.6rem 0.3rem 0;color:#6E6E6E;'
    f'vertical-align:top;">Aktive Aggregation</td>'
    f'<td><strong>{universe.n_banks} von 10 Banken</strong> · Σ EAD = €'
    f'{_selected_ead/1e12:.2f} Bio. ({_ead_share:.0f} % der Top-10-Σ-EAD) '
    f'· in der Sidebar konfigurierbar.</td></tr>'

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


# --- 2-Faktor-Stress aktiv auf das Universum anwenden ----------------
# Konsistent mit Kreditbuch und Eigenkapital: das Universum wird in-place
# gestresst, sodass jedes Segment in `.pd` und `.lgd` die gestressten Werte
# trägt; `_pd_base` / `_lgd_base` bleiben als Backup zugänglich.
from two_factor_stress import (apply_stress_to_universe,            # type: ignore
                                 stress_pd, stress_lgd,
                                 SENSITIVITY_MATRIX)


def _snapshot_segments_df(uni) -> pd.DataFrame:
    """Per-Segment-Snapshot des aktuellen Universums (PD/LGD/EAD/ρ)."""
    rows = []
    for bank_name, port in uni.banks.items():
        for s in port.segments:
            rows.append({
                "bank":           bank_name,
                "exposure_class": s.exposure_class,
                "ead":            float(s.ead),
                "pd":             float(s.pd),
                "lgd":            float(s.lgd),
                "rho":            float(getattr(s, "rho", 0.15)),
            })
    return pd.DataFrame(rows)


# Schritt A · Baseline-Snapshot vor dem Stress (PD/LGD = regulatorische
# bank-spezifische Pillar-3 EU-CR6 31.12.2024, ohne 2-Faktor-Verschiebung)
_seg_baseline = _snapshot_segments_df(universe)

# Schritt B · 2-Faktor-Stress in-place anwenden
_sens_overrides = config.get("sensitivity_overrides")
apply_stress_to_universe(
    universe, delta_brent_log, delta_r_pp,
    override_betas=_sens_overrides,
)
_is_stressed = (abs(delta_brent_log) > 1e-6) or (abs(delta_r_pp) > 1e-6)

# Schritt C · Stress-Snapshot nach dem 2-Faktor-Shift
_seg_stress = _snapshot_segments_df(universe)

# Vasicek-Pipeline läuft hier nur als "IRB-Capital-Rechner" — kein zusätzlicher
# M-Shift. Die Stress-Wirkung sitzt bereits in segment.pd / segment.lgd.
m_used = 0.0

# Anchor + corr_hat bleiben für die Korrelations-Sektion oben (Beweis-Block)
# und potentielle zukünftige Diagnostik erhalten.
anchor = anchor_from_eba("2025")


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
| **β_oil** | Oil-Sensitivität | ΔPD/ΔLGD-Reaktion pro Klasse auf Brent-Schock (NEU im 2-Faktor-Modell) |
| **β_rate** | Rate-Sensitivität | ΔPD/ΔLGD-Reaktion pro Klasse auf Δr_10y-Schock (z.B. negativ für Bank-Klasse = NIM-Effekt) |
| **M** | Systemic Factor (legacy) | Vorgänger-Methodik: Vasicek-Systemfaktor — durch 2-Faktor-Modell ersetzt, hier nur Referenz-View in Stage 2-5 |
| **ρ (rho)** | Asset Correlation | Vermögens-Korrelation pro Exposure-Class (Basel IRB-Formel) |
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
| **BCBS** | Basel Committee on Banking Supervision | Basler Ausschuss · Referenz für Basel-III-IRB (Vasicek-Modell)-IRB-Kalibrierung |
| **FRTB** | Fundamental Review of the Trading Book | Neues Markt-Risiko-Rahmenwerk (Basel III, Phase 2) |
""")

st.markdown(" ")  # vertical breathing room


# =====================================================================
# Ökonomisches Fundament · 5-Jahres-Korrelations-Analyse Brent vs Δr_10y
# =====================================================================
eyebrow("Ökonomisches Fundament · sind Brent und 10y-Zins überhaupt zwei "
        "getrennte Faktoren?")

st.markdown(
    '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
    'border-left:4px solid #034B6F;padding:0.95rem 1.2rem;'
    'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
    'font-size:0.9rem;line-height:1.7;">'
    '<strong>Warum diese Analyse zuerst?</strong><br>'
    'Bevor wir mit Brent und dem 10y-Zins als <em>zwei getrennten</em> '
    'Stress-Faktoren arbeiten, muss eine Frage empirisch beantwortet '
    'sein: <em>Sind die beiden überhaupt unabhängig genug, um sie '
    'separat zu modellieren?</em> Wenn ja, ist die zweidimensionale '
    'Stress-Steuerung ökonomisch fundiert. Wenn nein, würden wir '
    'denselben Effekt zweimal zählen.<br><br>'
    '<strong>Diese Frage adressiert direkt drei Kritikpunkte unseres '
    'betreuenden Professors:</strong> '
    'Punkt 5 (Faktor-Konstruktion), Punkt 8 (separate Modellierung) '
    'und Punkt 9 (empirische Korrelations-Analyse). Wir prüfen sie '
    'mit drei komplementären Tests über die letzten 5 Jahre täglicher '
    'Daten (die genaue Zahl der Handelstage steht oben im '
    'Kennzahlen-Streifen):'
    '<ul style="margin:0.5rem 0 0 1.2rem;padding:0;">'
    '<li><strong>Test 1:</strong> Pearson-Korrelation über das volle '
    'Sample (5 Jahre) — wie stark hängen Tagesveränderungen linear '
    'zusammen?</li>'
    '<li><strong>Test 2:</strong> Rolling-252d-Korrelation als '
    'Zeitreihe — ist die Korrelation über Zeit stabil oder ändert '
    'sie sich (z.B. nur in Krisen)?</li>'
    '<li><strong>Test 3:</strong> OLS-Regression Δr_10y ~ '
    'α + β·ΔBrent — wie viel Varianz erklärt der eine Faktor des '
    'anderen?</li>'
    '</ul>'
    '</div>',
    unsafe_allow_html=True,
)

# === Live correlation analysis ============================================
from factor_correlation import (                                  # type: ignore
    compute_factor_correlation,
    rolling_correlation_series,
    yearly_correlation_table,
)

@st.cache_data(ttl=24*3600, show_spinner="Berechne 5-Jahres-Korrelations-Analyse …")
def _run_correlation_analysis():
    _data = load_data_layer()
    if _data["brent"] is None or _data["svensson"] is None:
        return None
    res = compute_factor_correlation(_data["brent"], _data["svensson"],
                                      years=5)
    roll = rolling_correlation_series(_data["brent"], _data["svensson"],
                                       window=252, years=5)
    yearly = yearly_correlation_table(_data["brent"], _data["svensson"],
                                       years=5)
    return {"summary": res, "rolling": roll, "yearly": yearly}


_corr_res = _run_correlation_analysis()

if _corr_res is None:
    st.error("Daten-Cache fehlt — `python tools/data_fetch/03_fetch_brent_crude.py` und "
             "`python tools/data_fetch/04_fetch_bundesbank_svensson.py` zuerst ausführen.")
else:
    _summary = _corr_res["summary"]
    _rolling = _corr_res["rolling"]
    _yearly  = _corr_res["yearly"]

    # ----- KPI-Strip with the four headline numbers ---------------
    cc1, cc2, cc3, cc4, cc5 = st.columns(5, gap="small")
    cc1.metric("Beobachtungen", f"{_summary['n_obs']:,}",
               "Handelstage", delta_color="off")
    cc2.metric("Pearson ρ",
               f"{_summary['pearson_rho']:+.3f}",
               f"95%-KI: [{_summary['pearson_rho_ci_lo']:+.2f}, "
               f"{_summary['pearson_rho_ci_hi']:+.2f}]",
               delta_color="off")
    cc3.metric("Spearman ρ (Rang)",
               f"{_summary['spearman_rho']:+.3f}",
               "robust gegen Ausreißer", delta_color="off")
    cc4.metric("OLS β", f"{_summary['ols_beta']:+.3f}",
               f"t = {_summary['ols_t_beta']:+.2f}", delta_color="off")
    cc5.metric("OLS R²", f"{_summary['ols_r2']:.3f}",
               "erklärte Varianz", delta_color="off")

    # ----- Beweis-Block · drei Kennzahlen-Karten (ersetzt R-lm-Output) ----
    def _p_fmt(p):
        if p < 2e-16: return "<2e-16"
        if p < 0.001: return f"{p:.2e}"
        return f"{p:.4f}"

    _acc = (COLORS["teal"] if abs(_summary["pearson_rho"]) < 0.10
            else COLORS["mid_blue"] if abs(_summary["pearson_rho"]) < 0.30
            else COLORS["amber"])
    _ci_lo, _ci_hi = (_summary["pearson_rho_ci_lo"],
                      _summary["pearson_rho_ci_hi"])
    _ci_excl0 = (_ci_lo > 0) or (_ci_hi < 0)
    _ci_txt = ("schließt die Null aus — die (schwache) Korrelation ist "
               "statistisch belegt." if _ci_excl0 else
               "schließt die Null ein — die Korrelation ist statistisch "
               "nicht von null unterscheidbar.")
    _sig_word = ("hoch signifikant" if _summary["ols_p_beta"] < 0.001 else
                 "signifikant" if _summary["ols_p_beta"] < 0.05 else
                 "nicht signifikant")
    _r2_pct = _summary["ols_r2"] * 100.0

    def _proof_card(kicker, big, unit, body):
        return (
            f'<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
            f'border-top:4px solid {_acc};border-radius:8px;'
            f'padding:0.8rem 0.95rem;height:100%;">'
            f'<div style="font-size:0.64rem;font-weight:700;'
            f'letter-spacing:0.08em;text-transform:uppercase;'
            f'color:#6E6E6E;margin-bottom:0.15rem;">{kicker}</div>'
            f'<div style="font-family:Source Serif Pro,Georgia,serif;'
            f'font-size:1.5rem;font-weight:700;color:#051C2C;'
            f'line-height:1.1;margin-bottom:0.2rem;">{big}'
            f'<span style="font-size:0.85rem;color:#6E6E6E;font-weight:600;">'
            f' {unit}</span></div>'
            f'<div style="font-size:0.78rem;color:#3A4A57;line-height:1.45;">'
            f'{body}</div></div>'
        )

    _pc1, _pc2, _pc3 = st.columns(3, gap="small")
    with _pc1:
        st.markdown(_proof_card(
            "Test 1 · lineare Korrelation",
            f'{_summary["pearson_rho"]:+.3f}', "ρ",
            f'95%-Konfidenzintervall (Fisher r-to-z): '
            f'[{_ci_lo:+.3f}, {_ci_hi:+.3f}]. Das Intervall {_ci_txt}'),
            unsafe_allow_html=True)
    with _pc2:
        st.markdown(_proof_card(
            "Test 3 · OLS-Steigung β",
            f'{_summary["ols_beta"]:+.3f}', "pp je Δlog",
            f't = {_summary["ols_t_beta"]:+.2f}, p = '
            f'{_p_fmt(_summary["ols_p_beta"])} ({_sig_word}). β misst, um '
            f'wie viele Prozentpunkte sich Δr_10y je Brent-log-Return '
            f'verschiebt — hier praktisch flach.'),
            unsafe_allow_html=True)
    with _pc3:
        st.markdown(_proof_card(
            "Test 3 · erklärte Varianz R²",
            f'{_r2_pct:.2f}', "%",
            f'So viel der täglichen Zinsbewegung erklärt der Brent-Schock. '
            f'Der Rest ({100.0 - _r2_pct:.2f} %) ist unabhängige Information '
            f'— beide Faktoren tragen eigenständigen Gehalt.'),
            unsafe_allow_html=True)

    st.caption(
        "Diese drei Größen tragen den gesamten Unabhängigkeits-Schluss: "
        "ρ misst, wie stark sich Brent- und Zins-Tagesbewegungen linear "
        "gemeinsam bewegen (0 = unabhängig, ±1 = Gleichlauf); β ist die "
        "Steigung der Regression Δr_10y ~ α + β·ΔBrent und sagt, um wie "
        "viele Prozentpunkte sich der Zins je Brent-Schock verschiebt; "
        "R² ist der Anteil der Zinsbewegung, den der Brent-Faktor überhaupt "
        "erklärt. Die ausführliche Regressions-Diagnostik (t-, F-, "
        "RSE-Werte) steht unten im aufklappbaren Detail-Block."
    )

    # Interpretation der Regression in einer Tabelle (sehr verständlich)
    p_brent = _summary['ols_p_beta']
    sig_label = ("hoch signifikant" if p_brent < 0.001 else
                 "signifikant"      if p_brent < 0.05 else
                 "nicht signifikant")
    relev_label = (f"**erklärt nur {_summary['ols_r2']*100:.2f}%** der "
                   f"Zinsbewegung — fast null") if _summary['ols_r2'] < 0.05 else \
                  f"erklärt {_summary['ols_r2']*100:.1f}% der Zinsbewegung"

    reg_summary_table = pd.DataFrame([
        ("Pearson ρ", f"{_summary['pearson_rho']:+.4f}",
         f"Lineare Korrelation; nahe 0 = unabhängig, nahe ±1 = strenger Zusammenhang."),
        ("95%-KI für ρ",
         f"[{_summary['pearson_rho_ci_lo']:+.4f}, {_summary['pearson_rho_ci_hi']:+.4f}]",
         "Falls 0 im Intervall liegt: ρ ist nicht signifikant von null verschieden."),
        ("Spearman ρ (Rang)", f"{_summary['spearman_rho']:+.4f}",
         "Robuste Variante; identisches Vorzeichen wie Pearson, weniger ausreißer-anfällig."),
        ("OLS β (Steigung)", f"{_summary['ols_beta']:+.4f}",
         f"Pro +1.0 Brent-log-Return (+172%) verschiebt sich Δr_10y um {_summary['ols_beta']:+.3f} pp."),
        ("t-Wert für β", f"{_summary['ols_t_beta']:+.2f}",
         "Faustregel: |t| > 1.96 = statistisch signifikant auf 5%-Niveau."),
        ("p-Wert für β", _p_fmt(p_brent) + f"  ({sig_label})",
         "Wahrscheinlichkeit, β rein zufällig zu sehen, wenn wahres β = 0 ist."),
        ("R²", f"{_summary['ols_r2']:.4f}",
         f"Modell {relev_label}. Werte nahe 0 = kein linearer Erklärungswert."),
        ("F-Statistik", f"{_summary['ols_f_stat']:.2f}",
         "Gesamtmodell-Signifikanz; bei 1 Regressor identisch zu t²."),
        ("Residual Std. Error",
         f"{_summary['ols_rse']:.4f} pp",
         "Typische Abweichung der Vorhersage von der Realität (1-Sigma)."),
        ("Beobachtungen (n)", f"{_summary['n_obs']:,}",
         f"Tagesbeobachtungen über {_summary['years']} Jahre."),
    ], columns=["Kennzahl", "Wert", "Was sie bedeutet"])

    with st.expander("Alle Kennzahlen im Detail · Regressions-Diagnostik",
                     expanded=False):
        st.caption("So lesen Sie jede Kennzahl:")
        st.dataframe(reg_summary_table, hide_index=True,
                     use_container_width=True)

    # ----- Auto-Interpretation ------------------------------------
    _rho_abs = abs(_summary["pearson_rho"])
    if _rho_abs < 0.10:
        _verdict_color = COLORS["teal"]
        _verdict_title = "Faktoren empirisch nahezu unabhängig"
    elif _rho_abs < 0.30:
        _verdict_color = COLORS["mid_blue"]
        _verdict_title = "Schwache Korrelation — separate Modellierung gerechtfertigt"
    else:
        _verdict_color = COLORS["amber"]
        _verdict_title = "Moderate Korrelation — Multikollinearität beachten"

    st.markdown(
        f'<div style="background:#F4F4F4;border-left:4px solid {_verdict_color};'
        f'padding:0.85rem 1.1rem;border-radius:6px;margin:0.6rem 0 1.0rem 0;'
        f'color:#051C2C;font-size:0.9rem;line-height:1.65;">'
        f'<strong>Empirisches Urteil über {_summary["years"]} Jahre '
        f'({_summary["sample_start"].strftime("%Y-%m-%d")} → '
        f'{_summary["sample_end"].strftime("%Y-%m-%d")}):'
        f'</strong> {_verdict_title}.<br><br>'
        f'<strong>Was die Zahlen bedeuten:</strong><br>'
        f'• <strong>Pearson ρ = {_summary["pearson_rho"]:+.3f}</strong> — '
        f'lineare Korrelation der Tages-Veränderungen. Werte nahe 0 = '
        f'unabhängige Bewegungen. Werte nahe ±1 = quasi-deterministischer '
        f'Zusammenhang. Hier: nahezu unkorrelierte tägliche Bewegungen.<br>'
        f'• <strong>OLS R² = {_summary["ols_r2"]:.3f}</strong> — '
        f'Anteil der Δr-Varianz, der durch ΔBrent erklärt wird. '
        f'{_summary["ols_r2"]*100:.1f}% bedeutet: die Brent-Veränderung '
        f'erklärt fast nichts von der Zins-Bewegung — beide Faktoren '
        f'tragen weitgehend unabhängige Information.<br>'
        f'• <strong>Fazit:</strong> {_summary["interpretation"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ----- Two-panel chart: rolling correlation + scatter ----------
    cv_l, cv_r = st.columns(2, gap="medium")

    with cv_l:
        st.markdown("**Test 2 · Rolling-252d-Korrelation als Zeitreihe**")
        st.caption(
            "Ökonomisch: Wenn die Korrelation über Zeit stabil nahe null "
            "bleibt, ist die Unabhängigkeitsannahme robust. Springt sie "
            "phasenweise (z.B. in Krisen), wäre eine zeitvariable "
            "Modellierung angemessener. — Formel: ρ_t = Pearson-Korrelation "
            "über das Fenster [t−251, t]. Fenster = 252 Handelstage ≈ ein "
            "Kalenderjahr (Markt-Standard; J.P. Morgan RiskMetrics 1996). "
            "Kürzere Fenster sind verrauschter, längere glätten echte "
            "Regimewechsel weg."
        )
        fig_roll = go.Figure()
        roll_df = _rolling.dropna()
        fig_roll.add_trace(go.Scatter(
            x=roll_df.index, y=roll_df.values,
            mode="lines",
            line=dict(color=COLORS["mid_blue"], width=1.8),
            fill="tozeroy",
            fillcolor="rgba(3,75,111,0.10)",
            name="Rolling ρ(252d)",
            hovertemplate="<b>%{x|%Y-%m}</b><br>ρ = %{y:+.3f}<extra></extra>",
        ))
        fig_roll.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
        fig_roll.add_hline(y=_summary["pearson_rho"], line_dash="dash",
                            line_color=COLORS["crimson"], line_width=1.2,
                            annotation_text=f"5y-Mittel {_summary['pearson_rho']:+.2f}",
                            annotation_position="top right",
                            annotation_font=dict(color=COLORS["crimson"], size=10))
        fig_roll.update_layout(
            xaxis_title="Datum",
            yaxis_title="ρ — rollierende Korrelation [252 Handelstage]",
            height=340, showlegend=False, margin=dict(l=20, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    with cv_r:
        st.markdown("**Test 3 · Streudiagramm mit OLS-Regressionsgerade**")
        st.caption(
            "Jeder Punkt ist ein Handelstag. X-Achse: Brent-log-Return, "
            "Y-Achse: Δr_10y in Prozentpunkten. Die rote Linie ist die "
            "OLS-Regressionsgerade (nicht die 45°-Linie — die Achsen haben "
            "verschiedene Einheiten): ihre nahezu flache Steigung (β ≈ 0) "
            "zeigt, dass Brent-Schocks die Zinsbewegung kaum vorhersagen. "
            "Eine wolkenartige Punktverteilung ohne Trend ist genau das "
            "Bild zweier unabhängiger Faktoren."
        )
        # Build scatter data
        from factor_correlation import _build_daily_factors  # type: ignore
        _df_factors = _build_daily_factors(_data["brent"], _data["svensson"])
        cutoff = _df_factors.index.max() - pd.DateOffset(years=5)
        sub = _df_factors[_df_factors.index >= cutoff]
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scattergl(
            x=sub["d_brent_log"], y=sub["d_r_10y_pp"],
            mode="markers",
            marker=dict(size=3, color=COLORS["navy"], opacity=0.30),
            name="Daily obs",
            hovertemplate="ΔBrent: %{x:+.3f}<br>Δr_10y: %{y:+.3f} pp<extra></extra>",
        ))
        # OLS line
        xg = np.linspace(sub["d_brent_log"].min(),
                          sub["d_brent_log"].max(), 50)
        yg = _summary["ols_alpha"] + _summary["ols_beta"] * xg
        fig_sc.add_trace(go.Scatter(
            x=xg, y=yg, mode="lines",
            line=dict(color=COLORS["crimson"], width=2.5),
            name=(f"OLS: Δr = {_summary['ols_alpha']:+.3f} "
                  f"{_summary['ols_beta']:+.3f}·ΔBrent  (R² = "
                  f"{_summary['ols_r2']:.3f})"),
        ))
        fig_sc.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
        fig_sc.add_vline(x=0, line_color=COLORS["hairline"], line_width=1)
        fig_sc.update_layout(
            xaxis_title="ΔBrent — täglicher log-Return",
            yaxis_title="Δr_10y — Tagesänderung [pp]",
            height=340, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
            margin=dict(l=20, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ----- Per-year breakdown (stability test) --------------------
    with st.expander("Detail · Korrelation pro Kalenderjahr "
                     "(Stabilitäts-Test)", expanded=False):
        st.markdown(
            "Wir bestätigen die Gesamt-Korrelation pro Jahr separat. "
            "Wenn der ρ-Wert pro Jahr stark schwankt, wäre eine "
            "regime-abhängige Modellierung sinnvoller — sind die Werte "
            "stabil, bestätigt das die robuste Unabhängigkeitsannahme."
        )
        _yearly_disp = _yearly.copy()
        _yearly_disp["ρ (Pearson)"] = _yearly_disp["rho"].round(3)
        _yearly_disp["Ø ΔBrent (log)"] = _yearly_disp["mean_brent"].round(4)
        _yearly_disp["Ø Δr (pp)"]      = _yearly_disp["mean_rate_pp"].round(4)
        _yearly_disp["σ ΔBrent (log)"] = _yearly_disp["std_brent"].round(4)
        _yearly_disp["σ Δr (pp)"]       = _yearly_disp["std_rate_pp"].round(4)
        _yearly_disp = _yearly_disp.rename(columns={
            "year": "Jahr", "n_days": "n Tage"})
        st.dataframe(
            _yearly_disp[["Jahr", "n Tage", "ρ (Pearson)",
                          "Ø ΔBrent (log)", "Ø Δr (pp)",
                          "σ ΔBrent (log)", "σ Δr (pp)"]],
            hide_index=True, use_container_width=True,
        )

    # ----- Methodologie-Expander mit Quellen ----------------------
    with st.expander("Methodologie · was wird konkret gerechnet und welche "
                     "Quellen liegen zugrunde?", expanded=False):
        st.markdown("""
**Daten-Quellen (täglich, 5 Jahre)**

| Faktor | Quelle | Spalte / Item |
|---|---|---|
| Brent-Crude-Schlusskurs | ICE via `yfinance` | `Close_USD` |
| 10y-Zero-Coupon-Rate | Deutsche Bundesbank Svensson | `Beta0..Beta3, Tau1, Tau2` → `zero_rate(10y)` |

**Tages-Veränderungen**

- ΔBrent_log: `ln(Brent_t / Brent_{t-1})` — log-Return, geometrisch korrekt
  für Vergleich über Zeit (Tsay 2010, Kap. 1).
- Δr_10y_pp: `r_10y(t) − r_10y(t-1)` in Prozentpunkten — additive Definition
  per Bundesbank-Konvention.

**Statistiken**

- **Pearson ρ** misst die lineare Korrelation: ρ = Cov(X,Y) / (σ_X · σ_Y).
  95%-KI berechnet via Fisher-z-Transformation.
- **Spearman ρ** ist die Korrelation der Ränge — robust gegen Ausreißer
  und nicht-lineare monotone Zusammenhänge.
- **OLS β** ist der Steigungs-Koeffizient der Regression Δr ~ α + β·ΔBrent.
  Falls β statistisch nicht signifikant (|t| < 2), kann er nicht von null
  unterschieden werden.
- **R²** misst den erklärten Varianzanteil. Werte nahe 0 = keine
  Erklärungskraft, Werte nahe 1 = perfekte lineare Vorhersage.

**Konsequenz für unser Stress-Modell**

Mit ρ ≈ 0 über 5 Jahre nehmen wir an: ΔBrent und Δr_10y sind annähernd
orthogonale Faktoren. Damit ist es ökonomisch wie statistisch sauber, sie
als zwei separate Slider zu führen — kein doppeltes Zählen einer
gemeinsamen Schock-Komponente. Diese Annahme ist die fundamentale Basis
für das 2-Faktor-Stress-Modell, das ab Stufe 2 angewandt wird.

**Quellen**

- Tsay, R. (2010). *Analysis of Financial Time Series*, 3. Auflage,
  Wiley.
- Box, G. & Jenkins, G. (1970). *Time Series Analysis: Forecasting and
  Control*.
- Fisher, R. A. (1915). "Frequency distribution of the values of the
  correlation coefficient in samples from an indefinitely large
  population." *Biometrika*, 10(4), 507–521.
- Spearman, C. (1904). "The proof and measurement of association between
  two things." *American Journal of Psychology*, 15, 72–101.
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
# Stage 2 · ΔBrent + Δr_10y → ΔPD pro Exposure-Klasse
# =====================================================================
eyebrow("Stufe 2 · ΔBrent + Δr_10y → ΔPD pro Exposure-Klasse")

st.caption(
    "Pro Exposure-Klasse zeigt der Chart die EAD-gewichtete "
    "<strong>Baseline-PD</strong> (regulatorisch publizierte A-IRB-Werte) "
    "neben der <strong>gestressten PD</strong> aus dem 2-Faktor-Modell. "
    "Formel pro Klasse: ΔPD<sub>pp</sub> = β_oil · ΔBrent + β_rate · Δr_10y.",
    unsafe_allow_html=True,
)
st.caption(
    f"Datenbasis · **Aggregat über alle {N_TOTAL} "
    f"IRB-Banken** · EAD-gewichtete Mittelung der PD pro Exposure-Klasse "
    f"(Corporate · SME-Corporate · Mortgage · QRRE · Other Retail · Bank · "
    f"Sovereign)."
)


def _aggregate_class(df: pd.DataFrame) -> pd.DataFrame:
    """EAD-gewichteter Class-Snapshot aus einem Per-Segment-DataFrame."""
    g = df.groupby("exposure_class").apply(
        lambda x: pd.Series({
            "ead":     x["ead"].sum(),
            "pd_avg":  (np.average(x["pd"], weights=x["ead"])
                         if x["ead"].sum() > 0 else 0.0),
            "lgd_avg": (np.average(x["lgd"], weights=x["ead"])
                         if x["ead"].sum() > 0 else 0.0),
            "rho_avg": (np.average(x["rho"], weights=x["ead"])
                         if x["ead"].sum() > 0 else 0.0),
        }),
        include_groups=False,
    )
    return g.reset_index().sort_values("ead", ascending=False)


class_base   = _aggregate_class(_seg_baseline)
class_stress = _aggregate_class(_seg_stress)

# Merge into one frame for plotting: pd_avg = Baseline, pd_stress = nach 2-Faktor-Shift
class_view = class_base.merge(
    class_stress[["exposure_class", "pd_avg", "lgd_avg"]]
        .rename(columns={"pd_avg": "pd_stress", "lgd_avg": "lgd_stress"}),
    on="exposure_class",
)
# ΔPD / ΔLGD in Prozentpunkten — Single Source fuer Charts (Hover) + Metriken
class_view["pd_delta_pp"]  = (class_view["pd_stress"]  - class_view["pd_avg"])  * 100
class_view["lgd_delta_pp"] = (class_view["lgd_stress"] - class_view["lgd_avg"]) * 100

pd_col_l, pd_col_r = st.columns([3, 2], gap="medium")

with pd_col_l:
    fig_pd = go.Figure()
    fig_pd.add_trace(go.Bar(
        x=class_view["exposure_class"], y=class_view["pd_avg"] * 100,
        name="PD Baseline (A-IRB · Pillar-3, 31.12.2024)",
        marker_color=COLORS["mid_blue"],
        text=[f"{v*100:.2f}%" for v in class_view["pd_avg"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["navy"]),
        hovertemplate="<b>%{x}</b><br>PD Baseline: %{y:.2f}%<extra></extra>",
    ))
    fig_pd.add_trace(go.Bar(
        x=class_view["exposure_class"], y=class_view["pd_stress"] * 100,
        name="PD nach 2-Faktor-Stress", marker_color=COLORS["crimson"],
        text=[f"{v*100:.2f}%" for v in class_view["pd_stress"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["crimson"]),
        customdata=class_view["pd_delta_pp"],
        hovertemplate=("<b>%{x}</b><br>PD nach Stress: %{y:.2f}%"
                       "<br>ΔPD: %{customdata:+.2f} pp<extra></extra>"),
    ))
    _ttl_shock = (f"ΔBrent = {delta_brent_log:+.2f} log · "
                   f"Δr_10y = {delta_r_pp:+.1f} pp"
                   if _is_stressed else "kein Schock angelegt")
    fig_pd.update_layout(
        title=dict(
            text=f"PD pro Exposure-Klasse — Baseline (Pillar-3) vs. "
                  f"2-Faktor-Stress<br>"
                  f"<span style='font-size:0.85em;color:#6E6E6E'>"
                  f"{_ttl_shock}</span>",
            y=0.97, yanchor="top",
        ),
        xaxis_title="Exposure-Klasse",
        yaxis_title="PD — Probability of Default [%]",
        height=420, barmode="group", bargap=0.20,
        margin=dict(t=90, b=90, l=60, r=20),
        legend=dict(orientation="h", yanchor="top", y=-0.22,
                    xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_pd, use_container_width=True)

with pd_col_r:
    st.markdown("**2-Faktor-Sensitivität pro Klasse:**")
    st.latex(
        r"\Delta\text{PD}_{\text{pp}} = "
        r"\beta_{\text{oil}} \cdot \Delta\text{Brent}_{\text{log}} "
        r"+ \beta_{\text{rate}} \cdot \Delta r_{10y,\text{pp}}"
    )
    st.markdown(
        "**Woher kommen die β?** Pro Exposure-Klasse aus publizierten "
        "Quellen zum Modell-Stichtag 31.12.2024: ECB WP 2897/3207 "
        "(Corporate/KMU), ECB WP 3112 (Mortgage), ECB FSR 2024 + "
        "EBA Results Fig. 22 (Retail) — im EBA-Methodik-Rahmen §2.4.2 "
        "¶123 [published]. Vollständige β-Tabelle: Intro-Tab, Schritt 3b."
    )
    st.markdown(
        "**Wie liest man den Balken?** Blau = publizierte Baseline-PD, "
        "Rot = PD nach Schock; die Balkenhöhe ist das PD-Level, der "
        "Abstand Blau→Rot ist die ΔPD (siehe Hover, in pp). **Mortgage** "
        "reagiert primär auf Zinsen (β_rate hoch), **QRRE** auf Brent "
        "(β_oil hoch), **Bank** hat β_rate < 0 (NIM-Effekt: steigende "
        "Zinsen erhöhen die Marge und senken die PD)."
    )
    class_view["pd_delta_pp"] = (class_view["pd_stress"] - class_view["pd_avg"]) * 100
    if _is_stressed:
        top = class_view.sort_values("pd_delta_pp",
                                       key=lambda s: s.abs(),
                                       ascending=False).iloc[0]
        st.metric(
            f"Stärkste PD-Reaktion · Klasse: {top['exposure_class']}",
            f"{top['pd_delta_pp']:+.2f} pp (Prozentpunkte)",
            f"PD-Verschiebung: {top['pd_avg']*100:.2f} % → "
            f"{top['pd_stress']*100:.2f} %",
        )

st.divider()


# =====================================================================
# Stage 3 · ΔBrent + Δr_10y → ΔLGD pro Exposure-Klasse
# =====================================================================
eyebrow("Stufe 3 · ΔBrent + Δr_10y → ΔLGD pro Exposure-Klasse")

st.caption(
    "Parallel zur PD wird auch die LGD über das 2-Faktor-Schema gestresst: "
    "ΔLGD<sub>pp</sub> = γ_oil · ΔBrent + γ_rate · Δr_10y. Mortgage ist "
    "besonders rate-sensitiv (Property-Value-Haircut bei steigenden Zinsen), "
    "Corporate moderat über den Sicherheiten-Wertverlust.",
    unsafe_allow_html=True,
)

lgd_col_l, lgd_col_r = st.columns([3, 2], gap="medium")

with lgd_col_l:
    fig_lgd_class = go.Figure()
    fig_lgd_class.add_trace(go.Bar(
        x=class_view["exposure_class"], y=class_view["lgd_avg"] * 100,
        name="LGD Baseline (F-IRB-Konvention, 31.12.2024)",
        marker_color=COLORS["mid_blue"],
        text=[f"{v*100:.2f}%" for v in class_view["lgd_avg"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["navy"]),
        hovertemplate="<b>%{x}</b><br>LGD Baseline: %{y:.2f}%<extra></extra>",
    ))
    fig_lgd_class.add_trace(go.Bar(
        x=class_view["exposure_class"], y=class_view["lgd_stress"] * 100,
        name="LGD nach 2-Faktor-Stress", marker_color=COLORS["crimson"],
        text=[f"{v*100:.2f}%" for v in class_view["lgd_stress"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["crimson"]),
        customdata=class_view["lgd_delta_pp"],
        hovertemplate=("<b>%{x}</b><br>LGD nach Stress: %{y:.2f}%"
                       "<br>ΔLGD: %{customdata:+.2f} pp<extra></extra>"),
    ))
    _ttl_shock = (f"ΔBrent = {delta_brent_log:+.2f} log · "
                   f"Δr_10y = {delta_r_pp:+.1f} pp"
                   if _is_stressed else "kein Schock angelegt")
    fig_lgd_class.update_layout(
        title=dict(
            text=f"LGD pro Exposure-Klasse — Baseline vs. "
                  f"2-Faktor-Stress<br>"
                  f"<span style='font-size:0.85em;color:#6E6E6E'>"
                  f"{_ttl_shock}</span>",
            y=0.97, yanchor="top",
        ),
        xaxis_title="Exposure-Klasse",
        yaxis_title="LGD — Verlustquote bei Ausfall [%]",
        height=460, barmode="group", bargap=0.20,
        margin=dict(t=90, b=90, l=60, r=20),
        legend=dict(orientation="h", yanchor="top", y=-0.22,
                    xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_lgd_class, use_container_width=True)

with lgd_col_r:
    st.markdown("**2-Faktor-LGD-Stress:**")
    st.latex(
        r"\Delta\text{LGD}_{\text{pp}} = "
        r"\gamma_{\text{oil}} \cdot \Delta\text{Brent}_{\text{log}} "
        r"+ \gamma_{\text{rate}} \cdot \Delta r_{10y,\text{pp}}"
    )
    st.markdown(
        "**Woher kommt γ?** Nach EBA-2025-Methodik §2.4.2 ¶130 — die LGD "
        "spiegelt den Fair-Value-Verfall der Sicherheiten — kalibriert, in "
        "Downturn-Konvention nach CRR Art. 181 [published]. Werte gefloort "
        "bei 5 % und gecappt bei 100 %."
    )
    st.markdown(
        "**Wie liest man den Balken?** Blau = Baseline-LGD, Rot = LGD nach "
        "Schock; Abstand Blau→Rot = ΔLGD (Hover, in pp). **Mortgage** hat "
        "das größte γ_rate: steigende Zinsen senken direkt den Immobilien-"
        "Sicherheitenwert, also bekommt die Bank im Default weniger zurück; "
        "Corporate reagiert moderat über den Sicherheiten-Wertverlust."
    )
    class_view["lgd_delta_pp"] = (class_view["lgd_stress"]
                                   - class_view["lgd_avg"]) * 100
    if _is_stressed and class_view["lgd_delta_pp"].abs().sum() > 0.01:
        top = class_view.sort_values("lgd_delta_pp",
                                       key=lambda s: s.abs(),
                                       ascending=False).iloc[0]
        st.metric(
            f"Stärkste LGD-Reaktion · Klasse: {top['exposure_class']}",
            f"{top['lgd_delta_pp']:+.2f} pp",
            f"LGD-Verschiebung: {top['lgd_avg']*100:.1f} % → "
            f"{top['lgd_stress']*100:.1f} %",
        )

st.divider()


# =====================================================================
# Stage 4 · PD + LGD → IRB Capital (Basel-III-IRB)
# =====================================================================
eyebrow("Stufe 4 · PD + LGD → IRB-Capital (Basel III · BCBS 2017)")

st.caption(
    "Die gestressten PD- und LGD-Werte aus den Stufen 2 und 3 werden jetzt "
    "in die regulatorische Basel-III-IRB-Capital-Formel eingesetzt. Diese "
    "Formel ist unverändert der aufsichtsrechtliche Standard (CRR Art. "
    "153–155) — sie rechnet PD, LGD und Restlaufzeit in eine Kapital-"
    "Anforderung K und damit in RWA = K · 12.5 · EAD um."
)
st.caption(
    f"Datenbasis · **Aggregat über alle {N_TOTAL} "
    "IRB-Banken** · konsolidiert in ein virtuelles Aggregat-Portfolio "
    "(Summe aller Segmente). Capital-Bridge zeigt PD-Effekt + LGD-Effekt "
    "additiv."
)

st.markdown(
    '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
    'border-left:4px solid #034B6F;border-radius:6px;'
    'padding:0.8rem 1.1rem;margin:0.4rem 0 1rem 0;color:#051C2C;'
    'font-size:0.88rem;line-height:1.6;">'
    '<strong>Was ist die Capital-Bridge?</strong> Eine Brücke vom '
    'Kapitalbedarf <em>vor</em> dem Schock (Baseline) zum Kapitalbedarf '
    '<em>nach</em> dem Schock (Stress) — Säule für Säule. Jede Säule zeigt, '
    'wie viel ein einzelner Treiber zur Veränderung beiträgt.<br><br>'
    '<strong>Die drei Stufen:</strong> dieselbe Basel-III-IRB-Formel '
    '(CRR Art. 153–155), dreimal pro Segment gerechnet — nur die '
    'Eingangswerte wechseln: (1) Baseline = PD &amp; LGD wie gemeldet; '
    '(2) + PD-Stress = nur die gestresste PD aus Stufe 2, LGD noch '
    'unverändert; (3) + LGD-Stress = zusätzlich die gestresste LGD aus '
    'Stufe 3. Asset-Korrelation ρ und Maturity-Adjustment bleiben dabei '
    'konstant (klassenabhängig).'
    '</div>',
    unsafe_allow_html=True,
)

if _is_stressed:
    # Direct sequential bridge from baseline → PD-only → full-stress snapshots.
    # Uses the Basel-IRB capital formula on each segment for all 3 states.
    from vasicek import irb_capital_requirement                  # type: ignore

    _bridge_rows = []
    for _bank_name, _port in universe.banks.items():
        for _s in _port.segments:
            _pd_base  = float(getattr(_s, "_pd_base",  _s.pd))
            _lgd_base = float(getattr(_s, "_lgd_base", _s.lgd))
            _pd_str   = float(_s.pd)
            _lgd_str  = float(_s.lgd)

            def _K_RWA(pd_v: float, lgd_v: float,
                       _seg=_s) -> tuple[float, float]:
                m = irb_capital_requirement(
                    pd_v, lgd_v, _seg.exposure_class,
                    maturity_years=_seg.maturity_years,
                    sales_m_eur=_seg.sales_m_eur,
                    confidence=0.999,
                )
                return (float(m["K"])           * _seg.ead,
                        float(m["rwa_density"]) * _seg.ead)

            _K0, _RWA0 = _K_RWA(_pd_base, _lgd_base)
            _K1, _RWA1 = _K_RWA(_pd_str,  _lgd_base)   # nur PD gestresst
            _K2, _RWA2 = _K_RWA(_pd_str,  _lgd_str)    # PD + LGD gestresst
            _bridge_rows.append({
                "exposure_class": _s.exposure_class,
                "ead":            _s.ead,
                "K_base":         _K0, "K_pd_only":   _K1, "K_stress":   _K2,
                "RWA_base":       _RWA0, "RWA_pd_only": _RWA1, "RWA_stress": _RWA2,
                "dK_pd":          _K1 - _K0,   "dK_lgd":   _K2 - _K1,
                "dRWA_pd":        _RWA1 - _RWA0, "dRWA_lgd": _RWA2 - _RWA1,
            })
    _bridge_df = pd.DataFrame(_bridge_rows)
    bridge = {
        "K_base":      float(_bridge_df["K_base"].sum()),
        "K_pd_only":   float(_bridge_df["K_pd_only"].sum()),
        "K_stress":    float(_bridge_df["K_stress"].sum()),
        "delta_K_pd":  float(_bridge_df["dK_pd"].sum()),
        "delta_K_lgd": float(_bridge_df["dK_lgd"].sum()),
        "delta_K":     float(_bridge_df["dK_pd"].sum()
                              + _bridge_df["dK_lgd"].sum()),
        "rwa_base":    float(_bridge_df["RWA_base"].sum()),
        "rwa_pd_only": float(_bridge_df["RWA_pd_only"].sum()),
        "rwa_stress":  float(_bridge_df["RWA_stress"].sum()),
        "delta_rwa_pd":  float(_bridge_df["dRWA_pd"].sum()),
        "delta_rwa_lgd": float(_bridge_df["dRWA_lgd"].sum()),
        "delta_rwa":   float(_bridge_df["dRWA_pd"].sum()
                              + _bridge_df["dRWA_lgd"].sum()),
    }

    cb_col_l, cb_col_r = st.columns([3, 2], gap="medium")

    with cb_col_l:
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["K Baseline",
               "+ ΔK aus PD-Stress<br>(LGD noch unverändert)",
               "+ ΔK aus LGD-Stress<br>(PD bereits gestresst)",
               "K Stress"],
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
            title=dict(text=(
                f"Aggregat-Bank (Top-{universe.n_banks} EU-Banken) · "
                "sequentielle Capital-Bridge<br>"
                "<span style='font-size:0.85em;color:#6E6E6E'>"
                f"{_ttl_shock}</span>"),
                y=0.97, yanchor="top"),
            yaxis_title="K — IRB Capital Requirement [Mrd. EUR]",
            height=420, showlegend=False,
            margin=dict(t=96, b=40, l=60, r=20),
        )
        st.plotly_chart(wf, use_container_width=True)

    with cb_col_r:
        st.markdown("**IRB-Kapitalanforderung K** "
                    "(Basel III · CRR Art. 153–155):")
        st.latex(
            r"K = \text{LGD} \cdot \left[\,N\!\left("
            r"\tfrac{N^{-1}(\text{PD}) + \sqrt{\rho}\,N^{-1}(0.999)}"
            r"{\sqrt{1-\rho}}\right) - \text{PD}\right] \cdot \text{MA}"
        )
        st.markdown(
            "mit Asset-Korrelation ρ (klassenabhängig) und "
            "Maturity-Adjustment MA. K ist der Kapitalbedarf je Euro "
            "Exposure; multipliziert mit EAD ergibt sich der absolute "
            "Kapitalbedarf.\n\n"
            "**Die zwei mittleren Säulen des Waterfalls** zerlegen die "
            "Gesamtveränderung ΔK in genau zwei Beiträge:\n\n"
            "1. nur PD gestresst (LGD bleibt Baseline) → **ΔK_PD**\n"
            "2. zusätzlich LGD gestresst (PD bereits gestresst) → "
            "**ΔK_LGD**\n\n"
            "Weil die Stufen nacheinander gerechnet werden, sind die zwei "
            "Beiträge **exakt additiv** zu ΔK — es gibt keine Kreuz-Effekte."
        )
        total = bridge["delta_K"]
        if abs(total) > 1.0:
            pd_share  = bridge["delta_K_pd"]  / total * 100
            lgd_share = bridge["delta_K_lgd"] / total * 100
            st.metric(
                "Aufteilung ΔK · PD-Beitrag / LGD-Beitrag",
                f"{pd_share:.0f} %  /  {lgd_share:.0f} %",
                f"Gesamtveränderung Σ ΔK = €{total/1e9:+.1f} Mrd.",
            )
        st.caption(
            "RWA-Bridge analog · ΔRWA = ΔK · 12.5 · EAD. "
            f"Aggregat ΔRWA = €{bridge['delta_rwa']/1e9:+.0f} Mrd. "
            "(Basel-Standardumrechnung Faktor 12.5 = 1 / 8 % Mindest-"
            "Kapitalquote)."
        )
else:
    st.info("Sidebar-Slider bewegen, um die Capital-Bridge zu aktivieren — "
            "ohne Schock bleibt der Waterfall auf Baseline.")

st.divider()


# =====================================================================
# Stage 5 · Zwei Kanäle → CET1-Quote
# =====================================================================
eyebrow("Stufe 5 · Zwei Kanäle → CET1-Quote (vorher / nachher)")
st.caption(
    f"Datenbasis · **Aggregat über alle {N_TOTAL} "
    "IRB-Banken** · zwei Risiko-Kanäle zur CET1-Quote (Loan-Book + "
    "Sovereign-Bonds). Klicke unten auf die einzelnen "
    "Bridge-Positionen für die jeweilige Berechnung."
)


@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_channel_data():
    cap = parse_capital_overview(EBA_RAW_DIR / "tr_oth.csv", period=202506)
    sov = parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv", period=202506)
    # Bank-individuell gemeldeter IFRS-9-Split (Items 2520812-2520815)
    # statt Brutto-Maturity-Ladder: nur HfT/FVTPL/FVOCI sind CET1-wirksam.
    sov_acct = sovereign_by_accounting_class(sov, period=202506)
    bd  = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
    return cap, sov_acct, bd


if _is_stressed:
    cap_over, sov_acct, bd = _load_channel_data()

    # Per-Bank-Capital-Bridge aus den vorgestressten Segmenten
    # (analog zu Stage 4 — direkt via irb_capital_requirement)
    from vasicek import irb_capital_requirement                  # type: ignore

    def _bank_bridge(port) -> dict:
        """Sequentielle K/RWA/EL-Bridge für ein einzelnes Bank-Portfolio."""
        K0 = K1 = K2 = 0.0
        RWA0 = RWA1 = RWA2 = 0.0
        EL0 = EL1 = EL2 = 0.0
        for s in port.segments:
            pd_base  = float(getattr(s, "_pd_base",  s.pd))
            lgd_base = float(getattr(s, "_lgd_base", s.lgd))
            pd_str   = float(s.pd)
            lgd_str  = float(s.lgd)
            def _m(pd_v: float, lgd_v: float, _seg=s) -> tuple:
                r = irb_capital_requirement(
                    pd_v, lgd_v, _seg.exposure_class,
                    maturity_years=_seg.maturity_years,
                    sales_m_eur=_seg.sales_m_eur,
                    confidence=0.999,
                )
                return (float(r["K"])           * _seg.ead,
                        float(r["rwa_density"]) * _seg.ead,
                        float(r["EL"])          * _seg.ead)
            k0, rw0, el0 = _m(pd_base, lgd_base)
            k1, rw1, el1 = _m(pd_str,  lgd_base)
            k2, rw2, el2 = _m(pd_str,  lgd_str)
            K0 += k0;   K1 += k1;   K2 += k2
            RWA0 += rw0; RWA1 += rw1; RWA2 += rw2
            EL0 += el0; EL1 += el1; EL2 += el2
        return {
            "K_base": K0, "K_pd_only": K1, "K_stress": K2,
            "rwa_base": RWA0, "rwa_pd_only": RWA1, "rwa_stress": RWA2,
            "el_base": EL0, "el_pd_only": EL1, "el_stress": EL2,
            "delta_K":   K2 - K0,
            "delta_rwa": RWA2 - RWA0,
            "delta_el":  EL2 - EL0,
        }

    # Match top-N bank names to LEIs and build per-bank bridges
    name_to_lei = {}
    loan_bridges_lei = {}
    for bank_name, portfolio in universe.banks.items():
        rows = bd[bd["bank_name"] == bank_name]
        if rows.empty:
            continue
        lei = rows["lei"].iloc[0]
        name_to_lei[bank_name] = lei
        loan_bridges_lei[lei] = _bank_bridge(portfolio)

    # CET1-wirksamer Sovereign-MtM (nur HfT/FVTPL/FVOCI, echter
    # EBA-Split) — identische Datenbasis wie Tab 3 und Tab 4.
    sov_lookup = (sovereign_cet1_pnl_lookup(sov_acct, delta_r_pp=delta_r_pp)
                  if abs(delta_r_pp) > 1e-5 else {})

    # Zwei-Kanal-CET1-Bridge (Loan-Book + Sovereign) — Trading-Book-Kanal
    # in V1 entfernt (siehe cet1_ratio_bridge-Docstring).
    universe_leis = list(name_to_lei.values())
    cap_uni = cap_over[cap_over["LEI_Code"].isin(universe_leis)]
    cet1_bridge = cet1_ratio_bridge(cap_uni, loan_bridges_lei, sov_lookup)

    # Aggregate channel contributions
    cet1_b = float(cet1_bridge["cet1_base"].sum())
    cet1_s = float(cet1_bridge["cet1_stress"].sum())
    rwa_b  = float(cet1_bridge["rwa_total_base"].sum())
    rwa_s  = float(cet1_bridge["rwa_total_stress"].sum())
    delta_cet1 = cet1_s - cet1_b
    delta_rwa  = rwa_s  - rwa_b

    # Decompose ΔCET1 into channels (loan-EL, loan-RWA, sovereign-FV)
    delta_loan_el = -float(sum(b["delta_el"] for b in loan_bridges_lei.values()))
    delta_sov_fv  = float(sum(sov_lookup.values())) if sov_lookup else 0.0
    delta_loan_rwa   = float(sum(b["delta_rwa"] for b in loan_bridges_lei.values()))

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
        ch_loan_rwa = _pp_contrib(0.0, delta_loan_rwa)

        # Residual to ensure waterfall ends exactly on stressed ratio
        explained = ch_loan_el + ch_sov_fv + ch_loan_rwa
        actual_change_pp = (ratio_s - ratio_b) * 100
        residual = actual_change_pp - explained

        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative",
                     "relative", "total"],
            x=["CET1-Quote<br>Baseline",
               "Loan-Book<br>ΔEL (Expected Loss)",
               "Loan-Book<br>ΔRWA (Vasicek)",
               "Sovereign-Bonds<br>ΔFV (Duration-MtM)",
               "Kreuzterm /<br>Rundung",
               "CET1-Quote<br>nach Stress"],
            text=[f"{ratio_b*100:.2f}%",
                  f"{ch_loan_el:+.2f} pp",
                  f"{ch_loan_rwa:+.2f} pp",
                  f"{ch_sov_fv:+.2f} pp",
                  f"{residual:+.2f} pp",
                  f"{ratio_s*100:.2f}%"],
            textposition="outside",
            textfont=dict(size=10, color=COLORS["navy"]),
            y=[ratio_b * 100, ch_loan_el, ch_loan_rwa, ch_sov_fv,
               residual, ratio_s * 100],
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
            title=dict(text=(
                "CET1-Quote (Common Equity Tier 1) · 2-Kanal-Bridge · "
                f"Aggregat Top-{universe.n_banks} EU-Banken<br>"
                "<span style='font-size:0.85em;color:#6E6E6E'>"
                f"{_ttl_shock}</span>"),
                y=0.97, yanchor="top"),
            yaxis_title="CET1-Quote [%] = Hartes Kernkapital ÷ Risk-Weighted Assets",
            height=480, showlegend=False, margin=dict(t=96, b=40, l=60, r=20),
        )
        st.plotly_chart(wf, use_container_width=True)

    with cc_col_r:
        st.markdown(r"""
**CET1-Quote unter Stress** — 2 Kanäle:

1. **Loan-Book**:  ΔEL ↓ CET1-Kapital direkt, ΔRWA ↓ Quote über Nenner
2. **Sovereign**:  ΔFV (Modified-Duration · Δy) — CET1-wirksam ist nur der
   bank-individuell gemeldete HfT/FVTPL/FVOCI-Anteil (EBA-IFRS-9-Split,
   Items 2520812–2520815); AC bleibt zu Buchwert (latenter Verlust)

*(Trading-Book-Kanal in V1 entfernt — Markt-Risiko nur deskriptiv im
Marktbuch-Tab.)*

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
         ("Sovereign ΔFV", ch_sov_fv)],
        key=lambda x: x[1],
    )
    insight(
        f"<strong>Dominanter Kanal:</strong> {biggest_drop[0]} "
        f"({biggest_drop[1]:+.2f} pp) — der Kanal mit dem stärksten "
        f"(negativsten) Beitrag zur CET1-Quote. Der Loan-Book trägt "
        f"insgesamt {(ch_loan_el+ch_loan_rwa):+.2f} pp bei, der "
        f"Sovereign-Kanal {ch_sov_fv:+.2f} pp. Die vollständige "
        f"Aufschlüsselung Bank für Bank steht auf der "
        f"<em>Capital-Adequacy</em>-Page."
    )

    # =================================================================
    # Info-Ribbons · klickbare Erklärung pro Bridge-Position
    # =================================================================
    st.markdown(" ")
    st.markdown(
        '<div style="font-size:0.85rem;color:#034B6F;font-weight:600;'
        'margin:0.6rem 0 0.4rem 0;line-height:1.5;">'
        'Info-Ribbon — jede Kachel erklärt genau eine Säule des Waterfalls '
        'darüber: Definition, Formel, EBA-Quelle und der aktuelle Beitrag '
        'zur CET1-Quote.'
        '</div>',
        unsafe_allow_html=True,
    )

    info_cols = st.columns(5, gap="small")
    with info_cols[0]:
        with st.expander("CET1 Baseline", expanded=False):
            st.markdown(
                "**Definition.** Hartes Kernkapital (CET1) der "
                "Aggregat-Bank vor jedem Schock.\n\n**Berechnung.**"
            )
            st.latex(r"\text{CET1}^{\text{base}} = "
                     r"\sum_{b \in \text{Banken}} \text{CET1}_b")
            st.markdown(
                f"**Aktueller Wert.** €{cet1_b/1e9:,.1f} Mrd. EUR "
                f"(Summe über {universe.n_banks} IRB-Banken).\n\n"
                "**EBA-Datenquelle.** `tr_oth.csv` · Item 2520102 "
                "(Common Equity Tier 1 Capital, regulatorisch)."
            )
    with info_cols[1]:
        with st.expander("Loan-Book · ΔEL", expanded=False):
            st.markdown(
                "**Definition.** Veränderung des Expected Loss im "
                "Loan-Book durch höhere gestresste PD und LGD aus dem "
                "2-Faktor-Modell. Erhöht die Risiko-Vorsorge und "
                "reduziert direkt das CET1-Kapital.\n\n"
                "**Berechnung pro Segment.**"
            )
            st.latex(
                r"\Delta \text{EL} = "
                r"(\text{PD}_{\text{stress}} \cdot \text{LGD}_{\text{stress}}"
                r" - \text{PD} \cdot \text{LGD}) \cdot \text{EAD}"
            )
            st.markdown(
                "dann aggregiert pro Bank, dann über alle Top-10-IRB-"
                "Banken.\n\n"
                f"**Aktueller Beitrag zur CET1-Quote.** "
                f"{ch_loan_el:+.2f} pp = {delta_loan_el/1e9:+.1f} Mrd. "
                f"EUR (ΔEL_loan ÷ RWA_base).\n\n"
                "**Wirkung.** Reduziert CET1-Zähler."
            )
    with info_cols[2]:
        with st.expander("Loan-Book · ΔRWA", expanded=False):
            st.markdown(
                "**Definition.** Veränderung der Credit-RWA durch "
                "IRB-Capital-Bridge — gestresste PD + LGD ergeben "
                "höhere Capital-Anforderung K und damit höheres "
                "Risk-Weighted-Asset.\n\n**Berechnung.**"
            )
            st.latex(
                r"\Delta \text{RWA}_{\text{cr}} = "
                r"\Delta K \cdot 12.5 \cdot \text{EAD}"
            )
            st.markdown(
                "mit K aus der Basel-III-IRB-Formel (CRR Art. 153) und "
                "12.5 = 1 / 8 %-Mindest-Kapitalquote.\n\n"
                f"**Aktueller Beitrag.** {ch_loan_rwa:+.2f} pp = "
                f"ΔRWA {delta_loan_rwa/1e9:+.0f} Mrd. EUR (via "
                "Verdünnung im Nenner).\n\n"
                "**Wirkung.** Erhöht RWA-Nenner → Quote sinkt."
            )
    with info_cols[3]:
        with st.expander("Sovereign · ΔFV", expanded=False):
            st.markdown(
                "**Definition.** Mark-to-Market-Verlust auf Sovereign-"
                "Anleihen durch Zinsanstieg. CET1-wirksam ist nur der "
                "zum Marktwert geführte Teil: HfT + FVTPL (via GuV) und "
                "FVOCI (via OCI-Reserve). AC-Bestände bleiben zu "
                "Buchwert — latenter, nicht CET1-wirksamer Verlust.\n\n"
                "**Berechnung pro Bank** (nur CET1-wirksame Klassen):"
            )
            st.latex(
                r"\Delta \text{FV} = - \sum_{m,\,c \in \{\text{HfT,"
                r"FVTPL,FVOCI}\}} D_m \cdot \Delta y \cdot E_{b,m,c}"
            )
            st.markdown(
                "mit D_m = Modified-Duration des Maturity-Buckets, "
                "Δy = aktueller Δr_10y in Dezimal, c = IFRS-9-Klasse.\n\n"
                "**Datenquelle des Klassen-Splits.** Bank-individuell "
                "gemeldet in der EBA Transparency 2025 (`tr_sov.csv`, "
                "Items 2520812–2520815) — keine Annahme.\n\n"
                f"**Aktueller Beitrag.** {ch_sov_fv:+.2f} pp = "
                f"{delta_sov_fv/1e9:+.1f} Mrd. EUR CET1-wirksamer ΔFV.\n\n"
                "**Wirkung.** Reduziert CET1 wenn Zinsen steigen. "
                "Quelle: Tuckman/Serrat (2012), Kap. 4; IFRS 9 "
                "Klassifizierung."
            )
    with info_cols[4]:
        with st.expander("CET1 nach Stress", expanded=False):
            st.markdown(
                "**Definition.** CET1-Quote nach Stress = "
                "aggregierte Eigenkapital-Quote über beide "
                "Kanäle (Loan-Book + Sovereign).\n\n**Berechnung.**"
            )
            st.latex(
                r"\text{Quote}^{\text{stress}} = "
                r"\frac{\text{CET1}^{\text{base}} - "
                r"\Delta\text{EL}_{\text{loan}} + "
                r"\Delta\text{FV}_{\text{sov}}}"
                r"{\text{RWA}_{\text{base}} + "
                r"\Delta\text{RWA}_{\text{cr}}}"
            )
            st.markdown(
                f"**Aktueller Wert.** {ratio_s*100:.2f} % "
                f"({(ratio_s-ratio_b)*100:+.2f} pp gegenüber "
                f"{ratio_b*100:.2f} % Baseline).\n\n"
                "**Regulatorische Schwellen.** 4.5 % (Pillar 1) · "
                "7.0 % (inkl. CCB) · 8.0 % (typischer SREP-Target)."
            )

else:
    st.info("Schock anwenden, um die 2-Kanal-CET1-Bridge live zu sehen.")

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


eyebrow("Zusammenfassung · die fünf Stufen in einer Tabelle")

# Helper: stärkste PD- und LGD-Reaktion (oder Baseline-Text)
def _strongest_delta(view: pd.DataFrame, col: str, label: str,
                      digits: int = 2) -> str:
    if not _is_stressed:
        return "Baseline (kein Schock angelegt)"
    top = view.sort_values(col, key=lambda s: s.abs(), ascending=False).iloc[0]
    return (f"Stärkste Reaktion · {top['exposure_class']} "
            f"{top[col]:+.{digits}f} pp {label}")


# --- Live-Ergebnis pro Stufe (identische Ausdrücke wie zuvor, nur Darstellung neu) ---
_base_txt = "Baseline · kein Schock angelegt"
row1_live = (f"ΔBrent = {delta_brent_log:+.2f} log&nbsp;&nbsp;·&nbsp;&nbsp;"
             f"Δr_10y = {delta_r_pp:+.1f} pp")
row2_live = _strongest_delta(class_view, "pd_delta_pp", "PD")
row3_live = (_strongest_delta(class_view, "lgd_delta_pp", "LGD", 2)
             if "lgd_delta_pp" in class_view.columns else _base_txt)
row4_live = (f"ΔK = €{bridge['delta_K']/1e9:+.1f} Mrd.&nbsp;&nbsp;·&nbsp;&nbsp;"
             f"ΔRWA = €{bridge['delta_rwa']/1e9:+.0f} Mrd."
             if _is_stressed else _base_txt)
row5_live = (f"CET1-Quote {ratio_b*100:.2f}&nbsp;% → {ratio_s*100:.2f}&nbsp;% "
             f"({(ratio_s-ratio_b)*100:+.2f} pp)"
             if _is_stressed else _base_txt)

_C_INFO, _C_ADVERSE, _C_NEUTRAL = COLORS["mid_blue"], COLORS["crimson"], "#6E6E6E"


def _acc(adverse):
    """Live-Ergebnis-Farbe: adverse=rot, neutral-info=blau, Baseline=grau."""
    return (_C_ADVERSE if adverse else _C_INFO) if _is_stressed else _C_NEUTRAL


summary_rows = [
    ("1",
     "Eingangs-Schock: die zwei Macro-Faktoren aus der Sidebar gehen in "
     "die Bridge.",
     "ΔBrent (log-Return) &nbsp;+&nbsp; Δr_10y (pp)",
     "Sidebar-Slider · einheitlicher Forward-Stress (EBA-2025-Logik)",
     row1_live, _acc(False)),
    ("2",
     "Beide Faktoren verschieben die Ausfallwahrscheinlichkeit — je "
     "Exposure-Klasse unterschiedlich stark.",
     "ΔPD = β_oil · ΔBrent &nbsp;+&nbsp; β_rate · Δr_10y",
     "2-Faktor-Sensitivität · ECB WP 2897/3207/3112 + FSR 2024 · "
     "EBA §2.4.2 ¶123",
     row2_live, _acc(True)),
    ("3",
     "Dieselben Faktoren heben die Verlustquote bei Ausfall an (Mortgage "
     "besonders zins-sensitiv).",
     "ΔLGD = γ_oil · ΔBrent &nbsp;+&nbsp; γ_rate · Δr_10y",
     "2-Faktor-LGD-Stress (Downturn) · EBA §2.4.2 ¶130 · CRR Art. 181",
     row3_live, _acc(True)),
    ("4",
     "Gestresste PD und LGD fließen sequentiell in den IRB-Kapitalbedarf "
     "(PD-Effekt + LGD-Effekt, exakt additiv).",
     "K(PD, LGD) &nbsp;→&nbsp; ΔK = ΔK_PD + ΔK_LGD",
     "Basel-III-IRB-Capital-Bridge · BCBS 2017 · CRR Art. 153–155",
     row4_live, _acc(True)),
    ("5",
     "Zwei Kanäle treffen die harte Kernkapitalquote: das Kreditbuch "
     "(ΔRWA + ΔEL) und das Staatsanleihe-Buch (ΔFair-Value).",
     "ΔCET1 &nbsp;←&nbsp; Loan-Book (ΔRWA + ΔEL) &nbsp;+&nbsp; "
     "Sovereign (ΔFV)",
     "CET1-Bridge · Loan-Book + Sovereign mit IFRS-9-Filter",
     row5_live, _acc(True)),
]

if _is_stressed:
    _status_color = COLORS["crimson"]
    _status_txt = (f"Aktueller Schock angelegt — ΔBrent = "
                   f"{delta_brent_log:+.2f} log, Δr_10y = "
                   f"{delta_r_pp:+.1f} pp. Die Spalte „Live-Ergebnis“ zeigt "
                   f"die Wirkung pro Stufe.")
else:
    _status_color = COLORS["mid_blue"]
    _status_txt = ("Kein Schock angelegt — alle Stufen stehen auf Baseline. "
                   "Stelle ΔBrent oder Δr_10y in der Sidebar, um die Bridge "
                   "live durchzurechnen.")

_th = ("padding:0.55rem 0.7rem;text-align:left;font-weight:600;color:#FFFFFF;"
       "background:#051C2C;font-size:0.82rem;letter-spacing:0.02em;")
_td = "padding:0.7rem;vertical-align:top;border-bottom:1px solid #E6E6E6;"

_rows_html = ""
for nr, was, formel, quelle, live, acc in summary_rows:
    _rows_html += (
        f'<tr>'
        f'<td style="{_td}text-align:center;width:54px;">'
        f'<span style="display:inline-flex;align-items:center;'
        f'justify-content:center;width:1.7rem;height:1.7rem;border-radius:50%;'
        f'background:#051C2C;color:#FFFFFF;font-weight:600;'
        f'font-size:0.86rem;">{nr}</span></td>'
        f'<td style="{_td}width:30%;color:#051C2C;font-size:0.86rem;'
        f'line-height:1.5;">{was}</td>'
        f'<td style="{_td}width:30%;">'
        f'<code style="font-family:Consolas,monospace;font-size:0.82rem;'
        f'color:#034B6F;">{formel}</code>'
        f'<div style="color:#6E6E6E;font-size:0.78rem;margin-top:0.4rem;'
        f'line-height:1.45;">Quelle: {quelle}</div></td>'
        f'<td style="{_td}font-weight:600;color:{acc};font-size:0.86rem;'
        f'line-height:1.5;">{live}</td>'
        f'</tr>'
    )

st.markdown(
    f'<div style="background:#FAFAFA;border-left:4px solid {_status_color};'
    f'padding:0.6rem 1.0rem;border-radius:6px;margin:0.2rem 0 0.9rem 0;'
    f'color:#051C2C;font-size:0.86rem;line-height:1.55;">{_status_txt}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<table style="width:100%;border-collapse:collapse;border:1px solid '
    f'#E6E6E6;border-radius:8px;overflow:hidden;">'
    f'<thead><tr>'
    f'<th style="{_th}text-align:center;">Stufe</th>'
    f'<th style="{_th}">Was passiert</th>'
    f'<th style="{_th}">Formel &amp; Quelle</th>'
    f'<th style="{_th}">Live-Ergebnis</th>'
    f'</tr></thead><tbody>{_rows_html}</tbody></table>',
    unsafe_allow_html=True,
)
insight(
    "Das ist die ganze Kausalkette in fünf Schritten: <strong>zwei "
    "Macro-Faktoren (ΔBrent, Δr_10y) → ΔPD → ΔLGD → IRB-Kapitalbedarf K → "
    "CET1-Quote</strong>. Jede Zahl in der Spalte „Live-Ergebnis“ ist exakt "
    "das, was die Charts oben zeigen — nur verdichtet. Die Spalte "
    "„Formel &amp; Quelle“ macht transparent, woher jeder Schritt kommt "
    "(keine Black-Box)."
)


footer(
    f"Datenquelle: {universe.source} · Korrelations-Stichprobe: "
    f"{cov_source} · Aggregat über alle {N_TOTAL} "
    f"IRB-Banken · PDs/LGDs bank-spezifisch aus Pillar-3 EU-CR6 "
    f"(31.12.2024) · "
    f"β-Sensitivitäten kalibriert nach EBA-2025-Methodik §2.4.2 + "
    f"ECB WP 2897/3207/3112 + ECB FSR 2024 (Modell-Stichtag 31.12.2024)"
)
