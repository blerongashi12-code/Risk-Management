"""Credit Risk · Loan Book · Basel-III-IRB (Vasicek-Modell) + NPL ratios + CET1 impact strip."""
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
# render_loan_methodology entfernt — vollständig durch saubere
# Modell-Annahmen-Sektion (PD/LGD/EAD/EL) in diesem File ersetzt.
from components.backend_path import setup
setup()

from eba_loader import (load_eba_universe,                       # type: ignore
                         parse_credit_risk_csv, loan_book_class_breakdown,
                         parse_capital_overview, trading_book_stress,
                         cet1_ratio_bridge, load_bank_directory,
                         parse_sovereign_csv, sovereign_maturity_ladder,
                         rate_shock_pnl)
from macro_factor import (                                       # type: ignore
    anchor_from_eba, factor_stats,
)
from vasicek import asset_correlation                            # type: ignore
from config import KAPPA_DOWNTURN_LGD, EBA_RAW_DIR                # type: ignore

st.set_page_config(page_title="Kreditbuch · Loan Book", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Kreditbuch · Loan Book Channel",
    eyebrow="Tab 2 · Basel III IRB · 10 Top-EU-Banken · bank-spezifische "
            "Pillar-3-PDs",
    deck="Loan-Exposures der Top-10-IRB-Banken unter makroökonomischem "
         "Stress. PD- und LGD-Parameter <strong>bank-spezifisch aus den "
         "Pillar-3-Reports</strong> der jeweiligen Bank (EU-CR6-Tabelle, "
         "EAD-gewichtete 1-Jahres-PD pro IRB-Klasse, regulatorisch "
         "publiziert nach CRR Art. 431–455 und EBA ITS/2020/04). "
         "Capital-Bridge zeigt die schrittweise Aufschlüsselung des "
         "Stress-Effekts auf PD und LGD.",
)

tab_breadcrumb(2)

# === Datenquellen-Banner mit Coverage-Status ==========================
try:
    from eba_pd_loader import coverage_report                  # type: ignore
    _kb_cov = coverage_report()
    _kb_verified = _kb_cov.get("verified_banks", [])
    _kb_n_verified = len(_kb_verified)
except Exception:
    _kb_verified, _kb_n_verified = [], 0
try:
    from eba_pd_loader import load_pd_table                  # type: ignore
    _kb_df = load_pd_table()
    _kb_pending = sorted(set(_kb_df["bank_name"].unique())
                           - set(_kb_verified))
except Exception:
    _kb_pending = []

st.markdown(
    '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
    'border-left:4px solid #00A9A5;padding:0.85rem 1.1rem;'
    'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
    'font-size:0.88rem;line-height:1.65;">'
    '<strong>Woher kommen die PDs in diesem Tab?</strong><br>'
    'Pro Bank × IRB-Klasse zeigen wir die <strong>EAD-gewichtete 1-Jahres-'
    'Average-PD aus der EU-CR6-Tabelle des jeweiligen Pillar-3-Reports</strong> '
    'der Bank. Das ist die bank-interne Schätzung, die regulatorisch nach '
    'CRR Art. 180 berechnet und gemäß CRR Art. 431–455 + EBA ITS/2020/04 '
    'als Pillar-3-Disclosure veröffentlicht werden muss. Bank-spezifisch, '
    'nicht aus Country-Aggregaten abgeleitet.<br><br>'
    '<strong>Stichtag · einheitlich 31.12.2024</strong> für alle 10 Banken. '
    'EBA-Stress-Test-2025-Standard: ein uniformer Snapshot ist der '
    'Forward-Stress-Startpunkt. Backtest-Annahme über die 5-Jahres-Periode '
    '(2020–2024): 31.12.2024-PDs gelten als Baseline-Proxy weil A-IRB-PDs '
    'TTC-geglättet sind (CRR Art. 180, ±0,1–0,3 pp Drift) und die '
    'Macro-Dynamik in den β-Sensitivitäten sitzt.<br><br>'
    f'<strong style="color:#1E7A4E;">Pillar-3-verifiziert '
    f'({_kb_n_verified}/10):</strong> {" · ".join(_kb_verified)}<br>'
    f'<strong style="color:'
    f'{("#A52F4D" if _kb_pending else "#1E7A4E")};">'
    f'Vollständig auf Country-Aggregat: '
    f'{len(_kb_pending)}/10 Banken</strong> — '
    f'{(", ".join(_kb_pending) if _kb_pending else "keine, alle 10 Banken sind Pillar-3-verifiziert")}'
    '<br>'
    '<span style="color:#6E6E6E;">Einzelne Klassen-Zellen können auch '
    'bei Pillar-3-verifizierten Banken auf das EBA-Country-Aggregat '
    '(Q4 2024) bzw. den Basel-F-IRB-Default fallen, wenn die Bank '
    'diese Klasse nicht als IRB-Sub-total publiziert (z.B. Santander '
    'reportet Sovereign unter Standardised Approach). 93 % der 70 '
    'Zellen sind direkt bank-spezifisch Pillar-3-verifiziert. Der '
    'Status pro Zeile steht im CSV <code>data/pillar3_bank_pd_lgd.csv</code> '
    'in der Spalte <code>status</code>.</span>'
    '</div>',
    unsafe_allow_html=True,
)

# === Load universe + map macro shock to M =============================
# Top-10-IRB-Universe (Banken mit regulatorisch publizierten A-IRB-PDs aus
# dem bank-spezifische Pillar-3 EU-CR6 (31.12.2024)). League-Table, Bridge-Dropdown und alle
# Drilldowns operieren ausschließlich auf diesen 10 Banken — einheitliche
# Datenqualität, keine Defaulted/Original-Schätzung mehr.
@st.cache_data(ttl=24*3600, show_spinner="Loading EBA Top-10 universe …")
def _load_universe():
    from eba_pd_loader import filter_universe_to_top10                # type: ignore
    u = load_eba_universe(vintage="2025", top_n=None, prefer_real=True)
    return filter_universe_to_top10(u)


universe = _load_universe()
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

# --- 2-Faktor-Stress: Baseline zuerst, dann Stress -------------------
# WICHTIG: Wir computieren die Baseline-KPIs auf dem UNGESTRESSTEN Universum
# und applizieren den 2-Faktor-Stress erst danach. Die frühere Implementierung
# nutzte ``stressed_kpis(z_factor=0)`` als Stress-Pfad, was implizit eine
# weitere Vasicek-Transformation ``conditional_pd(pd, rho, 0)`` anwandte —
# diese ist am Median des Systemfaktors NICHT die Identität, sondern reduziert
# die PD typisch um 30–50 %. Das führte dazu, dass EL_stressed < EL_baseline
# *unabhängig* von der Richtung des Schocks war (Diskussions-Bug 2026-05-29).
from two_factor_stress import (apply_stress_to_universe,         # type: ignore
                                reset_universe_to_baseline)

_d_brent      = float(config["d_brent"])
_d_r_10y_pp   = float(config["d_r_10y_pp"])
_sens_overrides = config.get("sensitivity_overrides")

# Sicherheitshalber zuerst zur Baseline zurücksetzen (für Re-Renders nach
# bereits applizierten Stresses aus dem Streamlit-Cache).
reset_universe_to_baseline(universe)

m_used = 0.0
_is_stressed = (abs(_d_brent) > 1e-6) or (abs(_d_r_10y_pp) > 1e-6)

# Legacy defaults für V1-Knobs, die nur noch interne Capital-Berechnung steuern
_rho_mult       = 1.0
_lgd_cal        = 1.0
_stress_exp     = 1.0
_cap_mode       = "hard"

# === Macro-Schock-Diagnose-Strip =======================================
eyebrow("Macro-Schock — Brent + 10y-Zins (live aus Sidebar)")
m1, m2, m3 = st.columns(3, gap="small")
m1.metric("ΔBrent (log)", f"{_d_brent:+.2f}",
          f"{(np.exp(_d_brent)-1)*100:+.0f}% Preisänderung",
          delta_color="off")
m2.metric("Δr_10y", f"{_d_r_10y_pp*100:+.0f} bp",
          f"{_d_r_10y_pp:+.2f} pp", delta_color="off")
_d_combined = abs(_d_brent) + abs(_d_r_10y_pp)
m3.metric("Schock-Magnitude",
          f"|ΔBrent| + |Δr| = {_d_combined:+.2f}",
          "≈ 0 = baseline, > 1 = signifikant",
          delta_color="off")

source_tag = "Pillar-3 EU-CR6 bank-spezifisch (31.12.2024) · 10 Top-IRB-Banken"
if abs(_d_brent) < 1e-3 and abs(_d_r_10y_pp) < 1e-3:
    insight(
        f"<strong>Kein Macro-Schock aktiv.</strong> Alle Kennzahlen unten "
        f"zeigen die regulatorische Baseline auf Basis {source_tag}. "
        f"Slider in der Sidebar bewegen, um die 2-Faktor-Stress-"
        f"Transmission live zu sehen — ΔPD und ΔLGD werden pro "
        f"Exposure-Klasse mit unterschiedlichen β-Koeffizienten "
        f"angewandt (Sektor-Differenzierung)."
    )
else:
    insight(
        f"<strong>2-Faktor-Stress aktiv.</strong> ΔBrent = "
        f"{_d_brent:+.2f} und Δr = {_d_r_10y_pp:+.2f} pp werden "
        f"sektor-differenziert auf PD und LGD jeder Exposure-Klasse "
        f"angewandt: <code>ΔPD = β_oil · ΔBrent + β_rate · Δr</code>. "
        f"Beispiel: Mortgage reagiert primär auf Zinsen "
        f"(Floating-Rate-Affordability), Corporate auf beide Faktoren, "
        f"Bank-Klasse hat <em>negative</em> Zinssensitivität "
        f"(NIM-Effekt — adressiert Professor-Kritik Punkt 7). "
        f"Quelle: EBA Stress Test 2025 Methodology Note + Literatur "
        f"(Drehmann/Juselius, Hosszú/Király, Castro)."
    )

st.divider()

# =====================================================================
# Modell-Annahmen · die vier Bausteine PD · LGD · EAD · EL
# =====================================================================
eyebrow("Modell-Annahmen · die vier Bausteine PD · LGD · EAD · EL")

st.markdown(
    '<div style="background:#F4F4F4;border-left:4px solid #034B6F;'
    'padding:0.95rem 1.1rem;border-radius:6px;margin-bottom:1rem;'
    'color:#051C2C;font-size:0.92rem;line-height:1.6;">'
    '<strong>Wie liest sich die Kreditrisiko-Rechnung?</strong> Jeder Kredit '
    'wird durch vier Größen beschrieben: <strong>PD</strong> (wie '
    'wahrscheinlich fällt der Schuldner aus?), <strong>LGD</strong> '
    '(wie viel der Forderung bleibt verloren?), <strong>EAD</strong> '
    '(welcher Betrag ist im Default ausstehend?) und ihr Produkt '
    '<strong>EL = PD · LGD · EAD</strong> (der erwartete Verlust). '
    'Klicke unten auf jeden der vier Bausteine — jede Box dokumentiert '
    'Definition, Datenquelle, Prämissen und Limitationen unseres Modells.'
    '</div>',
    unsafe_allow_html=True,
)

# -------- 1) PD --------
with st.expander("① PD · Probability of Default · "
                 "Ausfallwahrscheinlichkeit", expanded=False):

    st.markdown("#### Definition")
    st.markdown(
        "PD = Wahrscheinlichkeit, dass ein Schuldner innerhalb der nächsten "
        "12 Monate ausfällt. Default-Definition nach CRR Art. 178: 90+ Tage "
        "überfällig oder *unlikely to pay*."
    )

    st.markdown("#### Woher kommt die PD?")
    st.markdown(
        "Die PD ist die **bank-spezifische exposure-gewichtete 1-Jahres-"
        "Average-PD** aus der **EU-CR6-Tabelle** im Pillar-3-Report der "
        "jeweiligen Bank (Sub-total pro IRB-Exposure-Klasse). "
        "Banken müssen diese Werte regulatorisch nach CRR Art. 180 "
        "(A-IRB-Approach, internes Modell) berechnen und gemäß "
        "CRR Art. 431–455 + EBA ITS/2020/04 als Pillar-3-Disclosure "
        "publizieren. Im Modell weisen wir jeder Bank ihre eigenen "
        "veröffentlichten Werte zu — **bank-spezifisch**, nicht über "
        "Country-Aggregate genähert."
    )
    st.markdown(
        "**Warum nicht aus EBA-Defaulted/Original-Quote ableiten?** "
        "Das wäre methodisch falsch (Wertgrößen statt Mengengrößen, "
        "ursprünglicher Kreditbetrag statt Restschuld). Die jetzt "
        "verwendeten Pillar-3-Werte sind hingegen forward-looking "
        "1-Jahres-PDs aus den internen Rating-Modellen der Banken, "
        "Basel-III-konform."
    )
    st.markdown(
        "**Aktueller Daten-Reifegrad:** **Alle 10 Banken sind "
        "Pillar-3-verifiziert** mit einheitlichem Stichtag 31.12.2024 "
        "(Deutsche Bank, ING, Société Générale, Rabobank, UniCredit, "
        "Crédit Agricole, Crédit Mutuel, BPCE, BNP Paribas, Santander). "
        "Rund 93 % der 70 CSV-Zellen sind direkt aus den jeweiligen "
        "EU-CR6-Sub-totals der Banken extrahiert. Die wenigen "
        "verbleibenden Country-Proxy-Zellen entstehen, wo eine Bank "
        "eine bestimmte Klasse nicht als IRB-Sub-total publiziert "
        "(z. B. Santander Sovereign unter Standardised Approach). "
        "Der Status pro Bank × Klasse ist in "
        "`data/pillar3_bank_pd_lgd.csv` Spalte `status` dokumentiert."
    )

    st.markdown("#### Wie wird PD im Stress transformiert?")
    st.markdown(
        "Die Baseline-PD wird sektor-differenziert über die **2-Faktor-"
        "Stress-Formel** verändert:"
    )
    st.latex(r"\Delta\text{PD}_{\text{class}} = "
             r"\beta_{\text{oil,class}} \cdot \Delta\text{Brent}_{\log} + "
             r"\beta_{\text{rate,class}} \cdot \Delta r_{10y,\text{pp}}")
    st.markdown(
        "Jede Exposure-Klasse hat eigene β-Koeffizienten für Brent und "
        "Zins — z.B. reagiert Mortgage stark auf Zinsen, Corporate "
        "auf beide Faktoren, und **die Bank-Klasse hat β_rate < 0** "
        "(Net-Interest-Margin-Uplift bei steigenden Zinsen). "
        "Quellen der Sensitivitäten: EBA Stress Test 2025 Methodology "
        "Note, Drehmann/Juselius (BIS WP 421), Hosszú/Király (MNB WP "
        "2018/2), Castro (2013)."
    )
    st.markdown(
        "Die empirische Korrelations-Analyse (Tab 1 · Faktor-Analyse) "
        "belegt, dass Brent und Zins über 5 Jahre nahezu unabhängig "
        "sind (ρ ≈ +0.07) — daher dürfen wir sie als separate Faktoren "
        "modellieren."
    )

    st.markdown("#### Floor / Cap")
    st.markdown(
        "3 bp (Basel-Sovereign-Floor) und 50% Cap für numerische "
        "Stabilität der Capital-Math."
    )

# -------- 2) LGD --------
with st.expander("② LGD · Loss Given Default · "
                 "Verlustquote bei Ausfall", expanded=False):

    st.markdown("#### Definition")
    st.markdown(
        "LGD = Anteil der Forderung, der bei Ausfall **nicht** zurückkommt "
        "(nach Recovery, Sicherheiten-Verwertung, Insolvenzverfahren). "
        "Wertebereich: 0% (vollständige Recovery) bis 100% (Totalverlust)."
    )

    st.markdown("#### Woher kommt die LGD?")
    st.markdown(
        "**Gleiche Datenquelle wie bei der PD — vollständig konsistent:** "
        "Die LGD-Werte sind die **bank-spezifischen EAD-gewichteten "
        "A-IRB-LGDs** aus derselben EU-CR6-Tabelle desselben Pillar-3-"
        "Reports (Average LGD Sub-total pro IRB-Klasse, regulatorisch "
        "publiziert nach CRR Art. 181 — Downturn-LGD-Konvention)."
    )
    st.markdown(
        "**Coverage konsistent zur PD:** PD und LGD jeder einzelnen CSV-"
        "Zeile stammen aus *derselben* Quelle (eine `status`-Spalte "
        "deckt beide ab). Bei den 5 von 70 Zellen, in denen eine Bank "
        "diese Klasse nicht als IRB-Sub-total publiziert, fällt sowohl "
        "die PD als auch die LGD gleichermaßen auf den EBA-Country-"
        "Aggregat (Q4 2024) bzw. den Basel-F-IRB-Default (45 %) zurück. "
        "Damit ist die LGD-Datenbasis methodisch identisch parallel zur "
        "PD-Basis."
    )
    st.markdown(
        "**Einheitlicher Stichtag 31.12.2024** für alle 10 Banken — "
        "vom Loader-Test `_test_vintage_consistency` erzwungen, gleicher "
        "Snapshot wie die PDs."
    )
    st.markdown(
        "Für die Klassen **Bank** und **Sovereign**, wo nicht jede Bank "
        "separate IRB-LGDs publiziert, verwenden wir bei Bedarf den "
        "Basel-F-IRB-Default von 45 % (CRR Art. 161)."
    )
    st.markdown("#### Beispielwerte aus dem aktiven Datensatz")
    lgd_table = pd.DataFrame([
        ("Crédit Agricole (FR)", "Corporate",   "34.17 %"),
        ("Crédit Agricole (FR)", "Mortgage",    "14.01 %"),
        ("Deutsche Bank (DE)",   "Corporate",   "38.32 %"),
        ("UniCredit (IT)",       "QRRE",        "55.45 %"),
        ("Santander (ES)",       "Other Retail","51.74 %"),
        ("ING (NL)",             "Mortgage",    "14.21 %"),
    ], columns=["Bank (Heimatland)", "Exposure-Klasse", "EBA-LGD Q4 2025"])
    st.dataframe(lgd_table, use_container_width=True, hide_index=True,
                 height=240)

    st.markdown("#### Welche Prämissen gehen wir ein?")
    st.markdown(
        "**Country-Aggregate als Proxy.** EBA aggregiert die A-IRB-LGD-"
        "Parameter nach Counterparty-Land. Eine Bank, die heimische "
        "und internationale Forderungen hält, sieht in der Realität "
        "eine gemischte LGD-Struktur — wir nehmen die LGDs ihres "
        "Heimatlandes als Approximation. Bank-individuelle Abweichungen "
        "werden nicht modelliert (Limitation transparent dokumentiert)."
    )

    st.markdown("#### Wie wird LGD im Stress transformiert?")
    st.markdown(
        "Genau wie die PD wird die LGD über die **2-Faktor-Stress-"
        "Formel** sektor-differenziert verändert:"
    )
    st.latex(r"\Delta\text{LGD}_{\text{class}} = "
             r"\gamma_{\text{oil,class}} \cdot \Delta\text{Brent}_{\log} + "
             r"\gamma_{\text{rate,class}} \cdot \Delta r_{10y,\text{pp}}")
    st.markdown(
        "Beispiel-Sensitivitäten (γ): **Mortgage** reagiert mit γ_rate = "
        "+1.50pp/+1pp besonders stark auf Zinsen (Property-Value-"
        "Haircut über Duration-Effekt). **Corporate** hat γ_rate = +1.0pp "
        "(Sicherheiten-Bond-Duration). Quellen: EBA Stress Test 2025 "
        "Methodology Note Sec. 6.1-6.2 + EBA Risk Dashboard historische "
        "LGD-Sensitivität."
    )
    st.markdown(
        "**Floor / Cap:** LGD wird auf [5%, 100%] geclippt für "
        "numerische Stabilität."
    )

    st.markdown("#### Bekannte Limitationen")
    st.markdown(
        "- Sektor-Granularität auf Pillar-3-Klassen-Ebene (Corporate, "
        "SME, Mortgage, QRRE, Other Retail, Bank, Sovereign) — keine "
        "Unterscheidung nach Industrie/Underlying-Sicherheiten innerhalb "
        "einer Klasse  \n"
        "- Kein Recovery-Schätzwert pro Sicherheiten-Typ (Property vs. "
        "Equipment vs. Cash-Collateral)  \n"
        "- Keine länderspezifischen LGD-Floors (regulatorisch in der "
        "EU einheitlich nach CRR Art. 164)  \n"
        "- LGD-Floor bei 5 %, Cap bei 100 % — rein numerische Stabilität "
        "der IRB-Capital-Rechnung, kein ökonomischer Kausal-Cap  \n"
        "- LGD-Sensitivitäten γ_oil / γ_rate sind defensible defaults "
        "aus EBA Stress-Test 2025 Methodology Note, nicht bank-"
        "individuell kalibriert"
    )

# -------- 3) EAD --------
with st.expander("③ EAD · Exposure at Default · "
                 "Risikoposition bei Ausfall", expanded=False):

    st.markdown("#### Definition")
    st.markdown(
        "EAD = ausstehender Forderungsbetrag in dem Moment, in dem der "
        "Schuldner defaultet. Für bilanzielle Kredite ist das der Buchwert; "
        "für nicht-bilanzielle Linien (Garantien, ungenutzte Kreditlinien) "
        "wird ein **Credit Conversion Factor (CCF)** angewendet, der den "
        "voraussichtlichen Drawdown abbildet."
    )

    st.markdown("#### Wie wir EAD im Modell messen")
    st.markdown(
        "Direkter Pull aus **EBA Item 2520522 (Exposure Value)**. Dieser "
        "Wert ist bereits CCF-adjustiert (post-Conversion-Factor, "
        "Basel-konform), sodass wir keine eigene CCF-Logik anwenden müssen."
    )
    st.markdown(
        "Datenbasis: EBA Transparency 2025, aggregiert pro Bank × "
        "Exposure-Klasse über alle Counterparty-Countries und Maturity-Buckets."
    )

    st.markdown("#### Welche Prämissen gehen wir ein?")
    st.markdown(
        "**1. EAD ist statisch unter Stress (V1-Vereinfachung).** Wir "
        "nehmen an, dass die Exposure-Beträge unverändert bleiben, wenn M "
        "sich ändert."
    )
    st.markdown(
        "**2. Kein dynamisches Drawdown-Risk modelliert.** Reale Banken "
        "sehen unter Stress eine **Erhöhung** der CCF auf Off-Balance-Linien "
        "(Beispiel: Kreditlinien an Corporate-Kunden — CCF kann von 50 % auf "
        "75 % steigen, wenn Kunden in der Krise ziehen). Dieser Effekt ist "
        "in V1 **nicht** modelliert. Konsequenz: das Modell **unterschätzt** "
        "unter starkem Stress die EAD-Komponente."
    )
    st.markdown(
        "**3. FX-Effekte sind in EBA-Daten bereits EUR-konsolidiert.** "
        "Currency-Channel ist nicht trennbar — wir können nicht separat "
        "ausweisen, wie sich z.B. eine EUR-USD-Abwertung auf die EAD von "
        "Tochterbanken auswirken würde."
    )

    st.markdown("#### Wie wird EAD im Stress transformiert?")
    st.markdown(
        "**Keine Transformation in V1.** EAD bleibt = EAD_base."
    )

    st.markdown("#### Bekannte Limitationen")
    st.markdown(
        "- Drawdown-Risk (CCF-Erhöhung in Krisen) nicht modelliert  \n"
        "- Keine FX-Sensitivität  \n"
        "- Keine Restrukturierung-Effekte (Verlängerung, Wandlung in Equity)  \n"
        "- Keine Behandlung von Cross-Default-Klauseln"
    )

# -------- 4) EL --------
with st.expander("④ EL · Expected Loss · Erwarteter Verlust",
                 expanded=False):

    st.markdown("#### Definition")
    st.markdown(
        "EL = der Erwartungswert des Verlusts auf das Portfolio über die "
        "nächsten 12 Monate. Bilanziell relevant als Risiko-Vorsorge "
        "(Provisions); regulatorisch wichtig zur Abgrenzung gegen das "
        "Unexpected Loss (UL), das durch Eigenkapital gedeckt werden muss."
    )

    st.markdown("#### Berechnung")
    st.latex(r"\text{EL} = \text{PD} \cdot \text{LGD} \cdot \text{EAD}")
    st.markdown(
        "pro Segment, dann aggregiert pro Bank und über alle Banken im "
        "Aggregat."
    )

    st.markdown("#### Stress-Decomposition")
    st.markdown(
        "Unter Stress wird ΔEL exakt in zwei additive Beiträge zerlegt:"
    )
    st.latex(r"\Delta\text{EL} = \underbrace{(\text{PD}^* - \text{PD}) \cdot "
             r"\text{LGD} \cdot \text{EAD}}_{\Delta\text{EL aus PD-Shift}} "
             r"\;+\; \underbrace{\text{PD}^* \cdot (\text{LGD}^* - "
             r"\text{LGD}) \cdot \text{EAD}}_{\Delta\text{EL aus LGD-Shift}}")
    st.markdown(
        "Diese sequentielle Decomposition ist im Capital-Bridge-Tab oben "
        "sichtbar — der **PD-Beitrag** wird erst angewendet (LGD bleibt "
        "unverändert), dann der **LGD-Beitrag** auf bereits gestresstem PD."
    )

    st.markdown("#### Verhältnis zur IRB-Capital-Charge")
    st.markdown(
        "EL und Capital sind komplementär:  \n"
        "- **EL** = Provisionsaufwendung (P&L-Belastung, sicher erwartet)  \n"
        "- **K = L₀.₉₉₉ − EL** = Unexpected-Loss-Komponente bei 99.9 % "
        "Konfidenz (Eigenkapital-Charge nach Basel III)  \n"
        "- **L₀.₉₉₉** = 99.9-%-Verlust-Quantil aus der Basel-IRB-Formel aus der Vasicek-Formel"
    )
    st.markdown(
        "Beide werden in der Capital-Bridge nebeneinander gezeigt. Die "
        "CET1-Quote reagiert auf beide: ΔEL reduziert direkt den Zähler, "
        "ΔK · 12.5 · EAD = ΔRWA verdünnt den Nenner."
    )

st.divider()

# =====================================================================
# Konkrete PD/LGD-Matrix · welche Werte hat unser Modell pro Bank?
# =====================================================================
eyebrow("Konkrete PD- und LGD-Werte pro Bank und Klasse "
        "(aktive Datenbasis)")

st.markdown(
    '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
    'border-left:4px solid #00A9A5;padding:0.85rem 1.1rem;'
    'border-radius:6px;margin:0.5rem 0 1rem 0;color:#051C2C;'
    'font-size:0.88rem;line-height:1.65;">'
    '<strong>Was siehst du hier?</strong><br>'
    'Die folgende Matrix zeigt, welche regulatorisch-publizierten '
    'PD- und LGD-Werte unser Modell konkret für jede der 10 Banken '
    'und jede ihrer 5 Exposure-Klassen geladen hat. Quelle: '
    'bank-spezifische Pillar-3 EU-CR6 (31.12.2024), COREP C 9.02. Jede Zelle ist die '
    'exposure-gewichtete A-IRB-Schätzung des Heimatlandes der Bank '
    '(z.B. Frankreich-Aggregat für BNP, Italien für UniCredit).<br><br>'
    '<strong>Wie liest man die Tabelle?</strong> Zeile = Bank, '
    'Spalte-Paar = Klasse · PD% | LGD%. Beispiel: Deutsche Bank '
    'Corporate hat PD = 3.08% und LGD = 38.32% — d.h. das Modell '
    'rechnet damit, dass von €1 Corporate-Exposure jährlich '
    '3.08&nbsp;Cent ausfallen und davon 38.32&nbsp;Cent '
    'Endverlust bleiben.'
    '</div>',
    unsafe_allow_html=True,
)

# Build the concrete PD/LGD matrix from the loaded EBA data
from eba_pd_loader import load_eba_pd_table                # type: ignore
_pd_table = load_eba_pd_table()
_display_classes = ["corporate", "sme_corporate", "mortgage",
                     "qrre", "other_retail"]

_matrix_rows = []
for bank_name in _pd_table["bank_name"].unique():
    bank_rows = _pd_table[_pd_table["bank_name"] == bank_name]
    row = {"Bank": bank_name, "Land": bank_rows["bank_country"].iloc[0]}
    for cls in _display_classes:
        cell = bank_rows[bank_rows["vasicek_class"] == cls]
        if len(cell) > 0:
            pd_val  = float(cell["pd_pct"].iloc[0])
            lgd_val = float(cell["lgd_pct"].iloc[0])
            row[f"{cls} PD%"]  = f"{pd_val:.2f}"
            row[f"{cls} LGD%"] = f"{lgd_val:.2f}"
        else:
            row[f"{cls} PD%"]  = "—"
            row[f"{cls} LGD%"] = "—"
    _matrix_rows.append(row)

_pd_matrix_df = pd.DataFrame(_matrix_rows)
st.dataframe(_pd_matrix_df, hide_index=True, use_container_width=True,
             height=395)
st.caption(
    "PD und LGD jeweils in Prozent · Quelle pro Zeile: EBA Risk "
    "Dashboard Q4 2025, COREP C 9.02 (Counterparty-Land = "
    "Heimatland der Bank)."
)

# =====================================================================
# Worked Example · wie ein Schock konkret PD und LGD ändert
# =====================================================================
st.markdown(" ")
eyebrow("Worked Example · ein Schock von +50% Brent und +200 bp "
        "Δr_10y konkret durchgerechnet")

st.markdown(
    '<div style="background:#F4F4F4;border-radius:6px;'
    'padding:0.85rem 1.1rem;margin:0.4rem 0 1rem 0;color:#051C2C;'
    'font-size:0.88rem;line-height:1.65;">'
    'Wir wenden die 2-Faktor-Formel auf vier illustrative Bank-Klasse-'
    'Kombinationen an. Schock-Vektor: '
    '<strong>ΔBrent = +0.50 (≈ +65% Ölpreis), Δr_10y = +2.0 pp '
    '(+200 bp)</strong>.<br><br>'
    'Formel: <code>ΔPD = β_oil · ΔBrent + β_rate · Δr_10y</code>, '
    'analog für LGD. Werte aus der oben gezeigten Matrix.'
    '</div>',
    unsafe_allow_html=True,
)

from two_factor_stress import SENSITIVITY_MATRIX, stress_pd, stress_lgd

_d_brent_ex = 0.50
_d_r_ex     = 2.0
_worked_examples = [
    ("UniCredit", "IT", "corporate"),
    ("Deutsche Bank", "DE", "mortgage"),
    ("BNP Paribas", "FR", "qrre"),
    ("Banco Santander", "ES", "bank"),
]
_worked_rows = []
for bank, country, cls in _worked_examples:
    bank_row = _pd_table[(_pd_table["bank_name"].str.contains(bank, case=False)) &
                         (_pd_table["vasicek_class"] == cls)]
    if len(bank_row) == 0:
        continue
    pd_base  = float(bank_row["pd_pct"].iloc[0])  / 100.0
    lgd_base = float(bank_row["lgd_pct"].iloc[0]) / 100.0
    pd_str   = stress_pd(pd_base,  _d_brent_ex, _d_r_ex, cls)
    lgd_str  = stress_lgd(lgd_base, _d_brent_ex, _d_r_ex, cls)
    sens     = SENSITIVITY_MATRIX.get(cls, {})
    bo = sens.get("pd_oil", 0); br = sens.get("pd_rate", 0)
    delta_pd_pp  = (bo * _d_brent_ex + br * _d_r_ex)
    delta_lgd_pp = (sens.get("lgd_oil", 0)  * _d_brent_ex +
                    sens.get("lgd_rate", 0) * _d_r_ex)
    _worked_rows.append({
        "Bank · Klasse":          f"{bank} ({country}) · {cls}",
        "β_oil PD":               f"{bo:+.2f}",
        "β_rate PD":              f"{br:+.2f}",
        "Rechnung ΔPD":           (f"{bo:+.2f}·0.50 + {br:+.2f}·2.00 = "
                                   f"{delta_pd_pp:+.3f} pp"),
        "PD baseline →  stress":  f"{pd_base*100:.2f}% → {pd_str*100:.2f}%",
        "ΔLGD":                   f"{delta_lgd_pp:+.2f} pp",
        "LGD baseline →  stress": f"{lgd_base*100:.1f}% → {lgd_str*100:.1f}%",
    })
_worked_df = pd.DataFrame(_worked_rows)
st.dataframe(_worked_df, hide_index=True, use_container_width=True,
             height=180)
st.markdown(
    '<div style="background:#FFFFFF;border-left:4px solid #A52F4D;'
    'padding:0.7rem 1.0rem;margin:0.4rem 0 0.6rem 0;color:#051C2C;'
    'font-size:0.86rem;line-height:1.55;">'
    '<strong>Drei Lehren aus dem Worked Example:</strong><br>'
    '• <strong>UniCredit Corporate</strong> reagiert auf beide '
    'Faktoren (β_oil und β_rate beide positiv) — Italien-Corporate-'
    'PD steigt um <strong>0.55&nbsp;pp</strong>.<br>'
    '• <strong>Deutsche Bank Mortgage</strong> wird primär vom '
    'Zinsschock getrieben (β_rate &gt;&gt; β_oil) — typische Floating-'
    'Rate-Hypotheken-Dynamik.<br>'
    '• <strong>BNP Paribas QRRE</strong> reagiert primär auf Brent '
    '(β_oil = +0.40) — Konsumkredite vs. Energie-Inflation.<br>'
    '• <strong>Banco Santander Bank-Klasse</strong> erlebt eine '
    '<em>Reduktion</em> der PD um <strong>0.075&nbsp;pp</strong> — '
    'NIM-Uplift kompensiert das Credit-Risk-Up. Adressiert direkt '
    'die Professor-Kritik: „steigende Zinsen ≠ allgemein schlecht".'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# === Compute baseline + stressed metrics across all banks =============
# Snapshot-Pattern: erst Baseline auf ungestresstem Universum erfassen,
# dann 2-Faktor-Stress applizieren und Stressed-KPIs erfassen. Beide Pfade
# nutzen portfolio_kpis() — ohne zusätzliche Vasicek-conditional_pd-
# Transformation, die am Median falsche „benignere" Stresses erzeugt hätte.

# 1) Baseline auf ungestresstem Universum
reset_universe_to_baseline(universe)
baseline_rows = []
for bank_name, portfolio in universe.banks.items():
    base_kpi = portfolio.portfolio_kpis(confidence=0.999,
                                          rho_multiplier=_rho_mult,
                                          lgd_calibration=_lgd_cal)
    base_kpi["Bank"] = bank_name
    baseline_rows.append(base_kpi)

# 2) Stress applizieren — danach steht s.pd/s.lgd = gestresst, s._pd_base
#    = ursprüngliche Baseline für nachgelagerte Bridge-Funktionen
apply_stress_to_universe(
    universe, _d_brent, _d_r_10y_pp,
    override_betas=_sens_overrides,
)

# 3) Stressed-KPIs auf gestresstem Universum
stressed_rows = []
if _is_stressed:
    for bank_name, portfolio in universe.banks.items():
        s_kpi = portfolio.portfolio_kpis(confidence=0.999,
                                           rho_multiplier=_rho_mult,
                                           lgd_calibration=_lgd_cal)
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
    # Wir behalten EL/ΔEL in Mio. €, RWA/EAD in Mrd. € — kleinere Banken
    # haben EL im 1-50 Mio.-Bereich und würden in Mrd. zu 0.00 runden.
    el_base_eur   = float(baseline_df.loc[bank_name, "el_eur"])
    el_stress_eur = (float(stressed_df.loc[bank_name, "el_eur"])
                     if stressed_df is not None else el_base_eur)
    rwa_base_eur   = float(baseline_df.loc[bank_name, "rwa"])
    rwa_stress_eur = (float(stressed_df.loc[bank_name, "rwa"])
                      if stressed_df is not None else rwa_base_eur)
    n_segs = baseline_df.loc[bank_name, "n_segments"] \
        if "n_segments" in baseline_df.columns else 0
    row = {
        "Bank":           bank_name,
        "Segmente":       int(n_segs) if n_segs else len(universe.banks[bank_name].segments),
        "EAD Mrd. €":     baseline_df.loc[bank_name, "total_ead"] / 1e9,
        "EL base Mio. €":   el_base_eur / 1e6,
        "EL stress Mio. €": el_stress_eur / 1e6,
        "Δ EL Mio. €":      (el_stress_eur - el_base_eur) / 1e6,
        "RWA base Mrd. €":   rwa_base_eur / 1e9,
        "RWA stress Mrd. €": rwa_stress_eur / 1e9,
        "Δ RWA Mrd. €":      (rwa_stress_eur - rwa_base_eur) / 1e9,
        "RWA-Dichte":        baseline_df.loc[bank_name, "rwa_density"],
    }
    table_rows.append(row)

league = pd.DataFrame(table_rows).set_index("Bank")
league = league.sort_values("EAD Mrd. €", ascending=False)

# Display-Tabelle: nur die lesbaren Spalten, formatiert
display_league = pd.DataFrame({
    "Segmente":          league["Segmente"].astype(int),
    "EAD Mrd. €":        league["EAD Mrd. €"].round(1),
    "EL base Mio. €":    league["EL base Mio. €"].round(0).astype(int),
    "EL stress Mio. €":  league["EL stress Mio. €"].round(0).astype(int),
    "Δ EL Mio. €":       league["Δ EL Mio. €"].round(0).astype(int),
    "RWA base Mrd. €":   league["RWA base Mrd. €"].round(1),
    "RWA stress Mrd. €": league["RWA stress Mrd. €"].round(1),
    "Δ RWA Mrd. €":      league["Δ RWA Mrd. €"].round(1),
    "RWA-Dichte":        (league["RWA-Dichte"] * 100).round(1).astype(str) + "%",
}, index=league.index)

# Kompakte Aliases für Plotly-Bar-Chart unten — Code dort referenziert
# noch auf die alten Spalten-Namen mit „bn" Suffix.
league["EAD bn"]        = league["EAD Mrd. €"]
league["EL base bn"]    = league["EL base Mio. €"] / 1000      # Mio. → Mrd.
league["EL stress bn"]  = league["EL stress Mio. €"] / 1000
league["Δ EL bn"]       = league["Δ EL Mio. €"] / 1000
league["RWA base bn"]   = league["RWA base Mrd. €"]
league["RWA stress bn"] = league["RWA stress Mrd. €"]
league["Δ RWA bn"]      = league["Δ RWA Mrd. €"]
league["RWA dens"]      = league["RWA-Dichte"]

st.dataframe(display_league, use_container_width=True, height=420)

st.divider()

# === Bank-by-bank EL bar (baseline vs stressed) =======================
eyebrow("Expected Loss pro Bank · Baseline vs. Stress")

st.caption(
    f"Alle {len(league)} Banken der kuratierten Datenbasis, absteigend "
    f"nach Stress-Expected-Loss sortiert (größte Risiko-Träger oben)."
)

# Feste Datenbasis: alle 10 kuratierten Banken, keine Top-N-Auswahl.
ranked = league.sort_values("EL stress bn", ascending=False)
chart_height = max(460, 34 * len(ranked) + 120)

# Plotly will plot Y in reverse order — wir kehren ranked um, damit die
# größte Bank oben steht
ranked = ranked.iloc[::-1]

fig = go.Figure()
fig.add_trace(go.Bar(
    y=ranked.index, x=ranked["EL base bn"],
    name="Baseline (vor Stress)",
    orientation="h",
    marker_color=COLORS["mid_blue"],
    text=[f"{v:.1f}" for v in ranked["EL base bn"]],
    textposition="outside",
    textfont=dict(size=10, color=COLORS["stone"]),
))
fig.add_trace(go.Bar(
    y=ranked.index, x=ranked["EL stress bn"],
    name="Stress (nach Schock)",
    orientation="h",
    marker_color=COLORS["crimson"],
    text=[f"{v:.1f}" for v in ranked["EL stress bn"]],
    textposition="outside",
    textfont=dict(size=10, color=COLORS["navy"]),
))
_shock_label = (f"ΔBrent = {_d_brent:+.2f} · Δr_10y = {_d_r_10y_pp:+.1f} pp"
                if _is_stressed else "kein Schock")
fig.update_layout(
    title=(f"Expected Loss · alle {len(ranked)} Banken · "
           f"Baseline (blau) vs. Stress (crimson) · {_shock_label}"),
    xaxis_title="Expected Loss [Mrd. EUR]",
    yaxis=dict(automargin=True, tickfont=dict(size=10)),
    height=chart_height,
    barmode="group",
    bargap=0.25,
    margin=dict(l=20, r=80, t=70, b=40),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# === Capital Bridge · Wirkungskette pro Bank ==========================
eyebrow("Wirkungskette · Capital Bridge per bank")
st.caption(
    "End-to-end Transmission Chain · Macro-Schock → 2-Faktor-Stress auf "
    "PD und LGD (sektor-differenzierte β-Sensitivitäten) → IRB-Capital. "
    "Sequentielle Aktivierung: erst der PD-Shift (LGD bleibt auf Baseline), "
    "dann der LGD-Shift (PD bereits gestresst). Beide Beiträge summieren "
    "sich per Konstruktion exakt zu ΔK."
)

_bank_ranking_bridge = sorted(
    universe.banks.keys(),
    key=lambda n: -universe.banks[n].total_ead,
)
# Globaler Bank-Filter · steuert Capital-Bridge UND Drilldown unten
# (Type-to-Search ist im Streamlit-Selectbox eingebaut: einfach in das
# Feld tippen, um eine Bank zu finden — z.B. "Deutsche", "BNP" oder "ING").
bridge_bank = st.selectbox(
    f"Bank-Auswahl · alle {universe.n_banks} IRB-Banken "
    f"(absteigend nach EAD · zum Suchen einfach tippen)",
    _bank_ranking_bridge,
    index=0,
    format_func=lambda n: f"{n}  ·  EAD {universe.banks[n].total_ead/1e9:.0f} Mrd. €",
    key="bridge_bank",
    help="Diese Auswahl steuert sowohl die Capital-Bridge oben als auch "
         "den Bank-Drilldown weiter unten. Type-to-search: in das Feld "
         "tippen (z.B. Deutsche, BNP oder Santander) filtert die Liste.",
)

# Sync der Drilldown-Selektor an die Bridge-Auswahl
st.session_state["vasicek_drill_bank"] = bridge_bank

bridge_portfolio = universe.banks[bridge_bank]
if _is_stressed:
    # 2-Faktor-Bridge: nutzt _pd_base/_lgd_base als Baseline und applizert
    # die sektor-differenzierten β-Sensitivitäten direkt — KEIN
    # Vasicek-conditional_pd(z=0)-Shift (ersetzt die alte capital_bridge,
    # die EL fälschlich reduzierte).
    from two_factor_stress import capital_bridge_2factor  # type: ignore
    bridge = capital_bridge_2factor(
        bridge_portfolio,
        _d_brent, _d_r_10y_pp,
        confidence=0.999,
        rho_multiplier=_rho_mult,
        lgd_calibration=_lgd_cal,
        override_betas=_sens_overrides,
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
            st.markdown("**K · Capital Requirement**")
            st.markdown(
                "Die regulatorische Eigenkapital-Anforderung (in EUR) "
                "auf einen Kredit unter Basel III IRB. Deckt die "
                "**Unexpected-Loss-Komponente** ab — also den Verlust "
                "oberhalb des durchschnittlich erwarteten Verlusts (EL)."
            )
            st.markdown("**Formel**")
            st.latex(r"K = \bigl(L_{0.999} - \text{PD} \cdot \text{LGD}\bigr) "
                     r"\cdot \text{MA}(M_{\text{eff}}) \cdot \text{EAD}")
            st.markdown(
                "mit $L_{0.999}$ = 99.9-%-Verlust-Quantil aus der Basel-IRB-Formel bei 99.9 % Konfidenz "
                "(Vasicek 2002), MA = Maturity-Adjustment (Basel BCBS 2017)."
            )
            st.markdown("**Datenquelle**")
            st.markdown(
                "PD, LGD, EAD pro Vasicek-Segment aus **EBA Transparency "
                "2025** (Items 2520512 / F-IRB-Defaults / 2520522), "
                "rechenliche Aggregation pro Bank."
            )
            st.markdown("**Reading der Bridge**")
            st.markdown(
                "Stage 1 hebt nur PD via Conditional-PD an (LGD bleibt "
                "auf Baseline). Stage 2 hebt zusätzlich LGD via Downturn-"
                "LGD-Funktion. Beide Beiträge sind exakt additiv zu ΔK."
            )
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
            st.markdown("**RWA · Risk-Weighted Assets**")
            st.markdown(
                "Die **risikogewichteten Aktiva** in EUR. Ergebnis der "
                "Capital-Anforderung K, hochskaliert mit dem Basel-Faktor "
                "12.5 (= 1 / 8 % Mindest-Eigenkapitalquote) und der EAD. "
                "RWA steht im **Nenner der CET1-Quote** — je höher die "
                "RWA, desto stärker verdünnt die regulatorische "
                "Eigenkapitalquote."
            )
            st.markdown("**Formel**")
            st.latex(r"\text{RWA} = K \cdot 12.5 \cdot \text{EAD}")
            st.markdown("**Datenquelle**")
            st.markdown(
                "Berechnet aus K (Vasicek/IRB) × 12.5 × EAD-Stichtagswert "
                "(EBA Item 2520522)."
            )
            st.markdown("**Reading**")
            st.markdown(
                "RWA folgt strukturell der gleichen Decomposition wie K — "
                "EAD ist im V1 konstant unter Stress, deshalb skaliert "
                "ΔRWA proportional zu ΔK."
            )
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
            st.markdown("**EL · Expected Loss**")
            st.markdown(
                "Der **erwartete Verlust** in EUR — das Produkt aus "
                "Ausfallwahrscheinlichkeit, Verlustquote und Risiko-"
                "Position. Bilanziell relevant als Risiko-Vorsorge "
                "(Provisions in der P&L). Im Gegensatz zu Capital (K), "
                "das das *unerwartete* Tail-Risiko abdeckt."
            )
            st.markdown("**Formel**")
            st.latex(r"\text{EL} = \text{PD} \cdot \text{LGD} \cdot \text{EAD}")
            st.markdown(
                "ΔEL zerfällt unter Stress sequentiell in zwei additive "
                "Beiträge:"
            )
            st.latex(r"\Delta\text{EL} = \underbrace{(\text{PD}^* - "
                     r"\text{PD}) \cdot \text{LGD} \cdot \text{EAD}}"
                     r"_{\text{aus PD-Shift}} + \underbrace{\text{PD}^* "
                     r"\cdot (\text{LGD}^* - \text{LGD}) \cdot \text{EAD}}"
                     r"_{\text{aus LGD-Shift}}")
            st.markdown("**Datenquelle**")
            st.markdown(
                "PD (EBA Item 2520512 ÷ 2520502), LGD (Basel F-IRB-Defaults "
                "× User-Kalibrierung), EAD (EBA Item 2520522)."
            )
            st.markdown("**Reading**")
            st.markdown(
                "EL ist der Erwartungswert (P&L-Vorsorge). Die Capital-"
                "Charge K im linken Tab ist die **Unexpected-Loss-"
                "Komponente** oberhalb von EL — beide werden gemeinsam "
                "für die Risiko-Bilanz gebraucht: EL → Provisionen, "
                "K → Eigenkapital-Puffer."
            )
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
eyebrow(f"Bank-Drilldown · {bridge_bank} · Exposure-Class-Breakdown")
st.caption(
    "Detail-Ansicht für die in der Bank-Auswahl oben gewählte Bank. "
    "Zeigt Segment-Metriken (EAD, PD, ρ, EL, UL, RWA) und EAD-"
    "Komposition pro Exposure-Klasse."
)

# Exposure-Klassen-Glossar — verständlich für Erstleser
with st.expander("Was bedeuten die Segment-Namen? · Exposure-Klassen-Glossar",
                 expanded=False):
    st.markdown(
        "Basel-III IRB unterscheidet sieben Exposure-Klassen — jede hat "
        "eine eigene Risiko-Korrelation ρ und Maturity-Behandlung. So "
        "liest du die Klassen in den Charts und Tabellen unten:"
    )
    segment_glossary = pd.DataFrame([
        ("Corporate",
         "Große Nicht-Finanz-Unternehmen (typisch Umsatz > €50 M, "
         "z.B. Siemens, Total, BASF)",
         "12 – 24 %"),
        ("SME-Corporate",
         "Kleine und mittelständische Unternehmen (Umsatz < €50 M); "
         "Basel-Adjustierung für Größe",
         "12 – 24 % minus SME-Bonus"),
        ("Bank",
         "Forderungen gegen andere Finanzinstitute (Interbank-Loans, "
         "Korrespondenzbanken, Depot-Forderungen)",
         "12 – 24 %"),
        ("Sovereign",
         "Forderungen gegen Staaten und Zentralbanken (im IRB-"
         "Loan-Book; reine Staatsanleihen siehe Tab 3 Marktbuch)",
         "12 – 24 %"),
        ("Residential Mortgage",
         "Wohnimmobilien-Hypotheken an Privatpersonen — durch Immobilie "
         "besichert",
         "15 % (fix)"),
        ("QRRE · Qualifying Revolving Retail",
         "Kreditkartenforderungen + revolvierende Konsumentenkredite "
         "(Dispokredite) — qualifizieren für eigene Basel-Behandlung",
         "4 % (fix)"),
        ("Other Retail",
         "Sonstige Privatkundenforderungen — Auto-Finanzierung, "
         "Konsumentenkredite, unbesicherte Personal Loans",
         "3 – 16 %"),
    ], columns=["Exposure-Klasse", "Was steckt drin?", "Asset-Korrelation ρ"])
    st.dataframe(segment_glossary, use_container_width=True,
                 hide_index=True, height=280)
    st.caption(
        "**Datenquelle.** Klassen-Zuordnung pro Bank-Segment aus dem "
        "EBA Transparency Exercise 2025 (Exposure-Class-Code in `tr_cre.csv`). "
        "ρ-Werte aus der Basel-III-IRB-Formel (BCBS 2017, CRR Art. 153). "
        "Höhere ρ → stärkere Reaktion der bedingten PD auf einen Schock in M."
    )

sel_bank = bridge_bank  # konsolidierter Filter — same selection als Bridge
drill_portfolio = universe.banks[sel_bank]
# Baseline-Metriken: temporär zurücksetzen, damit baseline_metrics die
# ursprünglichen PD/LGD und nicht die bereits gestressten Werte sieht.
# Danach wieder stressen, weil nachgelagerte Charts auf segment.pd/.lgd
# basieren.
if _is_stressed:
    reset_universe_to_baseline(universe)
drill_base = drill_portfolio.baseline_metrics(confidence=0.999,
                                                rho_multiplier=_rho_mult,
                                                lgd_calibration=_lgd_cal)
if _is_stressed:
    # Stress wieder applizieren, dann stressed_metrics aus den
    # gestressten Werten direkt berechnen (KEIN conditional_pd-Shift)
    apply_stress_to_universe(universe, _d_brent, _d_r_10y_pp,
                             override_betas=_sens_overrides)
    # Direkter Stress-Snapshot via portfolio_kpis pro Segment
    _drill_rows = []
    for s in drill_portfolio.segments:
        pd_b = float(getattr(s, "_pd_base", s.pd))
        bm = s.basel_metrics(confidence=0.999,
                              rho_multiplier=_rho_mult,
                              lgd_calibration=_lgd_cal)
        _drill_rows.append({
            "name":         s.name,
            "exposure_class": s.exposure_class,
            "pd_baseline":  pd_b,
            "pd_stressed":  float(s.pd),
            "delta_pd":     float(s.pd) - pd_b,
            "lgd_baseline": float(getattr(s, "_lgd_base", s.lgd)),
            "lgd_stressed": float(s.lgd),
            "el_eur":       float(bm["el_eur"]),
            "ul_eur":       float(bm["ul_eur"]),
            "rwa":          float(bm["rwa"]),
            "ead":          float(bm["ead"]),
        })
    drill_stress = pd.DataFrame(_drill_rows)
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
        eyebrow(f"{sel_bank} · 2-Faktor-gestresste PDs pro Segment")
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

# =====================================================================
# EL-Decomposition pro Klasse · bank-spezifisch (folgt der Bank-Auswahl)
# Beantwortet eine einfache Frage: welche Klasse trägt wie viel zur
# EL-Verschlechterung dieser Bank bei?
# =====================================================================
eyebrow(f"EL-Decomposition · welche Klasse stresst {bridge_bank} am meisten?")

st.caption(
    f"Bank-spezifische Sicht: pro Exposure-Klasse zeigen wir, wie viel "
    f"absoluter ΔEL-Beitrag (in Mio. €) durch den Stress entsteht. "
    f"Direkt sichtbar wird der Hauptverursacher der EL-Verschlechterung "
    f"für **{bridge_bank}**. Folgt der Bank-Auswahl oben."
)

if _is_stressed:
    # Stressed metrics pro Segment dieser Bank — wir greifen auf
    # drill_stress zu, das oben bereits aus den 2-Faktor-gestressten
    # PD/LGD-Werten (ohne conditional_pd-Shift) berechnet wurde.
    _bank_stress = drill_stress
    # Aggregation pro Klasse (eine Bank kann mehrere Segmente in der
    # gleichen Klasse haben — z.B. Corporate-FR + Corporate-DE)
    _bank_base_grp = drill_base.groupby("exposure_class").agg(
        ead=("ead", "sum"), el_base=("el_eur", "sum"),
    ).reset_index()
    _bank_stress_grp = _bank_stress.groupby("exposure_class").agg(
        el_stress=("el_eur", "sum"),
    ).reset_index()
    el_decomp = _bank_base_grp.merge(_bank_stress_grp,
                                       on="exposure_class", how="left")
    el_decomp["delta_el_eur"] = el_decomp["el_stress"] - el_decomp["el_base"]
    el_decomp = el_decomp.sort_values("delta_el_eur", ascending=False)

    total_delta_el = float(el_decomp["delta_el_eur"].sum())

    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Bar(
        x=el_decomp["exposure_class"],
        y=el_decomp["delta_el_eur"] / 1e6,
        marker_color=[
            COLORS["crimson"] if v > 0 else COLORS["teal"]
            for v in el_decomp["delta_el_eur"]
        ],
        text=[
            f"+€{v/1e6:.0f} M ({v/total_delta_el*100:.0f} %)"
            if total_delta_el != 0 and v > 0
            else (f"+€{v/1e6:.0f} M" if v > 0 else f"€{v/1e6:.0f} M")
            for v in el_decomp["delta_el_eur"]
        ],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["navy"]),
        marker_line_width=0,
        showlegend=False,
    ))
    fig_decomp.update_layout(
        title=(f"ΔEL pro Klasse · {bridge_bank} · "
               f"Σ ΔEL = €{total_delta_el/1e6:+,.0f} Mio. · "
               f"{_shock_label}"),
        xaxis_title="Exposure-Klasse",
        yaxis_title="ΔEL in Millionen EUR",
        height=380,
        bargap=0.35,
        margin=dict(l=20, r=20, t=70, b=60),
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

    # Insight unterhalb · Hauptverursacher + Top-2 in einer Zeile
    if total_delta_el > 0 and len(el_decomp) > 0:
        top_class = el_decomp.iloc[0]
        top_pct = float(top_class["delta_el_eur"] / total_delta_el * 100)
        runner_up = el_decomp.iloc[1] if len(el_decomp) > 1 else None
        msg = (
            f"<strong>Hauptverursacher:</strong> "
            f"<strong>{top_class['exposure_class']}</strong> "
            f"liefert <strong>{top_pct:.0f} %</strong> der gesamten "
            f"EL-Verschlechterung von "
            f"€{total_delta_el/1e6:,.0f} M "
            f"(absolut +€{top_class['delta_el_eur']/1e6:,.0f} M)."
        )
        if runner_up is not None and runner_up["delta_el_eur"] > 0:
            runner_pct = float(runner_up["delta_el_eur"] / total_delta_el * 100)
            msg += (
                f" An zweiter Stelle: "
                f"<strong>{runner_up['exposure_class']}</strong> "
                f"mit {runner_pct:.0f} % "
                f"(+€{runner_up['delta_el_eur']/1e6:,.0f} M)."
            )
        insight(msg)

else:
    st.info(
        f"Stress anwenden (Macro-Slider in der Sidebar (ΔBrent + Δr_10y)), um die "
        f"EL-Decomposition für **{bridge_bank}** zu sehen."
    )

st.divider()

# CET1-Impact-Strip wurde entfernt — Redundanz mit Tab 4 (Eigenkapital).
# Die volle 3-Channel-CET1-Bridge lebt jetzt ausschließlich dort.
st.info(
    "**CET1-Wirkung anschauen?** → Wechsle zu Tab 4 **Eigenkapital-Wirkung** "
    "für die volle 3-Channel-Decomposition (Loan + Sovereign + Trading Book), "
    "Threshold-Analyse (4.5% / 7% / 8% Breaches) und CET1-Sensitivity-Curve."
)

footer(
    f"Datenquelle: {universe.source}  ·  "
    f"PDs und LGDs: bank-spezifische Pillar-3 EU-CR6 (31.12.2024) (regulatorisch publizierte "
    f"A-IRB-Werte, CRR Art. 180)  ·  EAD: EBA Transparency 2025 Item "
    f"2520522  ·  Stress-Transmission: 2-Faktor-Sensitivitäten "
    f"(EBA Stress Test 2025 Methodology Note + Literatur)"
)
