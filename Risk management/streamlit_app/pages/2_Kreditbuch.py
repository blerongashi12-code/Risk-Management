"""Credit Risk · Loan Book · Vasicek/ASRF + NPL ratios + CET1 impact strip."""
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
    anchor_from_eba, hybrid_mapping, factor_stats,
)
from vasicek import conditional_pd, asset_correlation            # type: ignore
from config import KAPPA_DOWNTURN_LGD, EBA_RAW_DIR                # type: ignore

st.set_page_config(page_title="Kreditbuch · Loan Book", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Kreditbuch · Loan Book Channel",
    eyebrow="Tab 2 · Vasicek/ASRF · Basel III IRB · 67 IRB-Banken",
    deck="Loan-Exposures der EBA-IRB-Banken unter makroökonomischem Stress. "
         "Basel-III-IRB-Capital-Formeln Segment-by-Segment angewandt. "
         "Vier Bausteine — PD, LGD, EAD und EL — bilden die Risiko-Sicht; "
         "jeder ist unten transparent dokumentiert (Definition, Datenquelle, "
         "Prämissen, Limitationen). Capital-Bridge zeigt die sequentielle "
         "PD-+LGD-Decomposition unter Stress.",
)

# === Load universe + map macro shock to M =============================
# Vollständiges EBA-IRB-Universum (alle 67 Banken). Tab 1 Wirkungskette
# erlaubt eine separate Aggregat-Auswahl in der Sidebar; das Kreditbuch
# ist dagegen die comprehensive view — League-Table und Bridge-Dropdown
# zeigen IMMER alle IRB-Banken.
@st.cache_data(ttl=24*3600, show_spinner="Loading EBA Transparency exercise …")
def _load_universe():
    return load_eba_universe(vintage="2025", top_n=200, prefer_real=True)


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

_rho_mult       = float(config.get("rho_mult", 1.0))         # ρ-Override
_lgd_cal        = float(config.get("lgd_calibration", 1.0))  # V2 A-IRB-Proxy
_stress_exp     = float(config.get("stress_exponent", 1.0))  # V2 Tail-Risk
_cap_mode       = str(config.get("cap_mode", "hard"))        # V2 Cap-Modus
anchor = anchor_from_eba("2025")
mapping = hybrid_mapping(
    delta_brent_log=config["d_brent"],
    delta_rate_10y_pp=config["d_r_10y_pp"],   # already in pp
    anchor=anchor,
    cov_factors=cov_factors,
    horizon_days=H_DAYS,
)
# Single-factor mode: M is user-driven; override the mapping result
if config.get("m_direct") is not None:
    m_dir = float(config["m_direct"])
    mapping["m_anchor"] = m_dir
    mapping["m_data"]   = m_dir
    mapping["m_hybrid"] = m_dir

# === Mapping diagnostics strip ========================================
eyebrow("Macro shock mapped to Vasicek systematic factor M")
m1, m2, m3, m4, m5 = st.columns(5, gap="small")
m1.metric("ΔBrent (log)", f"{config['d_brent']:+.2f}",
          f"{(np.exp(config['d_brent'])-1)*100:+.0f}%", delta_color="off")
m2.metric("Δr_10y", f"{config['d_r_10y_pp']*100:+.0f} bp",
          f"{config['d_r_10y_pp']:+.2f} pp", delta_color="off")
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

    st.markdown("#### Wie wir PD im Modell messen")
    st.markdown(
        "Wir leiten PD aus der **beobachteten Default-Quote** der EBA "
        "Transparency Exercise 2025 ab:"
    )
    st.latex(r"\text{PD} = \frac{\text{Defaulted Exposure}}{\text{Original Exposure}} "
             r"= \frac{\text{EBA Item 2520512 (Status = 2)}}{\text{EBA Item 2520502}}")
    st.markdown(
        "Aggregiert pro Bank × Vasicek-Klasse. Quartals-Stichtag: Juni 2025."
    )

    st.markdown("#### Welche Prämisse gehen wir ein?")
    st.markdown(
        "**Stock-Größe statt Forward-Prognose.** Diese PD ist die "
        "Default-Quote *zum Stichtag* — sie sagt aus, *welcher Anteil* des "
        "Portfolios bereits ausgefallen ist. Eine forward-looking 1-Jahres-PD "
        "im Sinne von CRR Art. 180 ist sie **nicht**. Konsequenz: das absolute "
        "PD-Niveau ist konservativ (rückwärtsgewandt), aber die **Sensitivität "
        "gegen M** bleibt korrekt."
    )

    st.markdown("#### Wie wird PD im Stress transformiert?")
    st.markdown(
        "Die Baseline-PD wird über die **Vasicek-Conditional-PD-Formel** in "
        "eine bedingte 1-Jahres-PD überführt:"
    )
    st.latex(r"\text{PD}(M) = N\!\left(\frac{N^{-1}(\text{PD}) - "
             r"\sqrt{\rho}\,M}{\sqrt{1-\rho}}\right)")
    st.markdown("mit ρ = Basel-formelmäßige Asset-Korrelation pro Klasse.")

    st.markdown("#### Floor / Cap")
    st.markdown(
        "3 bp (Basel-Sovereign-Floor) und 50% Cap für numerische Stabilität."
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

    st.markdown("#### Wie wir LGD im Modell messen")
    st.markdown(
        "Wir verwenden die **Basel-F-IRB-Standardwerte** (regulatorisch "
        "vorgegeben in CRR Art. 161). Tabelle: Baseline-Werte links, "
        f"effektive Werte im Modell rechts (skaliert mit dem aktuellen "
        f"`lgd_calibration` = **{_lgd_cal:.2f}×** aus der Sidebar)."
    )
    lgd_table = pd.DataFrame([
        ("Corporate / Bank / Sovereign", "45 %", f"{0.45 * _lgd_cal * 100:.1f} %"),
        ("SME-Corporate",                "45 %", f"{0.45 * _lgd_cal * 100:.1f} %"),
        ("Residential Mortgage",          "20 %", f"{0.20 * _lgd_cal * 100:.1f} %"),
        ("QRRE (Revolving Retail)",       "65 %", f"{0.65 * _lgd_cal * 100:.1f} %"),
        ("Other Retail",                  "45 %", f"{0.45 * _lgd_cal * 100:.1f} %"),
    ], columns=["Vasicek-Klasse", "F-IRB Baseline (CRR Art. 161)",
                "Effektiv im Modell"])
    st.dataframe(lgd_table, use_container_width=True, hide_index=True,
                 height=220)

    st.markdown("#### Welche Prämissen gehen wir ein?")
    st.markdown(
        "**1. F-IRB statt A-IRB.** Die EBA-Public-Disclosure enthält "
        "**keine** bank-internen A-IRB-LGD-Schätzwerte. Wir nehmen deshalb "
        "die F-IRB-Standards, die regulatorisch konservativ sind. Reale "
        "A-IRB-LGDs mit einberechneten Sicherheiten liegen für "
        "Senior-Secured-Corporate bei 20–30 %, für Mortgage bei 10–15 %. "
        "Das Modell **überschätzt** deshalb tendenziell das Risiko."
    )
    st.markdown(
        "**2. Klassen-uniformes LGD (keine Sektor- oder Länder-"
        "Granularität).** Alle Corporate-Schuldner in Deutschland und "
        "Italien rechnen wir mit denselben 45 %. Sektor- und Länder-"
        "Recovery-Daten sind in EBA-Public-Disclosure nicht verfügbar — "
        "eine differenzierte Modellierung wäre nur mit bank-internen "
        "Daten möglich."
    )
    st.markdown(
        "**3. Sensitivity-Knob für A-IRB-Proxy in der Sidebar.** Der "
        "Multiplikator `lgd_calibration` erlaubt es, die F-IRB-LGDs uniform "
        "zu skalieren. **Wichtig:** dieser Multiplikator ist ein "
        "Sensitivity-Tool, keine Kalibrierung — real unterscheidet sich "
        "der A-IRB-Abschlag pro Klasse."
    )

    st.markdown("#### Wie wird LGD im Stress transformiert?")
    st.markdown(
        "**Downturn-LGD-Funktion** (EBA Stresstest 2023, CRR Art. 181):"
    )
    st.latex(r"\text{LGD}_{\text{stress}} = \text{LGD}_{\text{base}} "
             r"\cdot \bigl(1 + \kappa \cdot \max(-M, 0)^{p}\bigr)")
    st.markdown(
        f"mit κ = **{KAPPA_DOWNTURN_LGD:.2f}**, Stress-Exponent "
        f"p = **{_stress_exp:.2f}** (aus Sidebar), Cap-Modus = "
        f"**`{_cap_mode}`**."
    )
    st.markdown(
        "- p = 1.0 → linear (V1-Default, regulatorisch konform)  \n"
        "- p > 1.0 → konvex (Tail-Risk-Variante, PD-LGD-Korrelation in Krisen)  \n"
        "- `cap_mode = logit` → Sigmoid-Transformation statt Hard-Cap "
        "(akademische Variante, **nicht** regulatorisch zitierbar)"
    )

    st.markdown("#### Bekannte Limitationen")
    st.markdown(
        "- Kein bank-spezifisches LGD-Niveau (A-IRB nicht verfügbar)  \n"
        "- Kein Recovery-Schätzwert pro Sicherheiten-Klasse  \n"
        "- Keine länder-/sektor-spezifischen Recovery-Raten  \n"
        "- Asymmetrische Stress-Funktion: benigner M reduziert LGD **nicht** "
        "unter die Baseline (regulatorische Konvention)"
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
        "Vasicek-Klasse über alle Counterparty-Countries und Maturity-Buckets."
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
        "- **L₀.₉₉₉** = ASRF-Loss-Quantil aus der Vasicek-Formel"
    )
    st.markdown(
        "Beide werden in der Capital-Bridge nebeneinander gezeigt. Die "
        "CET1-Quote reagiert auf beide: ΔEL reduziert direkt den Zähler, "
        "ΔK · 12.5 · EAD = ΔRWA verdünnt den Nenner."
    )

st.markdown(
    f'<div style="background:#F4F4F4;border-radius:4px;padding:0.7rem 1.0rem;'
    f'margin-top:1rem;font-size:0.85rem;color:#6E6E6E;line-height:1.55;">'
    f'<strong style="color:#051C2C;">Aktive Sensitivity-Parameter</strong> '
    f'(Sidebar-Box „V2 · LGD-Modell-Risiko-Parameter"):  '
    f'A-IRB-Kalibrierung = <strong>{_lgd_cal:.2f}×</strong>  ·  '
    f'Stress-Exponent p = <strong>{_stress_exp:.2f}</strong>  ·  '
    f'LGD-Cap-Modus = <strong>{_cap_mode}</strong>. '
    f'{"Alle Werte = V1-Defaults (regulatorisch korrekte Basel-Variante)." if (abs(_lgd_cal-1.0)<0.01 and abs(_stress_exp-1.0)<0.05 and _cap_mode=="hard") else "Mindestens ein Parameter weicht vom V1-Default ab — Capital-Werte unten sind Sensitivity-Analyse, kein regulatorisches Ergebnis."}'
    f'</div>',
    unsafe_allow_html=True,
)

st.divider()

# === Compute baseline + stressed metrics across all banks =============
baseline_rows = []
stressed_rows = []
for bank_name, portfolio in universe.banks.items():
    base_kpi = portfolio.portfolio_kpis(confidence=0.999,
                                          rho_multiplier=_rho_mult,
                                          lgd_calibration=_lgd_cal)
    base_kpi["Bank"] = bank_name
    baseline_rows.append(base_kpi)

    if abs(m_used) > 1e-9:
        s_kpi = portfolio.stressed_kpis(z_factor=m_used, confidence=0.999,
                                          rho_multiplier=_rho_mult,
                                          lgd_calibration=_lgd_cal,
                                          stress_exponent=_stress_exp,
                                          cap_mode=_cap_mode)
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

# Top-N-Toggle: Default = Top 10 (lesbar) · optional alle 67 Banken
el_show_l, el_show_r = st.columns([1, 4], gap="small")
with el_show_l:
    el_view_mode = st.radio(
        "Ansicht",
        ["Top 10", f"Alle {len(league)}"],
        index=0, key="el_view_mode",
        horizontal=False,
        label_visibility="collapsed",
    )
with el_show_r:
    st.caption(
        "**Top 10** zeigt die zehn Banken mit dem höchsten Stress-EL "
        "(größte Risiko-Träger). **Alle 67** öffnet die volle Liste mit "
        "auto-skalierter Höhe — jede Bank bleibt lesbar."
    )

# Sortiert absteigend nach EL_stress (große oben)
ranked_full = league.sort_values("EL stress bn", ascending=False)

if el_view_mode == "Top 10":
    ranked = ranked_full.head(10)
    chart_height = 460
else:
    ranked = ranked_full
    # Skalierung: ~26 px pro Bank für saubere Lesbarkeit aller 67
    chart_height = max(560, 28 * len(ranked) + 120)

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
fig.update_layout(
    title=(f"Expected Loss · {el_view_mode} · "
           f"Baseline (blau) vs. Stress (crimson) @ M = {m_used:+.2f}"),
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
    "End-to-end transmission chain  Macro shock → conditional PD + downturn "
    "LGD → IRB Capital. Sequential activation: first the PD shift (LGD held "
    "at base), then the LGD shift (PD already stressed). The two contributions "
    "sum exactly to ΔK by construction."
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
if abs(m_used) > 1e-9:
    bridge = bridge_portfolio.capital_bridge(
        z_factor=m_used,
        kappa_lgd=KAPPA_DOWNTURN_LGD,
        confidence=0.999,
        rho_multiplier=_rho_mult,
        lgd_calibration=_lgd_cal,
        stress_exponent=_stress_exp,
        cap_mode=_cap_mode,
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
                "mit $L_{0.999}$ = ASRF-Loss-Quantil bei 99.9 % Konfidenz "
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
    "Komposition pro Vasicek-Klasse."
)

# Vasicek-Klassen-Glossar — verständlich für Erstleser
with st.expander("Was bedeuten die Segment-Namen? · Vasicek-Klassen-Glossar",
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
    ], columns=["Vasicek-Klasse", "Was steckt drin?", "Asset-Korrelation ρ"])
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
drill_base = drill_portfolio.baseline_metrics(confidence=0.999,
                                                rho_multiplier=_rho_mult,
                                                lgd_calibration=_lgd_cal)
if abs(m_used) > 1e-9:
    drill_stress = drill_portfolio.stressed_metrics(
        z_factor=m_used, confidence=0.999,
        rho_multiplier=_rho_mult,
        lgd_calibration=_lgd_cal,
        stress_exponent=_stress_exp,
        cap_mode=_cap_mode,
    )
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

st.divider()

# =====================================================================
# EL-Decomposition pro Klasse · bank-spezifisch (folgt der Bank-Auswahl)
# Beantwortet eine einfache Frage: welche Klasse trägt wie viel zur
# EL-Verschlechterung dieser Bank bei?
# =====================================================================
eyebrow(f"EL-Decomposition · welche Klasse stresst {bridge_bank} am meisten?")

st.caption(
    f"Bank-spezifische Sicht: pro Vasicek-Klasse zeigen wir, wie viel "
    f"absoluter ΔEL-Beitrag (in Mio. €) durch den Stress entsteht. "
    f"Direkt sichtbar wird der Hauptverursacher der EL-Verschlechterung "
    f"für **{bridge_bank}**. Folgt der Bank-Auswahl oben."
)

if abs(m_used) > 1e-9:
    # Stressed metrics pro Segment dieser Bank
    _bank_stress = drill_portfolio.stressed_metrics(
        z_factor=m_used, confidence=0.999,
        kappa_lgd=KAPPA_DOWNTURN_LGD,
        rho_multiplier=_rho_mult,
        lgd_calibration=_lgd_cal,
        stress_exponent=_stress_exp,
        cap_mode=_cap_mode,
    )
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
               f"Σ ΔEL = €{total_delta_el/1e6:+,.0f} M @ M = {m_used:+.2f}"),
        xaxis_title="Vasicek-Exposure-Klasse",
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
        f"Stress anwenden (M-Slider in der Sidebar), um die "
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
    f"Source: {universe.source} · Anchor: {anchor.label} "
    f"(z = {anchor.z_factor:+.2f}) · Factor Σ: {cov_source} · "
    f"LGD per Basel F-IRB defaults · "
    f"PD = observed default ratio (Item 2520512 ÷ Item 2520502)"
)
