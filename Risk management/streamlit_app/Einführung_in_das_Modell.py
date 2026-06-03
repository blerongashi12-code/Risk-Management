"""European Banking Credit Stress Cockpit — Streamlit-Frontend.

Entry-Point. Streamlit erkennt Pages automatisch in `pages/` und rendert
sie in der linken Navigation.

Diese Datei ist der Entry-Point; ihr Dateiname bestimmt zugleich das
Label des ersten Sidebar-Tabs ("Einführung in das Modell").

Start:
    streamlit run "streamlit_app/Einführung_in_das_Modell.py"
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import streamlit as st

from components.theme import (apply_theme, hero, eyebrow, insight, footer,
                              COLORS)
from components.sidebar import render_sidebar
from components.data_loader import load_data_layer
from components.backend_path import setup
setup()


st.set_page_config(
    page_title="EU Banking Credit Stress Cockpit",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

config = render_sidebar()

# ======================================================================
# Hero
# ======================================================================
hero(
    "EU Banking Credit Stress Cockpit",
    eyebrow="Quantitatives Credit-Risk-Modell · 2-Faktor-Stress · "
            "10 IRB-Banken · EBA Q4 2025",
    deck="Wie wirken sich Ölpreis- und Zinsänderungen auf die "
         "Eigenkapitalquote der zehn größten europäischen Banken aus? "
         "Das Cockpit liefert eine regulatorisch fundierte Antwort — "
         "auf Basis bank-publizierter A-IRB-Parameter aus dem EBA Risk "
         "Dashboard Q4 2025 und einer sektor-differenzierten "
         "Stress-Transmission.",
)


# ======================================================================
# 5-Minuten-Tour · Onboarding für Erstleser
# ======================================================================
# ----- Wiederverwendbare Onboarding-Bausteine (Stepper-Design) --------
_ONB_STEPS = [
    ("1", "Was das Modell misst"),
    ("2", "Die zwei Macro-Faktoren"),
    ("3", "Wirkung auf eine Bank"),
    ("4", "Banken & Datenbasis"),
    ("5", "Die drei Stress-Kanäle"),
]


def _onb_rail(active: int | None = None) -> None:
    """Horizontaler 5-Schritte-Stepper — macht sofort sichtbar,
    dass die Tour aus genau fünf Schritten besteht."""
    items = []
    for i, (num, label) in enumerate(_ONB_STEPS, start=1):
        is_on = (active is None) or (i <= active)
        circle_bg = "#051C2C" if is_on else "#FFFFFF"
        circle_fg = "#FFFFFF" if is_on else "#B7C2CB"
        circle_bd = "#051C2C" if is_on else "#D8DEE3"
        lbl_color = "#051C2C" if is_on else "#9BA7B0"
        items.append(
            '<div style="flex:1;min-width:104px;display:flex;'
            'flex-direction:column;align-items:center;text-align:center;'
            'gap:0.45rem;">'
            f'<div style="width:38px;height:38px;border-radius:50%;'
            f'background:{circle_bg};color:{circle_fg};'
            f'border:2px solid {circle_bd};display:flex;align-items:center;'
            'justify-content:center;font-weight:700;font-size:1.02rem;'
            "font-family:'Source Serif Pro',Georgia,serif;\">"
            f'{num}</div>'
            f'<div style="font-size:0.75rem;font-weight:600;'
            f'color:{lbl_color};line-height:1.3;">{label}</div>'
            '</div>'
        )
    arrow = ('<div style="display:flex;align-items:flex-start;color:#C4CDD4;'
             'font-size:1.1rem;padding-top:7px;font-weight:600;">→</div>')
    st.markdown(
        '<div style="display:flex;gap:6px;align-items:flex-start;'
        'flex-wrap:nowrap;overflow-x:auto;margin:0.2rem 0 1.7rem 0;'
        'padding:0.5rem 0.2rem;background:#FAFBFC;border:1px solid #ECEFF1;'
        'border-radius:10px;">'
        + arrow.join(items)
        + '</div>',
        unsafe_allow_html=True,
    )


def _onb_step(num: str, title: str, sub: str) -> None:
    """Markanter, nummerierter Schritt-Header (Badge + Titel + Subtitel)."""
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.95rem;'
        'margin:2.0rem 0 0.9rem 0;padding-bottom:0.65rem;'
        'border-bottom:2px solid #051C2C;">'
        '<div style="flex:none;width:46px;height:46px;border-radius:50%;'
        'background:#051C2C;color:#FFFFFF;display:flex;align-items:center;'
        'justify-content:center;font-weight:700;font-size:1.35rem;'
        "font-family:'Source Serif Pro',Georgia,serif;\">"
        f'{num}</div>'
        '<div style="flex:1;">'
        '<div style="font-size:0.67rem;font-weight:700;letter-spacing:0.14em;'
        'text-transform:uppercase;color:#A52F4D;margin-bottom:0.12rem;">'
        f'Schritt {num} von 5</div>'
        '<div style="font-family:\'Source Serif Pro\',Georgia,serif;'
        'font-size:1.5rem;font-weight:600;color:#051C2C;line-height:1.22;">'
        f'{title}</div>'
        f'<div style="font-size:0.89rem;color:#6E6E6E;margin-top:0.22rem;'
        f'line-height:1.5;">{sub}</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def _onb_sub(text: str) -> None:
    """Leichtgewichtiger Unter-Abschnitt innerhalb eines Schritts."""
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.55rem;'
        'margin:1.5rem 0 0.7rem 0;">'
        '<div style="width:9px;height:9px;border-radius:50%;'
        'background:#A52F4D;flex:none;"></div>'
        '<div style="font-size:1.04rem;font-weight:600;color:#051C2C;'
        "font-family:'Source Serif Pro',Georgia,serif;line-height:1.3;\">"
        f'{text}</div></div>',
        unsafe_allow_html=True,
    )


eyebrow("In 5 Minuten durch das Modell")

st.markdown(
    '<div style="background:linear-gradient(180deg,#FFFFFF 0%,#FAFAFA 100%);'
    'border:1px solid #E6E6E6;border-left:4px solid #051C2C;'
    'border-radius:8px;padding:1.05rem 1.4rem;margin:0.4rem 0 1.3rem 0;'
    'color:#051C2C;font-size:0.94rem;line-height:1.7;">'
    'Diese Tour erklärt das Modell in <strong>fünf aufeinander '
    'aufbauenden Schritten</strong> — von Grund auf, ohne '
    'Vorkenntnisse. Jeder Schritt ist klar nummeriert; danach kannst '
    'du dich durch die fünf Tabs in der linken Sidebar klicken und '
    'jeden Chart und jede Kennzahl konkret einordnen.'
    '</div>',
    unsafe_allow_html=True,
)

# Stepper-Überblick: die fünf Schritte auf einen Blick.
_onb_rail()

# --- Schritt 1 · Was das Modell misst ------------------------------
_onb_step("1", "Was das Modell misst",
          "Die regulatorische Eigenkapitalquote der zehn größten "
          "EU-Banken unter einem Macro-Schock.")
st.markdown("""
Das Cockpit zeigt, wie sich die **Eigenkapitalquote (CET1)** der zehn
größten EU-Banken verändert, wenn der **Ölpreis** und der
**10-Jahres-Zins** springen.

Die CET1-Quote ist die regulatorische Schlüsselkennzahl — sie sagt aus,
wie viel hartes Eigenkapital eine Bank im Verhältnis zu ihren
risikogewichteten Aktiva hat. Fällt sie unter
**4,5 % (Pillar 1) → 7,0 % (inkl. CCB) → 8,0 % (SREP)**, greifen gestufte
Aufsichtsmaßnahmen — bis hin zu Dividenden- und Bonus-Beschränkungen.
""")

# --- Schritt 2 · Die zwei Macro-Faktoren ---------------------------
_onb_step("2", "Die zwei Macro-Faktoren",
          "Ölpreis (Brent) und 10-Jahres-Zins — über die beiden Slider "
          "links unabhängig steuerbar.")
st.markdown("""
- **ΔBrent** — Ölpreis-Veränderung als log-Return.
  +0.50 ≈ +65 % höher, −0.50 ≈ −40 % niedriger.
- **Δr_10y** — 10-Jahres-Zins-Verschiebung in Prozentpunkten.
  +2.0 = +200 Basispunkte, −0.5 = −50 bp.

Die zwei Slider in der linken Sidebar steuern beide Faktoren
**unabhängig**. Empirisch ist ihre Korrelation über fünf Jahre nahezu
null (Pearson ρ ≈ +0.07) — deshalb behandeln wir sie als separate
Risikofaktoren. Im **Tab 1 · Faktor-Analyse & Transmission** findest du
die vollständige Regressions-Analyse mit R-Style-Output, p-Werten und
Konfidenzintervallen, die das empirisch belegt.
""")

# --- Schritt 2b: ökonomische + mathematische Fundierung ------------
_onb_sub("Vertiefung · Warum genau diese Faktoren")

found_l, found_m, found_r = st.columns(3, gap="medium")

with found_l:
    st.markdown(
        '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
        'border-top:2px solid #034B6F;border-radius:4px;'
        'padding:0.85rem 1.0rem;font-size:0.85rem;line-height:1.6;'
        'color:#051C2C;height:100%;">'
        '<div style="font-size:0.66rem;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.10em;color:#6E6E6E;'
        'margin-bottom:0.35rem;">Warum Brent (und nicht WTI)?</div>'
        '<strong>Ökonomisch:</strong> Brent ist der maßgebliche '
        '<em>europäische</em> Ölpreis-Referenzpunkt — über zwei Drittel '
        'des weltweit gehandelten Rohöls wird gegen Brent gepreist '
        '(ICE Futures Europe). Für eine EU-Banken-Stichprobe ist Brent '
        'damit der ökonomisch näherliegende Energie-Schock als WTI '
        '(US-Crude) oder Dubai (Asien).<br><br>'
        '<strong>Transmissionskanäle:</strong> Energie-Inputkosten für '
        'Corporates (insb. energieintensive Industrie), Konsumenten-'
        'Disposable-Income über Heiz- und Kraftstoff-Preise, sowie '
        'Headline-Inflation → Zentralbank-Reaktion.<br><br>'
        '<strong>Quellen:</strong> EBA Stress Test 2025 Methodology '
        'Note Sec. 3.2 (Energie als Macro-Treiber); Hamilton '
        '<em>JME</em> 1983, Kilian <em>AER</em> 2009 (Öl-Schock-'
        'Transmission); ICE Brent als Benchmark in EBA-Adverse-'
        'Szenarien seit 2014.'
        '</div>',
        unsafe_allow_html=True,
    )

with found_m:
    st.markdown(
        '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
        'border-top:2px solid #A52F4D;border-radius:4px;'
        'padding:0.85rem 1.0rem;font-size:0.85rem;line-height:1.6;'
        'color:#051C2C;height:100%;">'
        '<div style="font-size:0.66rem;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.10em;color:#6E6E6E;'
        'margin-bottom:0.35rem;">Warum 10-Jahres-Zins?</div>'
        '<strong>Ökonomisch:</strong> Der 10y-Zins ist der zentrale '
        '<em>Pricing-Anker</em> für Hypotheken (Fixzins-Refinanzierung), '
        'Unternehmensanleihen (Corporate-Spread baseline) und '
        'Staatsanleihen (Sovereign-Duration). Kurze Laufzeiten (2y/5y) '
        'reagieren primär auf Geldpolitik; sehr lange (30y) sind '
        'illiquide. 10y trifft die <em>Bilanz-Duration der EU-Banken</em>.'
        '<br><br>'
        '<strong>Transmissionskanäle:</strong> Hypotheken-Refi-Risiko '
        '(Mortgage-PD), Sovereign-Mark-to-Market (Modified Duration), '
        'Bank-NIM via Yield-Curve-Steepening, Corporate-Bond-Spreads.'
        '<br><br>'
        '<strong>Quellen:</strong> Bundesbank Svensson-Tagesreihe '
        'ab 1972 (offizielle Zero-Curve); EBA Stress Test 2025 nutzt '
        'identischen 10y-Sovereign-Punkt; Drehmann/Juselius <em>BIS WP</em> '
        '2014 zu DSR-Sensitivität auf 10y-Yield.'
        '</div>',
        unsafe_allow_html=True,
    )

with found_r:
    st.markdown(
        '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
        'border-top:2px solid #051C2C;border-radius:4px;'
        'padding:0.85rem 1.0rem;font-size:0.85rem;line-height:1.6;'
        'color:#051C2C;height:100%;">'
        '<div style="font-size:0.66rem;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.10em;color:#6E6E6E;'
        'margin-bottom:0.35rem;">Warum log-Return für Brent?</div>'
        '<strong>Mathematisch:</strong> Ein log-Return r = ln(P₁/P₀) '
        'ist <em>additiv über die Zeit</em> (r<sub>1→3</sub> = '
        'r<sub>1→2</sub> + r<sub>2→3</sub>), <em>symmetrisch</em> '
        '(+0.50 und −0.50 sind gleich groß in der Verteilung) und '
        'annähernd <em>normalverteilt</em> für Ölpreise — was robuste '
        'Regressions-Inferenz erlaubt. Einfache Prozent-Returns sind '
        'das nicht.<br><br>'
        '<strong>Praktisch:</strong> +0.50 entspricht +65 %, '
        '−0.50 entspricht −40 %. Damit ist der Slider <em>symmetrisch</em> '
        'um Null konfigurierbar — was der ökonomischen Realität von '
        'Öl-Boom/-Bust-Phasen entspricht.<br><br>'
        '<strong>Zinsen: keine log-Transformation</strong>, weil '
        'Zinsänderungen schon in pp linear sind und Bond-Pricing '
        '(Modified Duration) direkt auf Δy in pp wirkt. Quellen: '
        'Tuckman/Serrat 2012 (Bond-Pricing); Tsay <em>Analysis of '
        'Financial Time Series</em> 2010 zu log-Returns.'
        '</div>',
        unsafe_allow_html=True,
    )

# --- Schritt 3 · Wirkung auf eine konkrete Bank --------------------
_onb_step("3", "Wirkung auf eine Bank",
          "Sektor-spezifische ΔPD/ΔLGD je Exposure-Klasse — derselbe "
          "Schock trifft jede Klasse anders.")
st.markdown("""
Pro Exposure-Klasse (Corporate, Mortgage, Qualifying
Revolving Retail = QRRE, Other Retail, SME-Corporate, Bank,
Sovereign) gibt es eigene Sensitivitäten. Die zentrale Formel
ist immer dieselbe:
""")

st.markdown(
    '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
    'padding:0.6rem 1.0rem;border-radius:3px;margin:0.4rem 0;'
    'font-family:JetBrains Mono, Consolas, monospace;'
    'font-size:0.90rem;text-align:center;color:#051C2C;">'
    'ΔPD<sub>Klasse</sub> = β<sub>oil</sub> · ΔBrent + '
    'β<sub>rate</sub> · Δr<sub>10y</sub>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    "**Konkretes Beispiel** — was passiert, wenn der Ölpreis um "
    "+50 % steigt und der 10y-Zins um +200 Basispunkte hochgeht?"
)

import pandas as pd
example_df = pd.DataFrame([
    ("UniCredit", "Corporate (Italien)",
     "+0.30", "+0.20",
     "0.30·0.5 + 0.20·2.0 = +0.55 pp",
     "1.67 % → 2.22 %",
     "Beide Faktoren wirken — Ölpreis trifft Energie-Kosten, Zinsen Refi-Kosten"),
    ("Deutsche Bank", "Mortgage (Deutschland)",
     "+0.05", "+0.30",
     "0.05·0.5 + 0.30·2.0 = +0.62 pp",
     "0.87 % → 1.49 %",
     "Zinsen dominieren — Floating-Rate-Hypotheken belasten Hausbesitzer"),
    ("BNP Paribas", "QRRE (Frankreich)",
     "+0.40", "+0.15",
     "0.40·0.5 + 0.15·2.0 = +0.50 pp",
     "2.16 % → 2.66 %",
     "Ölpreis dominiert — Inflation belastet Konsumenten-Disposable-Income"),
    ("Santander", "Bank (Spanien)",
     "+0.05", "−0.05",
     "0.05·0.5 + (−0.05)·2.0 = −0.075 pp",
     "0.20 % → 0.125 %",
     "Bank PROFITIERT von steigenden Zinsen via Net Interest Margin — Kritik Punkt 7"),
], columns=["Bank", "Klasse · Land", "β_oil", "β_rate",
            "Berechnung ΔPD", "PD baseline → stress", "Ökonomische Logik"])
st.dataframe(example_df, hide_index=True, use_container_width=True,
             height=205)

st.markdown(
    "Die Sensitivitäten (β-Werte) sind aus dem **EBA Stress Test "
    "2025 Methodology Note** und der akademischen Literatur kalibriert "
    "(Drehmann/Juselius BIS 2014, Hosszú/Király MNB 2018, Castro "
    "Economic Modelling 2013). Komplette Tabelle in "
    "`MODEL_ASSUMPTIONS.md`."
)

# --- Schritt 3 · Vollbild: alle 7 Klassen einer Bank unter demselben Schock --
_onb_sub("Eine Bank, sieben Klassen, sieben Reaktionen")
st.markdown(
    "Derselbe Schock trifft jede Exposure-Klasse anders. Real-Werte aus "
    "UniCredit Pillar-3 EU-CR6 (31.12.2024) unter "
    "ΔBrent = +0,50 / Δr_10y = +200 bp:"
)

# Real Pillar-3 baselines (UniCredit, alle 7 IRB-Klassen) ·
# ΔPD / ΔLGD aus two_factor_stress.SENSITIVITY_MATRIX nach der Formel
# ΔX = β_oil · 0.50 + β_rate · 2.00 (siehe oben).
_uc_mech = pd.DataFrame([
    ("Corporate",     "+0,30 / +0,20", "+0,55 pp ↑", "2,42 % → 2,97 %",
     "+0,50 / +1,00", "+2,25 pp ↑",  "38,4 % → 40,7 %",
     "Ölpreis verteuert Input-Kosten und drückt Margen, höhere "
     "Zinsen erhöhen die Refinanzierungslast — beide Kanäle treiben "
     "die Ausfallrate moderat nach oben.",
     "EBA Stress-Test 2025, §5.3.2"),
    ("SME-Corporate", "+0,45 / +0,35", "+0,93 pp ↑", "3,93 % → 4,86 %",
     "+0,40 / +1,20", "+2,60 pp ↑",  "14,1 % → 16,7 %",
     "Reagiert ≈ 1,5× sensitiver als Large-Corporate: dünnere "
     "Liquiditätspuffer und geringere Preissetzungsmacht verstärken "
     "die Wirkung desselben Schocks.",
     "Drehmann & Juselius, BIS WP 2014"),
    ("Mortgage",      "+0,05 / +0,30", "+0,63 pp ↑", "1,41 % → 2,04 %",
     "+0,10 / +1,50", "+3,05 pp ↑",  "14,1 % → 17,1 %",
     "Zins-dominiert: variabel verzinste Darlehen erhöhen sofort die "
     "Schuldendienstquote, zugleich drückt die Immobilien-Neubewertung "
     "die Recovery (höchstes γ_rate).",
     "Hosszú & Király, MNB OP 2018"),
    ("QRRE",          "+0,40 / +0,15", "+0,50 pp ↑", "2,97 % → 3,47 %",
     "+0,30 / +0,50", "+1,15 pp ↑",  "54,1 % → 55,3 %",
     "Öl-dominiert: Inflation über Energiekosten belastet das "
     "verfügbare Haushaltseinkommen; unbesicherte Forderungen haben "
     "ohnehin hohe LGD, daher geringer LGD-Hub.",
     "Castro, Economic Modelling 2013"),
    ("Other Retail",  "+0,30 / +0,25", "+0,65 pp ↑", "5,68 % → 6,33 %",
     "+0,25 / +0,80", "+1,72 pp ↑",  "25,9 % → 27,6 %",
     "Mischprofil zwischen besichertem Mortgage und unbesichertem "
     "QRRE — Öl- und Zinskanal wirken etwa gleich stark auf PD und LGD.",
     "EBA Credit-Risk-Benchmark 2024"),
    ("Bank",          "+0,05 / −0,05", "−0,08 pp ↓", "2,74 % → 2,67 %",
     "+0,10 / +0,50", "+1,05 pp ↑",  "34,1 % → 35,2 %",
     "Vorzeichen-Umkehr: steigende Zinsen heben die Net Interest "
     "Margin, der NIM-Uplift überkompensiert den Kreditrisiko-Anstieg "
     "→ PD sinkt. Die LGD steigt dennoch, da höhere Zinsen Bond-Werte "
     "und damit die Recovery drücken.",
     "EBA Stress-Test 2025, §5.3.4"),
    ("Sovereign",     "  0,00 /   0,00", "  0,00 pp ·", "0,06 % → 0,06 %",
     "  0,00 /   0,00", "  0,00 pp ·", "8,25 % → 8,25 %",
     "Macro-orthogonal: Sovereign-Default ist ein fiskalisches, kein "
     "makroökonomisches Ereignis. Der Zinsschock wirkt stattdessen "
     "über den Marktbuch-MtM-Kanal (separate Modellebene, Tab "
     "Marktbuch).",
     "Reinhart & Rogoff, 2009"),
], columns=[
    "Klasse",
    "β_oil / β_rate (PD)",
    "ΔPD",
    "PD baseline → stress",
    "γ_oil / γ_rate (LGD)",
    "ΔLGD",
    "LGD baseline → stress",
    "Ökonomische Begründung",
    "Literatur-Quelle",
])


def _delta_html(val: str) -> str:
    """Farbcodierter Δ-Wert: rot = PD/LGD steigt (schlechter),
    grün = sinkt (besser), grau = neutral."""
    if "↑" in val:
        color = "#C0392B"
    elif "↓" in val:
        color = "#1E8449"
    else:
        color = "#95A5A6"
    return f'<span style="color:{color};font-weight:700;">{val}</span>'


_rows_html = ""
for i, _r in enumerate(_uc_mech.itertuples(index=False)):
    _bg = "#FFFFFF" if i % 2 == 0 else "#F4F7F9"
    _rows_html += (
        f'<tr style="background:{_bg};">'
        f'<td class="cls">{_r[0]}</td>'
        f'<td class="num">{_r[1]}</td>'
        f'<td class="num">{_delta_html(_r[2])}</td>'
        f'<td class="num">{_r[3]}</td>'
        f'<td class="num gsep">{_r[4]}</td>'
        f'<td class="num">{_delta_html(_r[5])}</td>'
        f'<td class="num">{_r[6]}</td>'
        f'<td class="txt gsep">{_r[7]}</td>'
        f'<td class="src">{_r[8]}</td>'
        f'</tr>'
    )

_table_html = (
    '<style>'
    '.mechtbl{border-collapse:collapse;width:100%;font-size:0.80rem;'
    'color:#051C2C;table-layout:auto;margin:0.2rem 0 0.4rem 0;}'
    '.mechtbl th,.mechtbl td{border-bottom:1px solid #E3E9ED;'
    'padding:7px 10px;vertical-align:top;}'
    '.mechtbl thead th{background:#034B6F;color:#FFFFFF;font-weight:600;'
    'text-align:center;border-bottom:none;line-height:1.25;}'
    '.mechtbl thead tr.grp th{background:#022F45;font-size:0.78rem;'
    'letter-spacing:0.04em;text-transform:uppercase;}'
    '.mechtbl td.cls{font-weight:700;white-space:nowrap;}'
    '.mechtbl td.num{text-align:center;white-space:nowrap;'
    "font-family:'Consolas','Courier New',monospace;font-size:0.76rem;}"
    '.mechtbl td.txt{text-align:left;white-space:normal;line-height:1.45;'
    'min-width:230px;}'
    '.mechtbl td.src{text-align:left;white-space:normal;font-style:italic;'
    'color:#5A6B76;font-size:0.74rem;line-height:1.4;min-width:130px;}'
    '.mechtbl .gsep{border-left:2px solid #B7C7D0;}'
    '</style>'
    '<div style="overflow-x:auto;">'
    '<table class="mechtbl">'
    '<thead>'
    '<tr class="grp">'
    '<th rowspan="2">Klasse</th>'
    '<th colspan="3">PD-Kanal</th>'
    '<th colspan="3">LGD-Kanal</th>'
    '<th rowspan="2">Ökonomische Begründung</th>'
    '<th rowspan="2">Literatur-Quelle</th>'
    '</tr>'
    '<tr>'
    '<th>β_oil / β_rate</th><th>ΔPD</th><th>PD: base&nbsp;→&nbsp;stress</th>'
    '<th>γ_oil / γ_rate</th><th>ΔLGD</th><th>LGD: base&nbsp;→&nbsp;stress</th>'
    '</tr>'
    '</thead>'
    f'<tbody>{_rows_html}</tbody>'
    '</table>'
    '</div>'
)
st.markdown(_table_html, unsafe_allow_html=True)

st.caption(
    "Annahmen: PD- und LGD-Baselines aus UniCredit Pillar-3-Report "
    "EU-CR6-Tabelle (Stichtag 31.12.2024, CRR Art. 431–455 + EBA "
    "ITS/2020/04). β/γ-Werte konstant über alle 10 Banken (Sektor-"
    "Konvention, nicht bank-individuell kalibriert — defensible "
    "defaults). PD-Floor 3 bp (Basel Sovereign), PD-Cap 50 %, "
    "LGD-Floor 5 %, LGD-Cap 100 % für numerische Stabilität der "
    "IRB-Capital-Formel."
)

# --- Schritt 3b: Sektor-Sensitivitäten · Übersicht + optionaler Override ---
_onb_sub("Vertiefung · Sektor-Sensitivitäten (β-Werte)")

st.markdown(
    "Diese Tabelle zeigt die Default-β-Werte je Exposure-Klasse. "
    "**β_oil** misst, um wie viele Prozentpunkte die PD steigt, wenn "
    "Brent um +100 % (log = +0.70) springt; **β_rate** entsprechend "
    "für +100 bp Δr_10y. Die Werte sind aus der Literatur abgeleitet, "
    "nicht aus dem Modell geschätzt — das ist methodisch wichtig: "
    "die β sind <em>aufsichtsrechtlich plausibilisiert</em>, nicht "
    "data-gefittet."
)

import pandas as pd
_beta_table = pd.DataFrame([
    ("Corporate",       "+0.30", "+0.20", "Hosszú/Király 2018; EBA ST 2025 §3.4"),
    ("SME Corporate",   "+0.40", "+0.25", "Castro 2013; höhere Sensitivität bei kleineren Firmen"),
    ("Mortgage",        "+0.05", "+0.30", "Drehmann/Juselius 2014 (DSR-Channel)"),
    ("QRRE",            "+0.40", "+0.15", "Castro 2013; Konsumenten-DSR via Energie-Inflation"),
    ("Other Retail",    "+0.25", "+0.20", "EBA ST 2025 §3.4"),
    ("Bank",            "+0.05", "−0.05", "NIM-Profit-Effekt bei steigenden Zinsen (Kritik 7)"),
    ("Sovereign",       "0.00",  "+0.00", "Sovereign-PD nur über Spread-Channel, hier nicht modelliert"),
], columns=["Exposure-Klasse", "β_oil", "β_rate", "Quelle / Logik"])
st.dataframe(_beta_table, hide_index=True, use_container_width=True,
             height=290)

with st.expander("β-Werte überschreiben (Sensitivitäts-Analyse, advanced)",
                 expanded=False):
    st.markdown(
        '<div style="font-size:0.84rem;color:#051C2C;line-height:1.6;'
        'margin:0 0 0.6rem 0;">'
        'Hier kannst du einzelne β-Werte überschreiben, um die '
        'Modell-Sensitivität gegenüber Annahmen zu testen. Die '
        'Änderungen wirken <strong>sofort und global</strong> auf '
        'alle Tabs (Kreditbuch, Eigenkapital, Validierung). '
        'Methodisch: damit prüfst du, wie robust das Endergebnis '
        'gegenüber der Literatur-Kalibrierung der β ist.'
        '</div>',
        unsafe_allow_html=True,
    )
    _CLASSES = ["corporate", "sme_corporate", "mortgage", "qrre",
                "other_retail", "bank"]


    def _default_beta(class_name: str, key: str) -> float:
        """Read default β from two_factor_stress.SENSITIVITY_MATRIX."""
        from pathlib import Path
        _root = Path(__file__).resolve().parent.parent / "backend"
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        try:
            from two_factor_stress import SENSITIVITY_MATRIX  # type: ignore
            return float(SENSITIVITY_MATRIX.get(class_name, {}).get(key, 0.0))
        except Exception:
            return 0.0

    _overrides = {}
    _cols = st.columns(3, gap="small")
    for i, c in enumerate(_CLASSES):
        with _cols[i % 3]:
            st.markdown(
                f'<div style="font-size:0.82rem;font-weight:600;'
                f'color:#051C2C;margin:0.4rem 0 0.2rem 0;">{c}</div>',
                unsafe_allow_html=True,
            )
            b_oil = st.number_input(
                f"β_oil · {c}",
                value=_default_beta(c, "pd_oil"),
                step=0.05, format="%.2f",
                key=f"intro_ov_pd_oil_{c}",
                label_visibility="collapsed",
            )
            st.caption("β_oil")
            b_rate = st.number_input(
                f"β_rate · {c}",
                value=_default_beta(c, "pd_rate"),
                step=0.05, format="%.2f",
                key=f"intro_ov_pd_rate_{c}",
                label_visibility="collapsed",
            )
            st.caption("β_rate")
            if (abs(b_oil - _default_beta(c, "pd_oil")) > 1e-6
                    or abs(b_rate - _default_beta(c, "pd_rate")) > 1e-6):
                _overrides[c] = {
                    "pd_oil":  b_oil,
                    "pd_rate": b_rate,
                    "lgd_oil": _default_beta(c, "lgd_oil"),
                    "lgd_rate": _default_beta(c, "lgd_rate"),
                }

    # Push to session_state so the sidebar picks it up
    st.session_state["sensitivity_overrides"] = (
        _overrides if _overrides else None)

    if _overrides:
        st.success(
            f"**{len(_overrides)} Sensitivität(en) überschrieben** — "
            "alle Tabs nutzen jetzt die geänderten β. Eintrag wieder "
            "auf Default-Wert setzen, um den Override zu entfernen.")
    else:
        st.caption(
            "Keine Overrides aktiv — alle Tabs nutzen die kalibrierten "
            "Default-Werte.")

# --- Schritt 4 · Banken und Datenbasis -----------------------------
_onb_step("4", "Banken & Datenbasis",
          "Zehn IRB-Banken · €8,28 Bio. Exposure · bank-spezifische "
          "Pillar-3-PDs/LGDs + EBA Transparency.")

step4_l, step4_r = st.columns([3, 2], gap="large")
with step4_l:
    st.markdown("""
**Zehn IRB-Banken** mit zusammen €8.28 Bio. Exposure (56 % des
EU-IRB-Banken-Systems):

Crédit Agricole · BNP Paribas · ING · Société Générale · BPCE ·
Deutsche Bank · Crédit Mutuel · Santander · Rabobank · UniCredit.

Die PDs und LGDs kommen **bank-spezifisch direkt aus den
EU-CR6-Tabellen der Pillar-3-Reports** der jeweiligen Banken
(regulatorisch publiziert nach CRR Art. 180 + EBA ITS/2020/04) — alle
10 Banken Pillar-3-verifiziert mit einheitlichem Stichtag 31.12.2024.

Keine Eigenrechnung der PD aus Defaulted/Original-Ratios mehr —
diese frühere Vorgehensweise war ein zentraler Kritikpunkt des
Professors (Mengen- vs. Wertgrößen), den wir mit der Umstellung
auf bank-publizierte A-IRB-Werte adressiert haben.
""")

with step4_r:
    st.markdown("""
**Drei Datenebenen:**

| Ebene | Quelle | Wofür |
|-------|--------|-------|
| PDs + LGDs | Pillar-3 EU-CR6 (bank-spezifisch, 31.12.2024) | Bank × Klasse |
| Exposures (EAD) | EBA Transparency 2025 | Bank × Klasse |
| Macro (Brent, Zins) | ICE / Bundesbank | täglich, 5 Jahre |

Alle Datenquellen sind frei verfügbar und im Cockpit pro
Sektion explizit zitiert.
""")

# --- Schritt 5 · Prozess-Map der drei Channels ---------------------
_onb_step("5", "Die drei Stress-Kanäle",
          "Kreditbuch, Sovereign-Buch und Handelsbuch — drei parallele "
          "Kanäle, die sich in der CET1-Quote addieren.")

# ----- 5.0  Großes „Worum geht es?"-Intro -----------------------
st.markdown(
    '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
    'border-left:4px solid #051C2C;padding:1.2rem 1.4rem;'
    'border-radius:4px;margin:0.6rem 0 1.4rem 0;'
    'color:#051C2C;font-size:0.96rem;line-height:1.75;">'
    '<div style="font-size:0.70rem;font-weight:600;letter-spacing:0.10em;'
    'text-transform:uppercase;color:#6E6E6E;margin-bottom:0.5rem;">'
    'Worum geht es?</div>'
    'Die <strong>CET1-Quote</strong> (Common Equity Tier 1) ist die '
    'wichtigste Eigenkapital-Kennzahl der Bankenaufsicht. Sie ist '
    'einfach ein Bruch:'
    '<div style="text-align:center;margin:0.9rem auto;padding:0.7rem 1rem;'
    'background:#FFFFFF;border:1px solid #E6E6E6;border-radius:4px;'
    'max-width:560px;font-size:0.92rem;line-height:1.5;">'
    'CET1-Quote &nbsp;=&nbsp; '
    '<span style="display:inline-block;text-align:center;'
    'vertical-align:middle;">'
    '<span style="display:block;">hartes Eigenkapital</span>'
    '<span style="display:block;border-top:1px solid #051C2C;'
    'padding-top:3px;margin-top:2px;">'
    'risiko­gewichtete Aktiva (RWA)</span>'
    '</span></div>'
    'Ein Macro-Schock (also ein Ölpreis- oder Zins-Sprung) kann <strong>'
    'beide Größen</strong> verändern:'
    '<ul style="margin:0.4rem 0 0.4rem 1.3rem;padding:0;">'
    '<li>Er kann den <em>Zähler senken</em> — etwa weil Bewertungs­verluste '
    'oder Kredit-Vorsorgen das Eigenkapital aufzehren.</li>'
    '<li>Er kann den <em>Nenner erhöhen</em> — weil das Portfolio '
    'risiko­reicher wird und Basel mehr Eigenkapital­unterlegung fordert.</li>'
    '</ul>'
    'Beide Bewegungen drücken die Quote in Richtung der Aufsichts­'
    'schwellen <strong>4,5 %</strong> (Mindest­anforderung) · '
    '<strong>7,0 %</strong> (mit Kapital­erhaltungs­puffer) · '
    '<strong>8,0 %</strong> (mit aufsichtlichem Zuschlag, SREP). '
    'Wird eine Schwelle unterschritten, greifen abgestufte '
    'Aufsichts­maßnahmen — bis hin zu Dividenden- und Bonus-Verboten.'
    '</div>',
    unsafe_allow_html=True,
)

# ----- 5.1 Mini-Glossar als Expander (mit voller IFRS-9-Erklärung) ---
with st.expander("Mini-Glossar (EL · RWA · MtM · IFRS-9 · FRTB · NIM) — "
                 "ausführlich, mit CET1-Wirkung",
                 expanded=False):
    st.markdown(
        '<table style="width:100%;border-collapse:collapse;'
        'font-size:0.86rem;line-height:1.65;">'

        # --- EL ---
        '<tr><td style="padding:0.45rem 0.5rem 0.45rem 0;vertical-align:top;'
        'width:140px;color:#6E6E6E;border-bottom:1px solid #EEEEEE;">'
        '<strong>EL</strong><br>'
        '<span style="font-size:0.74rem;">Expected Loss</span></td>'
        '<td style="border-bottom:1px solid #EEEEEE;padding:0.45rem 0;">'
        '<strong>Erwarteter Verlust</strong> auf ein Kredit-Portfolio: '
        'EL = PD × LGD × EAD. PD = Ausfallwahrscheinlichkeit, '
        'LGD = Verlust-bei-Ausfall-Quote, EAD = Exposure bei Ausfall.<br>'
        '<em>CET1-Wirkung:</em> wird als Risiko-Vorsorge gebucht → '
        'Aufwand in der GuV → reduziert direkt den <strong>Zähler</strong> '
        '(hartes Eigenkapital).</td></tr>'

        # --- RWA ---
        '<tr><td style="padding:0.45rem 0.5rem 0.45rem 0;vertical-align:top;'
        'color:#6E6E6E;border-bottom:1px solid #EEEEEE;">'
        '<strong>RWA</strong><br>'
        '<span style="font-size:0.74rem;">Risk-Weighted Assets</span></td>'
        '<td style="border-bottom:1px solid #EEEEEE;padding:0.45rem 0;">'
        '<strong>Risikogewichtete Aktiva</strong> — jede Forderung '
        'wird nach Basel mit einem Risikogewicht versehen (für IRB-Banken '
        'aus PD/LGD/Restlaufzeit berechnet). Σ RWA bestimmt, wie viel '
        'Eigenkapital die Bank halten muss.<br>'
        '<em>CET1-Wirkung:</em> steigt RWA, sinkt die Quote, weil der '
        '<strong>Nenner</strong> größer wird (gleiche Eigenkapitalbasis '
        'muss mehr Risiko abdecken).</td></tr>'

        # --- MtM ---
        '<tr><td style="padding:0.45rem 0.5rem 0.45rem 0;vertical-align:top;'
        'color:#6E6E6E;border-bottom:1px solid #EEEEEE;">'
        '<strong>MtM</strong><br>'
        '<span style="font-size:0.74rem;">Mark-to-Market</span></td>'
        '<td style="border-bottom:1px solid #EEEEEE;padding:0.45rem 0;">'
        'Bewertung eines Wertpapiers zum <em>aktuellen Marktpreis</em>, '
        'nicht zum Anschaffungs- oder Buchwert. Steigt der Marktzins, '
        'fallen die Kurse bestehender Anleihen (inverser Zins-Preis-'
        'Zusammenhang).<br>'
        '<em>CET1-Wirkung:</em> abhängig von der IFRS-9-Klasse — '
        'siehe nächste Zeile.</td></tr>'

        # --- IFRS-9 ---
        '<tr><td style="padding:0.55rem 0.5rem 0.55rem 0;vertical-align:top;'
        'color:#6E6E6E;border-bottom:1px solid #EEEEEE;">'
        '<strong>IFRS-9</strong><br>'
        '<span style="font-size:0.74rem;">Klassifizierung von '
        'Finanzinstrumenten</span></td>'
        '<td style="border-bottom:1px solid #EEEEEE;padding:0.55rem 0;">'
        '<strong>Internationaler Bilanz-Standard</strong> (IASB 2014, '
        'EU-pflichtig seit 2018). Teilt Wertpapiere in <strong>vier '
        'Klassen</strong> ein — die Klassifizierung entscheidet, wie '
        'ein MtM-Verlust durch die Bilanz wirkt:'
        '<table style="width:100%;margin-top:0.5rem;border-collapse:collapse;'
        'font-size:0.82rem;background:#FAFAFA;">'

        '<tr><td style="padding:0.35rem 0.5rem;vertical-align:top;'
        'width:90px;border-bottom:1px solid #E6E6E6;">'
        '<strong>HfT</strong><br><span style="font-size:0.70rem;color:#6E6E6E;">'
        'Held-for-Trading</span></td>'
        '<td style="padding:0.35rem 0.5rem;border-bottom:1px solid #E6E6E6;">'
        '<em>Handelsbestand</em> — Positionen mit kurzfristiger Verkaufs-'
        'Absicht. <strong>MtM täglich, voll GuV-wirksam.</strong> '
        'Verlust geht 1:1 durch die GuV ins Eigenkapital → reduziert '
        'sofort den CET1-Zähler.</td></tr>'

        '<tr><td style="padding:0.35rem 0.5rem;vertical-align:top;'
        'border-bottom:1px solid #E6E6E6;">'
        '<strong>FVTPL</strong><br><span style="font-size:0.70rem;'
        'color:#6E6E6E;">Fair Value Through P&amp;L</span></td>'
        '<td style="padding:0.35rem 0.5rem;border-bottom:1px solid #E6E6E6;">'
        '<em>Faire-Wert-Bewertung mit Erfolgswirkung</em> — Wertpapiere '
        'außerhalb des Handelsbuchs, die trotzdem zum Marktwert geführt '
        'werden (z. B. weil SPPI-Test scheitert). <strong>Voll GuV-'
        'wirksam wie HfT.</strong> Verlust mindert CET1-Zähler direkt.'
        '</td></tr>'

        '<tr><td style="padding:0.35rem 0.5rem;vertical-align:top;'
        'border-bottom:1px solid #E6E6E6;">'
        '<strong>FVOCI</strong><br><span style="font-size:0.70rem;'
        'color:#6E6E6E;">Fair Value Through Other Comprehensive Income</span></td>'
        '<td style="padding:0.35rem 0.5rem;border-bottom:1px solid #E6E6E6;">'
        '<em>Faire-Wert-Bewertung über das sonstige Ergebnis</em> — '
        'Bonds, die gehalten <em>und</em> ggf. verkauft werden sollen. '
        '<strong>MtM-Verlust geht NICHT durch die GuV</strong>, sondern '
        'in die OCI-Reserve (Other Comprehensive Income). Diese Reserve '
        'ist Teil des Eigenkapitals → CET1-Zähler wird trotzdem '
        'reduziert, nur an der GuV vorbei.</td></tr>'

        '<tr><td style="padding:0.35rem 0.5rem;vertical-align:top;">'
        '<strong>AC</strong><br><span style="font-size:0.70rem;'
        'color:#6E6E6E;">Amortised Cost</span></td>'
        '<td style="padding:0.35rem 0.5rem;">'
        '<em>Fortgeführter Anschaffungswert</em> (≈ Held-to-Maturity) — '
        'die Bank hält die Anleihe bis zur Fälligkeit. <strong>Keine '
        'MtM-Bewertung in der Bilanz</strong>, keine CET1-Wirkung unter '
        'Zinsstress. Der Wertverlust ist ökonomisch <em>real</em>, aber '
        'bilanziell unsichtbar (latenter Verlust). Aufsichtsrechtlich '
        'umstritten — siehe SVB-Krise 2023.</td></tr>'

        '</table>'
        '<div style="margin-top:0.55rem;padding:0.5rem 0.65rem;'
        'background:#FFF8E1;border:1px solid #FFE082;border-radius:3px;'
        'font-size:0.82rem;line-height:1.55;color:#051C2C;">'
        '<strong>Faustregel im Cockpit:</strong> Annahme 60 % der '
        'Sovereigns sind HfT/FVOCI (CET1-wirksam), 40 % AC (latent). '
        'Diese Aufteilung deckt sich grob mit der EBA-Transparency-'
        'Verteilung großer EU-IRB-Banken (siehe Marktbuch · '
        'Sovereign-Sub-Tab).'
        '</div>'
        '</td></tr>'

        # --- FRTB ---
        '<tr><td style="padding:0.45rem 0.5rem 0.45rem 0;vertical-align:top;'
        'color:#6E6E6E;border-bottom:1px solid #EEEEEE;">'
        '<strong>FRTB</strong><br>'
        '<span style="font-size:0.74rem;">Fundamental Review of the '
        'Trading Book</span></td>'
        '<td style="border-bottom:1px solid #EEEEEE;padding:0.45rem 0;">'
        'Basel-III-Standard (BCBS 2019) für die Berechnung der '
        '<strong>Markt-Risiko-RWA</strong> im Handelsbuch. Misst '
        'Sensitivitäten gegenüber regulatorisch vorgegebenen '
        'Risiko-Faktoren (Zinsen, Spreads, FX, Aktien, Rohstoffe) und '
        'aggregiert über Stress-Perioden.<br>'
        '<em>CET1-Wirkung:</em> bei Macro-Stress steigt die Markt-RWA → '
        '<strong>Nenner</strong> wächst → Quote sinkt.</td></tr>'

        # --- NIM ---
        '<tr><td style="padding:0.45rem 0.5rem 0.45rem 0;vertical-align:top;'
        'color:#6E6E6E;">'
        '<strong>NIM</strong><br>'
        '<span style="font-size:0.74rem;">Net Interest Margin</span></td>'
        '<td style="padding:0.45rem 0;">'
        '<strong>Netto-Zinsmarge</strong> — Differenz zwischen dem, was '
        'die Bank für Kredite verdient, und was sie für Einlagen zahlt, '
        'als Anteil der zinstragenden Aktiva. Kurzfristig profitieren '
        'Banken von steigenden Zinsen (Aktivseite passt sich schneller '
        'an als Passivseite), langfristig folgt aber höheres '
        'Kredit-Risiko.<br>'
        '<em>CET1-Wirkung:</em> indirekt — höhere NIM erhöht den '
        'einbehaltenen Gewinn → stärkt den <strong>Zähler</strong> '
        '(im Modell als negativer β_rate bei der Exposure-Klasse '
        '<em>Bank</em> abgebildet).</td></tr>'
        '</table>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="font-size:0.96rem;line-height:1.75;color:#051C2C;'
    'margin:0.4rem 0 1.4rem 0;">'
    'Im Modell wirken <strong>drei Kanäle parallel</strong> auf die '
    'CET1-Quote. Jeder Kanal hat seinen eigenen Bilanz-Bereich, seine '
    'eigene Formel und seine eigene Logik — die Effekte addieren sich '
    'am Ende auf. Wir gehen sie nacheinander durch, immer in derselben '
    'einfachen Reihenfolge: <strong>was passiert in Worten</strong> → '
    '<strong>Formel mit allen Begriffen erklärt</strong> → '
    '<strong>konkretes Zahlen-Beispiel</strong> → '
    '<strong>Auswirkung auf das Eigenkapital</strong>.'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Gemeinsames CSS für die drei Kanal-Karten
# ============================================================
st.markdown(
    '<style>'
    '.ch-card{background:#FFFFFF;border:1px solid #E6E6E6;'
    'border-radius:6px;padding:1.3rem 1.6rem;'
    'margin:0.6rem 0 1.4rem 0;}'
    '.ch-eyebrow{font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
    'text-transform:uppercase;color:#6E6E6E;margin-bottom:0.25rem;}'
    '.ch-title{font-family:"Source Serif Pro",Georgia,serif;'
    'font-size:1.45rem;font-weight:600;color:#051C2C;'
    'margin:0 0 0.6rem 0;line-height:1.3;}'
    '.ch-lead{font-size:0.96rem;line-height:1.8;color:#051C2C;'
    'margin-bottom:1.1rem;}'
    '.ch-formula-label{font-size:0.74rem;font-weight:600;'
    'letter-spacing:0.10em;text-transform:uppercase;color:#6E6E6E;'
    'margin:1.1rem 0 0.5rem 0;}'
    '.ch-formula{background:#FAFAFA;border:1px solid #E6E6E6;'
    'border-left:4px solid #034B6F;border-radius:4px;'
    'padding:0.9rem 1.2rem;font-family:"JetBrains Mono",Consolas,monospace;'
    'font-size:0.98rem;color:#051C2C;text-align:center;line-height:1.7;'
    'margin-bottom:0.9rem;}'
    '.ch-terms{margin:0 0 0.4rem 0;}'
    '.ch-terms li{margin:0.55rem 0;line-height:1.7;font-size:0.93rem;}'
    '.ch-example-label{font-size:0.74rem;font-weight:600;'
    'letter-spacing:0.10em;text-transform:uppercase;color:#A52F4D;'
    'margin:0.6rem 0 0.5rem 0;}'
    '.ch-example{background:#FAFAFA;border:1px solid #E6E6E6;'
    'border-left:4px solid #A52F4D;border-radius:4px;'
    'padding:1.0rem 1.3rem;font-size:0.94rem;line-height:1.85;'
    'color:#051C2C;}'
    '.ch-example .calc{font-family:"JetBrains Mono",Consolas,monospace;'
    'font-size:0.88rem;background:#FFFFFF;padding:0.15rem 0.4rem;'
    'border-radius:2px;color:#34495E;}'
    '.ch-example strong{color:#A52F4D;font-weight:600;}'
    '.ch-assumption-label{font-size:0.74rem;font-weight:600;'
    'letter-spacing:0.10em;text-transform:uppercase;color:#6E6E6E;'
    'margin:1.0rem 0 0.45rem 0;}'
    '.ch-assumption{background:#FFFFFF;border:1px dashed #C9C9C9;'
    'border-radius:4px;padding:0.85rem 1.1rem;font-size:0.88rem;'
    'line-height:1.75;color:#051C2C;}'
    '.ch-assumption table{width:100%;border-collapse:collapse;}'
    '.ch-assumption td{padding:0.32rem 0.4rem 0.32rem 0;'
    'vertical-align:top;border-bottom:1px solid #F2F2F2;}'
    '.ch-assumption tr:last-child td{border-bottom:none;}'
    '.ch-assumption .a-val{width:30%;font-weight:600;color:#034B6F;}'
    '.ch-assumption .a-src{color:#6E6E6E;font-size:0.84rem;}'
    '.ch-result{background:#051C2C;color:#FFFFFF;border-radius:4px;'
    'padding:0.95rem 1.3rem;margin-top:1.1rem;'
    'font-size:0.95rem;line-height:1.75;}'
    '.ch-result-label{font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
    'text-transform:uppercase;color:#C9C9C9;margin-bottom:0.35rem;}'
    '.ch-source{font-size:0.78rem;color:#6E6E6E;line-height:1.6;'
    'margin-top:0.9rem;padding-top:0.7rem;border-top:1px dotted #E6E6E6;}'
    '</style>',
    unsafe_allow_html=True,
)


# ============================================================
# KANAL 1 · Kreditbuch
# ============================================================
st.markdown(
    '<div class="ch-card">'
    '<div class="ch-eyebrow">Kanal 1 von 3 · Kreditbuch '
    '(englisch: Loan Book)</div>'
    '<div class="ch-title">Was die Bank an Krediten vergeben hat</div>'
    '<div class="ch-lead">'
    'Das <strong>Kreditbuch</strong> ist die größte Bilanzposition '
    'einer Bank — alle Kredite, die sie an Unternehmen, an Privat­'
    'kunden (Hypotheken, Konsumkredite) und an andere Banken vergeben '
    'hat. Wenn der Ölpreis oder der Zins springt, verschlechtert sich '
    'die Zahlungs­fähigkeit der Schuldner: mehr von ihnen fallen aus, '
    'und die Bank muss <em>vorsorglich</em> Geld zurücklegen — diese '
    'Risiko-Vorsorge mindert direkt den Gewinn und damit das '
    'Eigenkapital. Gleichzeitig fordert Basel, dass risiko­reichere '
    'Forderungen mit <em>mehr</em> Eigenkapital unterlegt werden — die '
    'risiko­gewichteten Aktiva steigen.'
    '</div>'

    '<div class="ch-formula-label">Die Formel</div>'
    '<div class="ch-formula">'
    'ΔEL = ΔPD · LGD · EAD &nbsp;&nbsp;|&nbsp;&nbsp; '
    'ΔRWA = f<sub>IRB</sub>(PD, LGD) · EAD'
    '</div>'

    '<div class="ch-formula-label">Was die Begriffe bedeuten</div>'
    '<ul class="ch-terms">'
    '<li><strong>EL</strong> · <em>Expected Loss</em>, also der '
    'erwartete Verlust — wie viel die Bank durch Kreditausfälle im '
    'Schnitt verliert. Δ bedeutet: die Veränderung gegenüber dem '
    'Normalzustand. ΔEL ist also der zusätzliche erwartete Verlust '
    'durch den Stress.</li>'
    '<li><strong>PD</strong> · <em>Probability of Default</em>, die '
    'Ausfall­wahrscheinlichkeit eines Schuldners. ΔPD ist der Anstieg '
    'der Ausfall­quote, den unser 2-Faktor-Modell vorhersagt (Brent + '
    'Zins, mit den sektor-spezifischen Sensitivitäten aus Schritt 3).</li>'
    '<li><strong>LGD</strong> · <em>Loss Given Default</em>, also die '
    'Verlustquote im Falle eines Ausfalls. Beispiel: LGD = 41 % heißt, '
    'die Bank bekommt im Schnitt 59 % einer notleidenden Forderung '
    'zurück (Sicherheiten, Verwertung).</li>'
    '<li><strong>EAD</strong> · <em>Exposure at Default</em>, der '
    'ausstehende Forderungsbetrag in Euro — also wie viel Geld bei '
    'einem hypothetischen Ausfall überhaupt im Risiko steht.</li>'
    '<li><strong>RWA</strong> · <em>Risk-Weighted Assets</em>, '
    'risiko­gewichtete Aktiva. Jede Forderung wird mit einem '
    'Risikogewicht versehen — riskante Kredite zählen in der '
    'Aufsichts-Bilanz mehr als sichere. Die Funktion '
    'f<sub>IRB</sub> ist die Basel-III-Formel, die aus PD, LGD und '
    'Restlaufzeit ein Risikogewicht macht (formal: BCBS 2017, '
    'Artikel 153 der EU-Kapital­anforderungs­verordnung CRR).</li>'
    '</ul>'

    '<div class="ch-example-label">Ein konkretes Zahlen-Beispiel</div>'
    '<div class="ch-example">'
    'Nehmen wir einen Unternehmenskredit-Bestand von €100&nbsp;Mrd. '
    'mit einer LGD von 41 % und einer Ausgangs-PD von 1,67 %. Der '
    'Schock: Brent +50 % (also Δlog = +0.50) und Δr_10y = +2,0&nbsp;pp. '
    'Mit den β-Werten für Corporate aus Schritt 3 (β_oil = +0.30, '
    'β_rate = +0.20) ergibt sich:'
    '<br><br>'
    '<span class="calc">ΔPD = 0.30 · 0.50 + 0.20 · 2.0 = +0.55 '
    'Prozentpunkte</span><br>'
    'Die PD steigt also von 1,67 % auf 2,22 %.<br><br>'
    'Daraus folgt:<br>'
    '<span class="calc">ΔEL = 0.0055 · 0.41 · €100 Mrd ≈ '
    '<strong>€226 Mio</strong></span> zusätzliche Risiko-Vorsorge.<br>'
    'Und die zusätzlichen risiko­gewichteten Aktiva betragen rund '
    '<strong>+€5&nbsp;Mrd.</strong> (aus der Basel-IRB-Formel: höhere '
    'PD → höheres Risikogewicht).'
    '</div>'

    '<div class="ch-assumption-label">Woher die Zahlen im Beispiel '
    'stammen</div>'
    '<div class="ch-assumption">'
    '<table>'
    '<tr><td class="a-val">PD = 1,67 %</td>'
    '<td class="a-src">Median der bank-spezifischen A-IRB-Corporate-'
    'PD aller zehn Top-IRB-Banken aus deren Pillar-3 EU-CR6-Tabellen '
    'am Stichtag 31.12.2024 — regulatorisch publizierter Wert nach '
    'CRR Art. 180 + EBA ITS/2020/04. Spannweite über die 10 Banken: '
    'rund 1,5–6,8 % (BNP Paribas bis Deutsche Bank).</td></tr>'
    '<tr><td class="a-val">LGD = 41 %</td>'
    '<td class="a-src">Median der bank-spezifischen A-IRB-Corporate-'
    'LGD aus den Pillar-3-EU-CR6-Tabellen (31.12.2024). Liegt '
    'innerhalb des typischen Basel-Foundation-IRB-Korridors von '
    '25–45 % für Senior Unsecured Corporate.</td></tr>'
    '<tr><td class="a-val">EAD = €100 Mrd.</td>'
    '<td class="a-src">Größenordnungs-Beispiel — der tatsächliche '
    'Corporate-EAD pro Bank im Cockpit-Universe reicht von '
    'rund €50 Mrd. (Crédit Mutuel) bis €250 Mrd. (BNP Paribas), '
    'Datenquelle EBA Transparency 2025 Item 2520522. €100 Mrd. ist '
    'die mediane Größenordnung.</td></tr>'
    '<tr><td class="a-val">β_oil = +0,30 · β_rate = +0,20</td>'
    '<td class="a-src">Sektor-Sensitivitäten der Klasse Corporate — '
    'kalibriert nach EBA Stress Test 2025 Methodology Note §3.4 sowie '
    'Hosszú/Király 2018 (MNB Working Paper, ungarisches Banken-'
    'System) und Castro 2013 (Economic Modelling, Eurozone-Panel). '
    'Vollständige Matrix in Schritt 3b oben.</td></tr>'
    '</table>'
    '</div>'

    '<div class="ch-result">'
    '<div class="ch-result-label">Auswirkung auf die CET1-Quote</div>'
    'Der <strong>Zähler</strong> (hartes Eigenkapital) sinkt um die '
    '<strong>€226&nbsp;Mio</strong> Risiko-Vorsorge — sie ist ein '
    'Aufwand in der Gewinn-und-Verlust-Rechnung. Gleichzeitig steigt '
    'der <strong>Nenner</strong> (RWA) um <strong>€5&nbsp;Mrd</strong>. '
    'Beide Bewegungen senken die Quote.'
    '</div>'

    '<div class="ch-source">'
    '<strong>Fundament:</strong> Basel III IRB-Ansatz (BCBS 2017, Art. '
    '153 CRR) · bank-spezifische Pillar-3 EU-CR6 (31.12.2024) für die regulatorisch '
    'publizierten A-IRB-PDs und -LGDs · β-Sensitivitäten aus EBA '
    'Stress Test 2025 Methodology Note und der Literatur '
    '(Hosszú/Király 2018, Castro 2013, Drehmann/Juselius 2014).'
    '</div>'

    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# KANAL 2 · Sovereign Book
# ============================================================
st.markdown(
    '<div class="ch-card">'
    '<div class="ch-eyebrow">Kanal 2 von 3 · Sovereign Book '
    '(deutsch: Staatsanleihen-Bestand)</div>'
    '<div class="ch-title">Was die Bank an Staatsanleihen hält</div>'
    '<div class="ch-lead">'
    'Jede große europäische Bank hält für hunderte Milliarden Euro '
    '<strong>Staatsanleihen</strong> — meist solche des eigenen '
    'Heimat-Staates und seiner Nachbarn. Diese Anleihen sind in '
    'der Bilanz unterschiedlich aufgeführt: ein Teil wird zum '
    'aktuellen Marktpreis bewertet (Mark-to-Market, also „zum '
    'Marktpreis"), der Rest zum ursprünglichen Anschaffungspreis. '
    'Wenn die Zinsen steigen, fallen automatisch die Kurse aller '
    'umlaufenden Anleihen (das ist eine mathematische Gesetz­mäßig­'
    'keit der Anleihen­bewertung). Ob dieser Verlust in der Bilanz '
    'sichtbar wird, hängt allein von der bilanziellen Einordnung ab.'
    '</div>'

    '<div class="ch-formula-label">Die Formel</div>'
    '<div class="ch-formula">'
    'ΔFV ≈ −D<sub>mod</sub> · Δr · Exposure '
    '&nbsp;&nbsp;|&nbsp;&nbsp; '
    'CET1-Effekt = ΔFV · IFRS-9-Filter'
    '</div>'

    '<div class="ch-formula-label">Was die Begriffe bedeuten</div>'
    '<ul class="ch-terms">'
    '<li><strong>FV</strong> · <em>Fair Value</em>, also der aktuelle '
    'Marktwert der Anleihen. ΔFV ist die Veränderung dieses Marktwerts '
    'durch den Zinsanstieg.</li>'
    '<li><strong>D<sub>mod</sub></strong> · <em>Modified Duration</em>, '
    'gemessen in Jahren. Sie ist das gängige Maß für die Zins­'
    'sensitivität einer Anleihe: eine Duration von 6 bedeutet — als '
    'Faustregel — dass der Anleihen­kurs um etwa 6 % fällt, wenn der '
    'Marktzins um +100 Basispunkte (+1 Prozentpunkt) steigt.</li>'
    '<li><strong>Δr</strong> · die Zinsänderung als Dezimal­zahl. '
    'Δr_10y = +2,0&nbsp;pp entspricht 0,020 in der Formel.</li>'
    '<li><strong>Exposure</strong> · der Nominalwert des Bond-Portfolios '
    'in Euro.</li>'
    '<li><strong>IFRS-9-Filter</strong> · der entscheidende Punkt. '
    'IFRS-9 ist der internationale Rechnungslegungs-Standard und teilt '
    'Anleihen in vier Klassen ein, die <em>unterschiedlich stark</em> '
    'auf das Eigenkapital wirken:'
    '<ul style="margin-top:0.5rem;">'
    '<li><strong>HfT</strong> (<em>Held-for-Trading</em>, Handels­'
    'bestand) · Bewertet zum Marktpreis, der Verlust geht <strong>direkt '
    'durch die Gewinn-und-Verlust-Rechnung</strong> ins Eigenkapital.</li>'
    '<li><strong>FVTPL</strong> (<em>Fair Value Through Profit &amp; '
    'Loss</em>) · Genauso behandelt wie HfT — voll erfolgswirksam.</li>'
    '<li><strong>FVOCI</strong> (<em>Fair Value Through Other '
    'Comprehensive Income</em>) · Verlust geht <strong>nicht durch die '
    'GuV</strong>, sondern in die OCI-Reserve (eine spezielle Rücklage '
    'innerhalb des Eigenkapitals). Wirkt trotzdem voll auf die '
    'CET1-Quote, nur an der GuV vorbei.</li>'
    '<li><strong>AC</strong> (<em>Amortised Cost</em>, fortgeführte '
    'Anschaffungs­kosten) · Wird <strong>nicht zum Marktpreis</strong>, '
    'sondern zum Buchwert geführt. Der Verlust ist wirtschaftlich '
    '<em>real</em>, in der Bilanz aber unsichtbar — ein „latenter '
    'Verlust", der erst beim Verkauf sichtbar würde (genau das wurde '
    'der Silicon Valley Bank 2023 zum Verhängnis).</li>'
    '</ul>'
    '</li>'
    '</ul>'

    '<div class="ch-example-label">Ein konkretes Zahlen-Beispiel</div>'
    '<div class="ch-example">'
    'Eine Bank hält €200&nbsp;Mrd. an Staatsanleihen mit einer '
    'durchschnittlichen Modified Duration von 6,2 Jahren. Der Schock: '
    'Δr_10y = +2,0&nbsp;pp.'
    '<br><br>'
    'Der reine Marktwert-Verlust auf das gesamte Portfolio:<br>'
    '<span class="calc">ΔFV = −6.2 · 0.020 · €200 Mrd ≈ '
    '<strong>−€25 Mrd.</strong></span><br><br>'
    'Wir nehmen die übliche IFRS-9-Aufteilung an: 60 % HfT/FVOCI '
    '(CET1-wirksam) und 40 % AC (nicht CET1-wirksam). Damit:<br>'
    '<span class="calc"><strong>CET1-Effekt ≈ −€15 Mrd.</strong></span> '
    '(60 % von −€25 Mrd.).'
    '<br><br>'
    'Die übrigen €10&nbsp;Mrd. Verlust existieren wirtschaftlich, '
    'erscheinen aber nicht in der Bilanz. Bei einem Notverkauf — etwa '
    'wegen Liquiditäts­problemen — würden sie sofort sichtbar.'
    '</div>'

    '<div class="ch-assumption-label">Woher die Zahlen im Beispiel '
    'stammen</div>'
    '<div class="ch-assumption">'
    '<table>'
    '<tr><td class="a-val">Bestand = €200 Mrd.</td>'
    '<td class="a-src">Größenordnungs-Beispiel — typischer Sovereign-'
    'Bestand einer großen EU-Bank laut EBA Transparency 2025 '
    '(Items 2520810/812/813/814/815, summiert über alle Laufzeit-'
    'Buckets). Spannweite im Universe: BNP Paribas hält rund '
    '€600 Mrd., kleinere Häuser wie Crédit Mutuel knapp €100 Mrd.</td></tr>'
    '<tr><td class="a-val">Modified Duration = 6,2 Jahre</td>'
    '<td class="a-src">Berechnet als nominal-gewichteter Durchschnitt '
    'der sieben EBA-Laufzeit-Buckets (&lt;3M, 3M–1J, 1–2J, 2–3J, '
    '3–5J, 5–10J, &gt;10J) für den medianen IRB-Bank — entspricht '
    'rund 6 Jahren und liegt im Bereich 4–8 Jahre über die zehn '
    'Banken hinweg. Methodik: Tuckman/Serrat 2012 (Modified-Duration-'
    'Approximation für Coupon-Bonds).</td></tr>'
    '<tr><td class="a-val">IFRS-9-Mix · 60 % HfT/FVOCI, 40 % AC</td>'
    '<td class="a-src">Stylized-fact-Annahme, gestützt auf die '
    'EBA-Reports „Implementation of IFRS 9 by EU Banks" (jährlich '
    'seit 2018) und die ECB-SSM-Supervisory-Banking-Statistics: für '
    'große EU-IRB-Banken liegt der Anteil der Sovereign-Anleihen, '
    'die zum Marktwert geführt werden (HfT + FVTPL + FVOCI), seit '
    '2020 stabil zwischen 50 % und 65 %; AC liegt entsprechend '
    'zwischen 35 % und 50 %. Wir nehmen 60 / 40 als typischen '
    'Mittelpunkt — die Sensitivität gegen diese Annahme wird im '
    'Tab Marktbuch → Sovereign-Sub-Tab geprüft.</td></tr>'
    '</table>'
    '</div>'

    '<div class="ch-result">'
    '<div class="ch-result-label">Auswirkung auf die CET1-Quote</div>'
    'Der <strong>Zähler</strong> (Eigenkapital) sinkt um '
    '<strong>€15&nbsp;Mrd</strong> — bei HfT/FVTPL über die GuV, '
    'bei FVOCI über die OCI-Reserve. Der <strong>Nenner</strong> '
    'bleibt praktisch unverändert (Staatsanleihen haben in der '
    'Basel-Welt meist 0 % Risikogewicht, der sogenannte „Sovereign-'
    'Floor").'
    '</div>'

    '<div class="ch-source">'
    '<strong>Fundament:</strong> Modified-Duration-Approximation '
    '(klassisches Bond-Pricing, Tuckman/Serrat 2012) · IFRS-9-Standard '
    '(IASB 2014, EU-pflichtig seit 2018) · EBA Report „Implementation '
    'of IFRS 9 by EU Banks" (jährlich) für den HfT/FVOCI/AC-Mix · '
    'EBA Transparency Exercise 2025, Sovereign-Items 2520810/812/813/'
    '814/815 für die maturity-bucket-spezifischen Bestände pro Bank.'
    '</div>'

    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# KANAL 3 · Trading Book
# ============================================================
st.markdown(
    '<div class="ch-card">'
    '<div class="ch-eyebrow">Kanal 3 von 3 · Trading Book '
    '(deutsch: Handelsbuch)</div>'
    '<div class="ch-title">Was die Bank aktiv handelt</div>'
    '<div class="ch-lead">'
    'Das <strong>Handelsbuch</strong> umfasst alle Positionen, die '
    'die Bank aktiv kauft und verkauft — Anleihen, Aktien, Derivate, '
    'Fremdwährungen, Rohstoffe. Anders als die Kredite im Kreditbuch '
    'oder die Halte-Bestände im Sovereign Book werden Handels­'
    'positionen <em>täglich</em> zum Marktpreis bewertet und sind '
    'sehr direkt von Markt­volatilität betroffen. Ein Schock hat hier '
    'zwei Effekte gleichzeitig: erstens steigt die regulatorisch '
    'vorgeschriebene Markt-Risiko-Kapital­unterlegung (mehr RWA), '
    'zweitens entstehen unmittelbar Bewertungs­verluste im laufenden '
    'Handels­ergebnis.'
    '</div>'

    '<div class="ch-formula-label">Die Formel</div>'
    '<div class="ch-formula">'
    'RWA<sub>market, stress</sub> = RWA<sub>market, 0</sub> · '
    '(1 + k · |Schock|) '
    '&nbsp;&nbsp;|&nbsp;&nbsp; '
    'ΔP&amp;L = −h · P&amp;L<sub>trading</sub>'
    '</div>'

    '<div class="ch-formula-label">Was die Begriffe bedeuten</div>'
    '<ul class="ch-terms">'
    '<li><strong>RWA<sub>market, 0</sub></strong> · die Markt-Risiko-'
    'gewichteten Aktiva der Bank im Normalfall — eine regulatorische '
    'Größe, die das aktuelle Verlustrisiko des Handelsbuchs misst. '
    'Wir lesen sie direkt aus dem EBA-Transparency-Datensatz '
    '(Item 2520210) ab.</li>'
    '<li><strong>|Schock|</strong> · die Summe der absoluten Faktor-'
    'Bewegungen: |ΔBrent| + |Δr_10y|. Wir nehmen die Beträge, weil '
    'Markt­stress unabhängig vom Vorzeichen wirkt — Volatilität ist '
    'symmetrisch, sowohl ein Crash als auch eine Rally erhöhen die '
    'Markt-RWA.</li>'
    '<li><strong>k</strong> · ein Skalierungs­faktor (Wert: 0,15). Er '
    'bildet vereinfacht die Mechanik der Markt-RWA-Modelle nach FRTB '
    'ab — FRTB steht für „Fundamental Review of the Trading Book" '
    'und ist der Basel-Standard für Markt-Risiko-Kapital seit 2019. '
    'Der konkrete Wert k = 0,15 ist an die EBA-Stress-Test-2025-'
    'Annahmen kalibriert.</li>'
    '<li><strong>P&amp;L<sub>trading</sub></strong> · das jährliche '
    'Handels­ergebnis (Profit-and-Loss, also Gewinn aus Trading-'
    'Aktivitäten). Wir lesen es ebenfalls aus den EBA-Daten '
    '(Item 2520311).</li>'
    '<li><strong>h</strong> · die Haircut-Quote (Wert: 0,50, also '
    '50 %). Annahme: in einem Stress-Szenario halbiert sich der '
    'Trading-Gewinn — die Bank verliert die Hälfte ihres üblichen '
    'Handels­einkommens.</li>'
    '</ul>'

    '<div class="ch-example-label">Ein konkretes Zahlen-Beispiel</div>'
    '<div class="ch-example">'
    'Eine Bank hat Markt-RWA von €40&nbsp;Mrd. und ein '
    'Trading-Jahres­ergebnis von €1,2&nbsp;Mrd. Schock: Brent +50 % '
    'und Δr_10y = +2,0&nbsp;pp.'
    '<br><br>'
    'Schock-Magnitude:<br>'
    '<span class="calc">|Schock| = |+0.50| + |+2.0| = 2.5</span><br><br>'
    'Daraus folgt:<br>'
    '<span class="calc">ΔRWA = €40 Mrd · 0.15 · 2.5 = '
    '<strong>+€15 Mrd.</strong></span> zusätzliche Markt-RWA.<br><br>'
    'Und der Bewertungs­verlust:<br>'
    '<span class="calc">ΔP&amp;L = −0.50 · €1.2 Mrd = '
    '<strong>−€0,6 Mrd.</strong></span>'
    '</div>'

    '<div class="ch-assumption-label">Woher die Zahlen im Beispiel '
    'stammen</div>'
    '<div class="ch-assumption">'
    '<table>'
    '<tr><td class="a-val">Markt-RWA = €40 Mrd.</td>'
    '<td class="a-src">Mediane Markt-Risiko-RWA der Top-10 EU-IRB-'
    'Banken aus EBA Transparency 2025, Item 2520210 (regulatorisch '
    'publizierte Größe). Spannweite: rund €8 Mrd. bei kleineren '
    'Häusern (Rabobank, Crédit Mutuel) bis ca. €70 Mrd. bei BNP '
    'Paribas oder Deutsche Bank.</td></tr>'
    '<tr><td class="a-val">Trading-Jahresergebnis = €1,2 Mrd.</td>'
    '<td class="a-src">Mediane Net Trading Income der Top-10 IRB-'
    'Banken laut EBA Transparency 2025, Item 2520311 (Geschäftsjahr '
    '2024). Spannweite €0,3 bis €3,5 Mrd. — stark konzentriert bei '
    'den großen Investment-Banken (DB, BNP, SocGen).</td></tr>'
    '<tr><td class="a-val">k = 0,15 (Skalierungs-Faktor Markt-RWA)</td>'
    '<td class="a-src">Kalibrierung: laut EBA Stress Test 2025 '
    '§7 (Market-Risk-Methodology) steigt die Markt-RWA großer EU-'
    'Banken unter dem Adverse-Szenario im Median um etwa 30–40 % '
    'gegenüber Baseline. Bei einer typischen Schock-Magnitude von '
    '|ΔBrent| + |Δr| ≈ 2 in Adverse-Szenarien entspricht das '
    '0,30 / 2 ≈ 0,15. Konsistent mit ECB-SSM-MR-RWA-Statistik '
    '(quartalsweise).</td></tr>'
    '<tr><td class="a-val">h = 50 % (Haircut Trading-Ergebnis)</td>'
    '<td class="a-src">Kalibrierung: EBA Stress Test 2025 Methodology '
    'Note §3.4 („Net Trading Income under stress") sieht für das '
    'Adverse-Szenario eine Reduktion der Net Trading Income um etwa '
    '40–60 % gegenüber Baseline vor; das EBA Stress Test 2023 '
    '(§3.2) nutzte denselben Korridor. Wir nehmen den Mittelpunkt '
    '50 %. Grundlage: BCBS „Principles for Sound Stress Testing" '
    '(2018) — Stress-Annahmen sollen <em>conservative</em>, aber '
    'plausibel sein.</td></tr>'
    '<tr><td class="a-val">|Schock| = |ΔBrent| + |Δr_10y|</td>'
    '<td class="a-src">Additive Aggregation gewählt, weil Brent und '
    '10y-Zins empirisch nahezu unkorreliert sind (Pearson '
    'ρ ≈ +0,07 über 5 Jahre, vgl. Schritt 2 und die Korrelations-'
    'Analyse in Tab 1) — ihre Volatilitäts-Beiträge addieren sich '
    'damit ohne Korrelations-Korrektur.</td></tr>'
    '</table>'
    '</div>'

    '<div class="ch-result">'
    '<div class="ch-result-label">Auswirkung auf die CET1-Quote</div>'
    'Der <strong>Zähler</strong> (Eigenkapital) sinkt um '
    '<strong>€0,6&nbsp;Mrd</strong> (der halbierte Handels­gewinn '
    'fließt direkt durch die GuV). Der <strong>Nenner</strong> '
    'steigt um <strong>€15&nbsp;Mrd</strong> zusätzliche Markt-RWA.'
    '</div>'

    '<div class="ch-source">'
    '<strong>Fundament:</strong> FRTB-Rahmenwerk (BCBS 2019, '
    'Basel III Phase 2) · EBA Transparency 2025: Markt-RWA aus '
    'Item 2520210, Trading-P&amp;L aus Item 2520311 · '
    'k-Multiplikator kalibriert an EBA Stress Test 2025 §7 · '
    'Haircut-Annahme an EBA Stress Test 2025 §3.4 / 2023 §3.2 sowie '
    'BCBS „Principles for Sound Stress Testing" (2018).'
    '</div>'

    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Zusammenführung: die drei Kanäle in der CET1-Formel
# ============================================================
st.markdown(
    '<div style="text-align:center;font-size:0.74rem;color:#6E6E6E;'
    'letter-spacing:0.12em;text-transform:uppercase;font-weight:600;'
    'margin:1.0rem 0 0.4rem 0;">Alles zusammen</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="background:#051C2C;color:#FFFFFF;border-radius:6px;'
    'padding:1.3rem 1.6rem;margin:0.2rem 0 1.0rem 0;">'
    '<div style="text-align:center;font-size:1.05rem;line-height:1.6;'
    'font-weight:600;margin-bottom:0.7rem;">'
    'So addieren sich die drei Kanäle in der CET1-Quote'
    '</div>'

    '<div style="background:rgba(255,255,255,0.06);'
    'border:1px solid #2A4054;border-radius:4px;'
    'padding:1.0rem 1.2rem;margin:0.7rem auto;max-width:760px;'
    'font-size:0.94rem;line-height:1.8;">'
    '<strong>Neuer Zähler</strong> &nbsp;=&nbsp; '
    'Eigenkapital<sub>vorher</sub> &nbsp;−&nbsp; '
    'Kredit-Vorsorge (Kanal&nbsp;1) &nbsp;−&nbsp; '
    'Marktwert-Verlust Sovereigns (Kanal&nbsp;2) &nbsp;−&nbsp; '
    'Trading-Bewertungsverlust (Kanal&nbsp;3)'
    '<div style="border-top:1px solid #FFFFFF;width:80%;margin:0.7rem auto;'
    'opacity:0.5;"></div>'
    '<strong>Neuer Nenner</strong> &nbsp;=&nbsp; '
    'RWA<sub>vorher</sub> &nbsp;+&nbsp; '
    'zusätzliche Kredit-RWA (Kanal&nbsp;1) &nbsp;+&nbsp; '
    'zusätzliche Markt-RWA (Kanal&nbsp;3)'
    '</div>'

    '<div style="font-size:0.88rem;color:#E6E6E6;margin-top:0.8rem;'
    'line-height:1.7;text-align:center;">'
    'Diese neue Quote wird dann gegen die drei Aufsichts­schwellen '
    'verglichen: '
    '<strong style="color:#FFFFFF;">4,5 %</strong> (Pillar 1), '
    '<strong style="color:#FFFFFF;">7,0 %</strong> (mit Kapital­'
    'erhaltungs­puffer) und '
    '<strong style="color:#FFFFFF;">8,0 %</strong> (mit aufsichtlichem '
    'Zuschlag SREP). Jede Unter­schreitung löst gestaffelte Maßnahmen '
    'aus — von Kapitalerhaltungs-Pflichten bis hin zu Dividenden- und '
    'Bonus-Verboten.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
    'padding:1.0rem 1.3rem;border-radius:4px;margin:1.0rem 0;'
    'font-size:0.92rem;color:#051C2C;line-height:1.75;">'
    '<strong>So liest du das Cockpit:</strong> Die zwei Slider links '
    'in der Sidebar bewegen die Macro-Faktoren — alle Zahlen, Charts '
    'und KPIs in den fünf Tabs reagieren sofort live darauf.'
    '<ul style="margin:0.5rem 0 0 1.2rem;padding:0;">'
    '<li><strong>Tab 1 · Faktor-Analyse &amp; Transmission</strong> — '
    'die empirische Korrelations­analyse plus die sequentielle '
    'Stress-Bridge ΔBrent + Δr → ΔPD → ΔLGD → Capital → CET1.</li>'
    '<li><strong>Tab 2 · Kreditbuch</strong> — Kanal 1 mit detailliertem '
    'Worked Example pro Bank.</li>'
    '<li><strong>Tab 3 · Marktbuch</strong> — Yield-Curve, Kanal 2 '
    '(Sovereigns mit IFRS-9-Aufteilung), Banking-Book-Bonds und '
    'Kanal 3 (Trading).</li>'
    '<li><strong>Tab 4 · Eigenkapital</strong> — alle drei Kanäle '
    'aggregiert, mit Schwellen-Analyse.</li>'
    '<li><strong>Tab 5 · Validierung</strong> — Backtest und '
    'Methodologie.</li>'
    '</ul>'
    '</div>',
    unsafe_allow_html=True,
)


# ======================================================================
# Data layer status + macro snapshot
# ======================================================================
data = load_data_layer()
sven = data["svensson"]
brent = data["brent"]

st.markdown(" ")  # vertical breathing room
eyebrow("System-Snapshot")

snap_cols = st.columns(5, gap="small")

# 1. Stichtag
if sven is not None:
    snap_cols[0].metric("Stichtag",
                        sven.index[-1].strftime("%Y-%m-%d"),
                        "Bundesbank-Reporting", delta_color="off")
else:
    snap_cols[0].metric("Stichtag", "—", "Daten fehlen", delta_color="off")

# 2. 10y-Zins (aktuell)
if sven is not None:
    from svensson import historical_curve, zero_rate  # type: ignore
    p = historical_curve(sven.index[-1], sven)
    r10 = float(zero_rate(10.0, p, as_decimal=False))
    if len(sven) > 22:
        p_prev = historical_curve(sven.index[-22], sven)
        r10_prev = float(zero_rate(10.0, p_prev, as_decimal=False))
        delta_bp = (r10 - r10_prev) * 100
        snap_cols[1].metric("10y-Zins",
                            f"{r10:.2f} %",
                            f"{delta_bp:+.0f} bp · 1 Monat")
    else:
        snap_cols[1].metric("10y-Zins", f"{r10:.2f} %", delta_color="off")
else:
    snap_cols[1].metric("10y-Zins", "—", delta_color="off")

# 3. Brent
if brent is not None and "Close_USD" in brent.columns:
    last_brent = float(brent["Close_USD"].iloc[-1])
    if len(brent) > 22:
        prev_brent = float(brent["Close_USD"].iloc[-22])
        delta_pct = (last_brent / prev_brent - 1) * 100
        snap_cols[2].metric("Brent-Crude",
                            f"${last_brent:.1f}",
                            f"{delta_pct:+.1f} % · 1 Monat")
    else:
        snap_cols[2].metric("Brent-Crude", f"${last_brent:.1f}",
                            delta_color="off")
else:
    snap_cols[2].metric("Brent-Crude", "—", delta_color="off")

# 4. Banken im Cockpit
snap_cols[3].metric("Banken-Universe", "10",
                    "Top-IRB EU · Coverage 56 %",
                    delta_color="off")

# 5. Σ EAD
snap_cols[4].metric("Σ Kredit-EAD",
                    "€8.28 tn",
                    "EBA Transparency 2025",
                    delta_color="off")

# Cache-Caption
status_parts = []
if sven is not None:
    status_parts.append(f"svensson {len(sven):,} Tage")
if brent is not None:
    status_parts.append(f"brent {len(brent):,} Tage")
st.caption("Macro-Cache · " + " · ".join(status_parts) if status_parts
           else "Cache leer — bitte zuerst die Daten-Fetcher ausführen")


st.divider()


# ======================================================================
# Modell-Architektur in einer Tabelle (statt zwei langen Spalten)
# ======================================================================
eyebrow("Modell-Architektur · die Wirkungskette in einem Bild")

st.markdown(
    "Drei voneinander unabhängige Stress-Kanäle, alle gespeist aus den "
    "zwei Macro-Faktoren — am Ende konvergieren sie zur einen "
    "regulatorischen Headline: der CET1-Quote."
)

import pandas as pd
arch_df = pd.DataFrame([
    ("①",
     "Kreditbuch",
     "ΔBrent + Δr → ΔPD + ΔLGD (sektor-differenziert)",
     "Δ Expected Loss + Δ RWA · Capital-Bridge"),
    ("②",
     "Sovereign-Buch",
     "Δr × Modified Duration",
     "Δ Fair Value (IFRS-9: HfT / FVTPL / FVOCI / AC)"),
    ("③",
     "Handelsbuch",
     "FRTB-style Multiplier auf Market-RWA",
     "Δ RWA + P&L-Haircut"),
], columns=["#", "Channel", "Treiber", "CET1-Wirkung"])
st.dataframe(arch_df, hide_index=True, use_container_width=True,
             height=145)

st.markdown(
    '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
    'border-radius:6px;padding:0.75rem 1.0rem;margin:0.6rem 0 0 0;'
    'color:#051C2C;font-size:0.86rem;line-height:1.6;">'
    'Die IRB-Capital-Formel (Basel-III-IRB (Vasicek-Modell), Basel III) bleibt der '
    'regulatorische Standard für die RWA-Berechnung — aber sie wird '
    'mit den oben hergeleiteten gestressten PD- und LGD-Werten '
    'gefüttert, nicht über einen aggregierten Einzelfaktor. Vollständige '
    'Dokumentation in <code>MODEL_ASSUMPTIONS.md</code>.'
    '</div>',
    unsafe_allow_html=True,
)


st.divider()


# ======================================================================
# Navigation · 5 Tabs
# ======================================================================
eyebrow("Navigation · die fünf Tabs")

st.markdown(
    '<div style="font-size:0.86rem;color:#6E6E6E;line-height:1.6;'
    'margin:0.2rem 0 1.0rem 0;">'
    'Reihenfolge entspricht der fachlichen Wirkungskette: Faktor-'
    'Analyse → Kreditrisiko → Marktrisiko → Eigenkapital → Validierung.'
    '</div>',
    unsafe_allow_html=True,
)

nav_row1 = st.columns(3, gap="small")
with nav_row1[0]:
    st.page_link("pages/1_Faktor_Analyse.py",
                 label="1 · Faktor-Analyse & Transmission",
                 help="Empirische Faktor-Korrelation · R-Style-Regression · "
                      "sequentielle Stress-Bridge",
                 use_container_width=True)
with nav_row1[1]:
    st.page_link("pages/2_Kreditbuch.py",
                 label="2 · Kreditbuch",
                 help="Konkrete PD/LGD-Matrix pro Bank · Worked Example · "
                      "Capital-Bridge · Bank-Drilldown",
                 use_container_width=True)
with nav_row1[2]:
    st.page_link("pages/3_Marktbuch.py",
                 label="3 · Marktbuch",
                 help="Yield-Curve · Sovereigns mit IFRS-9-Split · "
                      "Banking-Book Bonds · Trading-Book",
                 use_container_width=True)

st.markdown(" ")
nav_row2 = st.columns(2, gap="small")
with nav_row2[0]:
    st.page_link("pages/4_Eigenkapital.py",
                 label="4 · Eigenkapital",
                 help="3-Channel CET1-Waterfall · Threshold-Analyse · "
                      "Sensitivitäts-Curve",
                 use_container_width=True)
with nav_row2[1]:
    st.page_link("pages/5_Validierung.py",
                 label="5 · Validierung",
                 help="Walk-Forward-Backtest · Annahmen · Methodologie",
                 use_container_width=True)


footer(
    "Single Source of Truth: MODEL_ASSUMPTIONS.md · "
    "Datenbasis: Pillar-3 EU-CR6 bank-spezifisch (31.12.2024) + ICE Brent + Bundesbank "
    "Svensson · Backend-Module in `Risk management/backend/`"
)

# Build-Marker — bestätigt, welche Version in der Cloud live ist.
st.caption("Build 2026-06-02 · derived-cache fallback aktiv")
