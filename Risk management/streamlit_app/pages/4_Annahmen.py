"""Annahmen & Datenbasis · Three-tier governance disclosure.

Implements Critique 4 of the supervising professor: "Datenbasis und
wesentliche Modellannahmen müssen adressatengerecht dokumentiert werden".

Audience-tiered structure (EBA GL 14 §10):
  Tier 1 · Executive Summary  → Board / Senior Management (plain text)
  Tier 2 · Approximations     → Validator / Internal Audit (table)
  Tier 3 · Datenbasis         → Quant / Operations (data lineage)

This page is intentionally read-only in V1; an override console for the
core constants (κ, Bucket-Durations, EBA-Anchor) is planned for V2.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd
import streamlit as st

from components.theme import (apply_theme, hero, eyebrow, insight, footer,
                              COLORS)
from components.sidebar import render_sidebar
from components.backend_path import setup
setup()

from config import (                                              # type: ignore
    KAPPA_DOWNTURN_LGD, VASICEK_CONFIDENCE,
    VASICEK_DEFAULT_LGD, VASICEK_DEFAULT_MATURITY_YEARS,
    EBA_VINTAGE_PRIMARY, MACRO_FACTOR_ROUTE,
    EBA_RAW_DIR,
)
from eba_loader import (                                           # type: ignore
    LGD_BY_VASICEK_CLASS, MATURITY_BY_VASICEK_CLASS,
    DURATION_BY_BUCKET, MATURITY_BUCKETS,
    EBA_2025_ADVERSE_ANCHOR, EBA_2023_ADVERSE_ANCHOR,
)


st.set_page_config(page_title="Annahmen & Datenbasis", layout="wide")
apply_theme()
render_sidebar()

hero(
    "Annahmen & Datenbasis",
    eyebrow="Governance · Modell-Transparenz · EBA GL 14",
    deck="Adressatengerecht dokumentierte Single-Source-of-Truth für alle "
         "wesentlichen Modellannahmen, Approximationen und Datenquellen. "
         "Drei Layer: Executive Summary für Board und Senior Management, "
         "Approximations-Inventar für Validator und Internal Audit, "
         "Datenbasis-Lineage für Quant und Operations.",
)

# =====================================================================
# Layer 1 · Executive Summary (Board / Senior Management)
# =====================================================================
eyebrow("Layer 1 · Executive Summary")

st.markdown("""
Das Modell quantifiziert die Wirkung makroökonomischer Schocks (Brent-Preis,
10-Jahres-Sovereign-Yield) auf die regulatorische Eigenkapitalanforderung
und das Mark-to-Market-Risiko europäischer Banken. Die folgenden acht
Bullet-Points fassen die zentralen Modell-Eigenschaften für Board und
Senior Management zusammen.
""")

bullets = [
    ("PD-Schätzung",
     "**PD wird aus beobachteten Default-Quoten der EBA Transparency 2025 "
     "abgeleitet** (Stichtag Juni 2025). Es handelt sich um eine "
     "Stock-Größe, nicht um eine Forward-Prognose. Der Vasicek-Modell-"
     "Pfad projiziert diese Quote über die Conditional-PD-Funktion in "
     "eine bedingte 1-Jahres-PD."),
    ("LGD-Annahme",
     "**LGD ist regulatorischer F-IRB-Default** (Basel-III-Standard: 45% "
     "Senior Unsecured / 20% Mortgage / 65% QRRE / 45% Other Retail), "
     "**nicht bank-spezifisch**. A-IRB-LGDs sind in der EBA-Public-"
     "Disclosure nicht enthalten und werden konservativ durch die "
     "F-IRB-Defaults ersetzt."),
    ("LGD-Stress-Funktion",
     f"Unter adversem Stress wird LGD um den Faktor "
     f"**(1 + κ · |M|)** gehoben mit κ = {KAPPA_DOWNTURN_LGD:.2f} "
     "(EBA-2023-Stresstest-konsistent). Bei M = −2.5 (EBA-2025-Adverse-"
     f"Anker) entspricht das einer LGD-Erhöhung um {KAPPA_DOWNTURN_LGD*2.5*100:.0f}% relativ zur "
     "Baseline."),
    ("EAD-Annahme",
     "**EAD bleibt statisch unter Stress** in V1. Drawdown-Risk auf "
     "Off-Balance-Linien und FX-Effekte sind nicht modelliert. Eine "
     "CCF-Stress-Funktion ist für V2 vorgesehen."),
    ("Macro-Mapping (Brent + Δr → M)",
     "**Hybrid-Übersetzung** des Schock-Vektors auf den Vasicek-System-"
     "Faktor M: (i) Anchor-Skalierung gegen den EBA-2025-Stresstest-"
     "Adverse-Anker (regulatorisch zitierbar), (ii) Mahalanobis-Distanz "
     "in der empirischen 252-Tage-Brent/Δr-Kovarianz (datengetrieben), "
     "(iii) Mittel der beiden als Primär-Route. Beide Routen werden im "
     "Cockpit transparent ausgewiesen."),
    ("Sovereign-Bewertung",
     "**Mark-to-Market unter Parallel-Shift** der Yield-Kurve via "
     "Modified-Duration-Approximation (Bucket-Midpoint, kein Cashflow-"
     "Modell). Credit-Spread-Risiko und Hedging sind nicht abgebildet — "
     "Limitation in der Sovereign-Page transparent erklärt."),
    ("Backtesting-Status",
     "**EBA-Transparency-Vintages 2020-2024 lokal verfügbar** "
     "(~1.06 GB), Vintage-aware-Loader und KPI-Suite (Brier-Score, "
     "MAE-ΔPD, Stress-Episode-Vergleich) sind in Phase 5 in "
     "Vorbereitung. Ein Netto-EAD-Decomposition-Schritt (Volume × "
     "Mix × Stress-Residual) bereinigt um Portfolio-Wachstum und "
     "Class-Migration; FX bleibt limitiert (EBA aggregiert in EUR)."),
    ("Verwendungs-Scope",
     "**ICAAP-Validierungs-Use-Case** und Lehr-/Demo-Zwecke. "
     "Die Modell-Outputs sind regulatorische Risiko-Indikatoren — "
     "**keine** Investment- oder Geschäftsentscheidungs-Empfehlungen."),
]

for title, body in bullets:
    st.markdown(f"- **{title}.** {body}")

st.divider()

# =====================================================================
# Layer 2 · Approximations Inventory (Validator)
# =====================================================================
eyebrow("Layer 2 · Approximations-Inventar")

st.markdown(
    "Vollständige Liste der Approximationen, statistischen Schätzungen und "
    "hardcoded Annahmen im Modell. Jede Position trägt eine **Konfidenz-"
    "Klasse** und einen Verweis auf das aktivierende Modul. Die Tabelle "
    "ist die Single-Source-of-Truth für Validation Function (Second Line) "
    "und Internal Audit."
)

approx_inventory = pd.DataFrame([
    {
        "ID": "A-01",
        "Approximation": "PD = beobachteter Default-Ratio",
        "Modul": "eba_loader.aggregate_to_vasicek_segments",
        "Konfidenz": "approximation",
        "Auswirkung": "mittel",
        "Begründung": "Backward-looking; Stress-Sensitivität bleibt korrekt",
        "V2-Alternative": "Forward-PD via PiT-Inversion gegen Konjunktur",
    },
    {
        "ID": "A-02",
        "Approximation": "F-IRB-LGD ersetzt A-IRB",
        "Modul": "eba_loader.LGD_BY_VASICEK_CLASS",
        "Konfidenz": "assumption",
        "Auswirkung": "hoch",
        "Begründung": "A-IRB-LGD nicht in EBA-Public-Disclosure",
        "V2-Alternative": "nicht behebbar ohne bank-interne Daten",
    },
    {
        "ID": "A-03",
        "Approximation": "Downturn-LGD κ = 0.30",
        "Modul": "vasicek.downturn_lgd · config.KAPPA_DOWNTURN_LGD",
        "Konfidenz": "assumption",
        "Auswirkung": "hoch",
        "Begründung": "EBA-2023-Stresstest-Corporate-konsistent",
        "V2-Alternative": "sektor-spezifisches κ je Vasicek-Class",
    },
    {
        "ID": "A-04",
        "Approximation": "EAD konstant unter Stress",
        "Modul": "BankPortfolio.stressed_metrics",
        "Konfidenz": "approximation",
        "Auswirkung": "hoch",
        "Begründung": "CCF-Drawdown nicht modelliert",
        "V2-Alternative": "stress-elastische CCF-Funktion",
    },
    {
        "ID": "A-05",
        "Approximation": "Bucket-Midpoint = Duration",
        "Modul": "eba_loader.DURATION_BY_BUCKET",
        "Konfidenz": "approximation",
        "Auswirkung": "mittel",
        "Begründung": "Bullet-Bond-at-par-Annahme; ±10–15% Abweichung",
        "V2-Alternative": "Cashflow-Modell pro Coupon-Bond",
    },
    {
        "ID": "A-06",
        "Approximation": "Single-Factor Vasicek (eines M)",
        "Modul": "macro_factor",
        "Konfidenz": "approximation",
        "Auswirkung": "niedrig",
        "Begründung": "Standardansatz Basel-III-IRB",
        "V2-Alternative": "Multi-Factor CreditRisk+ (V3)",
    },
    {
        "ID": "A-07",
        "Approximation": "EBA-Anker als Single-Point",
        "Modul": "eba_loader.EBA_2025_ADVERSE_ANCHOR",
        "Konfidenz": "estimate",
        "Auswirkung": "niedrig",
        "Begründung": "Hybrid mit Mahalanobis-Route validiert",
        "V2-Alternative": "Multi-Anker (2023 + 2025 + 2021)",
    },
    {
        "ID": "A-08",
        "Approximation": "Parallel-Shift Yield-Kurve",
        "Modul": "eba_loader.rate_shock_pnl",
        "Konfidenz": "approximation",
        "Auswirkung": "niedrig",
        "Begründung": "Δβ₁/β₂/β₃-Slider treiben heute nur Tier 1",
        "V2-Alternative": "Bucket-spezifische Yield-Shifts (Slope/Curv.)",
    },
    {
        "ID": "A-09",
        "Approximation": "Empirische Σ stationär (252d)",
        "Modul": "macro_factor.factor_stats",
        "Konfidenz": "estimate",
        "Auswirkung": "niedrig",
        "Begründung": "Rolling 252-Tage-Window",
        "V2-Alternative": "DCC-GARCH oder Regime-Switching",
    },
    {
        "ID": "A-10",
        "Approximation": "Hedging nicht modelliert",
        "Modul": "Sovereign + Loan Book",
        "Konfidenz": "assumption",
        "Auswirkung": "hoch",
        "Begründung": "Bank-Swaps/Futures aus EBA-Public-Daten nicht rekonstruierbar",
        "V2-Alternative": "nicht behebbar ohne bank-interne Daten",
    },
])

# Color-code the Auswirkung column
def _impact_emoji(level: str) -> str:
    return {"hoch": "● hoch", "mittel": "◐ mittel", "niedrig": "○ niedrig"}.get(
        level, level
    )

approx_disp = approx_inventory.copy()
approx_disp["Auswirkung"] = approx_disp["Auswirkung"].map(_impact_emoji)
st.dataframe(approx_disp, use_container_width=True, hide_index=True,
             height=420)

# Confidence classes legend
st.caption(
    "**Konfidenz-Klassen:** "
    "`published` = veröffentlichte Messung, "
    "`estimate` = statistische Schätzung mit empirischer Datenbasis, "
    "`approximation` = strukturelle Vereinfachung mit dokumentierter Methode, "
    "`assumption` = hardcoded Annahme ohne empirische Kalibrierung."
)

st.divider()

# =====================================================================
# Layer 3 · Datenbasis & Lineage (Quant / Operations)
# =====================================================================
eyebrow("Layer 3 · Datenbasis · Lineage")

st.markdown(
    "Vollständige Liste der primären Datenquellen mit Stichtag, Frequenz, "
    "und Aggregations-Granularität. Quant-Owner kann jederzeit die "
    "Roh-Datei pro Quelle bis zur ursprünglichen EBA / Bundesbank-"
    "Veröffentlichung zurückverfolgen."
)

datenbasis = pd.DataFrame([
    {"Quelle": "EBA Transparency 2025",
     "File": "tr_cre.csv", "Stichtag": "Juni 2025",
     "Inhalt": "IRB-Loan-Book pro Bank × Class × Country × Status",
     "Größe": "~117 MB", "Frequenz": "jährlich"},
    {"Quelle": "EBA Transparency 2025",
     "File": "tr_sov.csv", "Stichtag": "Juni 2025",
     "Inhalt": "Sovereign-Exposures pro Bank × Country × Maturity",
     "Größe": "~87 MB", "Frequenz": "jährlich"},
    {"Quelle": "EBA Transparency 2025",
     "File": "tr_oth.csv", "Stichtag": "Juni 2025",
     "Inhalt": "Capital, RWA OV1, P&L, Leverage, Assets",
     "Größe": "~14 MB", "Frequenz": "jährlich"},
    {"Quelle": "EBA Transparency 2025",
     "File": "tr_mrk.csv", "Stichtag": "Juni 2025",
     "Inhalt": "Market Risk RWA + VaR/SVaR pro Bank",
     "Größe": "~3.6 MB", "Frequenz": "jährlich"},
    {"Quelle": "EBA Transparency 2025",
     "File": "TR_Metadata.xlsx", "Stichtag": "—",
     "Inhalt": "LEI ↔ Name, Country / Exposure / Status / Maturity Dim-Codes",
     "Größe": "~2.7 MB", "Frequenz": "—"},
    {"Quelle": "EBA Transparency 2025",
     "File": "SDD.xlsx", "Stichtag": "—",
     "Inhalt": "Single Data Dictionary inkl. Item-Translation 2014–2025",
     "Größe": "~55 KB", "Frequenz": "—"},
    {"Quelle": "EBA Transparency 2020–2024",
     "File": "transparency_YYYY/*.csv", "Stichtag": "Q4 2019 – Q2 2024",
     "Inhalt": "Historische Vintages für Backtesting (Phase 5)",
     "Größe": "~1.06 GB", "Frequenz": "jährlich"},
    {"Quelle": "EBA Stress Test 2025",
     "File": "Methodology Note (PDF)", "Stichtag": "Aug 2025",
     "Inhalt": "Adverse-Scenario-Anker (Brent +0.47, Δr +200 bp, M = −2.5)",
     "Größe": "—", "Frequenz": "bi-annual"},
    {"Quelle": "Deutsche Bundesbank",
     "File": "bundesbank_svensson.csv", "Stichtag": "tagesaktuell",
     "Inhalt": "Svensson-Parameter β₀–β₃, τ₁, τ₂",
     "Größe": "~212 KB", "Frequenz": "täglich"},
    {"Quelle": "ICE / yfinance Cache",
     "File": "brent_crude.parquet", "Stichtag": "tagesaktuell",
     "Inhalt": "Brent-Schluss + log-Returns (BZ=F)",
     "Größe": "(cached)", "Frequenz": "täglich"},
])

st.dataframe(datenbasis, use_container_width=True, hide_index=True,
             height=380)

# Constants reference card
st.divider()
eyebrow("Layer 3b · Konstanten-Referenz")

const_l, const_r = st.columns(2, gap="medium")

with const_l:
    st.markdown("**Vasicek / IRB**")
    const_basel = pd.DataFrame([
        ("Confidence α (ASRF)", f"{VASICEK_CONFIDENCE:.4f}"),
        ("Default LGD (override per segment)", f"{VASICEK_DEFAULT_LGD:.2f}"),
        ("Default Maturity (Corporate, M_eff)", f"{VASICEK_DEFAULT_MATURITY_YEARS:.1f} Jahre"),
        ("Downturn-LGD κ", f"{KAPPA_DOWNTURN_LGD:.2f}"),
        ("EBA-Vintage (Primary)", EBA_VINTAGE_PRIMARY),
        ("Macro→M Route", MACRO_FACTOR_ROUTE),
    ], columns=["Parameter", "Wert"])
    st.dataframe(const_basel, use_container_width=True, hide_index=True,
                 height=280)

    st.markdown("**LGD per Vasicek-Class**")
    const_lgd = pd.DataFrame(
        [(k, f"{v*100:.0f}%") for k, v in LGD_BY_VASICEK_CLASS.items()],
        columns=["Vasicek-Class", "F-IRB LGD"],
    )
    st.dataframe(const_lgd, use_container_width=True, hide_index=True,
                 height=280)

with const_r:
    st.markdown("**Bucket-Duration · Sovereign**")
    bucket_table = pd.DataFrame([
        (MATURITY_BUCKETS[k], f"{v:.3f} Jahre")
        for k, v in DURATION_BY_BUCKET.items()
    ], columns=["Bucket", "Modified Duration"])
    st.dataframe(bucket_table, use_container_width=True, hide_index=True,
                 height=280)

    st.markdown("**EBA-Stress-Test Adverse-Anchor**")
    anchor_2025 = EBA_2025_ADVERSE_ANCHOR
    anchor_2023 = EBA_2023_ADVERSE_ANCHOR
    anchor_table = pd.DataFrame([
        ("Vintage", str(anchor_2025["vintage"]),
         str(anchor_2023["vintage"])),
        ("Publication", anchor_2025["publication_date"],
         anchor_2023["publication_date"]),
        ("ΔBrent log-shock", f"{anchor_2025['brent_log_shock']:+.2f}",
         f"{anchor_2023['brent_log_shock']:+.2f}"),
        ("Δr_10y shock [pp]", f"{anchor_2025['rate_10y_pp_shock']:+.2f}",
         f"{anchor_2023['rate_10y_pp_shock']:+.2f}"),
        ("GDP-shock [pp]", f"{anchor_2025['gdp_pp_shock']:+.1f}",
         f"{anchor_2023['gdp_pp_shock']:+.1f}"),
        ("Implied M (z)", f"{anchor_2025['z_factor_implied']:+.2f}",
         f"{anchor_2023['z_factor_implied']:+.2f}"),
        ("Status", anchor_2025["status"][:30] + "…",
         anchor_2023["status"][:30] + "…"),
    ], columns=["Field", "EBA 2025", "EBA 2023"])
    st.dataframe(anchor_table, use_container_width=True, hide_index=True,
                 height=280)

# =====================================================================
# Limitations · what the model does NOT cover
# =====================================================================
st.divider()
eyebrow("Limitationen · was das Modell NICHT abdeckt")

insight(
    "<strong>Bewusste Out-of-Scope-Bereiche.</strong> Das Modell ist eine "
    "gezielte Vereinfachung. Folgende Risiko-Dimensionen werden nicht "
    "abgebildet und müssen bei der Interpretation der Outputs "
    "berücksichtigt werden:"
)

st.markdown("""
- **Operational Risk** wird konstant unter Stress angenommen (Item 2520215)
- **CVA / Counterparty Credit Risk** ist nicht im Stress-Szenario
- **Sovereign-Spread-Risk** (Italien-vs-Bund) ist out of scope
- **Concentration Risk** (HHI, Granularitäts-Add-on) wird nur als KPI gezeigt, nicht stress-gewichtet
- **Liquidity-Risk / LCR / NSFR** sind nicht modelliert
- **IFRS-9 Lifetime-EL** (Stage-2-Migration) ist out of scope — nur 1Y-Forward
- **Behavioral CCF unter Tail-Stress** literatur-basiert, nicht datenbasiert
- **Multi-Period-Stress-Pfade** (3-Jahres-EBA-Stress-Test-Logik) nicht modelliert
- **Hedging-Effekte** (Swaps, Futures, CDS) aus EBA-Public-Disclosure nicht rekonstruierbar
""")

footer(
    "Diese Page ist die single-source-of-truth für alle wesentlichen "
    "Modellannahmen. Override-Konsole für die zentralen Konstanten "
    "(κ, Bucket-Durations, EBA-Anchor) ist V2-Roadmap. Cross-Verweise "
    "zur quantitativen Doku in MODEL_ASSUMPTIONS.md."
)
