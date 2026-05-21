"""Legacy page-views as callable render-functions.

Three content blocks that used to be standalone pages are now exposed as
functions that can be embedded as sub-tabs inside the new 5-tab structure:

  - render_yield_curve_tab(config) — Bundesbank Svensson curve + β-shifts
                                       (lebt jetzt als Sub-Tab in Marktbuch)
  - render_annahmen_tab()          — 3-layer governance disclosure
                                       (Sub-Tab in Validierung)
  - render_methodology_tab()       — MODEL_ASSUMPTIONS.md renderer
                                       (Sub-Tab in Validierung)

Voraussetzung: aufrufender Code muss bereits apply_theme() + render_sidebar()
ausgeführt haben.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import (eyebrow, insight, COLORS)
from components.backend_path import setup
setup()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# =====================================================================
# 1. Yield Curve sub-tab
# =====================================================================
def render_yield_curve_tab(config: dict) -> None:
    from components.data_loader import baseline_yield_curve, load_data_layer
    from svensson import (historical_curve, shift_curve, curve_grid,        # type: ignore
                          zero_rate, params_from_row)

    base = baseline_yield_curve()
    if base.empty:
        st.error("Svensson cache missing — bitte `python 04_fetch_bundesbank_svensson.py` ausführen.")
        return

    data = load_data_layer()
    sven_df = data["svensson"]
    brent_df = data["brent"]

    base_params = historical_curve(sven_df.index[-1], sven_df)
    shifted = shift_curve(
        base_params,
        dlevel=config["d_b0"], dslope=config["d_b1"],
        dcurv1=config["d_b2"], dcurv2=config["d_b3"],
    )
    mats = np.arange(0.25, 30.25, 0.25)
    stressed = curve_grid(shifted, maturities=mats)

    st.markdown(
        '<div style="background:#F4F4F4;border-left:4px solid #034B6F;'
        'padding:0.7rem 1.0rem;border-radius:6px;margin-bottom:1rem;'
        'color:#051C2C;font-size:0.88rem;line-height:1.55;">'
        '<strong>Yield-Curve als Modell-Input.</strong> Bundesbank Svensson '
        'zero-coupon curve. Treibt drei Channels: (i) Δr_10y im Macro-Schock '
        'für Vasicek-M-Mapping, (ii) Modified-Duration-MtM auf Sovereign-Bonds, '
        '(iii) implizite Refinanzierungskosten der Banken (V2). Im '
        'Single-Factor-M-Modus wird die Δβ₀-Verschiebung aus M abgeleitet — '
        'für freie β-Slider in den Multi-Factor-Modus wechseln.'
        '</div>',
        unsafe_allow_html=True,
    )

    eyebrow("Key-Maturity-Raten · Baseline vs. Stress")
    key_mats = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    kpi_cols = st.columns(len(key_mats), gap="small")
    for col, m in zip(kpi_cols, key_mats):
        r_base = float(zero_rate(m, base_params, as_decimal=False))
        r_str  = float(zero_rate(m, shifted, as_decimal=False))
        delta_bp = (r_str - r_base) * 100
        label = f"{int(m*12)}m" if m < 1 else f"{int(m)}y"
        if abs(delta_bp) > 1e-2:
            col.metric(f"{label} rate", f"{r_str:.3f}%",
                       f"{delta_bp:+.0f} bp")
        else:
            col.metric(f"{label} rate", f"{r_base:.3f}%",
                       "baseline", delta_color="off")

    st.caption(
        f"Bundesbank Svensson zero-coupon curve · letzter Update "
        f"**{sven_df.index[-1].date()}** · Lookback "
        f"{len(sven_df):,} Handelstage "
        f"({sven_df.index.min().date()} → {sven_df.index.max().date()})"
    )
    st.divider()

    eyebrow("Curve-Topologie · Baseline / Stressed / Historical Reference")
    ref_dates = []
    today = sven_df.index[-1]
    for months_back, label in [(12, "12m ago"), (36, "3y ago")]:
        target = today - pd.DateOffset(months=months_back)
        nearest_idx = sven_df.index.get_indexer([target], method="nearest")[0]
        if nearest_idx >= 0 and nearest_idx < len(sven_df):
            ref_dates.append((sven_df.index[nearest_idx], label))

    fig = go.Figure()
    hist_colors = [COLORS["stone"], "#B0B0B0"]
    for (d, lbl), c in zip(ref_dates, hist_colors):
        p = historical_curve(d, sven_df, method="ffill")
        rates = curve_grid(p, maturities=mats)
        fig.add_trace(go.Scatter(
            x=rates.index, y=rates.values,
            name=f"{lbl} ({d.date()})",
            line=dict(color=c, width=1.2, dash="dot"),
            opacity=0.7,
        ))
    fig.add_trace(go.Scatter(
        x=base["maturity"], y=base["rate_pct"],
        name=f"Baseline ({sven_df.index[-1].date()})",
        line=dict(color=COLORS["navy"], width=2.8),
    ))
    shifted_at_zero = (config["d_b0"] == 0 and config["d_b1"] == 0
                       and config["d_b2"] == 0 and config["d_b3"] == 0)
    if not shifted_at_zero:
        fig.add_trace(go.Scatter(
            x=stressed.index, y=stressed.values,
            name="Stressed (live β)",
            line=dict(color=COLORS["crimson"], width=2.5, dash="dash"),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["hairline"], line_width=1)
    fig.update_layout(
        title=None,
        xaxis_title="Maturity [years]",
        yaxis_title="Zero rate [%]",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Svensson parameters table
    st.divider()
    eyebrow("Svensson-Parameter · Baseline vs. Stressed")
    params_table = pd.DataFrame({
        "Parameter": ["β₀ (level)", "β₁ (slope)", "β₂ (curvature 1)",
                      "β₃ (curvature 2)", "τ₁", "τ₂"],
        "Baseline": [base_params.beta0, base_params.beta1,
                     base_params.beta2, base_params.beta3,
                     base_params.tau1, base_params.tau2],
        "Stressed": [shifted.beta0, shifted.beta1, shifted.beta2,
                     shifted.beta3, shifted.tau1, shifted.tau2],
        "Δ":        [shifted.beta0 - base_params.beta0,
                     shifted.beta1 - base_params.beta1,
                     shifted.beta2 - base_params.beta2,
                     shifted.beta3 - base_params.beta3,
                     0.0, 0.0],
    })
    for c in ("Baseline", "Stressed", "Δ"):
        params_table[c] = params_table[c].round(4)
    st.dataframe(params_table, use_container_width=True, hide_index=True)


# =====================================================================
# 2. Annahmen sub-tab (3-Layer Governance)
# =====================================================================
def render_annahmen_tab() -> None:
    from config import (                                                    # type: ignore
        KAPPA_DOWNTURN_LGD, VASICEK_CONFIDENCE,
        VASICEK_DEFAULT_LGD, VASICEK_DEFAULT_MATURITY_YEARS,
        EBA_VINTAGE_PRIMARY, MACRO_FACTOR_ROUTE,
    )
    from eba_loader import (                                                # type: ignore
        LGD_BY_VASICEK_CLASS, DURATION_BY_BUCKET, MATURITY_BUCKETS,
        EBA_2025_ADVERSE_ANCHOR, EBA_2023_ADVERSE_ANCHOR,
    )

    # ---- Layer 1: Executive Summary ----
    eyebrow("Layer 1 · Executive Summary (Board / Senior Management)")
    st.markdown("""
Das Modell quantifiziert die Wirkung makroökonomischer Schocks (Brent-Preis,
10-Jahres-Sovereign-Yield) auf die regulatorische Eigenkapitalanforderung
und das Mark-to-Market-Risiko europäischer Banken.
""")
    bullets = [
        ("PD-Schätzung",
         "PD wird aus beobachteten Default-Quoten der EBA Transparency 2025 "
         "abgeleitet (Stichtag Juni 2025). Stock-Größe, nicht Forward-Prognose. "
         "Vasicek projiziert sie via Conditional-PD in eine bedingte 1-Jahres-PD."),
        ("LGD-Annahme",
         "F-IRB-Default (Basel-III: 45% Senior / 20% Mortgage / 65% QRRE / "
         "45% Other Retail). A-IRB-LGDs sind in der EBA-Public-Disclosure "
         "nicht enthalten — F-IRB als konservativer Proxy."),
        ("LGD-Stress",
         f"Downturn-LGD = LGD_base · (1 + κ · max(−M, 0)) mit κ = "
         f"{KAPPA_DOWNTURN_LGD:.2f} (EBA-2023-Konvention). Bei M = −2.5: "
         f"+{KAPPA_DOWNTURN_LGD*2.5*100:.0f}% LGD-Uplift."),
        ("EAD",
         "Statisch unter Stress in V1. Drawdown-Risk via CCF + FX-Effekte "
         "nicht modelliert. V2: stress-elastische CCF-Funktion."),
        ("Macro → M Mapping",
         "Single-Factor-Modus: M wird direkt vom User gesetzt; ΔBrent + "
         "Δr_10y aus EBA-Adverse-Anker-Direction abgeleitet. Multi-Factor-"
         "Modus: hybrid (Anchor + Mahalanobis)."),
        ("Sovereign Bonds",
         "Modified-Duration-MtM via Bucket-Midpoint, Parallel-Shift-Annahme. "
         "Credit-Spread + Hedging nicht abgebildet."),
        ("Backtesting",
         "22 Quartals-Stichtage Sep 2019 – Jun 2025, ~2 600 Bank-Quartal-"
         "Observationen, OLS-β + Episode-Diagnostik."),
        ("Verwendungs-Scope",
         "ICAAP-Validierungs-Use-Case und Lehr-/Demo-Zwecke. Keine Investment-"
         "Empfehlungen."),
    ]
    for title, body in bullets:
        st.markdown(f"- **{title}.** {body}")

    st.divider()

    # ---- Layer 2: Approximations Inventory ----
    eyebrow("Layer 2 · Approximations-Inventar (Validator / Internal Audit)")
    st.markdown(
        "Vollständige Liste aller Approximationen, statistischen Schätzungen "
        "und hardcoded Annahmen. Single-Source-of-Truth für Validation und Audit."
    )
    approx = pd.DataFrame([
        {"ID":"A-01","Approximation":"PD = beobachteter Default-Ratio",
         "Modul":"eba_loader","Konfidenz":"approximation","Auswirkung":"● hoch",
         "Begründung":"Backward-looking, Stress-Sensitivität korrekt",
         "V2-Alternative":"Forward-PD via PiT-Inversion"},
        {"ID":"A-02","Approximation":"F-IRB-LGD ersetzt A-IRB",
         "Modul":"eba_loader","Konfidenz":"assumption","Auswirkung":"● hoch",
         "Begründung":"A-IRB-LGD nicht in EBA-Disclosure",
         "V2-Alternative":"nicht behebbar ohne bank-interne Daten"},
        {"ID":"A-03","Approximation":f"Downturn-LGD κ = {KAPPA_DOWNTURN_LGD:.2f}",
         "Modul":"vasicek.downturn_lgd","Konfidenz":"assumption","Auswirkung":"● hoch",
         "Begründung":"EBA-2023-Stresstest-konsistent",
         "V2-Alternative":"Sektor-spezifisches κ je Vasicek-Class"},
        {"ID":"A-04","Approximation":"EAD konstant unter Stress",
         "Modul":"BankPortfolio.stressed_metrics","Konfidenz":"approximation",
         "Auswirkung":"● hoch","Begründung":"CCF-Drawdown nicht modelliert",
         "V2-Alternative":"Stress-elastische CCF-Funktion"},
        {"ID":"A-05","Approximation":"Bucket-Midpoint = Duration",
         "Modul":"eba_loader.DURATION_BY_BUCKET","Konfidenz":"approximation",
         "Auswirkung":"◐ mittel","Begründung":"Bullet-Bond-at-par, ±10-15% Drift",
         "V2-Alternative":"Cashflow-Modell pro Coupon-Bond"},
        {"ID":"A-06","Approximation":"Single-Factor Vasicek (ein M)",
         "Modul":"vasicek + macro_factor","Konfidenz":"approximation",
         "Auswirkung":"○ niedrig","Begründung":"Basel-III-IRB-Standard",
         "V2-Alternative":"Multi-Factor CreditRisk+ (V3)"},
        {"ID":"A-07","Approximation":"EBA-Anker als Single-Point",
         "Modul":"macro_factor","Konfidenz":"estimate","Auswirkung":"○ niedrig",
         "Begründung":"Hybrid mit Mahalanobis-Route validiert",
         "V2-Alternative":"Multi-Anker (2023 + 2025 + 2021)"},
        {"ID":"A-08","Approximation":"Parallel-Shift Yield-Kurve",
         "Modul":"eba_loader.rate_shock_pnl","Konfidenz":"approximation",
         "Auswirkung":"○ niedrig","Begründung":"β₁/β₂/β₃-Slider nur Tier 1",
         "V2-Alternative":"Bucket-spezifische Yield-Shifts"},
        {"ID":"A-09","Approximation":"Empirische Σ stationär (252d)",
         "Modul":"macro_factor.factor_stats","Konfidenz":"estimate",
         "Auswirkung":"○ niedrig","Begründung":"Rolling-Window",
         "V2-Alternative":"DCC-GARCH / Regime-Switching"},
        {"ID":"A-10","Approximation":"Hedging nicht modelliert",
         "Modul":"Sovereign + Loan-Book","Konfidenz":"assumption","Auswirkung":"● hoch",
         "Begründung":"Bank-Swaps/Futures nicht in EBA-Public-Disclosure",
         "V2-Alternative":"nicht behebbar ohne bank-interne Daten"},
        {"ID":"A-11","Approximation":"ρ-Multiplikator (User-Override)",
         "Modul":"vasicek.irb_capital_requirement","Konfidenz":"assumption",
         "Auswirkung":"● hoch","Begründung":"Modell-Risiko-Parameter, kein Basel-Standard",
         "V2-Alternative":"Bank-/Class-spezifische ρ-Kalibrierung"},
    ])
    st.dataframe(approx, use_container_width=True, hide_index=True, height=460)
    st.caption(
        "**Konfidenz-Klassen:** `published` = veröffentlichte Messung · "
        "`estimate` = statistische Schätzung · `approximation` = strukturelle "
        "Vereinfachung · `assumption` = hardcoded Annahme."
    )

    st.divider()

    # ---- Layer 3: Datenbasis & Lineage ----
    eyebrow("Layer 3 · Datenbasis · Lineage (Quant / Operations)")
    datenbasis = pd.DataFrame([
        ("EBA Transparency 2025","tr_cre.csv","Juni 2025",
         "IRB-Loan-Book pro Bank × Class × Country","~117 MB","jährlich"),
        ("EBA Transparency 2025","tr_sov.csv","Juni 2025",
         "Sovereign-Exposures pro Bank × Country × Maturity","~87 MB","jährlich"),
        ("EBA Transparency 2025","tr_oth.csv","Juni 2025",
         "Capital, RWA OV1, P&L, Leverage","~14 MB","jährlich"),
        ("EBA Transparency 2025","tr_mrk.csv","Juni 2025",
         "Market-Risk-RWA + VaR/SVaR","~3.6 MB","jährlich"),
        ("EBA Transparency 2025","TR_Metadata.xlsx","—",
         "LEI ↔ Name, Country/Class/Status Dim-Codes","~2.7 MB","—"),
        ("EBA Transparency 2025","SDD.xlsx","—",
         "Single Data Dictionary, Item-Translation 2014-2025","~55 KB","—"),
        ("EBA Transparency 2020-2024","transparency_YYYY/*.csv","Q4 2019 – Q2 2024",
         "Historische Vintages für Backtesting","~1.06 GB","jährlich"),
        ("EBA Stress Test 2025","Methodology Note","Aug 2025",
         "Adverse-Anker (Brent +0.47, Δr +200bp, M = −2.5)","—","bi-annual"),
        ("Deutsche Bundesbank","bundesbank_svensson.csv","tagesaktuell",
         "Svensson-Parameter β₀-β₃, τ₁, τ₂","~212 KB","täglich"),
        ("ICE / yfinance","brent_crude.parquet","tagesaktuell",
         "Brent-Schluss + log-Returns","(cached)","täglich"),
    ], columns=["Quelle","File","Stichtag","Inhalt","Größe","Frequenz"])
    st.dataframe(datenbasis, use_container_width=True, hide_index=True, height=380)

    st.divider()
    eyebrow("Konstanten-Referenz")
    c_l, c_r = st.columns(2, gap="medium")
    with c_l:
        st.markdown("**Vasicek / IRB**")
        st.dataframe(pd.DataFrame([
            ("Confidence α (ASRF)", f"{VASICEK_CONFIDENCE:.4f}"),
            ("Default LGD (override)", f"{VASICEK_DEFAULT_LGD:.2f}"),
            ("Default Maturity", f"{VASICEK_DEFAULT_MATURITY_YEARS:.1f} Jahre"),
            ("Downturn-LGD κ", f"{KAPPA_DOWNTURN_LGD:.2f}"),
            ("EBA-Vintage (Primary)", EBA_VINTAGE_PRIMARY),
            ("Macro→M Route", MACRO_FACTOR_ROUTE),
        ], columns=["Parameter","Wert"]),
            use_container_width=True, hide_index=True, height=240)

        st.markdown("**LGD per Vasicek-Class**")
        st.dataframe(pd.DataFrame(
            [(k, f"{v*100:.0f}%") for k, v in LGD_BY_VASICEK_CLASS.items()],
            columns=["Vasicek-Class","F-IRB LGD"]),
            use_container_width=True, hide_index=True, height=240)

    with c_r:
        st.markdown("**Bucket-Duration (Sovereign)**")
        st.dataframe(pd.DataFrame([
            (MATURITY_BUCKETS[k], f"{v:.3f} Jahre")
            for k, v in DURATION_BY_BUCKET.items()
        ], columns=["Bucket","Modified Duration"]),
            use_container_width=True, hide_index=True, height=240)

        st.markdown("**EBA-Adverse-Anchors**")
        a25, a23 = EBA_2025_ADVERSE_ANCHOR, EBA_2023_ADVERSE_ANCHOR
        st.dataframe(pd.DataFrame([
            ("ΔBrent log-shock", f"{a25['brent_log_shock']:+.2f}",
             f"{a23['brent_log_shock']:+.2f}"),
            ("Δr_10y shock [pp]", f"{a25['rate_10y_pp_shock']:+.2f}",
             f"{a23['rate_10y_pp_shock']:+.2f}"),
            ("GDP-shock [pp]", f"{a25['gdp_pp_shock']:+.1f}",
             f"{a23['gdp_pp_shock']:+.1f}"),
            ("Implied M (z)", f"{a25['z_factor_implied']:+.2f}",
             f"{a23['z_factor_implied']:+.2f}"),
        ], columns=["Field","EBA 2025","EBA 2023"]),
            use_container_width=True, hide_index=True, height=240)

    st.divider()
    eyebrow("Out-of-Scope · was das Modell NICHT abdeckt")
    insight(
        "Bewusste Vereinfachungen. Folgende Risiko-Dimensionen sind nicht "
        "modelliert und müssen bei der Interpretation berücksichtigt werden:"
    )
    st.markdown("""
- **Operational Risk** konstant unter Stress
- **CVA / Counterparty Credit Risk** nicht im Stress-Szenario
- **Sovereign-Spread-Risk** (Italien-vs-Bund) out of scope
- **Concentration Risk** (HHI) nur als KPI, nicht stress-gewichtet
- **Liquidity-Risk / LCR / NSFR** nicht modelliert
- **IFRS-9 Lifetime-EL** (Stage-2-Migration) out of scope — nur 1Y-Forward
- **Multi-Period-Stress-Pfade** (3-Jahres-EBA-Logik) nicht modelliert
- **Hedging-Effekte** (Swaps, Futures, CDS) nicht rekonstruierbar
""")


# =====================================================================
# 3. Methodology sub-tab (renders MODEL_ASSUMPTIONS.md)
# =====================================================================
def render_methodology_tab() -> None:
    md_path = _PROJECT_ROOT / "MODEL_ASSUMPTIONS.md"
    if not md_path.exists():
        st.error(f"MODEL_ASSUMPTIONS.md nicht gefunden unter {md_path}")
        return
    with open(md_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    st.markdown(content)
    st.caption("Live aus `MODEL_ASSUMPTIONS.md` gerendert — Single-Source-of-Truth.")
