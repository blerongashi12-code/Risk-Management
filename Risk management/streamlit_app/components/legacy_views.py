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

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #00A9A5;padding:0.8rem 1.1rem;border-radius:6px;'
        'margin:0.4rem 0 1rem 0;color:#051C2C;font-size:0.88rem;line-height:1.6;">'
        '<strong>Was zeigt diese Grafik?</strong> Die Zero-Coupon-Zinskurve der '
        'Eurozone (Bundesbank-Svensson-Schätzung) — heutiger Stand (navy) im '
        'Vergleich zu zwei historischen Referenzpunkten (12&nbsp;Monate und '
        '3&nbsp;Jahre zurück). Die x-Achse ist die Restlaufzeit, die y-Achse '
        'der annualisierte Zero-Coupon-Zins.<br><br>'
        '<strong>Warum relevant für das Stress-Modell?</strong> Ein '
        '<em>höherer</em> 10y-Zins entspricht im Modell einem '
        '<em>Δr<sub>10y</sub>&nbsp;&gt;&nbsp;0</em>-Schock — '
        'dies (a)&nbsp;verschiebt den systematischen Faktor M nach unten '
        '(Recessions-Signal → höhere bedingte PDs), '
        '(b)&nbsp;löst eine Modified-Duration-MtM-Verlustkomponente auf das '
        'Sovereign-Bond-Portfolio aus und (c)&nbsp;verteuert die '
        'Refinanzierungskosten der Banken. Die rote gestrichelte Linie zeigt — '
        'falls aktiv — die unter den Live-β-Reglern resultierende '
        'Stress-Kurve.'
        '</div>',
        unsafe_allow_html=True,
    )

    ref_dates = []
    today = sven_df.index[-1]
    for months_back, label in [(12, "12m ago"), (36, "3y ago")]:
        target = today - pd.DateOffset(months=months_back)
        nearest_idx = sven_df.index.get_indexer([target], method="nearest")[0]
        if nearest_idx >= 0 and nearest_idx < len(sven_df):
            ref_dates.append((sven_df.index[nearest_idx], label))

    fig = go.Figure()
    # Three distinct colors: 3y ago = amber (oldest), 12m ago = teal (mid),
    # Baseline = navy (today), Stressed = crimson dashed (override).
    hist_styles = {
        "12m ago": dict(color=COLORS["teal"],  dash="dash", width=1.8),
        "3y ago":  dict(color=COLORS["amber"], dash="dot",  width=1.8),
    }
    for d, lbl in ref_dates:
        style = hist_styles.get(lbl, dict(color=COLORS["stone"], dash="dot",
                                          width=1.4))
        p = historical_curve(d, sven_df, method="ffill")
        rates = curve_grid(p, maturities=mats)
        fig.add_trace(go.Scatter(
            x=rates.index, y=rates.values,
            name=f"{lbl} ({d.date()})",
            line=dict(color=style["color"], width=style["width"],
                      dash=style["dash"]),
            opacity=0.9,
        ))
    fig.add_trace(go.Scatter(
        x=base["maturity"], y=base["rate_pct"],
        name=f"Baseline ({sven_df.index[-1].date()})",
        line=dict(color=COLORS["navy"], width=3.0),
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

    # =================================================================
    # NEW · 10y rate evolution time series (recession/cycle context)
    # =================================================================
    st.divider()
    eyebrow("10y Zero-Rate · historische Entwicklung")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #2251FF;padding:0.8rem 1.1rem;border-radius:6px;'
        'margin:0.4rem 0 1rem 0;color:#051C2C;font-size:0.88rem;line-height:1.6;">'
        '<strong>Modell-Kontext.</strong> Der 10y-Zins ist die zentrale '
        'rate-Inputgröße des Modells. Die folgende Zeitreihe ordnet das '
        'aktuelle Niveau historisch ein und zeigt, wie weit man unter der '
        'live gesetzten Stress-Annahme von der Baseline abweicht — relativ '
        'zur Bandbreite, die der Markt in den letzten Jahren tatsächlich '
        'gesehen hat. Schraffierte Episoden markieren bekannte '
        'Stress-Phasen.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Build 10y series — sample monthly for performance
    hist_10y = sven_df[::5].copy()  # every ~week
    series_10y = []
    for ts, row in hist_10y.iterrows():
        try:
            p = params_from_row(row)
            series_10y.append((ts, float(zero_rate(10.0, p, as_decimal=False))))
        except Exception:
            continue
    ts_df = pd.DataFrame(series_10y, columns=["date", "r10"]).set_index("date")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts_df.index, y=ts_df["r10"],
        name="10y Zero rate",
        line=dict(color=COLORS["navy"], width=1.8),
        fill="tozeroy", fillcolor="rgba(5,28,44,0.05)",
    ))

    # Mark current baseline + stress level
    r10_base = float(zero_rate(10.0, base_params, as_decimal=False))
    r10_str  = float(zero_rate(10.0, shifted, as_decimal=False))
    fig_ts.add_hline(y=r10_base, line_dash="solid",
                     line_color=COLORS["navy"], line_width=1.2,
                     annotation_text=f"Baseline {r10_base:.2f}%",
                     annotation_position="top left",
                     annotation_font=dict(color=COLORS["navy"], size=10))
    if abs(r10_str - r10_base) > 1e-3:
        fig_ts.add_hline(y=r10_str, line_dash="dash",
                         line_color=COLORS["crimson"], line_width=1.4,
                         annotation_text=f"Stressed {r10_str:.2f}%",
                         annotation_position="bottom left",
                         annotation_font=dict(color=COLORS["crimson"],
                                              size=10))

    # Stress-Episoden-Schraffuren
    episodes = [
        ("2020-02-20", "2020-05-31", "COVID Shock"),
        ("2022-02-24", "2022-12-31", "Ukraine / Energy"),
        ("2023-03-08", "2023-05-31", "SVB / Credit Suisse"),
    ]
    for x0, x1, lbl in episodes:
        try:
            x0_dt = pd.Timestamp(x0)
            x1_dt = pd.Timestamp(x1)
            if x0_dt < ts_df.index.min() or x0_dt > ts_df.index.max():
                continue
            fig_ts.add_vrect(
                x0=x0_dt, x1=x1_dt,
                fillcolor=COLORS["crimson"], opacity=0.08,
                layer="below", line_width=0,
                annotation_text=lbl,
                annotation_position="top",
                annotation_font=dict(color=COLORS["crimson"], size=10),
            )
        except Exception:
            pass

    # Statistical context bands (10y min/max/median over series)
    r10_median = float(ts_df["r10"].median())
    r10_min = float(ts_df["r10"].min())
    r10_max = float(ts_df["r10"].max())
    fig_ts.add_hline(y=r10_median, line_dash="dot",
                     line_color=COLORS["stone"], line_width=1.0,
                     annotation_text=f"Median {r10_median:.2f}%",
                     annotation_position="top right",
                     annotation_font=dict(color=COLORS["stone"], size=10))

    fig_ts.update_layout(
        xaxis_title="Datum",
        yaxis_title="10y Zero rate [%]",
        height=380,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Quick stats strip
    perc_cols = st.columns(4, gap="small")
    pct_today = float((ts_df["r10"] <= r10_base).mean() * 100)
    perc_cols[0].metric("10y Range (Sample)",
                        f"{r10_min:.2f} – {r10_max:.2f}%",
                        f"Δ {r10_max - r10_min:.2f}pp",
                        delta_color="off")
    perc_cols[1].metric("Heute vs. Median",
                        f"{r10_base:.2f}%",
                        f"{r10_base - r10_median:+.2f}pp",
                        delta_color="off")
    perc_cols[2].metric("Perzentil heute",
                        f"{pct_today:.0f}%",
                        "Anteil Tage mit niedrigerem 10y",
                        delta_color="off")
    if abs(r10_str - r10_base) > 1e-3:
        # Where in history would stress level sit?
        pct_str = float((ts_df["r10"] <= r10_str).mean() * 100)
        perc_cols[3].metric("Perzentil Stress",
                            f"{pct_str:.0f}%",
                            f"r₁₀ = {r10_str:.2f}%",
                            delta_color="off")
    else:
        perc_cols[3].metric("Stress-Δ", "—",
                            "keine β-Verschiebung aktiv",
                            delta_color="off")

    # =================================================================
    # Svensson parameters table + explanation
    # =================================================================
    st.divider()
    eyebrow("Svensson-Parameter · Baseline vs. Stressed")

    with st.expander("Was bedeuten β₀, β₁, β₂, β₃, τ₁, τ₂?", expanded=False):
        st.markdown("""
Das **Svensson-Modell** (1994, Erweiterung von Nelson-Siegel 1987) ist die
Standard-Parametrisierung der EZB und der Deutschen Bundesbank für die
Zero-Coupon-Zinskurve. Statt jeden einzelnen Laufzeitpunkt zu speichern,
beschreiben sechs Parameter die gesamte Kurve über alle Laufzeiten:

- **β₀ (Level)** — langfristiges Zinsniveau, d.&nbsp;h. der Zins bei
  Laufzeit → ∞. Eine Erhöhung von β₀ verschiebt die *gesamte* Kurve nach
  oben (Parallel-Shift). Im Modell der wichtigste Schock-Hebel.
- **β₁ (Slope)** — kurzfristige Komponente. Negatives β₁ → steile Kurve
  (kurzes Ende tiefer als langes); positives β₁ → flache oder invertierte
  Kurve. Steuert das Verhältnis kurzer zu langer Zinsen.
- **β₂ (Curvature&nbsp;1)** — mittlere Krümmung. Beeinflusst den
  "Bauch" der Kurve (typisch 2y–5y-Bereich). Positives β₂ erzeugt einen
  Hump (mittlere Laufzeiten höher als das Average), negatives eine
  Mulde.
- **β₃ (Curvature&nbsp;2)** — sekundäre Krümmung. Erlaubt einen zweiten
  Buckel im längeren Bereich (typisch 5y–15y) — Svensson's Beitrag
  gegenüber Nelson-Siegel.
- **τ₁, τ₂ (Decay-Konstanten)** — bestimmen *wo* die Krümmungen wirken.
  Kleine τ → Wirkung am kurzen Ende; große τ → Wirkung am langen Ende.
  Diese werden in der Regel von der Zentralbank fix kalibriert, nicht
  täglich neu geschätzt.

**Formel.** Der Zero-Coupon-Zins bei Laufzeit *m* berechnet sich als
        """)
        st.latex(r"""
r(m) = \beta_0
     + \beta_1 \cdot \frac{1 - e^{-m/\tau_1}}{m/\tau_1}
     + \beta_2 \cdot \left(\frac{1 - e^{-m/\tau_1}}{m/\tau_1}
                            - e^{-m/\tau_1}\right)
     + \beta_3 \cdot \left(\frac{1 - e^{-m/\tau_2}}{m/\tau_2}
                            - e^{-m/\tau_2}\right)
""")
        st.markdown("""
**Im Modell** werden im Multi-Factor-Modus β₀-β₃ direkt vom Nutzer über
Slider geschoben — eine Δβ₀-Verschiebung ist ein Parallel-Shift der
gesamten Kurve und entspricht dem klassischen Duration-Schock-Setup. Im
Single-Factor-M-Modus wird der ΔBundeswert konsistent aus dem
EBA-Adverse-Anker abgeleitet.
""")

    params_table = pd.DataFrame({
        "Parameter": ["β₀ (Level — langfr. Niveau)",
                      "β₁ (Slope — kurz/lang-Spread)",
                      "β₂ (Curvature 1 — Bauch der Kurve)",
                      "β₃ (Curvature 2 — sek. Krümmung)",
                      "τ₁ (Decay — kurzes Ende)",
                      "τ₂ (Decay — langes Ende)"],
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
    st.caption(
        "τ₁ und τ₂ sind in der Bundesbank-Veröffentlichung "
        "fix kalibriert — daher kein Δ über die β-Slider abbildbar."
    )


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
