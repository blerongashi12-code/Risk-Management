"""Legacy page-views as callable render-functions.

Ein früher eigenständiger Page-View lebt als einbettbare Render-Funktion:

  - render_yield_curve_tab(config) — Bundesbank Svensson curve + β-shifts
                                       (Sub-Tab in Marktbuch)

Die früheren Sub-Tabs render_annahmen_tab()/render_methodology_tab()
(Validierung) wurden entfernt — die Annahmen-Dokumentation lebt vollständig
in docs/MODEL_ASSUMPTIONS.md bzw. der daraus generierten Word-Abgabefassung
(Abgabe-Files/Abgabedokumente/Modellannahmen.docx).

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
        st.error("Svensson cache missing — bitte `python tools/data_fetch/04_fetch_bundesbank_svensson.py` ausführen.")
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
