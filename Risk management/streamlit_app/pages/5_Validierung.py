"""Backtesting · Forecast vs Realized 1Y CET1 movements + RWA decomposition.

Critique 5 (Outcomes Analysis SR 11-7) und Critique 6 (Netto-EAD-Effekte
bereinigt um externe Faktoren) — auf Basis der historischen EBA-
Transparency-Vintages 2020-2025 (22 Quartals-Stichtage Sep 2019 → Jun 2025,
~120 Banken pro Stichtag).

Vier Diagnostik-Bereiche:
  1. Macro-Faktor-Timeline · realized Brent log-Return + Δr_10y + M_hybrid
  2. Empirische OLS-Sensitivität · Δ-CET1-Ratio = α + β · M_at_start
  3. Stress-Episode-Vergleich · Ukraine 2022, Rate-hike 2023
     (Corona 2020 nicht gerechnet — Brent-Cache reicht nicht zurück)
  4. RWA-Komponenten-Decomposition · Δ Credit / Market / Operational
     pro Bank-Period-Pair (EBA-publizierte Brutto-Decomposition)
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

from datetime import datetime, timedelta

from components.theme import (tab_breadcrumb, apply_theme, hero, eyebrow, insight, footer,
                              COLORS)
from components.sidebar import render_sidebar
from components.data_loader import load_data_layer
from components.backend_path import setup
setup()

from config import EBA_RAW_DIR                                     # type: ignore
from eba_loader import (load_historical_capital_panel,             # type: ignore
                        panel_to_wide, load_bank_directory,
                        load_eba_universe)
from macro_factor import factor_stats                              # type: ignore
from backtesting import (                                          # type: ignore
    compute_macro_panel, attach_m_factor,
    compute_realized_changes, build_forecast_panel,
    fit_empirical_sensitivity, episode_diagnostics, rwa_decomposition,
    DEFAULT_EPISODES,
)
from backtesting_walkforward import (                              # type: ignore
    compute_quarterly_macro, attach_m_factor_quarterly,
    build_walkforward_panel, walkforward_error_stats,
    per_bank_summary, system_aggregate_timeseries,
    EPISODES_QUARTERLY, m_to_credit_rwa_scale,
)


st.set_page_config(page_title="Validierung & Methodologie", layout="wide")
apply_theme()
render_sidebar()

hero(
    "Validierung & Methodologie",
    eyebrow="Tab 5 · Walk-Forward-Backtest · Annahmen · Methodologie",
    deck="Drei Validierungs-Ebenen kombiniert: (1) historisches Walk-"
         "Forward-Backtesting der Vorgänger-Methodik (Vasicek-Single-"
         "Factor-M) als Diagnose und Begründung für die Umstellung "
         "auf 2-Faktor, (2) 3-Layer-Annahmen-"
         "Disclosure (Executive · Validator · Quant), (3) vollständige "
         "Methodologie aus MODEL_ASSUMPTIONS.md. Outcomes Analysis nach "
         "SR 11-7 + Governance nach EBA GL 14.",
)



tab_breadcrumb(5)
# Top-level tabs · Backtesting + Annahmen + Methodologie
from components.legacy_views import render_annahmen_tab, render_methodology_tab

tab_bt, tab_an, tab_md = st.tabs([
    "1 · Backtesting (Forecast vs. Realized)",
    "2 · Annahmen & Datenbasis (Governance)",
    "3 · Methodologie (MODEL_ASSUMPTIONS.md)",
])

with tab_bt:
    # === Page intro · "Was ist das?" =====================================
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #034B6F;padding:0.9rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.65;">'
        '<strong>Hinweis zur Methodik:</strong> Dieser Walk-Forward-'
        'Backtest verwendet die <em>vorherige</em> Single-Factor-M-'
        'Formulierung des Modells (Basel-III-IRB (Vasicek-Modell)) als historischen '
        'Performance-Test. Er zeigt, wie ein M-aggregiertes Modell die '
        'tatsächlichen quartalsweisen RWA-Bewegungen vorhergesagt hätte. '
        'Im Live-Cockpit verwenden wir das überarbeitete 2-Faktor-Modell '
        '(ΔBrent + Δr_10y separat, sektor-differenzierte Sensitivitäten) '
        '— der Backtest unten ist die <strong>Diagnose</strong> der '
        'früheren Methodik und Begründung für die Umstellung.<br><br>'
        '<strong>Was ist Walk-Forward-Backtesting?</strong><br>'
        'Ein Stress-Modell ist nur so gut wie seine Vorhersagekraft auf '
        '<em>historische</em> Daten. Beim Walk-Forward-Backtest gehen wir '
        'die Vergangenheit Quartal für Quartal durch: '
        '(1)&nbsp;wir nehmen den Bilanz-Stand der Bank zu Beginn des Quartals, '
        '(2)&nbsp;messen den tatsächlich realisierten Macro-Schock '
        '(ΔBrent, Δr_10y) während des Quartals, '
        '(3)&nbsp;lassen das Modell <em>vorhersagen</em>, wie sich '
        'das Kredit-RWA dadurch ändern müsste, und '
        '(4)&nbsp;vergleichen mit der <em>tatsächlich beobachteten</em> '
        'Veränderung — bereinigt um den Volumeneffekt '
        '(siehe Bedienung „Bereinigung" weiter unten).<br><br>'
        '<strong>Datenbasis.</strong> 22 Quartals-Stichtage Sep 2019 – Jun '
        '2025, gefiltert auf die 67 IRB-Banken der EBA Transparency 2025. '
        'Macro: Brent (ICE) + Bundesbank-Svensson, daily.<br>'
        '<strong>Was leistet das?</strong> Quantifiziert, wo das Modell '
        'die Wirklichkeit gut/schlecht trifft — die wichtigste '
        'Voraussetzung dafür, dass ein/e Validator/in das Modell '
        'überhaupt einsetzen darf (SR 11-7, EBA GL 14).'
        '</div>',
        unsafe_allow_html=True,
    )

    # === Methodik · 5-Schritt-Ablauf =====================================
    with st.expander("Methodik · der 5-Schritt-Ablauf im Detail",
                     expanded=False):
        st.markdown("""
**Schritt A — Snapshot bei Quartal t.**
Aus den historischen EBA-Vintages 2020-2025 entnehmen wir pro Bank die
Kredit-RWA, Markt-RWA, Operational-RWA, CET1 und Total-RWA zum Stichtag.

**Schritt B — Realisierten Macro-Schock messen.**
Zwischen t und t+1 berechnen wir die *tatsächliche* Veränderung von:
- **ΔBrent (log)** — Ölpreis-Return über das Quartal
- **Δr_10y (pp)** — Bewegung des 10-Jahres-Zinses

**Schritt C — Implizites M ableiten.**
Unsere Macro→M-Funktion (`hybrid_mapping`) übersetzt das Schock-Tupel
(ΔBrent, Δr_10y) in einen einzigen Vasicek-System-Faktor M. Negatives
M = adverse Phase, positives M = benign.

**Schritt D — Modell-Vorhersage.**
Für jede Bank wenden wir den Skalierungsfaktor `scale(M)` an, der aus
einer einmaligen Kalibrierung auf dem EU-Aggregat-Portfolio kommt:
""")
        st.latex(r"""
\widehat{\Delta\mathrm{RWA}_{\text{credit}}}
    = \mathrm{RWA}_{\text{credit}, t}
      \cdot \bigl(\mathrm{scale}(M_q) - 1\bigr)
""")
        st.markdown("""
`scale(M)` ist die relative Veränderung des Loan-Book-RWA, die unser
Basel-III-IRB (Vasicek-Modell)-Modell mit M=0 als Referenz vorhersagt. Bei M = −2.5
(EBA-Adverse-Anker) liegt `scale` typisch bei ~1.4 (= +40% RWA).

**Schritt E — Volumeneffekt herausrechnen.**
Eine Bank kann ihr Kredit-RWA auch ohne Stress wachsen lassen — z.B.
durch mehr Neugeschäft. Diesen *Volumen*-Anteil müssen wir vom
beobachteten ΔRWA abziehen, um die *risiko-getriebene* Veränderung
zu isolieren:
""")
        st.latex(r"""
\Delta\mathrm{RWA}_{\text{credit}}^{\text{Volumen}}
    = \mathrm{RWA}_{\text{credit}, t}
      \cdot \left(\frac{\mathrm{RWA}_{\text{nicht-credit}, t+1}}
                       {\mathrm{RWA}_{\text{nicht-credit}, t}} - 1\right)
""")
        st.markdown("""
`RWA_nicht-credit` = RWA_market + RWA_operational. Diese spiegeln die
Geschäftsgröße der Bank wider, sind aber unempfindlich gegenüber dem
Kredit-Stress. Ihr Wachstum dient daher als sauberer Proxy für die
„organische" Bilanzausweitung.

**Vergleich.** Risiko-getriebenes Ist:
""")
        st.latex(r"""
\Delta\mathrm{RWA}_{\text{risk-driven}}
    = \Delta\mathrm{RWA}_{\text{credit}, \text{actual}}
    - \Delta\mathrm{RWA}_{\text{credit}}^{\text{Volumen}}
""")
        st.markdown("""
**Fehler-Metriken** über alle Bank-Quartal-Paare:
- **MAE** (Mean Absolute Error) — durchschnittlicher Betrag des Fehlers
- **Bias** — systematische Über-/Unterprognose des Modells
- **Hit-Rate** — Anteil der Beobachtungen, bei denen Modell und
  Realität dasselbe Vorzeichen haben
- **Korrelation** — Stärke des linearen Zusammenhangs Vorhersage vs. Ist

**Wichtige Limitation.** Wir nehmen einen *uniformen* Skalierungs-
Faktor `scale(M)` für alle Banken an. Bank-individuelle Heterogenität
(unterschiedliche Portfolio-Mix, Sektor-Konzentration) ist V2-Roadmap
und würde Segment-Level-EAD pro Vintage erfordern.
""")

    st.divider()

    # === Load data (cached) ==============================================
    @st.cache_data(ttl=24*3600, show_spinner="Loading historical EBA panel …")
    def _load_panel():
        panel = load_historical_capital_panel(EBA_RAW_DIR)
        wide  = panel_to_wide(panel)
        bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
        return panel, wide, bank_dir


    @st.cache_data(ttl=24*3600, show_spinner="Loading Top-10 IRB universe …")
    def _load_irb_leis(_bank_dir):
        # Top-10-Banken aus pillar3_bank_pd_lgd.csv (bank-spezifische Pillar-3 EU-CR6 31.12.2024)
        from eba_pd_loader import get_top10_leis                      # type: ignore
        return get_top10_leis()


    @st.cache_data(ttl=24*3600, show_spinner="Computing trailing-1Y macro state …")
    def _compute_macro(periods: tuple[int, ...]):
        data = load_data_layer()
        if data["brent"] is None or data["svensson"] is None:
            return None, None
        macro = compute_macro_panel(data["brent"], data["svensson"], list(periods))
        fs = factor_stats(data["brent"], data["svensson"], lookback=252)
        macro = attach_m_factor(macro, cov_factors=fs["sigma"])
        return macro, fs


    @st.cache_data(ttl=24*3600, show_spinner="Computing quarterly macro shocks …")
    def _compute_macro_quarterly(periods: tuple[int, ...]):
        data = load_data_layer()
        if data["brent"] is None or data["svensson"] is None:
            return None, None
        mq = compute_quarterly_macro(data["brent"], data["svensson"],
                                      list(periods))
        fs = factor_stats(data["brent"], data["svensson"], lookback=252)
        mq = attach_m_factor_quarterly(mq, cov_factors=fs["sigma"])
        return mq, fs


    @st.cache_data(ttl=24*3600, show_spinner="Walk-Forward Backtest läuft …")
    def _build_wf(_wide, _macro_q, _irb_leis):
        return build_walkforward_panel(_wide, _macro_q,
                                        irb_leis=_irb_leis)


    panel, wide, bank_dir = _load_panel()
    irb_leis = _load_irb_leis(bank_dir)
    periods = tuple(sorted(wide["Period"].unique()))
    macro, fs = _compute_macro(periods)
    macro_q, _ = _compute_macro_quarterly(periods)

    if macro is None or len(macro) == 0 or macro_q is None:
        st.error("Macro panel could not be computed — Brent or Svensson cache missing.")
        st.stop()

    # Build the walk-forward panel
    wf_panel_raw = _build_wf(wide, macro_q, irb_leis)
    wf_panel_raw = wf_panel_raw.merge(bank_dir[["lei", "bank_name"]],
                                       left_on="LEI_Code", right_on="lei",
                                       how="left")

    # === Horizont-Korrektur ==============================================
    # Wichtige methodische Anpassung: scale(M) ist aus einem 1-Jahres-
    # Stress-Modell kalibriert. Wir backtesten aber QUARTALSweise. Daher
    # darf nur ein Bruchteil der 1Y-Wirkung pro Quartal materialisieren.
    # Vorgabe: 1/4 als linearen Default. Slider erlaubt OLS-Kalibrierung.
    eyebrow("Modell-Kalibrierung · Horizont-Anpassung")
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #C9A227;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.6;">'
        '<strong>Wichtige Klarstellung — Backtest-Horizont.</strong> '
        'Unser strukturelles Modell kalibriert <code>scale(M)</code> '
        'auf eine <em>1-Jahres</em>-Stress-Wirkung: bei M = −2.5 '
        'projiziert es +40 % Kredit-RWA, weil die EBA-Adverse-Szenarien '
        '12-Monats-Horizonte sind. Im Quartal-Walk-Forward darf aber '
        'nur ein Bruchteil dieser Wirkung pro Quartal sichtbar werden — '
        'sonst überschießt das Modell systematisch um den Faktor 4. '
        'Die Horizont-Korrektur unten skaliert die Modell-Vorhersage '
        'auf den Backtest-Horizont.<br><br>'
        '<strong>Zwei Modi:</strong><br>'
        '• <em>Linear 1/4</em> — Annahme: jährliche Wirkung verteilt '
        'sich linear über 4 Quartale (Default 0.25)<br>'
        '• <em>OLS-kalibriert</em> — der Skalierungsfaktor wird aus '
        'den Daten selbst gefittet (regression-based)'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Compute OLS calibration: actual = a + b * pred
    # b ≈ optimal scale ratio. Use as recommended scaling.
    sub_clean = wf_panel_raw.dropna(subset=["pred_dRWA_credit_eur",
                                              "risk_driven_dRWA_credit_eur"])
    if len(sub_clean) > 30:
        x_pred = sub_clean["pred_dRWA_credit_eur"].to_numpy()
        y_act  = sub_clean["risk_driven_dRWA_credit_eur"].to_numpy()
        denom = float(np.sum(x_pred * x_pred))
        ols_scale = float(np.sum(x_pred * y_act) / max(denom, 1e-9))
    else:
        ols_scale = 0.25

    cal_l, cal_r = st.columns([1, 2], gap="medium")
    with cal_l:
        cal_mode = st.radio(
            "Skalierungs-Modus",
            ["Linear 1/4 (Default)", "OLS-kalibriert", "Roh (kein Scaling)"],
            index=0, key="wf_cal_mode",
        )
    if cal_mode.startswith("Linear"):
        scale_factor = 0.25
    elif cal_mode.startswith("OLS"):
        scale_factor = ols_scale
    else:
        scale_factor = 1.0
    with cal_r:
        st.markdown(
            f"**Aktiver Skalierungsfaktor: `{scale_factor:+.3f}`**  \n"
            f"OLS-Bestfit aus den Daten: `{ols_scale:+.3f}` "
            f"(Default 0.25 entspricht linearem Quartal/Jahr-Anteil). "
            f"Negative OLS-Werte würden bedeuten, dass das Modell-"
            f"Vorzeichen systematisch falsch ist — siehe Bias-KPI unten."
        )

    # Apply scaling to predictions
    wf_panel = wf_panel_raw.copy()
    wf_panel["pred_dRWA_credit_eur"] *= scale_factor
    wf_panel["pred_dK_eur"]          *= scale_factor
    # Recompute error + sign_match with scaled predictions
    wf_panel["error_eur"] = (wf_panel["pred_dRWA_credit_eur"]
                              - wf_panel["risk_driven_dRWA_credit_eur"])
    wf_panel["error_pct_of_start"] = (
        wf_panel["error_eur"] / wf_panel["rwa_credit_start"]
    )
    wf_panel["sign_match"] = (
        ((wf_panel["pred_dRWA_credit_eur"] >= 0)
         & (wf_panel["risk_driven_dRWA_credit_eur"] >= 0))
        | ((wf_panel["pred_dRWA_credit_eur"] < 0)
           & (wf_panel["risk_driven_dRWA_credit_eur"] < 0))
    )

    st.divider()

    # === Headline KPI strip ==============================================
    eyebrow("Backtest-Übersicht · die wichtigsten Kennzahlen auf einen Blick")
    stats = walkforward_error_stats(wf_panel)
    n_quarters = wf_panel["Period_end"].nunique()
    n_banks_wf = wf_panel["LEI_Code"].nunique()

    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    k1.metric("Bank-Quartal-Beobachtungen",
              f"{stats.get('n', 0):,}",
              f"{n_banks_wf} Banken × {n_quarters} Quartale",
              delta_color="off")
    k2.metric("Hit-Rate (Vorzeichen)",
              f"{stats.get('hit_rate', 0)*100:.1f}%",
              "Modell trifft Richtung",
              delta_color="off")
    mae_b = stats.get("mae_eur", 0) / 1e9
    k3.metric("MAE (Fehlerbetrag)",
              f"€{mae_b:,.1f} bn",
              "durchschnittlich, je Bank-Quartal",
              delta_color="off")
    bias_b = stats.get("bias_eur", 0) / 1e9
    k4.metric("Bias",
              f"€{bias_b:+,.1f} bn",
              "positive = Modell überschätzt",
              delta_color="off")
    k5.metric("Korrelation Modell-Ist",
              f"{stats.get('corr', float('nan')):+.2f}",
              "Pearson über alle Paare",
              delta_color="off")
    k6.metric("Fehler relativ",
              f"{stats.get('mae_pct', 0)*100:.1f}%",
              "MAE / RWA_credit_start",
              delta_color="off")

    # Interpretation
    bias_dir = "überschätzt" if bias_b > 0 else "unterschätzt"
    hit_label = "deutlich besser" if stats.get("hit_rate", 0) > 0.55 else (
                "leicht besser" if stats.get("hit_rate", 0) > 0.50 else
                "auf Münzwurf-Niveau")
    insight(
        f"<strong>Lesart.</strong> Bei <em>{stats.get('n', 0):,}</em> "
        f"Bank-Quartal-Paaren liegt die Vorzeichen-Hit-Rate bei "
        f"<strong>{stats.get('hit_rate', 0)*100:.1f}%</strong> — "
        f"{hit_label} als Zufall (50%). Das Modell {bias_dir} die "
        f"Realität durchschnittlich um "
        f"<strong>€{abs(bias_b):.1f} bn pro Bank-Quartal</strong>. "
        f"Wichtige Einordnung: 50 % über <em>alle</em> Quartale ist nicht "
        f"überraschend — in ruhigen Quartalen ohne Macro-Schock gibt es "
        f"praktisch kein Signal, das das Modell treffen könnte. Der "
        f"aussagekräftige Test ist die Hit-Rate <strong>nur in Stress-"
        f"Quartalen</strong> (|M| &gt; 0.5) — siehe nächster Abschnitt."
    )

    st.divider()

    # === Stress-Quartal-only Diagnose ====================================
    eyebrow("Konditionale Diagnose · Hit-Rate nur in Stress-Quartalen")
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #A52F4D;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.6;">'
        '<strong>Warum die rohe 50 %-Hit-Rate irreführend ist.</strong> '
        'Ein Stress-Modell ist <em>kein</em> Quartal-Prognose-Tool für '
        'Business-as-usual. Wenn das Quartal makroökonomisch ereignislos '
        'war (|M| nahe 0), ist die realisierte Veränderung des '
        'Kredit-RWA vor allem Rauschen aus Portfolio-Umschichtungen, '
        'CRM-Effekten und Bilanzpolitik. Der Modell-Beitrag ist klein '
        '→ der Erwartungswert der Hit-Rate ist 50&nbsp;%.<br><br>'
        'Aussagekräftig ist die Hit-Rate nur dann, wenn <em>tatsächlich</em> '
        'ein Macro-Schock vorlag. Wir filtern unten auf Quartale mit '
        '|M| &gt; 0.5 (= moderate adverse oder benign Phase) und |M| '
        '&gt; 1.0 (= ausgeprägter Stress).'
        '</div>',
        unsafe_allow_html=True,
    )

    thr_l, thr_r = st.columns([1, 3], gap="medium")
    with thr_l:
        m_thresholds = [0.0, 0.5, 1.0, 1.5]
        labels = ["Alle Quartale (|M| ≥ 0)",
                  "Schock-Quartale (|M| ≥ 0.5)",
                  "Stress-Quartale (|M| ≥ 1.0)",
                  "Schwer-Stress (|M| ≥ 1.5)"]
        sel_thr_label = st.radio(
            "Filter-Schwelle", labels, index=2, key="wf_thr",
        )
        m_thr = m_thresholds[labels.index(sel_thr_label)]

    with thr_r:
        sub_thr = wf_panel[wf_panel["m_quarter"].abs() >= m_thr]
        stats_thr = walkforward_error_stats(sub_thr)
        n_thr = stats_thr.get("n", 0)
        hr_thr = stats_thr.get("hit_rate", 0) * 100
        corr_thr = stats_thr.get("corr", float("nan"))
        mae_thr_b = stats_thr.get("mae_eur", 0) / 1e9
        bias_thr_b = stats_thr.get("bias_eur", 0) / 1e9
        tk1, tk2, tk3, tk4 = st.columns(4, gap="small")
        tk1.metric("n Paare (gefiltert)", f"{n_thr:,}",
                   f"von {stats.get('n', 0):,}", delta_color="off")
        tk2.metric("Hit-Rate", f"{hr_thr:.1f}%",
                   "vs. 50 % Zufall", delta_color="off")
        tk3.metric("Korrelation", f"{corr_thr:+.2f}",
                   "Modell vs. Ist", delta_color="off")
        tk4.metric("MAE (Mrd. EUR)",
                   f"€{mae_thr_b:.2f} bn",
                   f"Bias €{bias_thr_b:+.2f} bn", delta_color="off")

    # Comparison table: hit-rate by threshold
    diag_rows = []
    for lbl, t in zip(labels, m_thresholds):
        s = wf_panel[wf_panel["m_quarter"].abs() >= t]
        st_ = walkforward_error_stats(s)
        diag_rows.append({
            "Filter": lbl,
            "n Paare": st_.get("n", 0),
            "Hit-Rate": f"{st_.get('hit_rate', 0)*100:.1f}%",
            "Korrelation": f"{st_.get('corr', 0):+.2f}",
            "MAE bn": round(st_.get("mae_eur", 0)/1e9, 2),
            "Bias bn": round(st_.get("bias_eur", 0)/1e9, 2),
        })
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True,
                 hide_index=True, height=200)

    # Auto-interpretation
    hr_all = pd.DataFrame(diag_rows).iloc[0]
    hr_strong = pd.DataFrame(diag_rows).iloc[-1]
    insight(
        f"<strong>Was die Tabelle zeigt.</strong> Beim Filter auf <em>"
        f"alle</em> Quartale liegt die Hit-Rate bei {hr_all['Hit-Rate']}. "
        f"Beim Filter auf Schwer-Stress-Quartale "
        f"(|M| ≥ 1.5) liegt sie bei <strong>{hr_strong['Hit-Rate']}</strong> "
        f"— bei nur noch {hr_strong['n Paare']} Beobachtungen. "
        f"Steigt die Hit-Rate mit der Filter-Schwelle, ist das ein "
        f"valider Beleg, dass das Modell genau dann informativ wird, "
        f"wenn tatsächlich ein Schock vorliegt — der pragmatische "
        f"Use-Case eines Stress-Modells."
    )

    st.divider()

    # === M-Faktor-Timeline mit Episoden ===================================
    eyebrow("Macro-Schock-Timeline · realisiertes M pro Quartal")
    st.caption(
        "Aus den realisierten ΔBrent + Δr_10y pro Quartal abgeleiteter "
        "Vasicek-Faktor M. Negative Werte = adverse Phase, positive = benign. "
        "Crisis-Banner markieren bekannte Stress-Episoden."
    )

    fig_m = go.Figure()
    macro_q_sorted = macro_q.sort_values("Period_end").copy()
    macro_q_sorted["date_end"] = pd.to_datetime(macro_q_sorted["date_end"])
    fig_m.add_trace(go.Bar(
        x=macro_q_sorted["date_end"],
        y=macro_q_sorted["m_hybrid"],
        marker_color=[
            COLORS["crimson"] if v < -1.0 else
            COLORS["amber"]   if v < 0    else
            COLORS["teal"]
            for v in macro_q_sorted["m_hybrid"]
        ],
        text=[f"{v:+.2f}" for v in macro_q_sorted["m_hybrid"]],
        textposition="outside",
        textfont=dict(size=10),
        name="M (quarterly)",
        hovertemplate="<b>%{x|%Y-%m}</b><br>M = %{y:+.2f}<extra></extra>",
    ))
    fig_m.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
    for ep in EPISODES_QUARTERLY:
        ep_date = datetime(ep['period_end']//100, ep['period_end']%100, 1)
        fig_m.add_vline(
            x=ep_date,
            line_color=ep["color"], line_dash="dot", line_width=1.2,
            annotation_text=ep["label"],
            annotation_position="top",
            annotation_font=dict(color=ep["color"], size=10),
        )
    fig_m.update_layout(
        xaxis_title="Quartal", yaxis_title="M (Vasicek systemic factor)",
        height=320, showlegend=False, bargap=0.25,
    )
    st.plotly_chart(fig_m, use_container_width=True)

    st.divider()

    # === System-aggregate time-series · Modell vs Realität ===============
    eyebrow("System-Aggregat · Σ über alle 67 IRB-Banken pro Quartal")
    st.markdown(
        '<div style="background:#F4F4F4;padding:0.7rem 1.0rem;'
        'border-radius:6px;margin:0.3rem 0 0.9rem 0;color:#051C2C;'
        'font-size:0.86rem;line-height:1.55;">'
        '<strong>Was zeigt diese Grafik?</strong> Quartal für Quartal '
        'addieren wir Modell-Vorhersage und tatsächliche risiko-getriebene '
        'Veränderung über <em>alle</em> 67 IRB-Banken auf — das ist die '
        'System-View. Optimal: blaue (Modell) und rote (Ist bereinigt) '
        'Balken haben dieselbe Höhe und Richtung in jedem Quartal.'
        '</div>',
        unsafe_allow_html=True,
    )

    sys_ts = system_aggregate_timeseries(wf_panel).copy()
    sys_ts["date_end"] = pd.to_datetime(sys_ts["Period_end"].apply(
        lambda p: f"{p//100}-{p%100:02d}-01"))
    fig_sys = go.Figure()
    fig_sys.add_trace(go.Bar(
        x=sys_ts["date_end"], y=sys_ts["pred_sum"]/1e9,
        name="Modell-Vorhersage", marker_color=COLORS["navy"],
        marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x|%Y-%m}</b><br>Modell: %{y:+.1f} bn<extra></extra>",
    ))
    fig_sys.add_trace(go.Bar(
        x=sys_ts["date_end"], y=sys_ts["risk_driven_sum"]/1e9,
        name="Ist (risiko-bereinigt)", marker_color=COLORS["crimson"],
        marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x|%Y-%m}</b><br>Ist: %{y:+.1f} bn<extra></extra>",
    ))
    fig_sys.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
    for ep in EPISODES_QUARTERLY:
        ep_date = datetime(ep['period_end']//100, ep['period_end']%100, 1)
        fig_sys.add_vrect(
            x0=(ep_date - timedelta(days=30)).strftime("%Y-%m-%d"),
            x1=(ep_date + timedelta(days=30)).strftime("%Y-%m-%d"),
            fillcolor=ep["color"], opacity=0.06, layer="below",
            line_width=0,
            annotation_text=ep["label"],
            annotation_position="top",
            annotation_font=dict(color=ep["color"], size=10),
        )
    fig_sys.update_layout(
        xaxis_title="Quartal-Ende", yaxis_title="ΔRWA_credit [Mrd. EUR]",
        height=420, barmode="group", bargap=0.20,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
    )
    st.plotly_chart(fig_sys, use_container_width=True)

    st.divider()

    # === Bank Drilldown · Time-series pro Einzelbank ====================
    eyebrow("Bank-Drilldown · Modell vs. Realität pro Einzel-Institut")

    sov_per_bank = (wf_panel.groupby(["LEI_Code", "bank_name"])
                            ["rwa_credit_start"].mean()
                            .sort_values(ascending=False))

    bank_l, bank_r = st.columns([2, 5], gap="medium")
    with bank_l:
        st.markdown(
            "Wähle eine Bank — die Time-Series unten zeigt für jedes "
            "Quartal die Modell-Vorhersage neben dem tatsächlich "
            "risiko-bereinigten Wert. Banken sind nach durchschnittlicher "
            "Größe (Σ Kredit-RWA) sortiert."
        )

    with bank_r:
        bank_options = sov_per_bank.index.tolist()
        sel = st.selectbox(
            f"Bank wählen ({len(bank_options)} verfügbar · tippen zum Suchen)",
            bank_options,
            format_func=lambda x: f"{x[1]}  (Ø €{sov_per_bank.loc[x]/1e9:.0f} bn Kredit-RWA)",
            key="wf_bank",
        )
    sel_lei, sel_name = sel

    sub = wf_panel[wf_panel["LEI_Code"] == sel_lei].sort_values("Period_end").copy()
    sub["date_end"] = pd.to_datetime(sub["Period_end"].apply(
        lambda p: f"{p//100}-{p%100:02d}-01"))

    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        x=sub["date_end"], y=sub["pred_dRWA_credit_eur"]/1e9,
        name="Modell-Vorhersage", marker_color=COLORS["navy"],
        marker_line_width=0, opacity=0.88,
        hovertemplate="<b>%{x|%Y-%m}</b><br>Modell: %{y:+.2f} bn<extra></extra>",
    ))
    fig_b.add_trace(go.Bar(
        x=sub["date_end"], y=sub["risk_driven_dRWA_credit_eur"]/1e9,
        name="Ist (risiko-bereinigt)", marker_color=COLORS["crimson"],
        marker_line_width=0, opacity=0.88,
        hovertemplate="<b>%{x|%Y-%m}</b><br>Ist: %{y:+.2f} bn<extra></extra>",
    ))
    fig_b.add_trace(go.Scatter(
        x=sub["date_end"], y=sub["actual_dRWA_credit_eur"]/1e9,
        name="Ist (roh, inkl. Volumen)",
        mode="lines+markers",
        line=dict(color=COLORS["stone"], width=1.5, dash="dot"),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%Y-%m}</b><br>Ist roh: %{y:+.2f} bn<extra></extra>",
    ))
    fig_b.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
    for ep in EPISODES_QUARTERLY:
        ep_date = datetime(ep['period_end']//100, ep['period_end']%100, 1)
        fig_b.add_vrect(
            x0=(ep_date - timedelta(days=30)).strftime("%Y-%m-%d"),
            x1=(ep_date + timedelta(days=30)).strftime("%Y-%m-%d"),
            fillcolor=ep["color"], opacity=0.06, layer="below",
            line_width=0,
        )
    fig_b.update_layout(
        title=f"{sel_name} · ΔRWA_credit pro Quartal",
        xaxis_title="Quartal-Ende",
        yaxis_title="ΔRWA_credit [Mrd. EUR]",
        height=420, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
    )
    st.plotly_chart(fig_b, use_container_width=True)

    # Per-bank stats
    bank_stats = walkforward_error_stats(sub)
    bs1, bs2, bs3, bs4 = st.columns(4, gap="small")
    bs1.metric("Beobachtungen", f"{bank_stats.get('n', 0)}")
    bs2.metric("Hit-Rate", f"{bank_stats.get('hit_rate', 0)*100:.0f}%")
    bs3.metric("MAE", f"€{bank_stats.get('mae_eur', 0)/1e9:.2f} bn")
    bs4.metric("Bias", f"€{bank_stats.get('bias_eur', 0)/1e9:+.2f} bn",
               "+ = Modell überschätzt", delta_color="off")

    st.divider()

    # === Scatter Predicted vs Actual ====================================
    eyebrow("Scatter · Modell vs. risiko-bereinigte Realität (alle Paare)")
    st.caption(
        "Jeder Punkt = eine Bank in einem Quartal. Optimum: alle Punkte auf "
        "der 45°-Diagonale (Modell = Ist). Punkte oberhalb = Modell "
        "überschätzt, unterhalb = unterschätzt."
    )

    pred_x = wf_panel["pred_dRWA_credit_eur"] / 1e9
    actual_y = wf_panel["risk_driven_dRWA_credit_eur"] / 1e9
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scattergl(
        x=pred_x, y=actual_y, mode="markers",
        marker=dict(size=5, color=COLORS["navy"], opacity=0.35),
        text=wf_panel["bank_name"] + " · " + wf_panel["period_label_end"],
        hovertemplate="%{text}<br>Modell: %{x:+.2f} bn<br>Ist: %{y:+.2f} bn<extra></extra>",
        name="Bank-Quartal-Paar",
    ))
    # 45° line
    lim = max(abs(pred_x.min()), abs(pred_x.max()),
              abs(actual_y.min()), abs(actual_y.max())) * 1.05
    fig_sc.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color=COLORS["crimson"], width=2, dash="dash"),
        name="45° (perfekte Vorhersage)",
    ))
    fig_sc.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
    fig_sc.add_vline(x=0, line_color=COLORS["hairline"], line_width=1)
    fig_sc.update_layout(
        xaxis_title="Modell-Vorhersage ΔRWA_credit [Mrd. EUR]",
        yaxis_title="Ist (risiko-bereinigt) [Mrd. EUR]",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # === Per-Bank Summary Table =========================================
    eyebrow("Per-Bank Übersicht · Sortierung nach Modell-Fehler")

    pbs = per_bank_summary(wf_panel).merge(
        bank_dir[["lei", "bank_name"]],
        left_on="LEI_Code", right_on="lei", how="left",
    ).sort_values("mae_eur", ascending=False)

    pbs_disp = pd.DataFrame({
        "Bank":            pbs["bank_name"],
        "n Quartale":      pbs["n_obs"],
        "MAE bn":          (pbs["mae_eur"]/1e9).round(2),
        "Bias bn":         (pbs["bias_eur"]/1e9).round(2),
        "Hit-Rate":        (pbs["hit_rate"]*100).round(0).astype(int).astype(str) + "%",
        "Ø Kredit-RWA bn": (pbs["avg_rwa_credit_start"]/1e9).round(0).astype(int),
        "MAE / RWA":       (pbs["mae_pct_of_avg_rwa"]*100).round(1).astype(str) + "%",
    })
    st.dataframe(pbs_disp, use_container_width=True, hide_index=True,
                 height=500)
    st.caption(
        "Die größten Banken dominieren MAE in absoluten Beträgen. "
        "Spalte 'MAE / RWA' normalisiert auf die Bilanzgröße und ist daher "
        "der bessere Indikator für die *relative* Modell-Qualität "
        "pro Bank."
    )

    st.divider()

    # === Optional: alte Empirische OLS-Sensitivität als Challenger ======
    with st.expander(
            "Advanced · Empirische Sensitivität (Challenger-Modell, OLS)",
            expanded=False):
        st.caption(
            "Alternative Validierung: pro Bank wird die realisierte 1Y-"
            "Veränderung der CET1-Quote gegen den trailing-1Y-M-Faktor "
            "regressiert (OLS-Panel). Liefert ein empirisches β als "
            "Challenger zum strukturellen Modell."
        )

        # Forecast panel: realized + macro
        realized = compute_realized_changes(wide, lag_quarters=4)
        fp_full = build_forecast_panel(realized, macro)
        fp_full = fp_full.dropna(subset=["m_at_start", "delta_ratio_pp"])
        fp_full = fp_full.merge(bank_dir[["lei", "bank_name"]],
                                 left_on="LEI_Code", right_on="lei",
                                 how="left")


        # OLS regression
        fp_full_clean = fp_full[fp_full["delta_ratio_pp"].abs() <= 5.0].copy()
        sens = fit_empirical_sensitivity(fp_full_clean)
        ck1, ck2, ck3, ck4 = st.columns(4, gap="small")
        ck1.metric("Sample n", f"{sens.get('n', 0):,}",
                   delta_color="off")
        ck2.metric("β (Sens. zu M)",
                   f"{sens.get('beta', 0):+.3f}",
                   f"t = {sens.get('t_beta', 0):+.2f}",
                   delta_color="off")
        ck3.metric("R²", f"{sens.get('r2', 0):.3f}",
                   "Erklärte Varianz", delta_color="off")
        ck4.metric("RMSE", f"{sens.get('rmse', 0):.3f} pp",
                   delta_color="off")
        st.caption(
            "Empirisches β-Modell: ΔCET1-Ratio = α + β · M. Niedriges R² "
            "ist erwartet — bank-idiosynkratische Variation überlagert "
            "die Macro-Komponente. Dient als Sanity-Check der Vorzeichen-"
            "Konsistenz mit dem strukturellen Walk-Forward."
        )

    # === Methodik-Dokument zum Download ==================================
    st.divider()
    eyebrow("Methodik · Konzeptdokument")
    _project_root = Path(__file__).resolve().parent.parent.parent
    _docx_path = _project_root / "BACKTESTING_WALKFORWARD_KONZEPT.docx"
    dl_l, dl_r = st.columns([3, 2], gap="medium")
    with dl_l:
        st.markdown(
            "Das vollständige Konzeptpapier zum Walk-Forward-Backtest "
            "(7 Sektionen · Zielsetzung, Datenbasis, 5-Schritt-Ablauf, "
            "Fehler-Metriken, UI-Mapping, Limitationen, Interpretation) "
            "liegt als Word-Datei im Projekt-Root."
        )
    with dl_r:
        if _docx_path.exists():
            with open(_docx_path, "rb") as fh:
                st.download_button(
                    label="📄  BACKTESTING_WALKFORWARD_KONZEPT.docx",
                    data=fh.read(),
                    file_name="BACKTESTING_WALKFORWARD_KONZEPT.docx",
                    mime=("application/vnd.openxmlformats-officedocument."
                          "wordprocessingml.document"),
                    use_container_width=True,
                )
            st.caption(
                f"Pfad: `Risk management/BACKTESTING_WALKFORWARD_KONZEPT.docx`"
            )
        else:
            st.warning("Konzept-Dokument nicht gefunden.")

    footer(
        f"Walk-Forward Backtest · {stats.get('n', 0):,} Bank-Quartal-Paare · "
        f"{n_banks_wf} IRB-Banken × {n_quarters} Quartals-Übergänge · "
        f"Aktiver Skalierungsfaktor: {scale_factor:.3f} ({cal_mode}) · "
        f"Datenbasis: EBA Transparency 2020-2025 + Brent (ICE) + Bundesbank-"
        f"Svensson · Methodik: BACKTESTING_WALKFORWARD_KONZEPT.docx im "
        f"Projekt-Root"
    )

with tab_an:
    render_annahmen_tab()

with tab_md:
    render_methodology_tab()
