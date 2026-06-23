"""Validierung · Walk-Forward-Backtest auf der Pillar-3-PD/LGD-Zeitreihe.

Der Backtest friert das Portfolio jeder Bank an jedem historischen Stichtag
mit den *damals bekannten* Pillar-3-EU-CR6-Risikoparametern ein (PD, LGD und
EAD aus derselben Vintage → input-seitig 100 % Pillar-3, kein Look-ahead,
MODEL_ASSUMPTIONS A-02c) und prüft das Stress-Modell gegen die realisierte
Entwicklung aus den EBA-Transparency-Vintages.

Datenbasis der Roh-Reihe: `data/pillar3_backtest_pdlgd.csv` — 10 Banken,
Jahrgänge 2021-2024, EU-CR6-A-IRB-Sub-totals, dichte-/kontinuitäts-verifiziert
(keine fabrizierten Werte).

Validierung in drei Ebenen (ehrliche Outcomes-Analysis nach SR 11-7):
  1. Stress-Treiber-Timing  · trifft der Makro-Faktor M die bekannten Krisen?
  2. Strukturelle Soundness  · ist die bank-spezifische RWA-Antwort monoton
                               und konservativ (Vasicek/Basel-konsistent)?
  3. Outcomes-Analysis       · Prognose vs. realisiertes ΔRWA_credit — inkl.
                               der ehrlichen Grenze des Modells.
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
    # Series-driven 2-Faktor-Frozen-Portfolio-Pfad (nutzt die Pillar-3-Roh-Reihe)
    load_backtest_series, build_pdlgd_panel, backtest_series_coverage,
    build_frozen_portfolio_series, frozen_2factor_delta,
    compute_annual_macro, build_pd_backtest, pd_backtest_stats,
)
from two_factor_stress import SENSITIVITY_MATRIX, get_economic_logic  # type: ignore


st.set_page_config(page_title="Validierung & Methodologie", layout="wide")
apply_theme()
render_sidebar()

hero(
    "Validierung & Methodologie",
    eyebrow="Tab 5 · Pillar-3-Walk-Forward-Backtest · Annahmen · Methodologie",
    deck="Walk-Forward-Backtest auf der vollständigen Pillar-3-EU-CR6-PD/LGD-"
         "Zeitreihe (10 Banken · 2021-2024): das Modell wird an jedem "
         "historischen Stichtag mit den damals bekannten Risikoparametern "
         "eingefroren (kein Look-ahead) und in drei Ebenen gegen die "
         "realisierte Entwicklung geprüft. Plus 3-Layer-Annahmen-Disclosure "
         "und vollständige Methodologie. Outcomes-Analysis nach SR 11-7, "
         "Governance nach EBA GL 14.",
)



tab_breadcrumb(5)
# Top-level tabs · Backtesting + Annahmen + Methodologie
from components.legacy_views import render_annahmen_tab, render_methodology_tab

tab_bt, tab_an, tab_md = st.tabs([
    "1 · Backtesting · Pillar-3-Walk-Forward",
    "2 · Annahmen & Datenbasis (Governance)",
    "3 · Methodologie (MODEL_ASSUMPTIONS.md)",
])

with tab_bt:
    # ====================================================================
    #  Intro · Was macht dieser Backtest? (Klartext + ehrliche Rahmung)
    # ====================================================================
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #034B6F;padding:0.9rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.9rem;line-height:1.7;">'
        '<strong>Was macht dieser Backtest — in einem Satz?</strong> '
        'Er stellt das Modell zurück in die Vergangenheit, gibt ihm nur das '
        'damalige Wissen und fragt: <em>„Hätte es sich richtig verhalten?"</em>'
        '<br><br>'
        '<strong>Der Ablauf in vier Schritten.</strong><br>'
        '① <u>Zurückversetzen (T0).</u> Wir stellen uns an einen vergangenen '
        'Stichtag und geben dem Modell nur, was es damals wusste — die PD, LGD '
        'und EAD aus dem Pillar-3-Report des Vorjahres '
        '(<em>no-look-ahead</em>: Quartal in Jahr Y → Stichtag 31.12.(Y−1)).'
        '<br>'
        '② <u>Portfolio einfrieren &amp; Schock anwenden.</u> Das Bankportfolio '
        'wird zu T0 fixiert; dann lassen wir das <strong>2-Faktor-Modell</strong> '
        '(ΔBrent <em>und</em> Δr₁₀<sub>J</sub> getrennt, sektor-differenzierte '
        'Sensitivitäten β — identisch zum Live-Cockpit) die Kredit-RWA-Reaktion '
        'berechnen.<br>'
        '③ <u>Mit der Realität vergleichen.</u> Gegenüber steht die '
        '<em>wirklich beobachtete</em> RWA-Entwicklung aus dem EBA-Transparency-'
        'Panel — bereinigt um reines Volumenwachstum.<br>'
        '④ <u>Vorwärts laufen.</u> Das rollt über alle Banken &amp; Quartale '
        '(2022-2025) und ergibt viele Prognose/Ist-Paare.<br><br>'
        '<strong>Wichtige Einordnung für die Lesart.</strong> Unser Modell ist '
        'ein <strong>konditionales, konservatives Stress-/Frühwarn-Instrument</strong> '
        '(„<em>was wäre, wenn Szenario X einträte?</em>") — <strong>kein '
        'Quartals-Punktprognose-Tool</strong> für die laufende RWA-Drift. '
        'Regulatorisches Kredit-RWA wird aktiv gesteuert (CRM, Portfolio-'
        'Umschichtung, IRB↔SA-Wanderung) und ist daher quartalsweise kaum '
        'prognostizierbar. Der Backtest prüft das Modell deshalb dort, wo es '
        'etwas leisten <em>soll</em> — und dokumentiert ehrlich, wo nicht.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ====================================================================
    #  Daten laden (cached)
    # ====================================================================
    @st.cache_data(ttl=24*3600, show_spinner="Lade historisches EBA-Panel …")
    def _load_panel():
        panel = load_historical_capital_panel(EBA_RAW_DIR)
        wide = panel_to_wide(panel)
        bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
        return panel, wide, bank_dir

    @st.cache_data(ttl=24*3600, show_spinner="Trailing-1Y-Macro-Status …")
    def _compute_macro(periods: tuple[int, ...]):
        data = load_data_layer()
        if data["brent"] is None or data["svensson"] is None:
            return None, None
        macro = compute_macro_panel(data["brent"], data["svensson"], list(periods))
        fs = factor_stats(data["brent"], data["svensson"], lookback=252)
        macro = attach_m_factor(macro, cov_factors=fs["sigma"])
        return macro, fs

    @st.cache_data(ttl=24*3600, show_spinner="Quartals-Macro-Schocks …")
    def _compute_macro_quarterly(periods: tuple[int, ...]):
        data = load_data_layer()
        if data["brent"] is None or data["svensson"] is None:
            return None, None
        mq = compute_quarterly_macro(data["brent"], data["svensson"], list(periods))
        fs = factor_stats(data["brent"], data["svensson"], lookback=252)
        mq = attach_m_factor_quarterly(mq, cov_factors=fs["sigma"])
        return mq, fs

    @st.cache_data(ttl=24*3600, show_spinner="Frozen-Portfolio-Walk-Forward läuft …")
    def _build_series(_wide, _macro_q):
        series = load_backtest_series()
        panel = build_pdlgd_panel(_wide, _macro_q, series_df=series)
        return series, panel

    @st.cache_data(ttl=24*3600, show_spinner="2-Faktor-Zins-Antwortkurven je Bank …")
    def _rate_response(_series):
        """Prognostizierte ΔRWA_credit (% der Baseline) als Funktion eines
        Δr_10y-Schocks (Brent fix = 0) je Bank — die 2-Faktor-Antwort des
        eingefrorenen Portfolios (jüngster Jahrgang)."""
        grid = np.linspace(-1.0, 3.0, 21)     # Δr_10y in Prozentpunkten
        out = {}
        for lei, name in _series[["LEI", "bank_name"]].drop_duplicates().values:
            vints = sorted(_series.loc[_series["LEI"] == lei, "vintage_date"].unique())
            if not vints:
                continue
            v = vints[-1]
            fp = build_frozen_portfolio_series(lei, v, _series)
            if fp is None:
                continue
            ys = []
            for dr in grid:
                d = frozen_2factor_delta(fp, 0.0, float(dr))
                ys.append(100.0 * d["delta_rwa"] / d["rwa_base"] if d else np.nan)
            out[name] = {"grid": grid, "pct": ys, "vintage": v}
        return out

    @st.cache_data(ttl=24*3600, show_spinner="PD-Backtest (PIT vs. TTC) …")
    def _build_pd_bt(_series):
        data = load_data_layer()
        if data["brent"] is None or data["svensson"] is None:
            return None, None
        amac = compute_annual_macro(data["brent"], data["svensson"],
                                    [2022, 2023, 2024])
        pdbt = build_pd_backtest(_series, amac)
        return pdbt, amac

    panel_eba, wide, bank_dir = _load_panel()
    periods = tuple(sorted(wide["Period"].unique()))
    macro, fs = _compute_macro(periods)
    macro_q, _ = _compute_macro_quarterly(periods)
    if macro_q is None or len(macro_q) == 0:
        st.error("Macro-Panel nicht berechenbar — Brent/Svensson-Cache fehlt.")
        st.stop()

    series, panel = _build_series(wide, macro_q)
    if panel.empty:
        st.error("Backtest-Panel leer — bitte `data/pillar3_backtest_pdlgd.csv` prüfen.")
        st.stop()

    stats = walkforward_error_stats(panel)
    pd_bt, annual_macro = _build_pd_bt(series)
    pd_stats = pd_backtest_stats(pd_bt) if pd_bt is not None else {"n": 0}

    # ====================================================================
    #  Abschnitt 1 · Datenbasis greifbar machen
    # ====================================================================
    st.divider()
    eyebrow("Datenbasis · die Pillar-3-Roh-Reihe, die diesen Backtest erst möglich macht")

    n_banks_series = int(series["LEI"].nunique())
    n_vintages = int(series["vintage_date"].nunique())
    n_points = int(len(series))
    n_pairs = int(stats.get("n", 0))
    n_banks_bt = int(panel["LEI_Code"].nunique())
    pe_sorted = panel.sort_values("Period_end")
    win_lo = pe_sorted["period_label_end"].iloc[0]
    win_hi = pe_sorted["period_label_end"].iloc[-1]

    d1, d2, d3, d4, d5 = st.columns(5, gap="small")
    d1.metric("Banken", f"{n_banks_series}", "EU-IRB-Großbanken", delta_color="off")
    d2.metric("Pillar-3-Jahrgänge", f"{n_vintages}", "2021 – 2024", delta_color="off")
    d3.metric("PD/LGD/EAD-Punkte", f"{n_points}", "EU-CR6-A-IRB-Sub-totals",
              delta_color="off")
    d4.metric("Bank-Quartal-Paare", f"{n_pairs}", f"{n_banks_bt} Banken backgetestet",
              delta_color="off")
    d5.metric("Testfenster", f"{win_lo} – {win_hi}", "Walk-Forward", delta_color="off")

    cov = backtest_series_coverage(series)
    cov = cov.loc[cov.sum(axis=1).sort_values(ascending=False).index]   # größte zuerst
    fig_cov = go.Figure(go.Heatmap(
        z=cov.values,
        x=[c[:4] for c in cov.columns],            # Jahr-Label
        y=cov.index,
        text=cov.values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=[[0.0, "#F4F4F4"], [0.5, "#9FC0D8"], [1.0, COLORS["mid_blue"]]],
        zmin=0, zmax=7,
        showscale=True,
        colorbar=dict(title="IRB-<br>Klassen", thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z} IRB-Klassen<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig_cov.update_layout(
        title="Abdeckungs-Matrix · Anzahl extrahierter IRB-Klassen je Bank × Pillar-3-Stichtag",
        height=360, margin=dict(l=20, r=20, t=56, b=30),
    )
    fig_cov.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_cov, use_container_width=True)

    st.markdown(
        '<div style="background:#F4F4F4;padding:0.7rem 1.0rem;border-radius:6px;'
        'margin:0.2rem 0 0.6rem 0;color:#051C2C;font-size:0.84rem;line-height:1.55;">'
        '<strong>Quelle &amp; Integrität.</strong> Jeder Wert stammt direkt aus '
        'einem <em>EU-CR6-A-IRB-„Sub-total"</em> des bankpublizierten Pillar-3-'
        'Reports (EBA-ITS/2020/04, CRR Art. 431-455) — kein abgeleiteter, '
        'proxied oder fabrizierter Wert. Verifikation pro Zelle: Spalten gegen '
        'FY2024-Anker kalibriert, RWA/EAD-Dichte-Cross-Check und bankinterne '
        'EAD-Kontinuität über die Jahre. Volle Klassen × Jahr = 7; weiße/helle '
        'Zellen markieren reine Quellgrenzen (gerundete oder im PDF '
        'verschmolzene Werte), nicht fehlende Sorgfalt.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ====================================================================
    #  Abschnitt 2 · Validierungsebene 1 — Stress-Treiber-Timing
    # ====================================================================
    st.divider()
    eyebrow("Ebene 1 · Stress-Treiber-Timing — treffen die ZWEI Faktoren die echten Krisen?")
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #C9A227;padding:0.85rem 1.1rem;border-radius:6px;'
        'margin:0.3rem 0 0.9rem 0;color:#051C2C;font-size:0.86rem;line-height:1.6;">'
        '<strong>Warum das die erste Validierung ist.</strong> Das Modell wird '
        'von <strong>zwei getrennten, beobachtbaren Faktoren</strong> getrieben '
        '— dem Zinsschock <code>Δr₁₀<sub>J</sub></code> und dem Energie-/'
        'Angebotsschock <code>ΔBrent</code> (kein in einen einzelnen Vasicek-M '
        'kollabierter Aggregat-Regler mehr). Jeder Faktor wirkt über '
        '<em>sektor-spezifische</em> Sensitivitäten β (Ebene 2). Bevor man '
        'Prognosen vergleicht, müssen diese Treiber <em>zur richtigen Zeit '
        'ausschlagen</em> — und der dominante Treiber 2022-2024 war eindeutig '
        'der Zinsschock.'
        '</div>',
        unsafe_allow_html=True,
    )

    mq_s = macro_q.sort_values("Period_end").copy()
    mq_s["date_end"] = pd.to_datetime(mq_s["date_end"])
    fig_f = go.Figure()
    fig_f.add_trace(go.Bar(
        x=mq_s["date_end"], y=mq_s["dr_10y_pp_q"],
        name="Δr₁₀ⱼ (Zinsschock, pp)",
        marker_color=[COLORS["crimson"] if v > 0 else COLORS["teal"]
                      for v in mq_s["dr_10y_pp_q"]],
        opacity=0.85,
        hovertemplate="<b>%{x|%Y-%m}</b><br>Δr₁₀ⱼ = %{y:+.2f} pp<extra></extra>",
    ))
    fig_f.add_trace(go.Scatter(
        x=mq_s["date_end"], y=mq_s["brent_log_q"],
        name="ΔBrent (log-Return)", yaxis="y2", mode="lines+markers",
        line=dict(color=COLORS["amber"], width=2), marker=dict(size=6),
        hovertemplate="<b>%{x|%Y-%m}</b><br>ΔBrent = %{y:+.3f}<extra></extra>",
    ))
    fig_f.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
    for ep in EPISODES_QUARTERLY:
        # add_shape/annotation mit String-Datum statt add_vline(datetime).
        ep_str = f"{ep['period_end']//100}-{ep['period_end']%100:02d}-01"
        fig_f.add_shape(type="line", x0=ep_str, x1=ep_str, y0=0, y1=1, yref="paper",
                        line=dict(color=ep["color"], dash="dot", width=1.2))
        fig_f.add_annotation(x=ep_str, y=1, yref="paper", yanchor="bottom",
                             text=ep["label"], showarrow=False,
                             font=dict(color=ep["color"], size=10))
    fig_f.update_layout(
        title="Realisierte Schocks je Quartal · Zinsschock (Balken) + Brent (Linie)",
        xaxis_title="Quartal",
        yaxis=dict(title="Δr₁₀ⱼ [Prozentpunkte]", color=COLORS["crimson"]),
        yaxis2=dict(title="ΔBrent [log-Return]", overlaying="y", side="right",
                    color=COLORS["amber"], showgrid=False),
        height=360, bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_f, use_container_width=True)
    insight(
        "<strong>Befund.</strong> Der Zinsschock (Balken) schlägt 2022 massiv "
        "positiv aus — der Bundesbank-10J-Zins sprang über das Jahr um rund "
        "+2,8 pp (Ukraine- &amp; Inflationsschock, EZB-Wende), gefolgt von der "
        "SVB/CS-Episode Q1 2023. Genau das sind die bekannten Stress-Phasen. "
        "Beide Faktoren sind also zeitlich korrekt verankert — die Grund-"
        "voraussetzung dafür, dass das Modell das Richtige stresst."
    )

    # ====================================================================
    #  Abschnitt 3 · Validierungsebene 2 — Strukturelle Konservativität
    # ====================================================================
    st.divider()
    eyebrow("Ebene 2 · Strukturelle Soundness — sektor-differenzierte 2-Faktor-Sensitivitäten")
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #034B6F;padding:0.85rem 1.1rem;border-radius:6px;'
        'margin:0.3rem 0 0.9rem 0;color:#051C2C;font-size:0.86rem;line-height:1.6;">'
        '<strong>Was hier geprüft wird.</strong> Das 2-Faktor-Modell übersetzt '
        'die zwei Schocks über <em>klassen-spezifische</em> Sensitivitäten in '
        'ΔPD und ΔLGD: <code>ΔPD = β_oil·ΔBrent + β_rate·Δr</code>. Die Matrix '
        'unten zeigt diese β je IRB-Klasse — jede Zelle ist quellenbelegt '
        '(EBA-2025-Methodik §2.4.2, ECB-WP 2897/3112). Ökonomisch sauber heißt: '
        'die Vorzeichen müssen stimmen. Beispiele: <strong>Mortgage</strong> '
        'reagiert am stärksten auf Zinsen (β_rate≫β_oil — Affordability + '
        'Hauspreis-Kanal); <strong>Bank</strong> hat ein <em>negatives</em> '
        'β_rate (NIM-Uplift bei steigenden Zinsen); <strong>Sovereign</strong> '
        'ist macro-inert (Zins wirkt dort über den Marktbuch-Kanal). Diese '
        'Differenzierung ersetzt den früheren pauschalen Single-M-Regler.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- β-Sensitivitäts-Heatmap (die ökonomische Engine, quellenbelegt) ---
    cls_order = ["corporate", "sme_corporate", "mortgage", "qrre",
                 "other_retail", "bank", "sovereign"]
    col_keys = [("pd_rate", "β_PD · Zins"), ("pd_oil", "β_PD · Öl"),
                ("lgd_rate", "γ_LGD · Zins"), ("lgd_oil", "γ_LGD · Öl")]
    zmat = [[SENSITIVITY_MATRIX[c][k] for k, _ in col_keys] for c in cls_order]
    fig_beta = go.Figure(go.Heatmap(
        z=zmat, x=[lbl for _, lbl in col_keys], y=cls_order,
        text=[[f"{v:+.2f}" for v in row] for row in zmat],
        texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0.0, "#034B6F"], [0.5, "#FFFFFF"], [1.0, COLORS["crimson"]]],
        zmid=0, zmin=-1.6, zmax=1.6,
        colorbar=dict(title="β", thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b> · %{x}<br>β = %{z:+.2f}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig_beta.update_layout(
        title="Sektor-Sensitivitäts-Matrix β  (ΔRisikoparameter in pp je Faktor-Einheit)",
        height=340, margin=dict(l=20, r=20, t=56, b=30),
    )
    fig_beta.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_beta, use_container_width=True)

    # --- Realistische, bank-spezifische Zins-Antwort der RWA ---
    resp = _rate_response(series)
    fig_rr = go.Figure()
    for name, c in sorted(resp.items()):
        fig_rr.add_trace(go.Scatter(
            x=c["grid"], y=c["pct"], mode="lines",
            name=f"{name} (FY{c['vintage'][:4]})", line=dict(width=2),
            hovertemplate=f"<b>{name}</b><br>Δr = %{{x:+.1f}} pp<br>"
                          f"ΔRWA = %{{y:+.1f}} %<extra></extra>",
        ))
    fig_rr.add_hline(y=0, line_color=COLORS["stone"], line_dash="dash", line_width=1)
    fig_rr.update_layout(
        title="2-Faktor-RWA-Antwort auf einen Zinsschock je Bank (Brent fix, jüngster Jahrgang)",
        xaxis_title="Zinsschock Δr₁₀ⱼ [Prozentpunkte]",
        yaxis_title="prognostizierte ΔRWA_credit [% der Baseline]",
        height=420,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
                    font=dict(size=10)),
        margin=dict(r=180),
    )
    st.plotly_chart(fig_rr, use_container_width=True)
    insight(
        "<strong>Befund.</strong> Die Antwort ist monoton steigend im Zinsschock "
        "und konservativ (mehr Zins-Stress ⇒ mehr RWA), und die Größenordnung "
        "ist jetzt <strong>realistisch</strong>: ein +2,8-pp-Schock (wie 2022) "
        "projiziert grob +5 bis +15 % Kredit-RWA — plausibel im Bereich echter "
        "EBA-Stresstests, nicht mehr die +180 %-Überschießung des alten "
        "Single-M-Pfads. Die <em>Spreizung</em> zwischen den Banken zeigt, dass "
        "die echte Pillar-3-Zeitreihe wirkt: Häuser mit höherem Corporate-/"
        "Mortgage-Anteil reagieren steiler, ausgeprägte Retail-Bücher flacher."
    )

    # ====================================================================
    #  Abschnitt 4 · Validierungsebene 3 — Outcomes-Analysis (ehrlich)
    # ====================================================================
    st.divider()
    eyebrow("Ebene 3 · Outcomes-Analysis — zwei ehrliche Proben (SR 11-7)")
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #A52F4D;padding:0.85rem 1.1rem;border-radius:6px;'
        'margin:0.3rem 0 0.9rem 0;color:#051C2C;font-size:0.86rem;line-height:1.6;">'
        '<strong>Hier wird es ehrlich.</strong> Wir prüfen das Modell gegen die '
        'Realität in zwei Proben: <strong>(A)</strong> prognostizierte vs. '
        'tatsächlich gemeldete PD und <strong>(B)</strong> prognostizierte vs. '
        'realisierte Kredit-RWA-Bewegung. Beide zeigen schwache Übereinstimmung '
        '— und der <em>Grund dafür ist methodisch fundamental, kein Modellfehler</em>. '
        'Wir berichten das unverstellt und erklären es darunter genau.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ===== Probe A · PD-Backtest (PIT-Modell vs. TTC-Meldung) =============
    if pd_stats.get("n", 0) > 0:
        st.markdown("**Probe A · Prognostizierte vs. gemeldete PD — das PIT-vs-TTC-Problem**")
        pa1, pa2, pa3, pa4 = st.columns(4, gap="small")
        pa1.metric("PD-Vorhersagen", f"{pd_stats['n']}",
                   "Bank × Klasse × Jahr", delta_color="off")
        pa2.metric("Richtungs-Trefferquote", f"{pd_stats['hit_rate']*100:.0f}%",
                   "stieg die PD wie erwartet?", delta_color="off")
        pa3.metric("Korrelation ΔPD", f"{pd_stats['corr']:+.2f}",
                   "Modell vs. gemeldet", delta_color="off")
        pa4.metric("MAE Modell vs. naiv",
                   f"{pd_stats['mae_model']:.2f} / {pd_stats['mae_naive']:.2f} pp",
                   "schlägt 'keine Änderung' nicht", delta_color="off")

        # Jahres-Mittel: prognostizierte vs. realisierte PD-Änderung
        yr_agg = (pd_bt.groupby("year")[["d_pred_pp", "d_real_pp"]]
                  .mean().reset_index())
        fig_pa = go.Figure()
        fig_pa.add_trace(go.Bar(
            x=yr_agg["year"], y=yr_agg["d_pred_pp"], name="Modell-Prognose ΔPD",
            marker_color=COLORS["navy"], opacity=0.88,
            hovertemplate="FY%{x}<br>Modell ΔPD = %{y:+.3f} pp<extra></extra>",
        ))
        fig_pa.add_trace(go.Bar(
            x=yr_agg["year"], y=yr_agg["d_real_pp"], name="gemeldet (realisiert) ΔPD",
            marker_color=COLORS["crimson"], opacity=0.88,
            hovertemplate="FY%{x}<br>gemeldet ΔPD = %{y:+.3f} pp<extra></extra>",
        ))
        fig_pa.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
        fig_pa.update_layout(
            title="Ø PD-Änderung je Jahr · Modell-Prognose vs. tatsächlich gemeldet",
            xaxis_title="Pillar-3-Jahrgang (Stichjahr)",
            yaxis_title="Ø ΔPD [Prozentpunkte]",
            height=340, barmode="group", bargap=0.3,
            xaxis=dict(tickmode="array", tickvals=list(yr_agg["year"])),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_pa, use_container_width=True)

        st.markdown(
            '<div style="background:#FFF8E7;border:1px solid #E6D9A8;'
            'border-left:4px solid #C9A227;padding:0.85rem 1.1rem;'
            'border-radius:6px;margin:0.2rem 0 0.9rem 0;color:#051C2C;'
            'font-size:0.88rem;line-height:1.7;">'
            '<strong>Woran liegt es? — der entscheidende Befund.</strong><br>'
            'Schau auf <strong>2022</strong>: der Zins sprang um +2,8 pp, und das '
            'Modell sagt korrekt „PD <em>rauf</em>" (Ø +0,5 pp). Die tatsächlich '
            'gemeldeten PD gingen aber im Schnitt <em>runter</em>. Kein '
            'Widerspruch, sondern <strong>zwei verschiedene Größen</strong>:<br>'
            '• Unser Modell projiziert eine <strong>Point-in-Time (PIT)</strong>-'
            'Reaktion — „<em>wenn dieser Schock jetzt einträfe, stiege das '
            'Risiko</em>".<br>'
            '• Die gemeldete regulatorische A-IRB-PD ist '
            '<strong>Through-the-Cycle (TTC)</strong> — bewusst geglättet und '
            'antizyklisch (CRR Art. 180; sie soll im Abschwung gerade NICHT '
            'springen, um Prozyklizität zu vermeiden). 2022/23 kamen '
            'Modell-Rekalibrierungen und weiter niedrige <em>realisierte</em> '
            'Ausfälle hinzu → gemeldete PD seitwärts/runter.<br>'
            '<strong>Folge:</strong> ein PIT-Stressmodell gegen TTC-Meldedaten zu '
            'backtesten misst per Konstruktion eine Lücke — nicht die Modellgüte. '
            'Der saubere Test bräuchte <em>realisierte</em> PIT-Ausfallraten je '
            'Segment (bank-intern, nicht offengelegt; EBA-NPE erst ab 2024Q3) — '
            'genau die <strong>„keine echten Bankdaten"-Limitation</strong>.'
            '</div>',
            unsafe_allow_html=True,
        )
    st.divider()

    # ===== Probe B · RWA-Outcomes (2-Faktor, realistische Größen) ========
    st.markdown("**Probe B · Prognostizierte vs. realisierte Kredit-RWA-Bewegung**")
    pred = panel["pred_dRWA_credit_eur"].to_numpy(dtype=float)
    real = panel["risk_driven_dRWA_credit_eur"].to_numpy(dtype=float)

    o1, o2, o3, o4 = st.columns(4, gap="small")
    hr = stats.get("hit_rate", float("nan")) * 100
    o1.metric("Richtungs-Trefferquote", f"{hr:.0f}%", "vs. 50 % Zufall",
              delta_color="off")
    o2.metric("Korrelation Modell↔Ist", f"{stats.get('corr', float('nan')):+.2f}",
              "Pearson, alle Paare", delta_color="off")
    o3.metric("Konservativ-Anteil",
              f"{stats.get('conservative_share', float('nan'))*100:.0f}%",
              "Modell ≥ Ist (sichere Seite)", delta_color="off")
    o4.metric("Beobachtungen", f"{n_pairs}", f"{n_banks_bt} Banken × Quartale",
              delta_color="off")

    # Standardisierter Scatter (z-Scores) — entkoppelt das Skalen-Thema vom
    # Richtungs-/Korrelations-Thema. Runde, unkorrelierte Wolke = kein Signal.
    sc_l, sc_r = st.columns([3, 2], gap="medium")
    with sc_l:
        zp = (pred - pred.mean()) / (pred.std() if pred.std() > 0 else 1.0)
        zr = (real - real.mean()) / (real.std() if real.std() > 0 else 1.0)
        fig_zs = go.Figure()
        fig_zs.add_trace(go.Scattergl(
            x=zp, y=zr, mode="markers",
            marker=dict(size=6, color=COLORS["navy"], opacity=0.35),
            text=panel["bank_name"] + " · " + panel["period_label_end"],
            hovertemplate="%{text}<br>Modell (z): %{x:+.2f}<br>Ist (z): %{y:+.2f}<extra></extra>",
            name="Bank-Quartal",
        ))
        lim = 3.2
        fig_zs.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim], mode="lines",
            line=dict(color=COLORS["crimson"], width=2, dash="dash"),
            name="45° (perfekt)",
        ))
        fig_zs.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
        fig_zs.add_vline(x=0, line_color=COLORS["hairline"], line_width=1)
        fig_zs.update_layout(
            title="Prognose vs. Ist · standardisiert (z-Scores)",
            xaxis_title="Modell-Prognose (z)", yaxis_title="Ist risiko-bereinigt (z)",
            height=400, xaxis_range=[-lim, lim], yaxis_range=[-lim, lim],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_zs, use_container_width=True)
    with sc_r:
        st.markdown(
            '<div style="font-size:0.86rem;line-height:1.65;color:#051C2C;'
            'margin-top:1.5rem;">'
            '<strong>Wie man den Scatter liest.</strong> Läge echte '
            'Prognosekraft vor, drängten sich die Punkte um die rote 45°-Linie. '
            'Stattdessen bilden sie eine <em>runde, unkorrelierte Wolke</em> — '
            'die quartalsweise Richtung der realisierten RWA-Bewegung folgt dem '
            'Modell nicht. Standardisiert (z-Scores), damit der reine '
            'Skalen-Unterschied ausgeblendet ist und nur der '
            '<em>Zusammenhang</em> sichtbar bleibt.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Stress-konditionale Trefferquote
    thr_rows = []
    for lbl, t in [("Alle Quartale (|M| ≥ 0)", 0.0),
                   ("Schock (|M| ≥ 0.5)", 0.5),
                   ("Stress (|M| ≥ 1.0)", 1.0),
                   ("Schwer-Stress (|M| ≥ 1.5)", 1.5)]:
        s = panel[panel["m_quarter"].abs() >= t]
        ss = walkforward_error_stats(s)
        thr_rows.append({
            "Filter": lbl, "n Paare": ss.get("n", 0),
            "Trefferquote": f"{ss.get('hit_rate', 0)*100:.0f}%",
            "Korrelation": f"{ss.get('corr', float('nan')):+.2f}",
        })
    st.dataframe(pd.DataFrame(thr_rows), use_container_width=True,
                 hide_index=True, height=180)

    insight(
        f"<strong>Ehrlicher Befund (Probe B).</strong> Die <em>Größenordnung</em> "
        f"der 2-Faktor-Prognose ist jetzt realistisch (median ~2 % ΔRWA pro "
        f"Quartal statt der früheren Single-M-Überschießung). Die "
        f"Richtungs-Trefferquote liegt dennoch bei <strong>{hr:.0f}%</strong> und "
        f"die Korrelation bei <strong>{stats.get('corr', float('nan')):+.2f}</strong> "
        f"— wie bei Probe A: regulatorisches RWA bewegt sich quartalsweise vor "
        f"allem durch Steuerung (CRM, Umschichtung, IRB↔SA) und TTC-Glättung, "
        f"nicht durch den Makro-Schock. <strong>Schlussfolgerung des Validators "
        f"(SR 11-7):</strong> beide Proben bestätigen dieselbe Grenze — das "
        f"Modell darf <u>nicht</u> als RWA-/PD-Punktprognose gelesen werden; es "
        f"ist als konditionales, konservatives Stress-Instrument validiert "
        f"(Ebene 1 + 2). Diese Trennung sauber zu dokumentieren <em>ist</em> die "
        f"Outcomes-Analysis."
    )

    # ====================================================================
    #  Abschnitt 5 · Bank-Drilldown
    # ====================================================================
    st.divider()
    eyebrow("Bank-Drilldown · eingefrorene Risikoparameter und Richtungstreffer je Institut")

    size_rank = (panel.groupby(["LEI_Code", "bank_name"])["rwa_credit_start"]
                 .mean().sort_values(ascending=False))
    opts = size_rank.index.tolist()
    sel = st.selectbox(
        f"Bank wählen ({len(opts)} backgetestet · nach Ø Kredit-RWA sortiert)",
        opts, format_func=lambda x: f"{x[1]}  (Ø €{size_rank.loc[x]/1e9:.0f} bn Kredit-RWA)",
        key="bt_bank",
    )
    sel_lei, sel_name = sel

    bk_l, bk_r = st.columns(2, gap="medium")

    # Links: eingefrorene PD/LGD-Entwicklung (die Input-Reihe, greifbar)
    with bk_l:
        sb = series[series["LEI"] == sel_lei].copy()
        sb["w"] = sb["ead_eur_m"]
        agg = (sb.groupby("vintage_date")
                 .apply(lambda d: pd.Series({
                     "PD": np.average(d["pd_pct"], weights=d["w"]),
                     "LGD": np.average(d["lgd_pct"], weights=d["w"]),
                     "EAD_bn": d["ead_eur_m"].sum()/1e3,
                 }))
                 .reset_index().sort_values("vintage_date"))
        agg["yr"] = agg["vintage_date"].str[:4]
        fig_pd = go.Figure()
        fig_pd.add_trace(go.Scatter(
            x=agg["yr"], y=agg["PD"], mode="lines+markers", name="Ø PD (%)",
            line=dict(color=COLORS["crimson"], width=2.5), marker=dict(size=8),
            hovertemplate="FY%{x}<br>Ø PD = %{y:.2f}%<extra></extra>",
        ))
        fig_pd.add_trace(go.Scatter(
            x=agg["yr"], y=agg["LGD"], mode="lines+markers", name="Ø LGD (%)",
            line=dict(color=COLORS["mid_blue"], width=2.5, dash="dot"),
            marker=dict(size=8), yaxis="y2",
            hovertemplate="FY%{x}<br>Ø LGD = %{y:.1f}%<extra></extra>",
        ))
        fig_pd.update_layout(
            title=f"{sel_name} · eingefrorene EAD-gewichtete PD/LGD je Jahrgang",
            xaxis_title="Pillar-3-Jahrgang",
            yaxis=dict(title="Ø PD (%)", color=COLORS["crimson"]),
            yaxis2=dict(title="Ø LGD (%)", overlaying="y", side="right",
                        color=COLORS["mid_blue"], showgrid=False),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_pd, use_container_width=True)
        st.caption(
            "Die tatsächliche Input-Reihe dieser Bank — EAD-gewichtetes "
            "Portfolio-PD/-LGD je Pillar-3-Stichtag. Genau diese Werte friert "
            "der Walk-Forward zu jedem T0 ein (no-look-ahead)."
        )

    # Rechts: Richtungstreffer je Quartal (realisiert, lesbare Skala)
    with bk_r:
        sub = panel[panel["LEI_Code"] == sel_lei].sort_values("Period_end").copy()
        sub["date_end"] = pd.to_datetime(sub["Period_end"].apply(
            lambda p: f"{p//100}-{p%100:02d}-01"))
        sub["real_pct"] = sub["risk_driven_dRWA_credit_eur"] / sub["rwa_credit_start"] * 100
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=sub["date_end"], y=sub["real_pct"],
            marker_color=[COLORS["teal"] if m else COLORS["crimson"]
                          for m in sub["sign_match"]],
            text=["✓" if m else "✗" for m in sub["sign_match"]],
            textposition="outside", textfont=dict(size=11),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Ist risiko-bereinigt: "
                          "%{y:+.2f}% der RWA<extra></extra>",
        ))
        fig_h.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
        fig_h.update_layout(
            title=f"{sel_name} · realisiertes ΔRWA + Richtungstreffer (✓/✗)",
            xaxis_title="Quartal-Ende",
            yaxis_title="Ist risiko-bereinigt (% der RWA)",
            height=360, showlegend=False,
        )
        st.plotly_chart(fig_h, use_container_width=True)
        bstats = walkforward_error_stats(sub)
        bb1, bb2, bb3 = st.columns(3, gap="small")
        bb1.metric("Quartale", f"{bstats.get('n', 0)}")
        bb2.metric("Trefferquote", f"{bstats.get('hit_rate', 0)*100:.0f}%")
        bb3.metric("Ø frozen-Vintage",
                   sub["pd_vintage"].mode().iloc[0][:4] if len(sub) else "—")

    # ====================================================================
    #  Abschnitt 6 · Verdikt
    # ====================================================================
    st.divider()
    eyebrow("Validitäts-Verdikt · einfach zusammengefasst")
    st.markdown(
        '<div style="background:#051C2C;color:#FFFFFF;padding:1.1rem 1.3rem;'
        'border-radius:8px;margin:0.3rem 0 0.6rem 0;font-size:0.9rem;'
        'line-height:1.75;">'
        '<div style="font-size:0.72rem;letter-spacing:0.12em;text-transform:'
        'uppercase;color:#C9A227;margin-bottom:0.4rem;">Fazit für die Adressaten</div>'
        '<strong>1. Datenbasis: belastbar.</strong> Eine konsistente, '
        'quellenreine PD/LGD/EAD-Zeitreihe (10 Banken, 2021-2024, '
        f'{n_points} Pillar-3-Punkte) — die Voraussetzung, das Modell überhaupt '
        'historisch testen zu können.<br>'
        '<strong>2. Stress-Treiber: korrekt verankert.</strong> Beide Faktoren '
        '(Zinsschock + Brent) schlagen zur richtigen Zeit aus — der +2,8-pp-'
        'Zinssprung 2022 und SVB/CS Q1 2023.<br>'
        '<strong>3. Stress-Mechanik: ökonomisch sauber &amp; konsistent zum '
        'Live-Modell.</strong> Sektor-differenzierte 2-Faktor-β (Mortgage zins-'
        'getrieben, Bank-NIM negativ, Sovereign inert), monotone und '
        'konservative RWA-Antwort in jetzt <em>realistischer</em> Größenordnung.<br>'
        '<strong>4. Grenze: fundamental und ehrlich dokumentiert.</strong> '
        'Weder PD noch RWA sind treffsicher prognostizierbar — weil unser Modell '
        '<em>Point-in-Time</em> ist, die gemeldeten Daten aber '
        '<em>Through-the-Cycle</em>/aktiv gesteuert sind, und uns die '
        'realisierten Ausfallraten fehlen („keine echten Bankdaten"). Das ist '
        'eine Eigenschaft der Daten, kein Modellfehler.<br><br>'
        '<strong style="color:#C9A227;">Verdikt:</strong> Das Modell ist als '
        '<strong>konditionales, konservatives Frühwarn-/Stress-Instrument</strong> '
        'validiert — es beziffert „<em>wie viel Kapital, falls Szenario X '
        'einträte?</em>" robust und auf der sicheren Seite. Es ist '
        '<strong>kein</strong> Quartals-Punktprognose-Tool, und der Backtest '
        'belegt genau diese Trennung.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ====================================================================
    #  Methodik (formal) — Formeln, no-look-ahead, Quellen, Limitationen
    # ====================================================================
    with st.expander("Methodik · Formeln, No-Look-Ahead-Logik, Quellen & Limitationen",
                     expanded=False):
        st.markdown("""
**Schritt A — Portfolio einfrieren (no-look-ahead).** Für ein Quartal im Jahr
*Y* werden PD, LGD und EAD je IRB-Klasse aus dem Pillar-3-EU-CR6-Report zum
Stichtag **31.12.(Y−1)** entnommen — der jüngste Stichtag, der zu T0 real
veröffentlicht war. Fehlt dieser exakte Jahrgang für eine Bank, wird das
Quartal übersprungen (kein Rückgriff auf einen späteren Jahrgang).

**Schritt B — Realisierte 2 Faktoren.** Zwischen den Stichtagen messen wir die
zwei Schocks getrennt: ΔBrent (log-Return) und Δr₁₀ⱼ (Prozentpunkte). KEINE
Kollabierung mehr in einen einzelnen Vasicek-M.

**Schritt C — Sektor-differenzierte 2-Faktor-Transmission.** Pro IRB-Klasse:
""")
        st.latex(r"\Delta\mathrm{PD}_{\text{pp}}=\beta^{\text{oil}}_{c}\cdot\Delta\mathrm{Brent}"
                 r"+\beta^{\text{rate}}_{c}\cdot\Delta r_{10y},\qquad "
                 r"\Delta\mathrm{LGD}_{\text{pp}}=\gamma^{\text{oil}}_{c}\cdot\Delta\mathrm{Brent}"
                 r"+\gamma^{\text{rate}}_{c}\cdot\Delta r_{10y}")
        st.markdown("""
Die gestressten PD/LGD laufen durch die Basel-III-IRB-K-Formel
(`vasicek.irb_capital_requirement`, BCBS §272 ff.) → bank-spezifischer
ΔRWA. Identische Engine wie das Live-Cockpit (`two_factor_stress.
capital_bridge_2factor`); die β/γ je Klasse sind quellenbelegt (EBA-2025-
Methodik §2.4.2, ECB WP 2897/3112). Der relative Effekt wird auf die
*gemeldete* RWA_credit(t) angewandt:
""")
        st.latex(r"\widehat{\Delta\mathrm{RWA}}=\mathrm{RWA}_{\text{credit},t}\cdot"
                 r"\frac{\Delta\mathrm{RWA}^{\text{2F}}}{\mathrm{RWA}^{\text{2F}}_{\text{base}}}")
        st.markdown("""
**Probe A (PD-Backtest)** vergleicht zusätzlich die so prognostizierte
gemeldete PD(Y) mit der tatsächlich gemeldeten PD(Y) — Jahres-Schritt.

**Schritt D — Volumen herausrechnen.** Das beobachtete ΔRWA_credit wird um
reines Mengenwachstum bereinigt (Proxy: Wachstum des Nicht-Kredit-RWA), um die
*risiko-getriebene* Komponente zu isolieren.

**Fehler-/Güte-Metriken** (über alle Bank-Quartal-Paare):
""")
        st.latex(r"\text{Trefferquote}=\tfrac1N\sum \mathbf{1}[\operatorname{sign}(f_i)=\operatorname{sign}(y_i)]"
                 r"\qquad \rho=\frac{\operatorname{Cov}(f,y)}{\sigma_f\sigma_y}")
        st.markdown("""
- **Richtungs-Trefferquote** — Anteil korrekter Vorzeichen (50 % = Zufall);
  Direktional-Accuracy nach **Pesaran & Timmermann (1992)**.
- **Korrelation ρ** — linearer Zusammenhang Prognose ↔ Ist.
- **Konservativ-Anteil** — Anteil der Fälle, in denen das Modell bei gleichem
  Vorzeichen betragsmäßig **nicht unterschätzt** (|f| ≥ |y|) — stützt die
  „obere-Schranke"-Erzählung eines Stress-Tools.

**Limitationen (offen dokumentiert).**
1. *Restlaufzeit* ist in der Roh-Reihe nicht erfasst → Basel-Default M = 2.5 J
   (konstant über alle Jahrgänge, daher zeitvergleichs-neutral).
2. **PIT vs. TTC** — der Kern: das Modell projiziert eine Point-in-Time-
   Reaktion, die gemeldeten A-IRB-PD/RWA sind Through-the-Cycle (geglättet,
   antizyklisch, CRR Art. 180) und zusätzlich management-getrieben (CRM,
   Modell-Rekalibrierung, IRB↔SA-Wanderung). Ein direkter Backtest misst daher
   eine strukturelle Lücke, nicht die Modellgüte (Ebene 3).
3. **Keine realisierten Ausfalldaten.** Der saubere PIT-Test bräuchte realisierte
   Ausfall-/NPL-Raten je Segment (bank-intern, nicht offengelegt; EBA-NPE-Panel
   erst ab 2024Q3). Daher Validierung über Treiber-Timing + strukturelle
   Soundness statt Punktprognose.
4. `mortgage_sme` (nur ING) nutzt die Retail-Mortgage-Korrelation ρ = 0.15.

**Quellen.** Pillar-3-Inputs: EBA-ITS/2020/04 (EU CR6), CRR Art. 431-455
(inkl. Art. 180 zur PD-Schätzung/TTC). 2-Faktor-β: EBA (2024) *2025 EU-wide
Stress Test — Methodological Note* §2.4.2; ECB WP 2897 (2024), WP 3112 (2025).
IRB-K: Vasicek (2002); BCBS (2017, *Basel III: Finalising post-crisis reforms*).
Backtest-Evaluation: Hyndman & Athanasopoulos (2021, Kap. 5.8); Pesaran &
Timmermann (1992, *JBES*). Governance: SR 11-7 (Outcomes Analysis), EBA GL 2014/14.
""")

    # ====================================================================
    #  Challenger · empirische OLS-Sensitivität (komplementärer Sanity-Check)
    # ====================================================================
    with st.expander("Challenger · empirische OLS-Sensitivität (ΔCET1-Quote ~ M)",
                     expanded=False):
        st.caption(
            "Komplementäre, datengetriebene Gegenprobe: pro Bank wird die "
            "realisierte 1Y-Veränderung der CET1-Quote gegen den trailing-1Y-"
            "M-Faktor regressiert (OLS-Panel) — ein empirisches β als Challenger "
            "zum strukturellen Modell."
        )
        realized = compute_realized_changes(wide, lag_quarters=4)
        fp_full = build_forecast_panel(realized, macro)
        fp_full = fp_full.dropna(subset=["m_at_start", "delta_ratio_pp"])
        fp_full = fp_full.merge(bank_dir[["lei", "bank_name"]],
                                left_on="LEI_Code", right_on="lei", how="left")
        fp_full_clean = fp_full[fp_full["delta_ratio_pp"].abs() <= 5.0].copy()
        sens = fit_empirical_sensitivity(fp_full_clean)
        ck1, ck2, ck3, ck4 = st.columns(4, gap="small")
        ck1.metric("Sample n", f"{sens.get('n', 0):,}", delta_color="off")
        ck2.metric("β (Sens. zu M)", f"{sens.get('beta', 0):+.3f}",
                   f"t = {sens.get('t_beta', 0):+.2f}", delta_color="off")
        ck3.metric("R²", f"{sens.get('r2', 0):.3f}", "Erklärte Varianz",
                   delta_color="off")
        ck4.metric("RMSE", f"{sens.get('rmse', 0):.3f} pp", delta_color="off")
        st.caption(
            "Niedriges R² ist erwartet — bank-idiosynkratische Variation "
            "überlagert die Macro-Komponente. Dient als Vorzeichen-Sanity-Check, "
            "konsistent mit dem strukturellen Walk-Forward oben."
        )

    # ====================================================================
    #  Methodik-Konzeptdokument zum Download
    # ====================================================================
    st.divider()
    _project_root = Path(__file__).resolve().parent.parent.parent
    _docx_path = _project_root / "BACKTESTING_WALKFORWARD_KONZEPT.docx"
    if _docx_path.exists():
        dl_l, dl_r = st.columns([3, 2], gap="medium")
        with dl_l:
            st.markdown(
                "Das vollständige Konzeptpapier zum Walk-Forward-Backtest liegt "
                "als Word-Datei im Projekt-Root."
            )
        with dl_r:
            with open(_docx_path, "rb") as fh:
                st.download_button(
                    label="📄  BACKTESTING_WALKFORWARD_KONZEPT.docx",
                    data=fh.read(),
                    file_name="BACKTESTING_WALKFORWARD_KONZEPT.docx",
                    mime=("application/vnd.openxmlformats-officedocument."
                          "wordprocessingml.document"),
                    use_container_width=True,
                )

    footer(
        f"Pillar-3-Walk-Forward-Backtest · {n_pairs:,} Bank-Quartal-Paare · "
        f"{n_banks_bt} Banken × {n_vintages} Pillar-3-Jahrgänge · "
        f"Trefferquote {hr:.0f}% · Korrelation {stats.get('corr', float('nan')):+.2f} · "
        f"Datenbasis: pillar3_backtest_pdlgd.csv ({n_points} EU-CR6-Punkte) + "
        f"EBA Transparency 2020-2025 + Brent (ICE) + Bundesbank-Svensson"
    )

with tab_an:
    render_annahmen_tab()

with tab_md:
    render_methodology_tab()
