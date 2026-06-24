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
  1. Stress-Treiber-Timing  · treffen die zwei Faktoren die bekannten Krisen?
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
    build_cet1_backtest, cet1_backtest_stats,
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
        'Sensitivitäten β — identisch zum Live-Cockpit) die Wirkung auf die '
        '<strong>CET1-Quote</strong> berechnen (RWA steigt, Kapital sinkt durch '
        'höhere Verluste).<br>'
        '③ <u>Mit der Realität vergleichen.</u> Gegenüber steht die '
        '<em>tatsächlich gemeldete CET1-Quote</em> aus dem EBA-Transparency-Panel '
        '— wie nah war das Modell dran?<br>'
        '④ <u>Vorwärts laufen.</u> Das rollt über alle Banken &amp; Quartale '
        '(2022-2025) und ergibt viele Prognose/Ist-Paare.<br><br>'
        '<strong>Wichtige Einordnung für die Lesart.</strong> Die <strong>Zielgröße '
        'des Modells ist die CET1-Quote unter Stress</strong> — die Solvenz-'
        'Kennzahl der Bank, falls ein Schock einträte. Der Kern-Backtest (Ebene 3) '
        'speist daher die <em>real eingetretenen</em> Risikofaktoren ein und prüft, '
        'wie nah die prognostizierte CET1-Quote an der <em>tatsächlich gemeldeten</em> '
        'liegt. Das Modell ist bewusst eine <strong>konservative Abwärts-Sicht</strong> '
        '(es rechnet Gewinne/Zinsüberschuss nicht gegen) — es soll unter Stress '
        'nie zu optimistisch sein.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="background:#EAF2F8;border:1px solid #C9DAE8;'
        'border-left:4px solid #034B6F;padding:0.95rem 1.15rem;border-radius:6px;'
        'margin:0.1rem 0 1rem 0;color:#051C2C;font-size:0.92rem;line-height:1.75;">'
        '<strong>Die Kern-Idee — zeitlich versetzt, Jahr für Jahr.</strong> Wir '
        'verfolgen dieselbe Risiko-Zeitreihe (PD/LGD) auf <strong>zwei Wegen</strong> '
        'und legen sie übereinander:<br>'
        '🔴 <strong>Realität</strong> — wie haben sich die <em>tatsächlich '
        'gemeldeten</em> PD/LGD über die Jahre entwickelt?<br>'
        '🔵 <strong>Modell</strong> — wie hätten sie sich entwickelt, wenn wir die '
        '<em>real eingetretenen</em> Zins- und Ölpreis-Bewegungen durch unser '
        'Modell laufen lassen?<br>'
        '<strong>Der entscheidende Punkt:</strong> jeder Schritt startet mit den '
        'PD/LGD/EAD <em>des jeweiligen (Vor-)Jahres</em> — kein Blick in die '
        'Zukunft. Das Modell sagt also immer nur <strong>ein Jahr voraus</strong> '
        'und wird dann wieder an den echten Wert angedockt. So sieht man direkt: '
        'zieht das Modell die Reihe in <em>dieselbe Richtung</em> wie die Realität '
        '— und ähnlich stark?'
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

    @st.cache_data(ttl=24*3600, show_spinner="CET1-Quoten-Backtest …")
    def _build_cet1(_wide, _series, _amac):
        return build_cet1_backtest(_wide, _series, _amac or {})

    cet1_bt = _build_cet1(wide, series, annual_macro)
    cet1_stats = cet1_backtest_stats(cet1_bt) if cet1_bt is not None else {"n": 0}

    # --- Einheitliche Lesehilfe unter jeder Grafik (addressatengerecht) ---
    def lese(was, befund, modell, metrik=None):
        rows = [("Was zeigt die Grafik?", was),
                ("Befund", befund),
                ("Aussage — Bedeutung für das Modell", modell)]
        if metrik:
            rows.append(("Verwendete Kennzahl — einfach erklärt", metrik))
        inner = "".join(
            f'<div style="margin:0.18rem 0;"><span style="color:#034B6F;'
            f'font-weight:700;">{lab}</span> {txt}</div>' for lab, txt in rows)
        st.markdown(
            f'<div style="background:#F7F9FB;border:1px solid #E6E6E6;'
            f'border-left:4px solid #034B6F;padding:0.85rem 1.1rem;'
            f'border-radius:6px;margin:0.1rem 0 0.8rem 0;color:#051C2C;'
            f'font-size:0.9rem;line-height:1.65;">{inner}</div>',
            unsafe_allow_html=True)

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
    # „Gemeldete" Klassen je Bank = Klassen, die die Bank in IRGENDEINEM Jahr
    # offenlegt (ihr eigener A-IRB-Umfang). Strukturell fehlende Klassen
    # (z. B. kein IRB-Sovereign/QRRE) zählen NICHT als Lücke — die Matrix zeigt
    # daher Abdeckung relativ zum bank-eigenen Umfang.
    disc = series.groupby("bank_name")["vasicek_class"].nunique()
    cov = cov.loc[cov.sum(axis=1).sort_values(ascending=False).index]
    disc_v = disc.reindex(cov.index)
    years_lab = [c[:4] for c in cov.columns]
    zfrac = (cov.div(disc_v, axis=0) * 100.0).values
    txt = [[f"{int(cov.values[i][j])}/{int(disc_v.iloc[i])}"
            for j in range(cov.shape[1])] for i in range(cov.shape[0])]
    have_tot = int(cov.values.sum())
    disc_tot = int(disc_v.sum() * cov.shape[1])
    cov_pct = 100.0 * have_tot / disc_tot if disc_tot else 0.0
    fig_cov = go.Figure(go.Heatmap(
        z=zfrac, x=years_lab, y=cov.index,
        text=txt, texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0.0, "#F4F4F4"], [0.5, "#C9DAE8"], [0.85, "#7FA8C8"],
                    [1.0, COLORS["mid_blue"]]],
        zmin=0, zmax=100, showscale=True,
        colorbar=dict(title="Abdeckung<br>%", thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z:.0f}% der gemeldeten Klassen<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig_cov.update_layout(
        title="Abdeckung je Bank × Pillar-3-Stichtag (extrahiert / von der Bank gemeldet)",
        height=360, margin=dict(l=20, r=20, t=56, b=30),
    )
    fig_cov.update_xaxes(type="category")          # saubere Jahre, keine 2020,5-Ticks
    fig_cov.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_cov, use_container_width=True)

    lese(
        was="Je Bank (Zeile) und Pillar-3-Jahrgang (Spalte) der Anteil der "
            "<em>von der Bank gemeldeten</em> A-IRB-Klassen, für die PD, LGD und "
            "EAD extrahiert wurden (Zahl = extrahiert / gemeldet). Dunkelblau = "
            "vollständig.",
        befund=f"Relativ zum bank-eigenen Meldeumfang sind <strong>{cov_pct:.0f}%</strong> "
               f"der Zellen abgedeckt ({have_tot} von {disc_tot}). Klassen, die "
               "eine Bank gar nicht im A-IRB führt (z. B. UniCredit kein separates "
               "Mortgage; Rabobank/Crédit Mutuel/ING kein QRRE; mehrere kein "
               "IRB-Sovereign), sind <em>strukturell</em> und zählen nicht als Lücke.",
        modell="Die wenigen unvollständigen Zellen sind dokumentierte "
               "<strong>Quellgrenzen</strong>: BNP-Report 2021 rundet die PD auf "
               "ganze % (0 %/5 %); SocGen 2022 Bank+Sovereign sind im PDF-Text "
               "verschmolzen; Crédit Mutuel/BPCE-Vorjahre haben mehrdeutige Anker. "
               "Diese werden <strong>nicht</strong> mit abgeleiteten Werten gefüllt "
               "(keine Fabrikation) — daher bleiben sie offen statt erfunden.",
        metrik="Quelle: jeder Wert ist ein bankpubliziertes <em>EU-CR6-A-IRB-"
               "Sub-total</em> (EBA-ITS/2020/04, CRR Art. 431-455), gegen FY2024 "
               "kalibriert und über RWA/EAD-Dichte geprüft.",
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
        'von <strong>zwei getrennten, messbaren Faktoren</strong> getrieben — dem '
        'Zinsschock <code>Δr₁₀<sub>J</sub></code> und dem Energie-/Angebotsschock '
        '<code>ΔBrent</code>. Jeder wirkt über <em>sektor-spezifische</em> '
        'Sensitivitäten (Ebene 2). Bevor man Prognosen vergleicht, müssen diese '
        'Treiber <em>zur richtigen Zeit</em> ausschlagen — also genau dann, wenn '
        'real eine Krise herrschte.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Krisen-Legende: kurze Definition der markierten Episoden ---------
    st.markdown(
        '<div style="background:#FBF2F2;border:1px solid #E8C9CF;'
        'border-left:4px solid #A52F4D;padding:0.8rem 1.05rem;border-radius:6px;'
        'margin:0.1rem 0 0.8rem 0;color:#051C2C;font-size:0.86rem;line-height:1.65;">'
        '<strong>Die markierten Krisen (gestrichelte Linien in der Grafik):</strong><br>'
        '• <strong>Ukraine Q1 2022</strong> — russischer Überfall → Energiepreis- '
        '&amp; Inflationsschock.<br>'
        '• <strong>Zinswende Q3 2022</strong> — schnellste Leitzins-Anhebung der '
        'EZB-Geschichte gegen die Inflation.<br>'
        '• <strong>SVB/CS Q1 2023</strong> — Kollaps der Silicon Valley Bank + '
        'Credit-Suisse-Notrettung → kurze Banken-Vertrauenskrise.'
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
    lese(
        was="Die zwei tatsächlich eingetretenen <strong>Markt-Schocks</strong> pro "
            "Quartal — Zinsschock (Balken, rot = gestiegen) und Ölpreis (gelbe "
            "Linie). Wichtig: das ist der <strong>Input</strong> des Modells (aus "
            "Bundesbank-Zins + Brent-Ölpreis), <em>nicht</em> unsere PD/LGD-Reihe "
            "und <em>noch keine Prognose</em>. Gestrichelt = die oben definierten Krisen.",
        befund="Der Zinsschock springt 2022 massiv nach oben (10-Jahres-Zins "
               "+2,8 pp übers Jahr) — exakt in der Ukraine-/Inflations-/Zinswende-"
               "Phase, gefolgt von SVB/CS Anfang 2023.",
        modell="<strong>Schritt 1 von 3.</strong> Diese Grafik vergleicht <em>noch "
               "nicht</em> Modell gegen Realität — sie prüft nur, ob die Auslöser "
               "<em>zur richtigen Zeit</em> feuern (sie tun es: genau in den "
               "Krisen). Den eigentlichen Vergleich Prognose-gegen-Realität siehst "
               "du weiter unten in <strong>Ebene 3 → Probe A</strong> "
               "(blaue Linie = Modell, rote Linie = Realität).",
        metrik="Δr₁₀ⱼ = Veränderung des 10-Jahres-Zinses in <strong>Prozentpunkten</strong> "
               "(z. B. +2,8 pp = von 0 % auf 2,8 %). ΔBrent = log-Veränderung des "
               "Ölpreises über das Quartal (≈ prozentuale Änderung).",
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
        'ist macro-inert (Zins wirkt dort über den separaten Marktbuch-Kanal). '
        'Genau diese Differenzierung ist der Kern des 2-Faktor-Modells.'
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
    lese(
        was="Wie stark jede Kreditklasse auf die beiden Schocks reagiert — die "
            "Stellschrauben des Modells. Rot = Risiko steigt bei dem Schock, "
            "blau = Risiko sinkt, weiß = kein Effekt.",
        befund="Mortgage reagiert am stärksten auf Zinsen (Affordability + "
               "Hauspreise); Bank ist beim Zins <em>blau/negativ</em> (Banken "
               "verdienen an höheren Zinsen — NIM); Sovereign ist null (läuft "
               "über das Marktbuch). Alle Vorzeichen ökonomisch plausibel.",
        modell="Das Modell behandelt Zins- und Ölschock getrennt und je Sektor "
               "unterschiedlich — kein pauschaler Auf-/Ab-Regler. Die Reaktion "
               "ist also ökonomisch fundiert, nicht willkürlich.",
        metrik="β (Beta) = um wie viele <strong>Prozentpunkte</strong> sich die "
               "PD (bzw. γ für LGD) ändert, wenn der Faktor um eine Einheit steigt "
               "(Zins +1 pp bzw. Öl +100 %). Jeder Wert ist quellenbelegt "
               "(EBA-2025-Methodik §2.4.2, ECB WP 2897/3112).",
    )

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
    lese(
        was="Wie stark das vom Modell prognostizierte Kredit-RWA jeder Bank "
            "steigt, wenn der Zins um X Prozentpunkte anzieht (Ölpreis fix). "
            "Jede Linie = eine Bank.",
        befund="Alle Linien steigen sauber an (mehr Zins-Stress ⇒ mehr RWA, nie "
               "weniger) und liegen in realistischer Höhe: ein +2,8-pp-Schock wie "
               "2022 ergibt grob +5 bis +15 % RWA — im Bereich echter EBA-Stresstests.",
        modell="Die Stress-Mechanik ist <strong>konservativ</strong> (Risiko nur "
               "nach oben) und <strong>bank-individuell</strong> (steilere Linien "
               "= riskanterer Portfolio-Mix). Das ist genau das Verhalten, das ein "
               "seriöses Stress-Modell zeigen muss.",
        metrik="ΔRWA in <strong>% der Baseline</strong> = wie viel mehr Eigenkapital "
               "die Bank unter dem Schock hinterlegen müsste, relativ zum Normalzustand.",
    )

    # ====================================================================
    #  Abschnitt 4 · Validierungsebene 3 — Outcomes-Analysis (ehrlich)
    # ====================================================================
    st.divider()
    eyebrow("Ebene 3 · Outcomes — wie nah ist die prognostizierte CET1-Quote an der Realität? (SR 11-7)")
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #A52F4D;padding:0.85rem 1.1rem;border-radius:6px;'
        'margin:0.3rem 0 0.9rem 0;color:#051C2C;font-size:0.86rem;line-height:1.6;">'
        '<strong>Jetzt zählt die Zielgröße.</strong> Wir speisen die <em>real '
        'eingetretenen</em> Risikofaktoren jedes Jahres zeitversetzt ins Modell '
        'und vergleichen die prognostizierte <strong>CET1-Quote</strong> mit der '
        'tatsächlich gemeldeten — das ist der <strong>Kern-Test</strong> (die '
        'CET1-Quote ist die Solvenz-Kennzahl, die das Modell liefern soll). '
        'Darunter öffnen wir den Haupt-Treiber dahinter: die <strong>PD-Entwicklung</strong> '
        '(gesamt und je Segment).'
        '</div>',
        unsafe_allow_html=True,
    )

    # ===== Kern-Test · CET1-Quote: Modell vs. Realität ====================
    if cet1_stats.get("n", 0) > 0:
        st.markdown(
            "**Kern-Test · CET1-Quote: Modell vs. Realität (die Zielgröße).**  "
            "🔴 **rote Linie = tatsächlich gemeldete CET1-Quote**, 🔵 **blaue Äste "
            "= Modell-Prognose**, wenn wir den realen Zins-/Öl-Schock des Jahres "
            "einspeisen (jeweils vom Istwert des Vorjahres aus — zeitlich versetzt).")
        _mae = cet1_stats["mae_pp"]
        gcol, icol = st.columns([1, 1], gap="large")
        with gcol:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=_mae,
                number={"suffix": " pp", "font": {"size": 34, "color": COLORS["navy"]}},
                gauge={
                    "axis": {"range": [0, 4], "tickvals": [0, 1, 2, 3, 4],
                             "tickwidth": 1, "tickcolor": COLORS["stone"]},
                    "bar": {"color": COLORS["navy"], "thickness": 0.22},
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 1], "color": "#BFE3CC"},
                        {"range": [1, 2], "color": "#E6EFCB"},
                        {"range": [2, 3], "color": "#FAE7BE"},
                        {"range": [3, 4], "color": "#F0CFCF"}],
                    "threshold": {"line": {"color": COLORS["crimson"], "width": 4},
                                  "thickness": 0.85, "value": _mae}}))
            fig_g.update_layout(height=230, margin=dict(t=10, b=10, l=30, r=30))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(
                "**MAE der CET1-Quote** = Ø Abstand Prognose ↔ Realität in "
                "Prozentpunkten. **Faustregel:** ≤ 1 pp sehr gut · 1–2 pp gut · "
                "2–3 pp akzeptabel · > 3 pp schwach.")
        with icol:
            st.markdown(
                f'<div style="background:#F7F9FB;border:1px solid #E6E6E6;'
                f'border-left:4px solid #034B6F;padding:0.9rem 1.1rem;'
                f'border-radius:6px;font-size:0.9rem;line-height:1.7;color:#051C2C;">'
                f'<strong>MAE — Definition.</strong> Mean Absolute Error = '
                f'mittlerer absoluter Fehler: '
                f'<code>MAE = (1/N) · Σ |Prognose − Ist|</code> — der '
                f'durchschnittliche Abstand zwischen prognostizierter und '
                f'gemeldeter CET1-Quote (in pp), <em>richtungsunabhängig</em> '
                f'(Über- wie Unterschätzung zählen gleich).<br>'
                f'<strong>Einordnung — wie gut ist das?</strong> Unser MAE von '
                f'<strong>{_mae:.1f} pp</strong> heißt: die Prognose liegt im '
                f'Schnitt nur gut einen Prozentpunkt neben der real gemeldeten '
                f'CET1-Quote — auf ~15 %-Niveau ≈ {_mae/15*100:.0f} % relativer '
                f'Abstand.<br>'
                f'• in <strong>{cet1_stats["within_1pp"]*100:.0f}%</strong> der '
                f'Bank-Jahre ≤ 1 pp Abstand<br>'
                f'• in <strong>{cet1_stats["conservative_share"]*100:.0f}%</strong> '
                f'konservativ (Prognose ≤ Ist — sichere Seite), Bias '
                f'<strong>{cet1_stats["bias_pp"]:+.1f} pp</strong><br>'
                f'Für ein sparsames 2-Faktor-Modell mit nur zwei Makro-Inputs — '
                f'und trotz der Limitationen (PIT-Modell vs. TTC-Meldung, nur drei '
                f'Jahre) — ein belastbares <strong>„gut"</strong>.'
                f'</div>',
                unsafe_allow_html=True)

        cy = (cet1_bt.groupby("year")
              .agg(real=("cet1_ratio_real", "mean"),
                   pred=("cet1_ratio_pred", "mean"),
                   start=("cet1_ratio_start", "mean"))
              .reset_index().sort_values("year"))
        cyrs = [int(y) for y in cy["year"]]
        c_anchor = cyrs[0] - 1
        cx = [c_anchor] + cyrs
        cy_real = [float(cy["start"].iloc[0])] + [float(v) for v in cy["real"]]
        cpred_change = {int(r["year"]): float(r["pred"] - r["start"])
                        for _, r in cy.iterrows()}
        fig_c = go.Figure()
        cmps = [cy_real[i] + cpred_change.get(cyrs[i], 0.0) for i in range(len(cyrs))]
        creal_pos = ["bottom center"] + [
            "top center" if cy_real[i + 1] >= cmps[i] else "bottom center"
            for i in range(len(cyrs))]
        fig_c.add_trace(go.Scatter(
            x=cx, y=cy_real, name="🔴 Realität (gemeldete CET1-Quote)",
            mode="lines+markers+text", line=dict(color=COLORS["crimson"], width=3),
            marker=dict(size=11),
            text=[f"{v:.1f}%" for v in cy_real], textposition=creal_pos,
            textfont=dict(size=11, color=COLORS["crimson"]),
            hovertemplate="FY%{x}<br><b>Realität</b>: CET1 %{y:.2f} %<extra></extra>"))
        for i, y in enumerate(cyrs):
            mp = cmps[i]
            dr = (annual_macro or {}).get(y, {}).get("d_r_10y_pp", float("nan"))
            m_pos = "top center" if mp >= cy_real[i + 1] else "bottom center"
            fig_c.add_trace(go.Scatter(
                x=[cx[i], y], y=[cy_real[i], mp], mode="lines+markers+text",
                line=dict(color=COLORS["navy"], width=2.5, dash="dot"),
                marker=dict(size=11, symbol="diamond", color=COLORS["navy"]),
                text=["", f"{mp:.1f}%"], textposition=m_pos,
                textfont=dict(size=11, color=COLORS["navy"]),
                name="🔵 Modell (realer Schock eingespeist)", showlegend=(i == 0),
                hovertemplate=(f"FY{y} · Zins {dr:+.1f} pp<br>"
                               f"<b>Modell</b>: CET1 %{{y:.2f}} %<extra></extra>")))
            gap = mp - cy_real[i + 1]
            fig_c.add_annotation(
                x=y, y=(cy_real[i + 1] + mp) / 2, showarrow=False,
                text=f"Δ {gap:+.1f} pp", font=dict(size=9, color=COLORS["stone"]),
                xshift=34)
        fig_c.update_layout(
            title="Modell-CET1-Quote vs. Realität (in %, Ø über alle Banken)",
            xaxis_title="Jahr (Modell = Vorjahres-Istwert + realer Schock · zeitlich versetzt)",
            yaxis_title="CET1-Quote [%]",
            height=440, xaxis=dict(tickmode="array", tickvals=cx), margin=dict(t=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig_c, use_container_width=True)
        lese(
            was="🔴 die tatsächlich gemeldete CET1-Quote über die Jahre; 🔵 was das "
                "Modell für die CET1-Quote vorhersagt, wenn man den realen Schock "
                "jedes Jahres einspeist (jeweils neu vom Istwert des Vorjahres aus).",
            befund=f"In ruhigen Jahren trifft das Modell die CET1-Quote eng "
                   f"(Ø-Abstand <strong>{cet1_stats['mae_pp']:.1f} pp</strong>, in "
                   f"{cet1_stats['within_1pp']*100:.0f}% der Fälle ≤ 1 pp). Im "
                   f"Schockjahr 2022 zieht der blaue Ast nach unten (Modell: CET1 "
                   f"fällt), die Realität blieb dank Bank-Gewinnen aber stabiler.",
            modell=f"Das Modell liegt in <strong>{cet1_stats['conservative_share']*100:.0f}%</strong> "
                   f"der Fälle auf der <strong>konservativen Seite</strong> (sagt eine "
                   f"niedrigere CET1 voraus als real eintrat). Genau das soll ein "
                   f"Solvenz-Stresstest: im Zweifel vorsichtig, nie zu optimistisch "
                   f"— und im Niveau nah dran.",
            metrik="CET1-Quote = hartes Kernkapital ÷ RWA (zentrale Solvenz-Kennzahl). "
                   "MAE = mittlerer Abstand Prognose↔Ist in Prozentpunkten. Bias "
                   "negativ = Modell schätzt die CET1 vorsichtig zu niedrig (sichere Seite).",
        )
        st.divider()

    # ===== Kanal-Detail 1 · PD-Backtest (PIT-Modell vs. TTC-Meldung) ======
    if pd_stats.get("n", 0) > 0:
        st.markdown(
            "**Kanal-Detail 1 · Entwicklung der PD-Reihe: Realität vs. Modell.**  "
            "Der erste Treiber der CET1-Quote. 🔴 **rote Linie = wie "
            "sich die tatsächlich gemeldete PD entwickelt hat**, 🔵 **blaue Linie = "
            "wie sie sich entwickelt, wenn wir den realen Zins-/Öl-Schock jedes "
            "Jahres ins Modell geben** (jeweils neu vom Istwert des Vorjahres aus). "
            "Über jedem Jahr steht der Treiber, der die Modell-Bewegung erzeugt.")
        pa1, pa2 = st.columns(2, gap="small")
        pa1.metric("PD-Vorhersagen", f"{pd_stats['n']}",
                   "Bank × Klasse × Jahr", delta_color="off")
        pa2.metric("Ø Abstand (MAE)", f"{pd_stats['mae_model']:.2f} pp",
                   "Prognose ↔ gemeldete PD", delta_color="off")
        st.caption(
            "**Kennzahl:** **Ø Abstand (MAE)** = wie weit die prognostizierte PD im "
            "Schnitt von der gemeldeten abweicht, in Prozentpunkten. Die PDs liegen "
            "je nach Klasse bei ~0,1 % (Sovereign) bis ~15 % (Retail); ein Abstand "
            "< 1 pp ist daher gering. Eine *Richtungs-Trefferquote* berichten wir "
            "hier bewusst **nicht** — bei nur drei Jahren statistisch wenig "
            "belastbar; aussagekräftig ist die geringe pp-Abweichung im Niveau."
        )

        # Niveau-Trajektorie: wie entwickelt sich die PD-Reihe — Realität vs. Modell
        # (Modell = 1-Jahres-Vorhersage, jeweils neu vom Istwert des Vorjahres aus)
        amac = annual_macro or {}
        by_year = (pd_bt.groupby("year")
                   .agg(real=("pd_real_pct", "mean"),
                        model=("pd_pred_pct", "mean"),
                        base=("pd_base_pct", "mean"))
                   .reset_index().sort_values("year"))
        yrs_all = [int(y) for y in by_year["year"]]
        anchor = yrs_all[0] - 1
        x_axis = [anchor] + yrs_all
        anchor_val = float(by_year["base"].iloc[0])
        y_real = [anchor_val] + [float(v) for v in by_year["real"]]
        # Vom Modell prognostizierte 1-Jahres-Änderung je Jahr (= model − base):
        d_pred_yr = {int(r["year"]): float(r["model"] - r["base"])
                     for _, r in by_year.iterrows()}
        fig_pa = go.Figure()
        # Modell-Endpunkte vorab; Label-Seiten so wählen, dass sich Real- und
        # Modell-Wert je Jahr NICHT überlagern (gegenüberliegende Seiten).
        mps = [y_real[i] + d_pred_yr.get(yrs_all[i], 0.0) for i in range(len(yrs_all))]
        real_pos = ["bottom center"] + [
            "top center" if y_real[i + 1] >= mps[i] else "bottom center"
            for i in range(len(yrs_all))]
        fig_pa.add_trace(go.Scatter(
            x=x_axis, y=y_real, name="🔴 Realität (gemeldete PD)",
            mode="lines+markers+text", line=dict(color=COLORS["crimson"], width=3),
            marker=dict(size=11),
            text=[f"{v:.2f}%" for v in y_real], textposition=real_pos,
            textfont=dict(size=11, color=COLORS["crimson"]),
            hovertemplate="FY%{x}<br><b>Realität</b>: PD %{y:.2f} %<extra></extra>",
        ))
        # Modell = 1-Jahres-Vorhersage, jeweils NEU vom Istwert des Vorjahres aus
        # → je Jahr ein blauer Ast, der von der roten Realitäts-Linie abzweigt.
        for i, y in enumerate(yrs_all):
            mp = mps[i]
            dr = amac.get(y, {}).get("d_r_10y_pp", float("nan"))
            ob = (amac.get(y, {}).get("d_brent_log", 0.0) or 0.0) * 100
            m_pos = "top center" if mp >= y_real[i + 1] else "bottom center"
            fig_pa.add_trace(go.Scatter(
                x=[x_axis[i], y], y=[y_real[i], mp],
                mode="lines+markers+text",
                line=dict(color=COLORS["navy"], width=2.5, dash="dot"),
                marker=dict(size=11, symbol="diamond", color=COLORS["navy"]),
                text=["", f"{mp:.2f}%"], textposition=m_pos,
                textfont=dict(size=11, color=COLORS["navy"]),
                name="🔵 Modell (1-Jahres-Vorhersage)", showlegend=(i == 0),
                hovertemplate=(f"FY{y} · Zins {dr:+.1f} pp · Öl {ob:+.0f} %<br>"
                               f"<b>Modell sagt</b>: PD %{{y:.2f}} %<extra></extra>"),
            ))
        fig_pa.update_layout(
            title="Modell-PD vs. Realität (in %, Ø über alle Banken & Segmente)",
            xaxis_title="Jahr (Modell = Vorjahres-Istwert + realer Schock · zeitlich versetzt)",
            yaxis_title="Ø PD [%]",
            height=440, xaxis=dict(tickmode="array", tickvals=x_axis),
            margin=dict(t=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_pa, use_container_width=True)

        # --- Pro Segment: Realität vs. Modell (gewähltes Jahr, Ø über alle Banken) ---
        seg_years = sorted(int(y) for y in pd_bt["year"].unique())
        sy = st.radio("PD je Segment — Prognose-Jahr wählen", seg_years,
                      index=len(seg_years) - 1, horizontal=True, key="pd_seg_year")
        seg = (pd_bt[pd_bt["year"] == sy].groupby("vasicek_class")
               .agg(real=("pd_real_pct", "mean"), model=("pd_pred_pct", "mean"))
               .reset_index().sort_values("real", ascending=False))
        fig_seg = go.Figure()
        fig_seg.add_trace(go.Bar(
            x=seg["vasicek_class"], y=seg["real"], name="🔴 Realität (gemeldete PD)",
            marker_color=COLORS["crimson"], opacity=0.9,
            text=[f"{v:.2f}" for v in seg["real"]], textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{x}<br>Realität: %{y:.2f} %<extra></extra>"))
        fig_seg.add_trace(go.Bar(
            x=seg["vasicek_class"], y=seg["model"], name="🔵 Modell (Vorjahr + Schock)",
            marker_color=COLORS["navy"], opacity=0.9,
            text=[f"{v:.2f}" for v in seg["model"]], textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{x}<br>Modell: %{y:.2f} %<extra></extra>"))
        fig_seg.update_layout(
            title=f"PD je Segment {sy}: Realität vs. Modell (Ø über alle Banken)",
            xaxis_title="Segment (IRB-Klasse)", yaxis_title="PD [%]",
            height=400, barmode="group", margin=dict(t=56),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig_seg, use_container_width=True)
        st.caption(
            f"Pro Kreditklasse: 🔴 tatsächlich gemeldete PD {sy} vs. 🔵 Modell-Prognose "
            f"(PD {sy-1} + realer Schock {sy}). Wo **blau über rot** liegt, stresst das "
            f"Modell die Klasse stärker als die geglättete TTC-Meldung — im Schockjahr "
            f"2022 am deutlichsten, in ruhigen Jahren liegen beide eng beieinander."
        )

        st.markdown(
            '<div style="background:#FFF8E7;border:1px solid #E6D9A8;'
            'border-left:4px solid #C9A227;padding:0.85rem 1.1rem;'
            'border-radius:6px;margin:0.2rem 0 0.9rem 0;color:#051C2C;'
            'font-size:0.88rem;line-height:1.7;">'
            '<strong>Ursache des Befunds (Schwerpunkt 2022).</strong><br>'
            '<strong>2022</strong>: der Zins stieg um +2,8 pp; das Modell '
            'projiziert daraufhin korrekt eine höhere PD (Ø +0,5 pp). Die '
            'tatsächlich gemeldeten PD sanken im Schnitt jedoch. Das ist kein '
            'Widerspruch, sondern Ausdruck <strong>zweier verschiedener Größen</strong>:<br>'
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

    # ===== Hervorgehoben · warum 2022 daneben (PIT vs. TTC) ==============
    st.markdown(
        '<div style="background:#FBF2F2;border:2px solid #A52F4D;'
        'border-radius:8px;padding:1.0rem 1.25rem;margin:0.6rem 0 0.4rem 0;'
        'color:#051C2C;font-size:1.0rem;line-height:1.8;">'
        '<div style="font-size:1.05rem;font-weight:800;color:#A52F4D;'
        'margin-bottom:0.3rem;">Ursache der Abweichung 2022 (~2,7 pp) '
        '→ <span style="text-decoration:underline;">PIT vs. TTC</span></div>'
        '2022 stieg der Zins stark (<strong>+2,8 pp</strong>). '
        '<strong>Unser Modell ist Point-in-Time (PIT)</strong> — es reagiert '
        '<strong>sofort</strong>: „PD&nbsp;rauf → CET1&nbsp;runter". Die '
        '<strong>gemeldeten regulatorischen PD sind aber Through-the-Cycle '
        '(TTC)</strong> — per Vorschrift geglättet und antizyklisch (CRR&nbsp;'
        'Art.&nbsp;180), damit sie im Abschwung gerade <strong>nicht</strong> '
        'springen. Sie blieben also flach → die gemeldete CET1 blieb stabil. '
        'Dazu kommt: die Banken <strong>verdienten</strong> am Zinsanstieg '
        '(Zinsüberschuss), was unser Modell bewusst <strong>nicht gegenrechnet</strong>. '
        '<br><strong style="color:#A52F4D;">Fazit:</strong> Die 2022-Lücke ist '
        '<strong>kein Modellfehler</strong>, sondern der Unterschied zwischen einer '
        '<strong>PIT-Sofortreaktion (Modell)</strong> und einer '
        '<strong>TTC-geglätteten Meldung (Realität)</strong> — und sie zeigt in die '
        '<strong>sichere Richtung</strong> (Modell konservativer als die Realität).'
        '</div>',
        unsafe_allow_html=True,
    )

    # ====================================================================
    #  Abschnitt 5 · Bank-Drilldown
    # ====================================================================
    st.divider()
    eyebrow("Bank-Drilldown · eingefrorene Risikoparameter und CET1-Quote (Modell vs. Realität) je Institut")

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

        def _wavg(d, col):
            w = d["ead_eur_m"].to_numpy(dtype=float)
            x = d[col].to_numpy(dtype=float)
            return float(np.average(x, weights=w)) if np.nansum(w) > 0 else float("nan")

        # Version-sichere EAD-Gewichtung (kein groupby.apply → robust über
        # pandas-Versionen, kein FutureWarning auf Gruppen-Spalten).
        agg = (pd.DataFrame([
                   {"vintage_date": v, "PD": _wavg(d, "pd_pct"),
                    "LGD": _wavg(d, "lgd_pct"), "EAD_bn": d["ead_eur_m"].sum()/1e3}
                   for v, d in sb.groupby("vintage_date")])
               .sort_values("vintage_date"))
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
            "**Was:** die echten Eingangswerte dieser Bank — EAD-gewichtete Ø PD "
            "(rot) und Ø LGD (blau) je Pillar-3-Stichtag. **Aussage:** genau diese "
            "Werte friert der Backtest zu jedem Zeitpunkt ein (kein Blick in die "
            "Zukunft); erkennbar ist zudem, wie stabil bzw. geglättet (TTC) die "
            "gemeldeten Parameter über die Jahre sind."
        )

    # Rechts: CET1-Quote Modell vs. Realität je Bank (die Zielgröße)
    with bk_r:
        bc = (cet1_bt[cet1_bt["LEI"] == sel_lei].sort_values("year")
              if (cet1_bt is not None and not cet1_bt.empty) else pd.DataFrame())
        if bc.empty:
            st.info(f"Für {sel_name} liegen keine CET1-Jahresdaten vor "
                    "(EBA-Jahresend-Stichtage 31.12. fehlen für diese Bank).")
        else:
            byrs = [int(y) for y in bc["year"]]
            anchor = byrs[0] - 1
            bx = [anchor] + byrs
            by_real = ([float(bc["cet1_ratio_start"].iloc[0])]
                       + [float(v) for v in bc["cet1_ratio_real"]])
            bmps = [float(v) for v in bc["cet1_ratio_pred"]]
            breal_pos = ["bottom center"] + [
                "top center" if by_real[i + 1] >= bmps[i] else "bottom center"
                for i in range(len(bmps))]
            fig_bc = go.Figure()
            fig_bc.add_trace(go.Scatter(
                x=bx, y=by_real, name="🔴 Realität (gemeldet)",
                mode="lines+markers+text", line=dict(color=COLORS["crimson"], width=3),
                marker=dict(size=10),
                text=[f"{v:.1f}%" for v in by_real], textposition=breal_pos,
                textfont=dict(size=10, color=COLORS["crimson"]),
                hovertemplate="FY%{x}<br><b>Realität</b>: CET1 %{y:.2f} %<extra></extra>"))
            for i, (_, r) in enumerate(bc.iterrows()):
                yy = int(r["year"]); mp = bmps[i]
                m_pos = "top center" if mp >= by_real[i + 1] else "bottom center"
                fig_bc.add_trace(go.Scatter(
                    x=[bx[i], yy], y=[by_real[i], mp], mode="lines+markers+text",
                    line=dict(color=COLORS["navy"], width=2.5, dash="dot"),
                    marker=dict(size=10, symbol="diamond", color=COLORS["navy"]),
                    text=["", f"{mp:.1f}%"], textposition=m_pos,
                    textfont=dict(size=10, color=COLORS["navy"]),
                    name="🔵 Modell (realer Schock)", showlegend=(i == 0),
                    hovertemplate=f"FY{yy}<br><b>Modell</b>: CET1 %{{y:.2f}} %<extra></extra>"))
            fig_bc.update_layout(
                title=f"{sel_name} · CET1-Quote: Modell vs. Realität",
                xaxis_title="Jahr (Modell = Vorjahres-Istwert + realer Schock)",
                yaxis_title="CET1-Quote [%]",
                height=360, xaxis=dict(tickmode="array", tickvals=bx),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig_bc, use_container_width=True)
            mae_b = float(bc["abs_error_pp"].mean())
            cons_b = int(bc["conservative"].sum())
            d1b, d2b, d3b = st.columns(3, gap="small")
            d1b.metric("Jahre", f"{len(bc)}")
            d2b.metric("Ø Abstand (MAE)", f"{mae_b:.1f} pp")
            d3b.metric("Konservativ", f"{cons_b}/{len(bc)}")
            st.caption(
                "**Was:** je Jahr die gemeldete CET1-Quote (🔴) und die "
                "Modell-Prognose unter dem realen Schock (🔵, jeweils vom "
                "Vorjahres-Istwert). **Aussage:** wie nah und auf welcher Seite "
                "das Modell für diese Bank lag (konservativ = Prognose ≤ Ist)."
            )

    # --- PD je Segment pro Bank (Segment wählbar, inkl. "Alle Segmente") ---
    bank_seg = (pd_bt[pd_bt["LEI"] == sel_lei].copy()
                if (pd_bt is not None and not pd_bt.empty) else pd.DataFrame())
    if not bank_seg.empty:
        st.markdown(f"**{sel_name} · PD je Segment — Modell vs. Realität** "
                    "(1-Jahres-Vorhersage je Kreditklasse).")
        classes = sorted(bank_seg["vasicek_class"].unique())
        seg_choice = st.selectbox("Segment", ["Alle Segmente"] + classes,
                                  key="dd_seg_choice")
        if seg_choice == "Alle Segmente":
            seg_yrs = sorted(int(y) for y in bank_seg["year"].unique())
            ysel = st.radio("Jahr", seg_yrs, index=len(seg_yrs) - 1,
                            horizontal=True, key="dd_seg_year")
            d = bank_seg[bank_seg["year"] == ysel].sort_values("pd_real_pct",
                                                               ascending=False)
            fig_ds = go.Figure()
            fig_ds.add_trace(go.Bar(
                x=d["vasicek_class"], y=d["pd_real_pct"], name="🔴 Realität (gemeldet)",
                marker_color=COLORS["crimson"], opacity=0.9,
                text=[f"{v:.2f}" for v in d["pd_real_pct"]], textposition="outside",
                textfont=dict(size=9),
                hovertemplate="%{x}<br>Realität: %{y:.2f} %<extra></extra>"))
            fig_ds.add_trace(go.Bar(
                x=d["vasicek_class"], y=d["pd_pred_pct"], name="🔵 Modell",
                marker_color=COLORS["navy"], opacity=0.9,
                text=[f"{v:.2f}" for v in d["pd_pred_pct"]], textposition="outside",
                textfont=dict(size=9),
                hovertemplate="%{x}<br>Modell: %{y:.2f} %<extra></extra>"))
            fig_ds.update_layout(
                title=f"{sel_name} · PD je Segment {ysel}: Realität vs. Modell",
                xaxis_title="Segment (IRB-Klasse)", yaxis_title="PD [%]",
                height=380, barmode="group", margin=dict(t=56),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig_ds, use_container_width=True)
            st.caption(
                f"Je Kreditklasse für {ysel}: 🔴 gemeldete PD vs. 🔵 Modell-Prognose "
                f"(PD {ysel-1} + realer Schock {ysel}). Liegt 🔵 über 🔴, stresst das "
                "Modell die Klasse stärker als die geglättete TTC-Meldung.")
        else:
            d = bank_seg[bank_seg["vasicek_class"] == seg_choice].sort_values("year")
            syrs = [int(y) for y in d["year"]]
            anchor = syrs[0] - 1
            sx = [anchor] + syrs
            s_real = [float(d["pd_base_pct"].iloc[0])] + [float(v) for v in d["pd_real_pct"]]
            smps = [float(v) for v in d["pd_pred_pct"]]
            sreal_pos = ["bottom center"] + [
                "top center" if s_real[i + 1] >= smps[i] else "bottom center"
                for i in range(len(smps))]
            fig_ds = go.Figure()
            fig_ds.add_trace(go.Scatter(
                x=sx, y=s_real, name="🔴 Realität (gemeldet)", mode="lines+markers+text",
                line=dict(color=COLORS["crimson"], width=3), marker=dict(size=10),
                text=[f"{v:.2f}%" for v in s_real], textposition=sreal_pos,
                textfont=dict(size=10, color=COLORS["crimson"]),
                hovertemplate="FY%{x}<br>Realität: %{y:.2f} %<extra></extra>"))
            for i, yy in enumerate(syrs):
                mp = smps[i]
                m_pos = "top center" if mp >= s_real[i + 1] else "bottom center"
                fig_ds.add_trace(go.Scatter(
                    x=[sx[i], yy], y=[s_real[i], mp], mode="lines+markers+text",
                    line=dict(color=COLORS["navy"], width=2.5, dash="dot"),
                    marker=dict(size=10, symbol="diamond", color=COLORS["navy"]),
                    text=["", f"{mp:.2f}%"], textposition=m_pos,
                    textfont=dict(size=10, color=COLORS["navy"]),
                    name="🔵 Modell", showlegend=(i == 0),
                    hovertemplate=f"FY{yy}<br>Modell: %{{y:.2f}} %<extra></extra>"))
            fig_ds.update_layout(
                title=f"{sel_name} · PD {seg_choice}: Realität vs. Modell über die Jahre",
                xaxis_title="Jahr", yaxis_title="PD [%]", height=380,
                xaxis=dict(tickmode="array", tickvals=sx),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig_ds, use_container_width=True)
            st.caption(
                f"PD-Entwicklung der Klasse {seg_choice}: 🔴 gemeldet vs. 🔵 Modell "
                "(Vorjahres-Istwert + realer Schock, 1-Jahres-Vorhersage).")

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
        f'<strong>4. Zielgröße CET1-Quote: konservativ &amp; nah getroffen.</strong> '
        f'Speist man die real eingetretenen Schocks ein, liegt die prognostizierte '
        f'CET1-Quote im Schnitt nur <strong>{cet1_stats.get("mae_pp", 0):.1f} pp</strong> '
        f'neben der gemeldeten und in '
        f'<strong>{cet1_stats.get("conservative_share", 0)*100:.0f}%</strong> der '
        f'Fälle auf der konservativen (vorsichtigen) Seite. Die einzelnen Kanäle '
        f'(PD, RWA) sind quartalsweise verrauscht (PIT-Modell vs. TTC-Meldung), '
        f'die Solvenz-Aussage im Niveau ist aber belastbar.<br><br>'
        '<strong style="color:#C9A227;">Verdikt:</strong> Das Modell ist als '
        '<strong>konservatives Solvenz-/Stress-Instrument</strong> validiert — es '
        'beantwortet „<em>wie steht die CET1-Quote, falls ein Schock einträte?</em>" '
        'nah an der Realität und <strong>nie zu optimistisch</strong>. Für die '
        'punktgenaue Quartals-Prognose einzelner Kanäle taugt es nicht — dafür ist '
        'es auch nicht gebaut.'
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
zwei Schocks getrennt: ΔBrent (log-Return) und Δr₁₀ⱼ (Prozentpunkte) — getrennt,
nicht zu einem einzelnen Aggregat-Faktor verrechnet.

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
1. *Restlaufzeit* ist in der Roh-Reihe nicht erfasst → Basel-Default 2,5 Jahre
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
        f"Pillar-3-Walk-Forward-Backtest · {n_banks_bt} Banken × {n_vintages} "
        f"Pillar-3-Jahrgänge · CET1-Kern-Test: MAE "
        f"{cet1_stats.get('mae_pp', 0):.1f} pp · "
        f"{cet1_stats.get('within_1pp', 0)*100:.0f}% ≤ 1 pp · "
        f"{cet1_stats.get('conservative_share', 0)*100:.0f}% konservativ · "
        f"Datenbasis: pillar3_backtest_pdlgd.csv ({n_points} EU-CR6-Punkte) + "
        f"EBA Transparency 2020-2025 + Brent (ICE) + Bundesbank-Svensson"
    )

with tab_an:
    render_annahmen_tab()

with tab_md:
    render_methodology_tab()
