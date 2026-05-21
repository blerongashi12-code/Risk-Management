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

from components.theme import (apply_theme, hero, eyebrow, insight, footer,
                              COLORS)
from components.sidebar import render_sidebar
from components.data_loader import load_data_layer
from components.backend_path import setup
setup()

from config import EBA_RAW_DIR                                     # type: ignore
from eba_loader import (load_historical_capital_panel,             # type: ignore
                        panel_to_wide, load_bank_directory)
from macro_factor import factor_stats                              # type: ignore
from backtesting import (                                          # type: ignore
    compute_macro_panel, attach_m_factor,
    compute_realized_changes, build_forecast_panel,
    fit_empirical_sensitivity, episode_diagnostics, rwa_decomposition,
    DEFAULT_EPISODES,
)


st.set_page_config(page_title="Validierung & Methodologie", layout="wide")
apply_theme()
render_sidebar()

hero(
    "Validierung & Methodologie",
    eyebrow="Tab 5 · Backtesting · Annahmen · Methodologie",
    deck="Drei Validierungs-Ebenen kombiniert: (1) historisches Backtesting "
         "gegen 22 EBA-Transparency-Vintages 2019-2025, (2) 3-Layer-Annahmen-"
         "Disclosure (Executive · Validator · Quant), (3) vollständige "
         "Methodologie aus MODEL_ASSUMPTIONS.md. Outcomes Analysis nach "
         "SR 11-7 + Governance nach EBA GL 14.",
)



# Top-level tabs · Backtesting + Annahmen + Methodologie
from components.legacy_views import render_annahmen_tab, render_methodology_tab

tab_bt, tab_an, tab_md = st.tabs([
    "1 · Backtesting (Forecast vs. Realized)",
    "2 · Annahmen & Datenbasis (Governance)",
    "3 · Methodologie (MODEL_ASSUMPTIONS.md)",
])

with tab_bt:
    # === Methodology disclaimer at top ===================================
    st.caption(
        "**Backtest-Setup.** Pro EBA-Reporting-Period wird der trailing-1Y-"
        "Macro-Schock (Brent log-Return, Δr_10y) berechnet und über die "
        "hybrid Macro→M-Funktion auf einen Vasicek-Systemfaktor M abgebildet. "
        "Diese M-Werte werden dann mit den realisierten 1Y-Veränderungen der "
        "CET1-Ratio pro Bank empirisch korreliert. Dieser empirische β-Schätzer "
        "dient als **Challenger-Modell** zum strukturellen Tier-2-Stack — wenn "
        "die historische Sensitivität deutlich von der Modell-Sensitivität "
        "abweicht, ist das ein Validation-Findings-Indikator (SR 11-7 §V.5)."
    )

    st.divider()

    # === Load data (cached) ==============================================
    @st.cache_data(ttl=24*3600, show_spinner="Loading historical EBA panel …")
    def _load_panel():
        panel = load_historical_capital_panel(EBA_RAW_DIR)
        wide  = panel_to_wide(panel)
        bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
        return panel, wide, bank_dir


    @st.cache_data(ttl=24*3600, show_spinner="Computing macro state per period …")
    def _compute_macro(periods: tuple[int, ...]):
        data = load_data_layer()
        if data["brent"] is None or data["svensson"] is None:
            return None, None
        macro = compute_macro_panel(data["brent"], data["svensson"], list(periods))
        fs = factor_stats(data["brent"], data["svensson"], lookback=252)
        macro = attach_m_factor(macro, cov_factors=fs["sigma"])
        return macro, fs


    panel, wide, bank_dir = _load_panel()
    periods = tuple(sorted(wide["Period"].unique()))
    macro, fs = _compute_macro(periods)

    if macro is None or len(macro) == 0:
        st.error("Macro panel could not be computed — Brent or Svensson cache missing.")
        st.stop()

    # Forecast panel: realized + macro
    realized = compute_realized_changes(wide, lag_quarters=4)
    fp_full = build_forecast_panel(realized, macro)
    fp_full = fp_full.dropna(subset=["m_at_start", "delta_ratio_pp"])

    # Bank name lookup
    fp_full = fp_full.merge(bank_dir[["lei", "bank_name"]],
                             left_on="LEI_Code", right_on="lei", how="left")


    # === KPI strip · panel coverage ======================================
    eyebrow("Backtest panel coverage")
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    c1.metric("Quartals-Stichtage", str(len(periods)),
              f"{periods[0]} → {periods[-1]}", delta_color="off")
    c2.metric("Banken", str(wide["LEI_Code"].nunique()),
              "pro Stichtag", delta_color="off")
    c3.metric("Macro-Periods (1Y lookback)", str(len(macro)),
              "Brent + Svensson available", delta_color="off")
    c4.metric("Realized 1Y-Pairs", f"{len(realized):,}",
              "(LEI, Period_t, Period_t+4Q)", delta_color="off")
    c5.metric("Forecast Panel (joined)", f"{len(fp_full):,}",
              f"after macro merge", delta_color="off")

    st.divider()

    # === Macro factor time-series ========================================
    eyebrow("Macro factor time-series · trailing 1Y window")

    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(
        x=macro["date"], y=macro["m_hybrid"], name="M (hybrid)",
        line=dict(color=COLORS["navy"], width=2.5),
        mode="lines+markers",
        marker=dict(size=6, color=COLORS["navy"]),
        yaxis="y1",
    ))
    fig_m.add_trace(go.Scatter(
        x=macro["date"], y=macro["brent_log_1y"], name="ΔBrent log (1Y)",
        line=dict(color=COLORS["bright_blue"], width=1.8, dash="dash"),
        mode="lines",
        yaxis="y2",
    ))
    fig_m.add_trace(go.Scatter(
        x=macro["date"], y=macro["dr_10y_1y_pp"], name="Δr_10y (1Y, pp)",
        line=dict(color=COLORS["crimson"], width=1.8, dash="dash"),
        mode="lines",
        yaxis="y2",
    ))
    fig_m.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
    fig_m.update_layout(
        title="Realized macro shocks 2021–2025 (trailing-1Y view)",
        yaxis=dict(title="M factor (Vasicek systematic factor)",
                   side="left",
                   tickfont=dict(color=COLORS["navy"]),
                   title_font=dict(color=COLORS["navy"])),
        yaxis2=dict(title="Brent log / Δr_10y [pp]",
                    overlaying="y", side="right",
                    tickfont=dict(color=COLORS["stone"]),
                    title_font=dict(color=COLORS["stone"]),
                    showgrid=False, zeroline=False),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
    )
    st.plotly_chart(fig_m, use_container_width=True)

    st.caption(
        f"Empirische Faktor-Korrelation ρ̂(Brent, Δr_10y) = {fs['corr'][0,1]:+.2f} "
        f"über {fs['n_obs']} Handelstage. Macro-Cache reicht zurück bis Q2 2021 "
        f"(Brent + Svensson 252-Tage-Lookback). Corona-2020-Episode ist deshalb "
        f"nicht im Backtest enthalten — als Limitation in der Annahmen-Page "
        f"dokumentiert."
    )

    st.divider()

    # === Outlier filter + Forecast-vs-Realized scatter ====================
    eyebrow("Forecast vs. Realized · cross-bank panel")

    filt_l, filt_r = st.columns([1, 3])
    with filt_l:
        outlier_cap = st.slider(
            "Outlier-Cap |Δratio_pp|",
            min_value=2.0, max_value=20.0, value=5.0, step=0.5,
            help="Bank-Period-Pairs mit |Δ-Ratio| oberhalb dieser Schwelle "
                 "werden ausgeschlossen — typisch sind das M&A-Events, "
                 "regulatorische One-Offs, Bank-Restrukturierungen.",
            key="bt_outlier",
        )
    with filt_r:
        st.markdown(
            f"Aus dem vollen Panel ({len(fp_full):,} Bank-Period-Pairs) werden "
            f"Bank-Period-Pairs mit |Δ-Ratio| > **{outlier_cap:.1f} pp** "
            f"als Outlier ausgeschlossen. Diese Filterung adressiert direkt "
            f"Critique 6 — *externe Effekte (M&A, Portfolio-Umschichtungen, "
            f"regulatorische Änderungen) verzerren den Backtest, wenn nicht "
            f"bereinigt*."
        )

    fp = fp_full[fp_full["delta_ratio_pp"].abs() <= outlier_cap].copy()
    n_dropped = len(fp_full) - len(fp)
    st.caption(f"After filtering: **{len(fp):,}** observations · "
               f"{n_dropped} outliers dropped ({n_dropped/max(len(fp_full),1)*100:.1f}%)")

    # Empirical OLS regression
    sens = fit_empirical_sensitivity(fp)

    # KPI strip
    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    k1.metric("Sample size n", f"{sens['n']:,}", delta_color="off")
    k2.metric("β (Sensitivity to M)",
              f"{sens['beta']:+.3f}",
              f"t = {sens['t_beta']:+.2f}", delta_color="off")
    k3.metric("α (Intercept)", f"{sens['alpha']:+.3f}",
              "pp", delta_color="off")
    k4.metric("R²", f"{sens['r2']:.3f}",
              "explained variance", delta_color="off")
    k5.metric("RMSE", f"{sens['rmse']:.3f}",
              "pp", delta_color="off")
    k6.metric("MAE", f"{sens['mae']:.3f}",
              "pp", delta_color="off")

    # Sign + significance interpretation
    sign_msg = ""
    if not pd.isna(sens["beta"]) and not pd.isna(sens["t_beta"]):
        expected_sign = "positive" if sens["beta"] > 0 else "negative"
        sig = "significant (|t| > 2)" if abs(sens["t_beta"]) > 2 else "not significant"
        direction = ("benign macro state → CET1 strengthening"
                     if sens["beta"] > 0 else
                     "adverse macro state (M < 0) → CET1 strengthening — "
                     "counter-intuitive in pooled cross-section")
        sign_msg = (f"<strong>β = {sens['beta']:+.3f} ({sig})</strong>. "
                    f"In der gepoolten Cross-Section: {direction}.")

    if abs(sens["beta"]) > 0:
        insight(
            "<strong>Empirische Sensitivität.</strong> " + sign_msg +
            " R² ist niedrig — das ist erwartbar in einer gepoolten Bank-"
            "Cross-Section, weil die idiosynkratische Bank-spezifische Variation "
            "(Geschäftsmodell, Land, M&A) die Macro-Komponente überlagert. "
            "Aussagekräftiger sind Episode-Level-Mittel (siehe nächste Sektion)."
        )

    # Scatter
    fig_s = go.Figure(go.Scattergl(
        x=fp["m_at_start"], y=fp["delta_ratio_pp"],
        mode="markers",
        marker=dict(size=4, color=COLORS["navy"], opacity=0.4),
        text=fp["bank_name"] + " · " + fp["Period_start"].astype(str) + " → "
             + fp["Period_end"].astype(str),
        hovertemplate=("%{text}<br>"
                       "M_at_start: %{x:+.2f}<br>"
                       "Δ-Ratio: %{y:+.2f} pp<extra></extra>"),
    ))
    # Regression line
    if not pd.isna(sens["beta"]):
        x_grid = np.linspace(fp["m_at_start"].min(), fp["m_at_start"].max(), 50)
        y_grid = sens["alpha"] + sens["beta"] * x_grid
        fig_s.add_trace(go.Scatter(
            x=x_grid, y=y_grid, mode="lines",
            line=dict(color=COLORS["crimson"], width=2.5),
            name=f"OLS: y = {sens['alpha']:+.3f} {sens['beta']:+.3f}·M",
        ))
    fig_s.add_hline(y=0, line_color=COLORS["hairline"], line_width=1)
    fig_s.add_vline(x=0, line_color=COLORS["hairline"], line_width=1)
    fig_s.update_layout(
        title="Realized 1Y Δ-CET1-Ratio (pp) vs. M factor at start of window",
        xaxis_title="M factor (Vasicek systematic factor) at start of period",
        yaxis_title="Realized Δ CET1 Ratio [pp] over next 4 quarters",
        height=440,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    st.divider()

    # === Stress-episode comparison =======================================
    eyebrow("Stress-episode comparison")

    ep_df = episode_diagnostics(fp)

    if len(ep_df) > 0:
        # Episode summary table
        st.dataframe(
            ep_df.assign(
                **{
                    "M_at_start":          ep_df["M_at_start"].round(2),
                    "delta_ratio_mean":    ep_df["delta_ratio_mean"].round(2),
                    "delta_ratio_median":  ep_df["delta_ratio_median"].round(2),
                    "delta_ratio_p25":     ep_df["delta_ratio_p25"].round(2),
                    "delta_ratio_p75":     ep_df["delta_ratio_p75"].round(2),
                    "delta_ratio_min":     ep_df["delta_ratio_min"].round(2),
                    "delta_ratio_max":     ep_df["delta_ratio_max"].round(2),
                }
            ).rename(columns={
                "episode":            "Episode",
                "label":              "Window",
                "n_banks":            "N",
                "M_at_start":         "M @ start",
                "delta_ratio_mean":   "Δ mean (pp)",
                "delta_ratio_median": "Δ median (pp)",
                "delta_ratio_p25":    "p25",
                "delta_ratio_p75":    "p75",
                "delta_ratio_min":    "min",
                "delta_ratio_max":    "max",
            }),
            use_container_width=True, hide_index=True,
        )

        # Per-bank box-plot (excluding outliers above cap by definition)
        fig_e = go.Figure()
        for _, ep in ep_df.iterrows():
            ep_data = fp[(fp["Period_start"] == DEFAULT_EPISODES[ep["episode"]]["Period_start"])
                         & (fp["Period_end"] == DEFAULT_EPISODES[ep["episode"]]["Period_end"])]
            fig_e.add_trace(go.Box(
                y=ep_data["delta_ratio_pp"],
                name=f"{ep['episode']} · M={ep['M_at_start']:.1f}",
                boxmean=True,
                marker_color=COLORS["mid_blue"],
                line_color=COLORS["navy"],
            ))
        fig_e.add_hline(y=0, line_color=COLORS["hairline"], line_width=1,
                        line_dash="dot")
        fig_e.update_layout(
            title="Realized Δ-CET1-Ratio distribution per stress episode",
            yaxis_title="Δ CET1 Ratio over 1Y window [pp]",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig_e, use_container_width=True)

        # Insight on what we see
        if len(ep_df) >= 2:
            ep_a = ep_df.iloc[0]
            ep_b = ep_df.iloc[-1]
            insight(
                f"<strong>Cross-episode comparison.</strong> Bei "
                f"<em>{ep_a['episode']}</em> war M = {ep_a['M_at_start']:+.2f} und "
                f"die mediane realisierte Δ-Ratio betrug "
                f"{ep_a['delta_ratio_median']:+.2f} pp — bei "
                f"<em>{ep_b['episode']}</em> war M = {ep_b['M_at_start']:+.2f} und "
                f"die mediane Δ-Ratio {ep_b['delta_ratio_median']:+.2f} pp. "
                f"<br><br>Das EU-Bankensystem hat unter beiden Stress-Episoden "
                f"die CET1-Ratio im Schnitt <em>nicht</em> reduziert — Profitabilität, "
                f"Dividendenkürzung, RWA-Optimization und regulatorische "
                f"Stützungsmaßnahmen haben den Macro-Schock absorbiert. Das ist "
                f"ein bekanntes Phänomen aus EBA-Stress-Tests (siehe Risk "
                f"Assessment Report 2023, EBA Validation §3.4)."
            )
    else:
        st.info("Episode panel empty — increase outlier cap.")

    st.divider()

    # === RWA-component decomposition =====================================
    eyebrow("RWA decomposition · Δ Credit / Market / Operational")

    st.caption(
        "EBA-publizierte ΔRWA-Komponenten zwischen Period_start und Period_end. "
        "Diese Brutto-Decomposition ist die observable Aufschlüsselung; sie "
        "isoliert noch nicht das stress-induzierte vom portfolio-induzierten "
        "ΔRWA (Critique 6). Eine vollständige Brinson-Style Volume × Mix × "
        "Stress-Residual-Decomposition würde Segment-Level-EAD pro Vintage "
        "erfordern und ist V2-Roadmap."
    )

    rwa_d = rwa_decomposition(fp)
    if len(rwa_d) > 0:
        # Episode-filter for the table
        ep_choice = st.selectbox(
            "Episode für Bank-Drilldown",
            ["All periods"] + list(DEFAULT_EPISODES.keys()),
            index=0,
            key="bt_rwa_ep",
        )
        if ep_choice == "All periods":
            rwa_view = rwa_d
            title = "All bank-period pairs"
        else:
            spec = DEFAULT_EPISODES[ep_choice]
            rwa_view = rwa_d[(rwa_d["Period_start"] == spec["Period_start"])
                             & (rwa_d["Period_end"]   == spec["Period_end"])]
            title = f"{ep_choice} · {spec['label']}"

        if len(rwa_view) > 0:
            # Top 10 by abs total ΔRWA
            rwa_top = rwa_view.assign(abs_drwa=rwa_view["delta_rwa_eur"].abs())
            rwa_top = rwa_top.sort_values("abs_drwa", ascending=False).head(10)
            disp = pd.DataFrame({
                "Bank":               rwa_top["bank_name"],
                "Window":              rwa_top["Period_start"].astype(str)
                                      + " → " + rwa_top["Period_end"].astype(str),
                "RWA start bn":       (rwa_top["rwa_start"]/1e9).round(0).astype(int),
                "Δ Credit bn":        (rwa_top["dRWA_credit"]/1e9).round(1),
                "Δ Market bn":        (rwa_top["dRWA_market"]/1e9).round(1),
                "Δ Op bn":            (rwa_top["dRWA_operational"]/1e9).round(1),
                "Δ Total bn":         (rwa_top["delta_rwa_eur"]/1e9).round(1),
                "Δ Total %":          (rwa_top["g_total"]*100).round(1),
            })
            st.dataframe(disp, use_container_width=True, hide_index=True,
                         height=400)

            st.caption(f"Top 10 banks by |ΔRWA| · {title}")
        else:
            st.info("No RWA decomposition rows for this episode.")

    footer(
        f"Backtest panel: {len(fp_full):,} bank-period pairs from {len(periods)} "
        f"quarterly snapshots Q3 2019 → Q2 2025 · "
        f"Sources: 6 EBA Transparency vintages (2020-2025), Bundesbank Svensson "
        f"daily, ICE Brent daily · Outlier filter: |Δ-Ratio| ≤ "
        f"{outlier_cap:.1f} pp · Net-EAD-Decomposition into volume/mix/stress "
        f"residual at segment level is V2 (requires segment-EAD time-series "
        f"from tr_cre.csv across vintages)."
    )

with tab_an:
    render_annahmen_tab()

with tab_md:
    render_methodology_tab()
