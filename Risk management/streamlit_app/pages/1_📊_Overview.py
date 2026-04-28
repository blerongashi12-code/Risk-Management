"""Page 1 · Overview · Portfolio-KPIs, Waterfall, Heatmaps, Top-N."""
import sys
from pathlib import Path

# Make components/ importable
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import streamlit as st

from components.sidebar import render_sidebar
from components.data_loader import run_baseline, run_scenarios
from components.stress_engine import (run_custom_scenario, run_custom_mc,
                                      decompose_pd, portfolio_kpis)
from components.charts import (waterfall_decomposition, top_n_baseline_vs_stress,
                               firm_scenario_heatmap, sector_quantile_heatmap,
                               critical_thresholds_chart)

# Page-Setup
st.set_page_config(
    page_title="Overview · DAX Credit Stress",
    page_icon="📊", layout="wide",
)

# Sidebar
config = render_sidebar()

# Header
st.title("📊 DAX Credit Stress · Overview")
base = run_baseline()
if base["merton_df"] is None:
    st.error("Cache-Files fehlen. Bitte zuerst `00_build_data_layer.py` ausführen.")
    st.stop()

st.caption(
    f"Stand: **{base['as_of'].date()}** · "
    f"Stress-Konfig: ΔBrent **{config['d_brent']:+.2f}** "
    f"(≈ {(np.exp(config['d_brent'])-1)*100:+.0f}%), "
    f"Δr_10y **{config['d_r_10y_pp']*100:+.1f} bp**, "
    f"Horizon **{config['horizon_days']}d**, "
    f"MC **{config['n_sims']:,} Pfade**"
)

# === Pipeline aufrufen ============================================
kpi = portfolio_kpis(
    config["d_brent"], config["d_r_10y_pp"],
    config["horizon_days"], config["n_sims"],
    config["corr_override"],
)
custom_df = run_custom_scenario(
    config["d_brent"], config["d_r_10y_pp"], config["horizon_days"],
)
mc_df = run_custom_mc(
    config["n_sims"], config["horizon_days"], config["corr_override"],
)

# === KPI-Bar ======================================================
st.subheader("Portfolio-KPIs")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Σ EAD (Σ DPT)",
          f"{kpi['total_ead']/1e9:.1f} bn €",
          help="Default Point = ShortTermDebt + 0.5·LongTermDebt (Moody's KMV)")
c2.metric("EL Baseline",
          f"{kpi['el_baseline']/1e6:.1f} m €",
          help=f"LGD = {kpi['lgd']*100:.0f}% (Basel-Standard)")
c3.metric("EL Custom-Stress",
          f"{kpi['el_custom']/1e6:.1f} m €",
          delta=f"{kpi['delta_el_custom']/1e6:+.1f} m €",
          help="Deterministischer Schock aus den Sliders")
c4.metric("EL MC-Mean",
          f"{kpi['el_mc_mean']/1e6:.1f} m €",
          help=f"Mean über {config['n_sims']:,} MC-Pfade")
c5.metric("EL MC p99",
          f"{kpi['el_mc_p99']/1e6:.1f} m €",
          delta=f"{kpi['delta_el_p99']/1e6:+.1f} m €",
          help="99%-Quantil der Stress-PD pro Firma, EAD-gewichtet — adverse Indikator")
c6.metric("HHI / N-eff",
          f"{kpi['hhi']:.3f} / {1/kpi['hhi']:.1f}" if kpi['hhi'] > 0 else "–",
          help=f"Herfindahl auf EAD-Anteilen. {kpi['n_converged']}/{kpi['n_firms']} konvergiert.")

st.divider()

# === Waterfall + Reverse Stress ===================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("PD-Decomposition (Stress-Waterfall)")
    wf_target = st.radio(
        "Ebene",
        ["Portfolio (EL in €)", "Einzelfirma (PD in %)"],
        horizontal=True, label_visibility="collapsed",
        key="wf_target",
    )
    if wf_target.startswith("Portfolio"):
        decomp = decompose_pd(
            None, config["d_brent"], config["d_r_10y_pp"],
            config["horizon_days"],
        )
    else:
        all_tickers = list(base["merton_df"].index)
        # Default = Top-PD-Firma
        default_idx = (
            base["merton_df"]["PD"].fillna(0).idxmax()
            if base["merton_df"]["PD"].notna().any() else all_tickers[0]
        )
        sel_tk = st.selectbox(
            "Firma:", all_tickers,
            index=all_tickers.index(default_idx),
            key="wf_ticker", label_visibility="collapsed",
        )
        decomp = decompose_pd(
            sel_tk, config["d_brent"], config["d_r_10y_pp"],
            config["horizon_days"],
        )
    st.plotly_chart(waterfall_decomposition(decomp), use_container_width=True)

with col_right:
    st.subheader("Reverse Stress · Kritische Schwellen")
    st.caption(
        "Welcher ΔBrent / Δr drückt eine Firma auf PD = 5 %? "
        "NaN = nicht im Bereich [-3, +3] / [-5, +5pp] erreichbar."
    )
    # Reverse-Stress lazy laden
    @st.cache_data(ttl=24*3600, show_spinner="Berechne Reverse-Stress …")
    def _rev():
        from components.data_loader import load_data_layer
        from reverse_stress import run_dax40 as run_rev  # type: ignore
        d = load_data_layer()
        return run_rev(d["prices"], d["brent"], d["svensson"], d["fundamentals"])
    rev_df = _rev()
    st.plotly_chart(critical_thresholds_chart(rev_df), use_container_width=True)

st.divider()

# === Firm × Scenario Heatmap ======================================
st.subheader("Firm × Scenario Heatmap")
st.caption(
    "Stress-PD jeder Firma unter den vier Library-Szenarien "
    "(Corona, Ukraine, Ukraine-Peak, Iran). Live-Slider werden "
    "hier nicht reflektiert — diese Heatmap zeigt die fixen Library-Szenarien."
)
scen_res = run_scenarios()
st.plotly_chart(firm_scenario_heatmap(scen_res), use_container_width=True)

st.divider()

# === Sektor-Quantil-Heatmap =======================================
st.subheader("Sektor-Risiko über MC-Quantile")
st.plotly_chart(sector_quantile_heatmap(mc_df), use_container_width=True)

st.divider()

# === Top-15 Bar Chart =============================================
st.subheader("Top-15 Firmen · Baseline vs. Stress")
# Custom-Scenario-Output hat Spalte 'ScenarioPD'
top15_fig = top_n_baseline_vs_stress(custom_df, n=15)
st.plotly_chart(top15_fig, use_container_width=True)

st.divider()

# === Top-10 Tabelle ===============================================
st.subheader("Top-10 Stress-PD Ranking")
top10 = (custom_df.dropna(subset=["ScenarioPD"])
                  .sort_values("ScenarioPD", ascending=False)
                  .head(10)
                  .copy())
top10["BaselinePD"] = (top10["BaselinePD"] * 100).round(4)
top10["ScenarioPD"] = (top10["ScenarioPD"] * 100).round(4)
top10["DeltaPD"] = (top10["DeltaPD"] * 100).round(4)
top10 = top10[["Name", "Sector", "BaselinePD", "ScenarioPD",
               "DeltaPD", "BrentContrib", "RateContrib"]]
top10.columns = ["Name", "Sektor", "Base PD %", "Stress PD %",
                 "Δ PD pp", "Brent-Beitrag (log)", "Rate-Beitrag (log)"]
st.dataframe(top10, use_container_width=True, height=400)

st.divider()

# === Footer ======================================================
st.caption(
    "ℹ Methodik in `MODEL_ASSUMPTIONS.md`. "
    "Diese Page nutzt Live-Slider (Sidebar) für ΔBrent + Svensson-Shifts. "
    "Die anderen Pages (Drill-Down, Reverse Stress, Yield Curve, Methodology) "
    "sind in den Tabs links."
)
