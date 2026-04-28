"""Page 3 · Scenario Heatmap (Stub für V1)."""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st
from components.sidebar import render_sidebar
from components.data_loader import run_scenarios
from components.charts import firm_scenario_heatmap

st.set_page_config(page_title="Scenario Heatmap", page_icon="🔥", layout="wide")
render_sidebar()

st.title("🔥 Scenario Heatmap")
st.caption("Library-Szenarien (Corona, Ukraine, Iran) auf alle DAX-Firmen.")
res = run_scenarios()
st.plotly_chart(firm_scenario_heatmap(res), use_container_width=True)

st.divider()
st.subheader("Schock-Werte pro Szenario")
import pandas as pd
rows = []
for name, sc in res["scenarios"].items():
    rows.append({
        "Szenario": name,
        "Source": sc["source"],
        "ΔBrent_log": f"{sc['delta_brent_log']:+.3f}",
        "Δr_10y [pp]": f"{sc['delta_rate_10y_pp']:+.2f}",
        "Horizon [d]": sc["horizon_days"],
        "Beschreibung": sc["description"],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
