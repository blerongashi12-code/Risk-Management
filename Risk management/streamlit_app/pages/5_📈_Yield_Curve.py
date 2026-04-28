"""Page 5 · Yield Curve (Stub mit Live-Plot der Svensson-Kurve)."""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.sidebar import render_sidebar
from components.data_loader import baseline_yield_curve, delta_r_from_beta_shifts

st.set_page_config(page_title="Yield Curve", page_icon="📈", layout="wide")
config = render_sidebar()

st.title("📈 Bundesbank Svensson Yield Curve")

base = baseline_yield_curve()
if base.empty:
    st.error("Svensson-Cache fehlt.")
    st.stop()

# Stressed Curve berechnen
from components.backend_path import setup
setup()
from svensson import historical_curve, shift_curve, curve_grid  # type: ignore
from components.data_loader import load_data_layer

data = load_data_layer()
base_params = historical_curve(data["svensson"].index[-1], data["svensson"])
shifted = shift_curve(
    base_params,
    dlevel=config["d_b0"], dslope=config["d_b1"],
    dcurv1=config["d_b2"], dcurv2=config["d_b3"],
)
mats = np.arange(0.25, 30.25, 0.25)
stressed = curve_grid(shifted, maturities=mats)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=base["maturity"], y=base["rate_pct"],
    name="Baseline (heute)", line=dict(color="#1a365d", width=3),
))
fig.add_trace(go.Scatter(
    x=stressed.index, y=stressed.values,
    name="Stressed (mit Slider-Shifts)",
    line=dict(color="#d97706", width=3, dash="dash"),
))
fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
fig.update_layout(
    title=f"Svensson-Zinskurve · Stand {data['svensson'].index[-1].date()}",
    xaxis_title="Maturity [Jahre]", yaxis_title="Zero-Rate [%]",
    height=500, plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Aktuelle Svensson-Parameter")
st.dataframe(
    {"Parameter": ["β₀", "β₁", "β₂", "β₃", "τ₁", "τ₂"],
     "Baseline": [base_params.beta0, base_params.beta1,
                  base_params.beta2, base_params.beta3,
                  base_params.tau1, base_params.tau2],
     "Stressed": [shifted.beta0, shifted.beta1, shifted.beta2,
                  shifted.beta3, shifted.tau1, shifted.tau2],
     "Δ":        [shifted.beta0 - base_params.beta0,
                  shifted.beta1 - base_params.beta1,
                  shifted.beta2 - base_params.beta2,
                  shifted.beta3 - base_params.beta3,
                  0.0, 0.0]},
    use_container_width=True, hide_index=True,
)
