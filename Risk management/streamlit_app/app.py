"""DAX Credit Stress Cockpit — Streamlit-Frontend.

Entry-Point. Streamlit erkennt automatisch die Pages in `pages/` und
zeigt sie in der linken Navigation.

Start:
    streamlit run streamlit_app/app.py
"""
import sys
from pathlib import Path

# Make components/ importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st

from components.sidebar import render_sidebar
from components.data_loader import load_data_layer, run_baseline


st.set_page_config(
    page_title="DAX Credit Stress Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar (auch auf Welcome-Page sichtbar — globaler State)
config = render_sidebar()

# === Welcome ==========================================================
st.title("DAX Credit Stress Cockpit")
st.subheader("Strukturmodell-basierter Stresstest mit Svensson, Faktor-Modell und Monte-Carlo")

st.markdown("""
**Wofür ist dieses Cockpit?**
Quantitative Analyse der Default-Wahrscheinlichkeit (PD) und Distance-to-Default (DD)
für die DAX-40-Aktien unter Energiepreis- und Zinsschocks. Pipeline:

1. **Daten-Layer** — yfinance + Bundesbank Svensson
2. **Svensson** — Zero-Rate-Engine
3. **Merton/KMV** — Strukturmodell für PD pro Firma
4. **Faktor-Modell** — 2-Faktor-OLS (Brent + Δr_10y) mit Sektor-Multipliern
5. **Monte-Carlo** — multivariate Normal-Pfade
6. **Szenarien** — Library (Corona / Ukraine / Iran) + Live-Custom
7. **Reverse Stress** — kritische Schock-Schwellen
8. **Portfolio-Aggregation** — EL, VaR/CVaR, HHI

→ Methodik vollständig dokumentiert in `MODEL_ASSUMPTIONS.md`.
""")

st.divider()

# === Quick Status =====================================================
data = load_data_layer()
base = run_baseline()

col1, col2, col3 = st.columns(3)
with col1:
    if data["prices"] is not None:
        st.success(f"✅ DAX-Prices: {data['prices'].shape[0]} Tage × "
                   f"{data['prices'].shape[1]} Tickers")
        st.caption(f"{data['prices'].index.min().date()} → "
                   f"{data['prices'].index.max().date()}")
    else:
        st.error("❌ DAX-Prices nicht im Cache")

with col2:
    if data["svensson"] is not None:
        st.success(f"✅ Svensson: {len(data['svensson'])} Handelstage")
        st.caption(f"{data['svensson'].index.min().date()} → "
                   f"{data['svensson'].index.max().date()}")
    else:
        st.error("❌ Svensson nicht im Cache")

with col3:
    if base["merton_df"] is not None:
        n_total = len(base["merton_df"])
        n_ok = int(base["merton_df"]["Converged"].fillna(False).sum())
        st.success(f"✅ Merton-Baseline: {n_ok}/{n_total} konvergiert")
        st.caption(f"Stand {base['as_of'].date()}")
    else:
        st.error("❌ Merton-Pipeline failed")

st.divider()

# === Navigation-Hinweis ==============================================
st.subheader("Navigation")
st.markdown("""
- **📊 Overview** — Portfolio-KPIs, Waterfall, Heatmaps, Top-N
- **🔍 Firm Drill-Down** — pro-Firma Merton, Faktor-Betas, Stress-PD-Histogramm
- **🔥 Scenario Heatmap** — alle Library-Szenarien mit Decomposition
- **⚡ Reverse Stress** — kritische Schock-Schwellen pro Firma
- **📈 Yield Curve** — Svensson interaktiv, Faktor-Returns Plots
- **📚 Methodology** — gerendertes `MODEL_ASSUMPTIONS.md`
""")

st.caption(
    "Sidebar links: Live-Stress-Slider. Alle Pages reagieren in Echtzeit "
    "auf Slider-Änderungen (gecached, ~0.5–2 s pro Update)."
)
