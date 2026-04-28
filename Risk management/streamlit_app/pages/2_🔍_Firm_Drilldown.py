"""Page 2 · Firm Drill-Down (Stub für V1)."""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st
from components.sidebar import render_sidebar

st.set_page_config(page_title="Firm Drilldown", page_icon="🔍", layout="wide")
render_sidebar()
st.title("🔍 Firm Drill-Down")
st.info("Coming soon · pro-Firma Merton-Inputs, Faktor-Betas, Stress-PD-Histogramm.")
