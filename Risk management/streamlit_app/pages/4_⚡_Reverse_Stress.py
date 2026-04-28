"""Page 4 · Reverse Stress (Stub für V1)."""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st
from components.sidebar import render_sidebar

st.set_page_config(page_title="Reverse Stress", page_icon="⚡", layout="wide")
render_sidebar()

st.title("⚡ Reverse Stress")
st.info("Coming soon · interaktive Iso-PD-Kurven + adaptive Targets pro Firma.")
