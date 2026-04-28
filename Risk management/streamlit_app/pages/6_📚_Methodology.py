"""Page 6 · Methodology · rendert MODEL_ASSUMPTIONS.md."""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st
from components.sidebar import render_sidebar

st.set_page_config(page_title="Methodology", page_icon="📚", layout="wide")
render_sidebar()

st.title("📚 Methodology")

# Repo-Root: streamlit_app/../MODEL_ASSUMPTIONS.md
md_path = _HERE.parent / "MODEL_ASSUMPTIONS.md"
if not md_path.exists():
    st.error(f"`MODEL_ASSUMPTIONS.md` nicht gefunden unter {md_path}")
    st.stop()

with open(md_path, "r", encoding="utf-8") as fh:
    content = fh.read()

st.markdown(content)
