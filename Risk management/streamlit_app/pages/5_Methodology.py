"""Methodology · full assumptions document."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st

from components.theme import apply_theme, hero, footer
from components.sidebar import render_sidebar

st.set_page_config(page_title="Methodology", layout="wide")
apply_theme()
render_sidebar()

hero(
    "Methodology",
    eyebrow="Full model documentation",
    deck="Every assumption, parameter, formula and known limitation, with "
         "source references. The single source of truth for model behaviour.",
)

md_path = _HERE.parent / "MODEL_ASSUMPTIONS.md"
if not md_path.exists():
    st.error(f"MODEL_ASSUMPTIONS.md not found at {md_path}")
    st.stop()

with open(md_path, "r", encoding="utf-8") as fh:
    content = fh.read()

st.markdown(content)

footer("This document is rendered live from MODEL_ASSUMPTIONS.md.")
