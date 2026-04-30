"""McKinsey-Aesthetic · zentrales Theme-Modul.

Ein Aufruf von `apply_theme()` am Anfang jeder Page injiziert die globale
CSS-Schicht und registriert ein Plotly-Default-Template, sodass alle
Charts ohne weitere Anpassungen konsistent aussehen.

Usage
-----
>>> from components.theme import apply_theme, hero, eyebrow, insight, footer
>>> apply_theme()
>>> hero("Bank Portfolio View",
...      eyebrow="Vasicek · ASRF",
...      deck="EBA-anchored exposure-class stress test under macro shocks.")
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# =====================================================================
# 1. Color Tokens
# =====================================================================
COLORS = {
    "navy":        "#051C2C",
    "mid_blue":    "#034B6F",
    "bright_blue": "#2251FF",
    "teal":        "#00A9A5",
    "crimson":     "#A52F4D",
    "amber":       "#C9A227",
    "stone":       "#6E6E6E",
    "hairline":    "#E6E6E6",
    "canvas":      "#F4F4F4",
    "white":       "#FFFFFF",
}

# Sequential palette for ordered/heatmap data (cool → warm)
SEQ_COOL_TO_WARM = [
    [0.00, "#F4F4F4"],
    [0.20, "#C9DAE8"],
    [0.40, "#7FA8C8"],
    [0.60, "#3F6FA0"],
    [0.80, "#A52F4D"],
    [1.00, "#7A1F38"],
]

# Discrete palette (use first n entries)
PALETTE_DISCRETE = [
    "#2251FF",  # bright blue
    "#051C2C",  # navy
    "#A52F4D",  # crimson
    "#00A9A5",  # teal
    "#C9A227",  # amber
    "#034B6F",  # mid blue
    "#6E6E6E",  # stone
]


# =====================================================================
# 2. Plotly Template
# =====================================================================
def _build_template() -> go.layout.Template:
    """Baut das McKinsey-Plotly-Template auf."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(
            family="Inter, -apple-system, Segoe UI, sans-serif",
            color=COLORS["navy"],
            size=12,
        ),
        title=dict(
            font=dict(
                family="Source Serif Pro, Georgia, serif",
                size=16,
                color=COLORS["navy"],
            ),
            x=0.0,
            xanchor="left",
            pad=dict(t=10, b=10),
        ),
        paper_bgcolor=COLORS["white"],
        plot_bgcolor=COLORS["white"],
        colorway=PALETTE_DISCRETE,
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor=COLORS["hairline"],
            ticks="outside",
            tickcolor=COLORS["hairline"],
            tickfont=dict(size=11, color=COLORS["stone"]),
            title=dict(font=dict(size=12, color=COLORS["navy"])),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["hairline"],
            gridwidth=1,
            zeroline=False,
            showline=False,
            tickfont=dict(size=11, color=COLORS["stone"]),
            title=dict(font=dict(size=12, color=COLORS["navy"])),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11, color=COLORS["navy"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=20, r=20, t=60, b=40),
        hoverlabel=dict(
            bgcolor=COLORS["white"],
            bordercolor=COLORS["navy"],
            font=dict(family="Inter", size=12, color=COLORS["navy"]),
        ),
    )
    return template


def _register_plotly_template() -> None:
    pio.templates["mckinsey"] = _build_template()
    pio.templates.default = "mckinsey"


# =====================================================================
# 3. CSS Injection
# =====================================================================
_CSS_PATH = Path(__file__).resolve().parent.parent / "static" / "mckinsey.css"


def _inject_css() -> None:
    if not _CSS_PATH.exists():
        return
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# =====================================================================
# 4. Public API
# =====================================================================
_THEME_APPLIED_KEY = "_mc_theme_applied"


def apply_theme() -> None:
    """Idempotent — erst CSS, dann Plotly-Template registrieren."""
    _register_plotly_template()
    _inject_css()
    st.session_state[_THEME_APPLIED_KEY] = True


def hero(title: str, *, eyebrow: str | None = None, deck: str | None = None) -> None:
    """Rendert den Page-Hero im Konsulting-Stil."""
    parts = ['<div class="mc-hero">']
    if eyebrow:
        parts.append(f'<div class="mc-eyebrow">{eyebrow}</div>')
    parts.append(f"<h1>{title}</h1>")
    if deck:
        parts.append(f'<p class="mc-deck">{deck}</p>')
    parts.append("</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    """Kleine Section-Überschrift im Eyebrow-Stil (UPPERCASE)."""
    st.markdown(
        f'<span class="mc-eyebrow-inline">{text}</span>',
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    """Hervorgehobene Insight-Box (links Navy-Akzent)."""
    st.markdown(f'<div class="mc-insight">{text}</div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    """Page-Footer mit Hairline."""
    st.markdown(f'<div class="mc-footer">{text}</div>', unsafe_allow_html=True)


def kpi_columns(n: int):
    """Wrapper um st.columns mit konsistenten Gaps für KPI-Strips."""
    return st.columns(n, gap="small")


# =====================================================================
# 5. Color helpers (für ad-hoc Plotly-Calls)
# =====================================================================
def color(name: str) -> str:
    """Liefert einen Farbwert per Token-Name (`navy`, `bright_blue`, …)."""
    return COLORS[name]
