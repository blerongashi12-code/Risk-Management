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


# Nav-Hook: setzt beim Klick auf einen Sidebar-Link die Klasse `mc-nav` auf
# stApp (→ Boot-Overlay aus mckinsey.css erscheint) und entfernt sie, sobald
# der Script-Run beendet ist. Läuft im Parent-Dokument (same-origin-iframe);
# Guard `__mcNavHook` verhindert Doppel-Registrierung über Seitenwechsel.
# Klicks auf den bereits aktiven Tab (aria-current="page") lösen nichts aus.
_NAV_HOOK_JS = """
<script>
(function () {
  try {
    var P = window.parent, D = P.document;
    if (P.__mcNavHook) return;
    P.__mcNavHook = true;
    // WICHTIG: Die Logik muss im PARENT-Realm leben — dieser Komponenten-
    // iframe wird bei jedem Seitenwechsel zerstoert (Timer/Closures aus dem
    // iframe-Realm sterben mit). Darum ein <script> ins Parent-Dokument.
    var code = [
      "(function () {",
      "  var t0 = 0;",
      "  document.addEventListener('click', function (e) {",
      "    var a = e.target && e.target.closest",
      "            ? e.target.closest('[data-testid=\\"stSidebarNavLink\\"]') : null;",
      "    if (!a || a.getAttribute('aria-current') === 'page') return;",
      "    var app = document.querySelector('[data-testid=\\"stApp\\"]');",
      "    if (app) { app.classList.add('mc-nav'); t0 = Date.now(); }",
      "  }, true);",
      "  setInterval(function () {",
      "    var app = document.querySelector('[data-testid=\\"stApp\\"]');",
      "    if (!app || !app.classList.contains('mc-nav')) return;",
      "    var running = app.getAttribute('data-test-script-state') === 'running';",
      "    if (!running && Date.now() - t0 > 900) app.classList.remove('mc-nav');",
      "  }, 250);",
      "})();"
    ].join("\\n");
    var s = D.createElement("script");
    s.textContent = code;
    D.head.appendChild(s);
  } catch (err) { /* nie die App blockieren */ }
})();
</script>
"""


def _inject_nav_hook() -> None:
    """0-Pixel-Komponenten-iframe, dessen Skript den Seitenwechsel-Overlay
    steuert (st.markdown führt kein JS aus, Komponenten-iframes schon; der
    Layout-Slot wird per CSS ausgeblendet)."""
    from streamlit.components.v1 import html as _html
    _html(_NAV_HOOK_JS, height=0)


# =====================================================================
# 4. Public API
# =====================================================================
_THEME_APPLIED_KEY = "_mc_theme_applied"


def apply_theme() -> None:
    """Idempotent — erst CSS, dann Plotly-Template registrieren."""
    _register_plotly_template()
    _inject_css()
    _inject_nav_hook()
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


def tab_breadcrumb(current: int) -> None:
    """Roter-Faden-Indikator · zeigt Position in der 5-Tab-Pipeline.

    Pro Tab ein Breadcrumb-Streifen mit allen 5 Stationen — die aktuelle
    Position ist hervorgehoben, die anderen dezenter. Direkt unter dem
    Hero einsetzen.

    Parameters
    ----------
    current : int
        Tab-Nummer (1-5).
    """
    tabs = [
        ("1", "Faktor-Analyse & Transmission",
         "Korrelations-Analyse + sequentielle Bridge"),
        ("2", "Kreditbuch",
         "Loan-Book · ΔPD/ΔLGD · Capital-Bridge"),
        ("3", "Marktbuch",
         "Yield-Curve · Sovereign-Kanal (ΔMtM via OCI)"),
        ("4", "Eigenkapital",
         "2-Kanal · CET1-Quote · Schwellen"),
        ("5", "Validierung",
         "Walk-Forward-Backtest · Annahmen"),
    ]
    items = []
    for idx, (num, name, sub) in enumerate(tabs, start=1):
        is_active = (idx == current)
        bg = "#051C2C" if is_active else "#FFFFFF"
        fg = "#FFFFFF" if is_active else "#051C2C"
        border = "#051C2C" if is_active else "#E6E6E6"
        sub_color = "#C9C9C9" if is_active else "#6E6E6E"
        item = (
            f'<div style="flex:1;min-width:130px;background:{bg};'
            f'border:1px solid {border};border-radius:4px;'
            f'padding:0.55rem 0.7rem;color:{fg};font-size:0.78rem;'
            f'line-height:1.35;">'
            f'<div style="font-size:0.66rem;letter-spacing:0.10em;'
            f'text-transform:uppercase;opacity:0.85;">Tab {num}</div>'
            f'<div style="font-weight:600;font-size:0.86rem;'
            f'margin:0.1rem 0;">{name}</div>'
            f'<div style="color:{sub_color};font-size:0.72rem;'
            f'line-height:1.35;">{sub}</div>'
            f'</div>'
        )
        items.append(item)
    arrow = ('<div style="display:flex;align-items:center;color:#9B9B9B;'
             'font-size:0.95rem;padding:0 4px;">→</div>')
    html = (
        '<div style="display:flex;gap:6px;flex-wrap:nowrap;overflow-x:auto;'
        'margin:0.4rem 0 1.0rem 0;">'
        + arrow.join(items)
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def kpi_columns(n: int):
    """Wrapper um st.columns mit konsistenten Gaps für KPI-Strips."""
    return st.columns(n, gap="small")


# =====================================================================
# 5. Color helpers (für ad-hoc Plotly-Calls)
# =====================================================================
def color(name: str) -> str:
    """Liefert einen Farbwert per Token-Name (`navy`, `bright_blue`, …)."""
    return COLORS[name]
