"""European Banking Credit Stress Cockpit — Streamlit frontend.

Entry point. Streamlit auto-discovers pages in `pages/` and renders
them in the left navigation.

Start:
    streamlit run streamlit_app/app.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import streamlit as st

from components.theme import (apply_theme, hero, eyebrow, insight, footer,
                              COLORS)
from components.sidebar import render_sidebar
from components.data_loader import load_data_layer
from components.backend_path import setup
setup()


st.set_page_config(
    page_title="EU Banking Credit Stress Cockpit",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# Sidebar
config = render_sidebar()

# === Hero =============================================================
hero(
    "Credit Stress Cockpit",
    eyebrow="Vasicek / ASRF · EBA Transparency 2025",
    deck="Regulatory credit-risk view of the European banking system. "
         "Basel-III IRB capital formulas applied to EBA-anchored bank "
         "portfolios; the same Δr_10y shock that drives the loan-book "
         "PDs also drives a duration-based mark-to-market on the "
         "sovereign book.",
)

# === Data layer status + macro snapshot ===============================
data = load_data_layer()
sven = data["svensson"]
brent = data["brent"]

eyebrow("System snapshot")

snap_cols = st.columns(5, gap="small")

# 1. As-of date
if sven is not None:
    snap_cols[0].metric("As of",
                        sven.index[-1].strftime("%Y-%m-%d"),
                        "Bundesbank reporting day", delta_color="off")
else:
    snap_cols[0].metric("As of", "—", "data missing", delta_color="off")

# 2. 10y zero rate (latest with 1m delta)
if sven is not None:
    from svensson import historical_curve, zero_rate  # type: ignore
    p = historical_curve(sven.index[-1], sven)
    r10 = float(zero_rate(10.0, p, as_decimal=False))
    if len(sven) > 22:
        p_prev = historical_curve(sven.index[-22], sven)
        r10_prev = float(zero_rate(10.0, p_prev, as_decimal=False))
        delta_bp = (r10 - r10_prev) * 100
        snap_cols[1].metric("10y zero rate",
                            f"{r10:.2f}%",
                            f"{delta_bp:+.0f} bp · 1m")
    else:
        snap_cols[1].metric("10y zero rate", f"{r10:.2f}%", delta_color="off")
else:
    snap_cols[1].metric("10y zero rate", "—", delta_color="off")

# 3. Brent crude
if brent is not None and "Close_USD" in brent.columns:
    last_brent = float(brent["Close_USD"].iloc[-1])
    if len(brent) > 22:
        prev_brent = float(brent["Close_USD"].iloc[-22])
        delta_pct = (last_brent / prev_brent - 1) * 100
        snap_cols[2].metric("Brent crude",
                            f"${last_brent:.1f}",
                            f"{delta_pct:+.1f}% · 1m")
    else:
        snap_cols[2].metric("Brent crude", f"${last_brent:.1f}",
                            delta_color="off")
else:
    snap_cols[2].metric("Brent crude", "—", delta_color="off")

# 4. EBA universe coverage
@st.cache_data(ttl=24*3600, show_spinner=False)
def _quick_eba_kpis(top_n: int):
    from eba_loader import load_eba_universe                # type: ignore
    u = load_eba_universe(vintage="2025", top_n=top_n)
    return {
        "n_banks":       u.n_banks,
        "total_ead":     float(u.total_ead_eur),
        "source":        u.source,
        "is_real":       u.source.lower().startswith("eba transparency"),
    }

try:
    eba = _quick_eba_kpis(top_n=10)
    snap_cols[3].metric("EBA banks loaded",
                        f"{eba['n_banks']}",
                        "Top by IRB EAD", delta_color="off")
    snap_cols[4].metric("Σ EAD (top-10)",
                        f"€{eba['total_ead']/1e12:.2f} tn",
                        "EBA Transparency 2025"
                        if eba["is_real"] else "Synthetic anchors",
                        delta_color="off")
except Exception:
    snap_cols[3].metric("EBA banks loaded", "—", "data missing",
                        delta_color="off")
    snap_cols[4].metric("Σ EAD", "—", delta_color="off")

# Cache caption
status_parts = []
if sven is not None:
    status_parts.append(f"svensson {len(sven):,}d")
if brent is not None:
    status_parts.append(f"brent {len(brent):,}d")
st.caption("Cache · " + " · ".join(status_parts) if status_parts
           else "Cache empty — run the data fetchers first")

st.divider()

# === Methodology overview ============================================
eyebrow("Pipeline architecture")

arch_l, arch_r = st.columns([3, 2], gap="medium")
with arch_l:
    st.markdown("""
The cockpit operates a regulatory credit-risk lens on the European
banking system, driven by a single macroeconomic shock vector — Brent
log-returns and a parallel 10-year sovereign yield shift.

**Loan-book channel.** Vasicek single-factor (ASRF) — the mathematical
foundation of Basel III IRB — applied to EBA-anchored bank portfolios
by exposure class. Conditional PDs under macro stress translate into
expected loss, unexpected loss and regulatory capital. The macro shock
is mapped onto the systematic factor M via an EBA stress-test anchor,
validated against the empirical Brent / Δr covariance.

**Sovereign-book channel.** The same Δr_10y also drives a duration-based
Mark-to-Market on the sovereign portfolio. Per-bank maturity ladders
from the EBA Transparency Exercise yield the modified-duration approxi-
mation P&L = −D · Δy · Exposure, summed across the seven maturity
buckets.

**Trading-book channel.** A third channel scales the published Market-
Risk RWA (Item 2520210) under stress (FRTB-style multiplier on VaR/SVaR)
and applies a Trading-Book P&L haircut. All three channels feed the
CET1-Ratio numerator and denominator on the Capital Adequacy page.
""")

with arch_r:
    insight(
        "<strong>Three-channel consistency.</strong> The same Δr_10y drives "
        "Loan-Book PDs (Vasicek M), Sovereign-Book Mark-to-Market "
        "(modified duration), and Trading-Book Market-RWA + P&L. "
        "Three independent risk lenses, one shock — converging on the "
        "regulatory CET1 ratio."
    )
    insight(
        "<strong>Why Vasicek?</strong> It is the literal regulatory model: "
        "the closed-form Basel-III IRB capital formula derives directly "
        "from the Vasicek 2002 portfolio-loss quantile. EBA-anchored "
        "calibration makes the cockpit citable to regulatory reviewers."
    )

st.divider()

# === Navigation cards ================================================
eyebrow("Navigation")

st.markdown(
    "<div style='font-family: var(--mc-sans); font-size: 0.85rem; "
    "color: var(--mc-stone); margin: 0.4rem 0 1rem 0; "
    "letter-spacing: 0.04em; text-transform: uppercase;'>Risk channels"
    "</div>",
    unsafe_allow_html=True,
)
nav1 = st.columns(3, gap="small")
with nav1[0]:
    st.page_link("pages/1_Bank_Portfolio.py", label="Bank Portfolio",
                 help="EBA Top-N banks · IRB capital under macro stress",
                 use_container_width=True)
with nav1[1]:
    st.page_link("pages/2_Sovereign_Risk.py", label="Sovereign Risk",
                 help="Doom-loop concentration · duration-shock P&L",
                 use_container_width=True)
with nav1[2]:
    st.page_link("pages/3_Capital_Adequacy.py", label="Capital Adequacy",
                 help="Three-channel CET1 ratio · Loan + Sovereign + "
                      "Trading Book",
                 use_container_width=True)

st.markdown(
    "<div style='font-family: var(--mc-sans); font-size: 0.85rem; "
    "color: var(--mc-stone); margin: 1.4rem 0 1rem 0; "
    "letter-spacing: 0.04em; text-transform: uppercase;'>Engine &amp; reference"
    "</div>",
    unsafe_allow_html=True,
)
nav2 = st.columns(3, gap="small")
with nav2[0]:
    st.page_link("pages/4_Yield_Curve.py", label="Yield Curve",
                 help="Bundesbank Svensson · multi-maturity shifts · "
                      "factor history",
                 use_container_width=True)
with nav2[1]:
    st.page_link("pages/5_Annahmen.py", label="Annahmen & Datenbasis",
                 help="Three-tier governance disclosure · "
                      "Executive · Validator · Quant",
                 use_container_width=True)
with nav2[2]:
    st.page_link("pages/6_Methodology.py", label="Methodology",
                 help="Full assumptions document · all formulas + sources",
                 use_container_width=True)

footer(
    "Methodology in MODEL_ASSUMPTIONS.md · Live sliders in the sidebar drive "
    "every chart on every page · Cache TTL 24 h · "
    "Backend modules in `Risk management/backend/`"
)
