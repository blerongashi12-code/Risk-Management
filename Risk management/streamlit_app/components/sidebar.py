"""Global sidebar with macro-shock controls.

Rendered first on every page. Returns a config dict consumed by all
pages to drive the live macroeconomic stress.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from components.data_loader import delta_r_from_beta_shifts


def render_sidebar() -> dict:
    """Renders the macro-shock sliders and returns the live config."""
    with st.sidebar:
        st.title("Macro stress")
        st.caption("Live controls — every chart re-computes on change.")

        # ---------- Energy price shock ----------
        with st.expander("Energy price shock", expanded=True):
            d_brent = st.slider(
                "ΔBrent (cumulative log-return)",
                min_value=-2.0, max_value=2.0, value=0.0, step=0.05,
                help="Log-return of Brent over the stress horizon. "
                     "−1.20 ≈ Corona crash, +0.55 ≈ Iran escalation.",
                key="d_brent",
            )
            pct = (np.exp(d_brent) - 1) * 100
            st.caption(f"≈ {pct:+.0f}% Brent price change")

        # ---------- Yield curve shift ----------
        with st.expander("Yield-curve shift (Svensson β)", expanded=True):
            d_b0 = st.slider("Δβ₀ — level (parallel)",
                             -3.0, 3.0, 0.0, 0.05,
                             help="Shifts the entire curve.", key="d_b0")
            d_b1 = st.slider("Δβ₁ — slope (short end)",
                             -3.0, 3.0, 0.0, 0.05,
                             help="Mainly short-end effect.", key="d_b1")
            d_b2 = st.slider("Δβ₂ — curvature 1",
                             -3.0, 3.0, 0.0, 0.05,
                             help="Mid-maturity hump around τ₁.", key="d_b2")
            d_b3 = st.slider("Δβ₃ — curvature 2",
                             -3.0, 3.0, 0.0, 0.05,
                             help="Long-end hump around τ₂.", key="d_b3")

            d_r10 = delta_r_from_beta_shifts(d_b0, d_b1, d_b2, d_b3)
            st.metric("Implied Δr_10y",
                      f"{d_r10*100:+.1f} bp",
                      help="Δr at τ = 10y from the β shifts.")

        st.divider()

        # ---------- Reset & quick scenarios ----------
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reset", use_container_width=True):
                for k in ("d_brent", "d_b0", "d_b1", "d_b2", "d_b3",
                          "quick_select", "_last_quick"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        with col_b:
            scenario_quick = st.selectbox(
                "Quick scenario",
                ["—", "Corona 2020", "Ukraine 2022", "Iran 2026",
                 "EBA 2025 adverse"],
                label_visibility="collapsed", key="quick_select",
            )
            _apply_quick_scenario(scenario_quick)

        st.divider()
        st.caption("**Pipeline**")
        st.caption(
            "Δβ → Δr_10y · ΔBrent  →  M (Vasicek) · duration P&L (sovereign)"
        )

    return {
        "d_brent":     d_brent,
        "d_b0": d_b0, "d_b1": d_b1, "d_b2": d_b2, "d_b3": d_b3,
        "d_r_10y_pp":  d_r10,
    }


def _apply_quick_scenario(name: str) -> None:
    """Applies a preset (ΔBrent, Δβ₀) on the sliders."""
    if name == "—":
        return
    presets = {
        "Corona 2020":       {"d_brent": -1.20, "d_b0": -0.65,
                              "d_b1": 0.0, "d_b2": 0.0, "d_b3": 0.0},
        "Ukraine 2022":      {"d_brent": +0.12, "d_b0": +0.67,
                              "d_b1": 0.0, "d_b2": 0.0, "d_b3": 0.0},
        "Iran 2026":         {"d_brent": +0.55, "d_b0": +0.50,
                              "d_b1": 0.0, "d_b2": 0.0, "d_b3": 0.0},
        "EBA 2025 adverse":  {"d_brent": +0.47, "d_b0": +2.00,
                              "d_b1": 0.0, "d_b2": 0.0, "d_b3": 0.0},
    }
    if name not in presets:
        return
    if st.session_state.get("_last_quick") == name:
        return
    for k, v in presets[name].items():
        st.session_state[k] = v
    st.session_state["_last_quick"] = name
    st.rerun()
