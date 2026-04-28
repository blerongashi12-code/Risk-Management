"""Globale Sidebar mit Stress-Konfigurations-Slidern.

Wird auf jeder Page als erstes gerendert. Liefert ein Config-Dict zurück,
das alle Pages zur Stress-Berechnung verwenden.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from components.data_loader import delta_r_from_beta_shifts


def render_sidebar() -> dict:
    """Rendert die globalen Stress-Slider und gibt das Config-Dict zurück."""
    with st.sidebar:
        st.title("Stress-Konfiguration")
        st.caption("Live-Regler — alle Charts reagieren in Echtzeit.")

        # ---------- Energiepreis ----------
        with st.expander("⚡ Energiepreis-Schock", expanded=True):
            d_brent = st.slider(
                "ΔBrent (kumulativer log-Return)",
                min_value=-2.0, max_value=2.0, value=0.0, step=0.05,
                help="Log-Return des Brent-Preises über den Stress-Horizon. "
                     "−1.20 ≈ Corona-Crash, +0.55 ≈ Iran-Eskalation",
                key="d_brent",
            )
            pct = (np.exp(d_brent) - 1) * 100
            st.caption(f"≈ {pct:+.0f} % Brent-Preisänderung "
                       f"(0 = unverändert, −100% = Verlust, +100 % = Verdopplung)")

        # ---------- Svensson-β-Shifts ----------
        with st.expander("📈 Zinskurven-Shift (Svensson β-Parameter)",
                         expanded=True):
            d_b0 = st.slider("Δβ₀ (Level — parallel)",
                             -3.0, 3.0, 0.0, 0.05,
                             help="Verschiebt die gesamte Kurve.", key="d_b0")
            d_b1 = st.slider("Δβ₁ (Slope — kurzes Ende)",
                             -3.0, 3.0, 0.0, 0.05,
                             help="Wirkt vor allem kurzfristig.", key="d_b1")
            d_b2 = st.slider("Δβ₂ (Curvature 1)",
                             -3.0, 3.0, 0.0, 0.05,
                             help="Mittel-Maturity-Buckel um τ₁.", key="d_b2")
            d_b3 = st.slider("Δβ₃ (Curvature 2)",
                             -3.0, 3.0, 0.0, 0.05,
                             help="Langes Ende-Buckel um τ₂.", key="d_b3")

            d_r10 = delta_r_from_beta_shifts(d_b0, d_b1, d_b2, d_b3)
            st.metric("→ Δr_10y resultierend",
                      f"{d_r10*100:+.1f} bp",
                      help="Aus den β-Shifts berechnetes Δr bei τ = 10y.")

        # ---------- Korrelation & MC ----------
        with st.expander("🎲 MC & Korrelation", expanded=False):
            corr_mode = st.radio(
                "Brent ↔ Δr_10y Korrelation",
                ["historisch (252-Tage)", "manuell"],
                horizontal=True, key="corr_mode",
            )
            if corr_mode == "manuell":
                corr_override = st.slider(
                    "Korrelation",
                    -0.95, 0.95, 0.0, 0.05,
                    help="Override der historischen Korrelation in der MC-Σ",
                    key="corr_override",
                )
            else:
                corr_override = None

            horizon_days = st.slider(
                "Time Horizon (Handelstage)",
                30, 504, 252, 1,
                help="MC-Pfad-Länge. 252 = 1 Jahr, 504 = 2 Jahre",
                key="horizon_days",
            )
            n_sims = st.select_slider(
                "Monte-Carlo-Pfade",
                options=[1_000, 5_000, 10_000, 25_000, 50_000, 100_000],
                value=10_000,
                help="Mehr Pfade = stabilere Quantile, aber langsamer.",
                key="n_sims",
            )

        st.divider()

        # ---------- Reset & Quick-Apply ----------
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Reset", use_container_width=True):
                for k in ("d_brent", "d_b0", "d_b1", "d_b2", "d_b3",
                          "corr_override", "horizon_days", "n_sims",
                          "corr_mode"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        with col_b:
            scenario_quick = st.selectbox(
                "Quick:",
                ["–", "Corona 2020", "Ukraine 2022", "Iran 2026"],
                label_visibility="collapsed", key="quick_select",
            )
            # Apply quick scenario via session_state on rerun
            _apply_quick_scenario(scenario_quick)

        st.divider()
        st.caption("**Stress-Pipeline:**")
        st.caption("Δβ → Δr → β_R · Δr  +  β_E · ΔBrent → ΔE → Stress-PD")

    return {
        "d_brent": d_brent,
        "d_b0": d_b0, "d_b1": d_b1, "d_b2": d_b2, "d_b3": d_b3,
        "d_r_10y_pp": d_r10,
        "corr_override": corr_override,
        "horizon_days": horizon_days,
        "n_sims": n_sims,
    }


def _apply_quick_scenario(name: str) -> None:
    """Setzt Slider auf vordefinierte Werte aus der scenarios-Library."""
    if name == "–":
        return
    presets = {
        "Corona 2020":  {"d_brent": -1.20, "d_b0": -0.65, "d_b1": 0.0,
                         "d_b2": 0.0, "d_b3": 0.0},
        "Ukraine 2022": {"d_brent": +0.12, "d_b0": +0.67, "d_b1": 0.0,
                         "d_b2": 0.0, "d_b3": 0.0},
        "Iran 2026":    {"d_brent": +0.55, "d_b0": +0.50, "d_b1": 0.0,
                         "d_b2": 0.0, "d_b3": 0.0},
    }
    if name not in presets:
        return
    if st.session_state.get("_last_quick") == name:
        return  # bereits angewendet
    for k, v in presets[name].items():
        st.session_state[k] = v
    st.session_state["_last_quick"] = name
    st.rerun()
