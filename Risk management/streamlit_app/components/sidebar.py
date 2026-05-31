"""Sidebar · 2-Faktor-Macro-Steuerung (Brent + 10y-Zins).

Schlankes Design nach Removal des Vasicek-Single-Factor-M:
- Genau 2 Slider: ΔBrent log-Return + Δr_10y in Prozentpunkten
- Keine Quick-Scenarios, keine M-Aggregation, keine V2-LGD-Knobs
- KEINE Sektor-Sensitivitäts-Override-UI im Sidebar — diese liegt
  methodisch im Intro-Tab (Schritt 3), wo die β-Werte ökonomisch
  begründet werden. Die Sidebar liest die Overrides aus
  ``st.session_state["sensitivity_overrides"]`` (vom Intro-Tab gesetzt).
- KEINE Yield-Curve-spezifischen β-Slider — der Zins-Slider wirkt
  als reiner Parallel-Shift (Δr_10y). Slope/Curvature-Stress wird
  nicht angeboten, um die UI minimal zu halten.

Return-Dict-Schema (kompatibel zu allen 5 Pages):
    d_brent       — Brent-log-Return-Schock (Dezimal, z.B. +0.50 = +50%)
    d_r_10y_pp    — Δ 10y-Zins in Prozentpunkten (z.B. +2.0 = +200bp)
    d_b0..d_b3    — Svensson-β-Shifts (immer 0 — Legacy-Kompat für
                    Yield-Curve-Plot in der Marktbuch-Page)
    sensitivity_overrides — dict mit per-Klasse-β-Overrides (oder None);
                            wird im Intro-Tab gesetzt und via
                            session_state propagiert
"""
from __future__ import annotations

import numpy as np
import streamlit as st


# ============================================================================
# Public API
# ============================================================================
def render_sidebar() -> dict:
    """Render-Funktion mit Return-Dict für alle Pages.

    Returns
    -------
    dict mit Keys:
        d_brent              : float — ΔBrent log-Return
        d_r_10y_pp           : float — Δr_10y in Prozentpunkten
        d_b0, d_b1, d_b2, d_b3 : float — Svensson-β-Shifts (immer 0)
        sensitivity_overrides : dict | None — Per-Class-β-Overrides
                                (aus st.session_state, gesetzt im Intro)
    """
    with st.sidebar:
        st.title("Makro-Stress")
        st.caption("Zwei direkte Faktoren steuern alle Analysen live. "
                   "Keine Aggregation in einen Einzelfaktor.")

        # ------------------------------------------------------------------
        # 1. Die zwei Haupt-Slider
        # ------------------------------------------------------------------
        st.markdown(
            '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
            'border-left:3px solid #034B6F;padding:0.5rem 0.7rem;'
            'border-radius:5px;margin:0.2rem 0 0.8rem 0;'
            'font-size:0.78rem;color:#051C2C;line-height:1.45;">'
            '<strong>Faktor 1 · Brent (Öl)</strong><br>'
            'Log-Return des Brent-Crude-Preises (geometric).<br>'
            '+0.50 ≈ +50%, −0.50 ≈ −40%.'
            '</div>',
            unsafe_allow_html=True,
        )
        d_brent = st.slider(
            "ΔBrent log-Return",
            min_value=-1.0, max_value=+1.5,
            value=0.0, step=0.05,
            key="d_brent_main",
            help="Brent-Schock als logarithmische Veränderung. Positiv = "
                 "Ölpreis steigt (Energie-Schock), negativ = Ölpreis fällt.",
        )
        brent_pct = (np.exp(d_brent) - 1) * 100
        st.caption(f"= **{brent_pct:+.0f}%** Ölpreis-Veränderung")

        st.markdown(
            '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
            'border-left:3px solid #A52F4D;padding:0.5rem 0.7rem;'
            'border-radius:5px;margin:0.8rem 0 0.6rem 0;'
            'font-size:0.78rem;color:#051C2C;line-height:1.45;">'
            '<strong>Faktor 2 · 10y-Zins</strong><br>'
            'Verschiebung des 10-Jahres-Zinses in Prozentpunkten.<br>'
            '+1.00 = +100bp, −0.50 = −50bp.'
            '</div>',
            unsafe_allow_html=True,
        )
        d_r_10y_pp = st.slider(
            "Δr_10y (Prozentpunkte)",
            min_value=-3.0, max_value=+5.0,
            value=0.0, step=0.1,
            key="d_r_10y_pp_main",
            help="Parallel-Shift der Zinskurve am 10y-Punkt, in pp.",
        )
        st.caption(f"= **{d_r_10y_pp*100:+.0f} bp** Δr_10y")

        # ------------------------------------------------------------------
        # 2. Reset-Knopf
        # ------------------------------------------------------------------
        if st.button("Reset (alle Slider auf 0)", use_container_width=True):
            for k in ("d_brent_main", "d_r_10y_pp_main"):
                if k in st.session_state:
                    del st.session_state[k]
            # Sektor-Overrides werden im Intro-Tab zurückgesetzt
            st.rerun()

        st.divider()

        # ------------------------------------------------------------------
        # 3. Hinweis: Sektor-Sensitivitäten liegen im Intro-Tab
        # ------------------------------------------------------------------
        _ov = st.session_state.get("sensitivity_overrides")
        if _ov:
            st.markdown(
                '<div style="background:#FFF8E1;border:1px solid #FFE082;'
                'border-radius:5px;padding:0.5rem 0.7rem;margin:0.2rem 0;'
                'font-size:0.74rem;color:#051C2C;line-height:1.45;">'
                f'<strong>{len(_ov)} Sektor-Sensitivität(en) '
                'überschrieben</strong><br>'
                'Override-UI liegt im Intro-Tab (Schritt 3).'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "Sektor-Sensitivitäten (β-Werte pro Exposure-Klasse) sind "
                "im **Intro-Tab · Schritt 3** dokumentiert und dort "
                "optional überschreibbar."
            )

        st.divider()

        # ------------------------------------------------------------------
        # 4. Footer: Datenquellen-Banner
        # ------------------------------------------------------------------
        st.markdown(
            '<div style="font-size:0.72rem;color:#6E6E6E;line-height:1.4;">'
            '<strong>Datenbasis:</strong><br>'
            '• PDs/LGDs: Pillar-3 EU-CR6 bank-spezifisch (31.12.2024)<br>'
            '• Brent: ICE (yfinance), täglich<br>'
            '• 10y-Zins: Bundesbank Svensson, täglich<br>'
            '• Banken-Universe: 10 IRB-Banken<br>'
            '• Stress-Sensitivitäten: EBA + Literatur'
            '</div>',
            unsafe_allow_html=True,
        )

    return {
        "d_brent":               float(d_brent),
        "d_r_10y_pp":            float(d_r_10y_pp),
        # Yield-Curve-β-Shifts entfernt aus UI — Defaults beibehalten
        # für Kompatibilität mit legacy_views.py (Marktbuch · Yield-Curve)
        "d_b0":                  0.0,
        "d_b1":                  0.0,
        "d_b2":                  0.0,
        "d_b3":                  0.0,
        "sensitivity_overrides": st.session_state.get(
            "sensitivity_overrides"),
    }
