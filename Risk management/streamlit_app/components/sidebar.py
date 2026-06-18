"""Sidebar · 2-Faktor-Macro-Steuerung (Brent + 10y-Zins).

Schlankes Design nach Removal des Vasicek-Single-Factor-M:
- Genau 2 Slider: ΔBrent log-Return + Δr_10y in Prozentpunkten
- Keine Quick-Scenarios, keine M-Aggregation, keine V2-LGD-Knobs
- KEINE Sektor-Sensitivitäts-Override-UI im Sidebar — diese liegt
  methodisch im Intro-Tab, wo die β-Werte ökonomisch
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
        st.markdown(
            '<div class="rm-sidebar-head">'
            '<div class="rm-sidebar-kicker">Model inputs</div>'
            '<div class="rm-sidebar-title">Makro-Stress</div>'
            '<div class="rm-sidebar-sub">Zwei direkte Faktoren steuern alle '
            'Analysen live. Keine Aggregation in einen Einzelfaktor.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # 1. Die zwei Haupt-Slider
        # ------------------------------------------------------------------
        st.markdown(
            '<div class="rm-sidebar-card rm-sidebar-card-oil">'
            '<div class="rm-sidebar-card-title">Brent-Ölpreis</div>'
            '<div class="rm-sidebar-card-text">Log-Return des Brent-Crude-'
            'Preises. +0.50 ≈ +50%, −0.50 ≈ −40%.</div>'
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
        st.markdown(
            f'<div class="rm-sidebar-readout">= <strong>{brent_pct:+.0f}%</strong> '
            'Ölpreis-Veränderung</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="rm-sidebar-card rm-sidebar-card-rate">'
            '<div class="rm-sidebar-card-title">10-Jahres-Zins</div>'
            '<div class="rm-sidebar-card-text">Verschiebung des '
            '10-Jahres-Zinses in Prozentpunkten. +1.00 = +100bp, '
            '−0.50 = −50bp.</div>'
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
        st.markdown(
            f'<div class="rm-sidebar-readout">= <strong>{d_r_10y_pp*100:+.0f} bp</strong> '
            'Δr_10y</div>',
            unsafe_allow_html=True,
        )

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
        # Hinweis: Sektor-Sensitivitäten liegen im Intro-Tab
        # ------------------------------------------------------------------
        _ov = st.session_state.get("sensitivity_overrides")
        if _ov:
            st.markdown(
                '<div class="rm-sidebar-note rm-sidebar-note-warn">'
                f'<strong>{len(_ov)} Sektor-Sensitivität(en) '
                'überschrieben</strong><br>'
                'Override-UI liegt im Intro-Tab · Sensitivitäten.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="rm-sidebar-note">Sektor-Sensitivitäten '
                '(β-Werte pro Exposure-Klasse) sind im <strong>Intro-Tab · '
                'Sensitivitäten</strong> dokumentiert und dort optional '
                'überschreibbar.</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ------------------------------------------------------------------
        # 4. Footer: Datenquellen-Banner
        # ------------------------------------------------------------------
        st.markdown(
            '<div class="rm-sidebar-data">'
            '<div class="rm-sidebar-data-title">Aktive Datenbasis</div>'
            '<div>PDs/LGDs · Pillar-3 EU-CR6 bank-spezifisch (31.12.2024)</div>'
            '<div>Brent · ICE (yfinance), täglich</div>'
            '<div>10y-Zins · Bundesbank Svensson, täglich</div>'
            '<div>Banken-Universe · 10 IRB-Banken</div>'
            '<div>Sensitivitäten · EBA §2.4.2 ¶123 + ECB WP 2897/3207/3112 + FSR 2024</div>'
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
