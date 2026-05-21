"""Global Sidebar · Single-Factor M (Vasicek/ASRF) als Hauptregler.

Methodischer Hintergrund
------------------------
Das Basel-III-IRB-Modell ist per Konstruktion ein **Ein-Faktor-Modell**
(Vasicek/ASRF). Es gibt genau einen systematischen Faktor

    M ~ N(0, 1)

der alle PDs bedingt. Adverse Schocks → M < 0. Der EBA-2025-Adverse-Stress-
Test entspricht M ≈ −2.50.

Die Sidebar bietet zwei Modi:
  · **Single-Factor M** (Default) — ein einziger Slider treibt M direkt.
    Brent- und Δr-Werte werden aus M via EBA-Anker-Kopplung abgeleitet.
  · **Multi-Factor (Advanced)** — Originaldarstellung mit getrennten
    Brent- und Svensson-β-Slidern. Für Yield-Curve-Page nötig.

Backward-Kompatibilität: der zurückgegebene Config-Dict hat in beiden
Modi exakt dieselben Keys, sodass alle Pages unverändert weiterlaufen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

from components.data_loader import delta_r_from_beta_shifts
from components.backend_path import setup
setup()

from macro_factor import anchor_from_eba  # type: ignore


# Anchor-Direction für M → (Brent, Δr)-Ableitung
_ANCHOR = anchor_from_eba("2025")


def _derive_from_m(m: float) -> tuple[float, float]:
    """M → (ΔBrent_log, Δr_10y in pp) via EBA-Adverse-Anker-Direction.

    Der EBA-2025-Adverse-Anker spezifiziert die "adverse Richtung" im
    (Brent, Δr)-Raum: Brent +0.47 log, Δr +2.0 pp, M = −2.5.
    Wir koppeln M linear an diese Richtung:

        Brent(M)  = − M · (anchor.brent_log    / |anchor.z|)
        Δr_10y(M) = − M · (anchor.rate_10y_pp  / |anchor.z|)

    Negatives Vorzeichen, weil der Anker-z negativ ist (adverse).
    M = −2.5 → exakt die Anker-Werte. M = 0 → keine Schocks.
    M = +1.0 → benigne Schocks (Brent runter, Rates runter).
    """
    if abs(_ANCHOR.z_factor) < 1e-9:
        return 0.0, 0.0
    scale = -m / abs(_ANCHOR.z_factor)
    return (
        _ANCHOR.brent_log    * scale * np.sign(-_ANCHOR.z_factor),  # log return
        _ANCHOR.rate_10y_pp  * scale * np.sign(-_ANCHOR.z_factor),  # in pp
    )


def render_sidebar() -> dict:
    """Render the sidebar controls and return a config dict.

    Returns
    -------
    dict with keys:
        d_brent       — ΔBrent log-return (für Vasicek-Mapping + Bonds)
        d_b0…d_b3     — Svensson-β-Shifts in pp (für Yield-Curve-Page)
        d_r_10y_pp    — Δr_10y in PERCENTAGE POINTS (pp); für sov-MtM
        m_direct      — User-gewählter M (oder None im Multi-Factor-Modus)
    """
    with st.sidebar:
        st.title("Makro-Stress")
        st.caption("Vasicek/ASRF Ein-Faktor-Modell  ·  alle Charts "
                   "rechnen live nach.")

        # ---------------- Mode toggle ----------------
        mode = st.radio(
            "Eingabe-Modus",
            ["Single-Factor M (Standard)",
             "Multi-Factor (Advanced)"],
            index=0, key="stress_mode",
            help="Single-Factor: ein Slider treibt M direkt — empfohlen, "
                 "weil das Modell intern Single-Factor ist. "
                 "Multi-Factor: separate Brent- und β-Slider; für "
                 "Yield-Curve-Analyse nötig.",
        )

        if mode == "Single-Factor M (Standard)":
            cfg = _render_single_factor()
        else:
            cfg = _render_multi_factor()

        # ---------------- V2 · LGD-Modell-Risiko-Parameter -----------------
        with st.expander("V2 · LGD-Modell-Risiko-Parameter (4 Knobs)",
                         expanded=False):
            st.markdown(
                '<div style="font-size:0.78rem;color:#A52F4D;'
                'background:#FAECE7;border-left:3px solid #A52F4D;'
                'padding:0.4rem 0.6rem;border-radius:4px;margin-bottom:0.6rem;'
                'line-height:1.4;">'
                '<strong>V2-Modell-Erweiterungen</strong> — adressieren '
                'vier akademische Kritikpunkte am F-IRB-LGD-Modell. '
                'Defaults = V1-Verhalten (keine Veränderung gegenüber Basel-'
                'Standard ohne explizite Aktivierung).'
                '</div>',
                unsafe_allow_html=True,
            )

            lgd_calibration = st.slider(
                "1 · A-IRB-Kalibrierung · LGD-Multiplikator",
                min_value=0.40, max_value=1.50,
                value=float(st.session_state.get("lgd_cal", 1.0)),
                step=0.05, key="lgd_cal",
                help="F-IRB-LGDs (Basel-Default 45% Corp / 20% Mortgage / "
                     "65% QRRE) sind regulatorisch konservativ. Reale "
                     "A-IRB-LGDs mit Sicherheiten liegen bei 20-30% für "
                     "Corporates. 0.6× ≈ ein typischer A-IRB-Abschlag.",
            )
            if abs(lgd_calibration - 1.0) < 0.01:
                st.caption("Zustand: **F-IRB-Default** (Basel-konservativ)")
            elif lgd_calibration < 1.0:
                _eff_corp = 0.45 * lgd_calibration * 100
                st.caption(
                    f"Zustand: **A-IRB-Proxy** · effektive Corporate-LGD = "
                    f"{_eff_corp:.0f}% (statt 45%) · Sicherheiten "
                    f"eingerechnet"
                )
            else:
                st.caption(f"Zustand: **Conservative-Up** · "
                           f"+{(lgd_calibration-1)*100:.0f}% über F-IRB")

            stress_exponent = st.slider(
                "2 · Non-Linearer Stress-Exponent (Tail-Risk)",
                min_value=0.5, max_value=2.5,
                value=float(st.session_state.get("stress_exp", 1.0)),
                step=0.1, key="stress_exp",
                help="LGD_stress = LGD_base · (1 + κ · |M|^p). "
                     "p = 1.0 = lineare V1 · p > 1.0 = überproportionaler "
                     "Anstieg bei extremen |M| (Tail-Risk / Crisis-Markets) · "
                     "p < 1.0 = abgeflachte Stress-Antwort.",
            )
            if abs(stress_exponent - 1.0) < 0.05:
                st.caption("Zustand: **linear** (V1-Default)")
            elif stress_exponent > 1.0:
                _ex_mult = 0.30 * (2.5 ** stress_exponent)
                st.caption(
                    f"Zustand: **konvex (Tail-Risk)** · bei |M|=2.5 wirkt "
                    f"κ·|M|^p = {_ex_mult:.2f} statt linear 0.75"
                )
            else:
                st.caption(f"Zustand: **konkav** · gedämpfte Tail-Response")

            cap_mode = st.radio(
                "3 · LGD-Cap-Modus",
                ["Hard-Cap bei 100% (V1)",
                 "Logit-Transformation (asymptotisch)"],
                index=0, key="cap_mode_radio",
                help="Bei QRRE (Baseline 65%) wird LGD im V1 ab |M| ≈ 1.79 "
                     "auf 100% gecappt — Informationsverlust im Extrembereich. "
                     "Logit-Modus ersetzt den Cap durch eine Sigmoid-"
                     "Transformation, die sich 100% asymptotisch nähert.",
            )
            cap_mode_value = "logit" if "Logit" in cap_mode else "hard"
            if cap_mode_value == "logit":
                st.caption("Zustand: **Logit** · keine Diskontinuität bei 100%")
            else:
                st.caption("Zustand: **Hard-Cap** (regulatorisch konform)")

            st.markdown(
                '<div style="font-size:0.75rem;color:#6E6E6E;'
                'background:#F4F4F4;border-radius:3px;padding:0.4rem 0.55rem;'
                'margin-top:0.5rem;line-height:1.4;">'
                '<strong>4. Sektor-/Länder-Granularität:</strong> derzeit '
                'einheitliches κ pro Class. V3-Roadmap: branchen- und '
                'länderspezifisches κ_i (höher für zyklische Sektoren).'
                '</div>',
                unsafe_allow_html=True,
            )

            cfg.update({
                "lgd_calibration": float(lgd_calibration),
                "stress_exponent": float(stress_exponent),
                "cap_mode":        cap_mode_value,
            })

        # ---------------- ρ · Asset Correlation Override ----------------
        with st.expander("ρ-Override · Asset-Korrelations-Multiplikator",
                         expanded=False):
            st.markdown(
                '<div style="font-size:0.78rem;color:#A52F4D;'
                'background:#FAECE7;border-left:3px solid #A52F4D;'
                'padding:0.4rem 0.6rem;border-radius:4px;margin-bottom:0.6rem;'
                'line-height:1.4;">'
                '<strong>Modell-Risiko-Parameter</strong> — kein Basel-Standard. '
                'Default 1.0× = Basel-III-formelmäßige ρ. '
                'Werte > 1.0× simulieren höhere Asset-Korrelation '
                '(z.B. Krisen-Stress), < 1.0× geringere.'
                '</div>',
                unsafe_allow_html=True,
            )
            rho_mult = st.slider(
                "ρ-Multiplikator (Stress auf die Basel-Korrelation)",
                min_value=0.5, max_value=2.0,
                value=float(st.session_state.get("rho_mult", 1.0)),
                step=0.05, key="rho_mult",
                help="Skaliert ρ pro Klasse uniformement. Wirkt sowohl im "
                     "Conditional-PD als auch im ASRF-Loss-Quantil und damit "
                     "in der gesamten Capital-Bridge.",
            )
            if abs(rho_mult - 1.0) < 0.01:
                st.caption("Zustand: **Basel-Standard** (keine ρ-Manipulation)")
            elif rho_mult > 1.0:
                st.caption(
                    f"Zustand: **ρ erhöht** um {(rho_mult-1)*100:.0f}% — "
                    "stärkere Co-Bewegung, höhere Konzentrations-Anfälligkeit"
                )
            else:
                st.caption(
                    f"Zustand: **ρ reduziert** um {(1-rho_mult)*100:.0f}% — "
                    "schwächere Co-Bewegung, niedrigere Capital-Charge"
                )
            cfg["rho_mult"] = float(rho_mult)

        # ---------------- ρ Reference ----------------
        with st.expander("ρ · Asset-Korrelation — Referenz pro Klasse"):
            st.markdown(r"""
**Was ist ρ?**
Die *Asset-Korrelation* ρ misst, wie stark der einzelne Schuldner an den
systematischen Faktor M koppelt. Höhere ρ → stärkere Reaktion der PD auf
einen Schock in M.

**Basel-III-IRB-Kalibrierung** (BCBS 2017, Art. 153 CRR):

| Exposure-Class | Asset-Korrelation ρ |
|---|---|
| Corporate / Sovereign / Bank | **12 – 24 %** (PD-abhängig, Formel) |
| SME-Corporate | 12 – 24 % minus Größen-Adjustment |
| Residential Mortgage | **15 %** (fix) |
| Qualifying Revolving Retail (QRRE) | **4 %** (fix) |
| Other Retail | **3 – 16 %** (PD-abhängig) |

**Formel Corporate:**

$$\rho(\text{PD}) = 0.12 \cdot \frac{1 - e^{-50\,\text{PD}}}{1 - e^{-50}} + 0.24 \cdot \!\left(1 - \frac{1 - e^{-50\,\text{PD}}}{1 - e^{-50}}\right)$$

Niedrige PD → ρ nahe 24 %  ·  hohe PD → ρ nahe 12 %.

**Conditional-PD-Formel (Vasicek 2002):**

$$\text{PD}(M) = N\!\!\left(\frac{N^{-1}(\text{PD}) - \sqrt{\rho}\,M}{\sqrt{1-\rho}}\right)$$

ρ steht im Nenner — je größer ρ, desto sensitiver die PD gegen M.
""")

        # ---------------- Reset ----------------
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reset", use_container_width=True,
                         help="Alle Slider auf Baseline (kein Schock)"):
                for k in list(st.session_state.keys()):
                    if k.startswith(("m_input", "d_brent", "d_b",
                                      "quick_select", "_last_quick",
                                      "rho_mult", "lgd_cal", "stress_exp",
                                      "cap_mode_radio")):
                        del st.session_state[k]
                st.rerun()
        with col_b:
            scenario_quick = st.selectbox(
                "Quick-Szenario",
                ["—", "Corona 2020", "Ukraine 2022",
                 "Iran 2026", "EBA 2025 adverse"],
                label_visibility="collapsed", key="quick_select",
            )
            _apply_quick_scenario(scenario_quick, mode)

        st.divider()
        st.caption(
            "Pipeline · M → Vasicek-Conditional-PD → Capital-Bridge → CET1"
        )

    return cfg


# =====================================================================
# Single-Factor mode
# =====================================================================
def _render_single_factor() -> dict:
    """One slider drives M directly. Brent & Δr are derived via anchor."""
    with st.expander("Systemfaktor M (Hauptregler)", expanded=True):
        m_input = st.slider(
            "Vasicek-Systemfaktor M  (Standard-Normal · N(0, 1))",
            min_value=-3.0, max_value=3.0,
            value=float(st.session_state.get("m_input", 0.0)),
            step=0.05,
            key="m_input",
            help="M < 0 = adverse (Default-Wahrscheinlichkeiten steigen). "
                 "M > 0 = benigne (PDs sinken). "
                 "EBA 2025 Adverse Anker ≈ −2.50.",
        )
        # Severity context
        if abs(m_input) < 0.05:
            st.caption("Zustand: **Baseline** — keine Schocks angelegt")
        elif m_input < 0:
            share = abs(m_input / _ANCHOR.z_factor) * 100
            st.caption(
                f"Zustand: **adverse** · {share:.0f}% der EBA-2025-Adverse-"
                f"Schwere (Anker: M = {_ANCHOR.z_factor:.2f})"
            )
        else:
            st.caption(f"Zustand: **benigne** · M = +{m_input:.2f}σ "
                       "→ PDs unter Baseline, LGD bleibt auf Baseline")

        # Derived macro echoes
        d_brent, d_r_pp = _derive_from_m(m_input)
        brent_pct = (np.exp(d_brent) - 1) * 100
        st.markdown(
            f"<div style='font-size:0.78rem;color:#6E6E6E;"
            f"border-top:1px solid #E6E6E6;padding-top:0.4rem;"
            f"margin-top:0.5rem;line-height:1.5;'>"
            f"<strong>Aus M abgeleitete Markt-Schocks</strong> "
            f"(via EBA-Adverse-Direction):<br/>"
            f"ΔBrent = {d_brent:+.2f} log  (≈ {brent_pct:+.0f}%)  ·  "
            f"Δr_10y = {d_r_pp*100:+.0f} bp"
            f"</div>",
            unsafe_allow_html=True,
        )

    return {
        "d_brent":     d_brent,
        "d_b0":        d_r_pp, "d_b1": 0.0, "d_b2": 0.0, "d_b3": 0.0,
        "d_r_10y_pp":  d_r_pp,
        "m_direct":    m_input,
        "mode":        "single_factor",
    }


# =====================================================================
# Multi-Factor mode (legacy, for Yield-Curve page)
# =====================================================================
def _render_multi_factor() -> dict:
    with st.expander("Energy-Price-Schock (Brent)", expanded=True):
        d_brent = st.slider(
            "ΔBrent (kumulativer log-Return)",
            min_value=-2.0, max_value=2.0, value=0.0, step=0.05,
            key="d_brent_multi",
            help="−1.20 ≈ Corona-Crash · +0.55 ≈ Iran-Eskalation",
        )
        pct = (np.exp(d_brent) - 1) * 100
        st.caption(f"≈ {pct:+.0f}% Preis-Änderung über Horizon")

    with st.expander("Yield-Curve · Svensson-β-Shifts", expanded=True):
        d_b0 = st.slider("Δβ₀ — Level (parallel shift)", -3.0, 3.0, 0.0, 0.05,
                         key="d_b0_multi",
                         help="Verschiebt die gesamte Kurve in pp")
        d_b1 = st.slider("Δβ₁ — Slope (kurzes Ende)", -3.0, 3.0, 0.0, 0.05,
                         key="d_b1_multi")
        d_b2 = st.slider("Δβ₂ — Curvature 1 (Mid-Maturity-Hump bei τ₁)",
                         -3.0, 3.0, 0.0, 0.05, key="d_b2_multi")
        d_b3 = st.slider("Δβ₃ — Curvature 2 (Long-End-Hump bei τ₂)",
                         -3.0, 3.0, 0.0, 0.05, key="d_b3_multi")
        d_r_pp = float(delta_r_from_beta_shifts(d_b0, d_b1, d_b2, d_b3))
        st.metric("Resultierender Δr_10y", f"{d_r_pp*100:+.0f} bp",
                  help="In Basispunkten — aus den β-Shifts via "
                       "Svensson-Funktion an τ = 10 Jahre.")

    return {
        "d_brent":     d_brent,
        "d_b0":        d_b0, "d_b1": d_b1, "d_b2": d_b2, "d_b3": d_b3,
        "d_r_10y_pp":  d_r_pp,
        "m_direct":    None,           # signal: page should map (Brent, Δr) → M
        "mode":        "multi_factor",
    }


# =====================================================================
# Quick scenarios — map to M directly in single-factor, to (Brent, β) in multi
# =====================================================================
def _apply_quick_scenario(name: str, mode: str) -> None:
    if name == "—":
        return
    # M-values calibrated so the EBA preset matches the anchor exactly
    m_by_scenario = {
        "Corona 2020":      +0.7,   # benign for credit (rate cut + oil crash)
        "Ukraine 2022":     -0.9,
        "Iran 2026":        -1.6,
        "EBA 2025 adverse": _ANCHOR.z_factor,   # −2.50
    }
    multi_presets = {
        "Corona 2020":      {"d_brent": -1.20, "d_b0": -0.65},
        "Ukraine 2022":     {"d_brent": +0.12, "d_b0": +0.67},
        "Iran 2026":        {"d_brent": +0.55, "d_b0": +0.50},
        "EBA 2025 adverse": {"d_brent": +0.47, "d_b0": +2.00},
    }
    if name not in m_by_scenario:
        return
    if st.session_state.get("_last_quick") == name:
        return

    if mode == "Single-Factor M (Standard)":
        st.session_state["m_input"] = float(m_by_scenario[name])
    else:
        for k, v in multi_presets[name].items():
            st.session_state[f"{k}_multi"] = v
        st.session_state["d_b1_multi"] = 0.0
        st.session_state["d_b2_multi"] = 0.0
        st.session_state["d_b3_multi"] = 0.0

    st.session_state["_last_quick"] = name
    st.rerun()
