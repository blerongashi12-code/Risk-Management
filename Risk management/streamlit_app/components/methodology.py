"""Reusable methodology / disclosure boxes (Critique 2 + 3).

Each Streamlit page that displays risk metrics derived from approximations
should call one of the helpers below to make the derivation chain
transparent to the audience (auditor, validator, board).

The boxes are intentionally compact and rendered as collapsible expanders
so they do not dominate the page; they expand on demand.
"""
from __future__ import annotations

import streamlit as st


# Confidence badges — kept inline-text only (no emoji per project policy).
# 🟢 published / actual measurement
# 🟡 statistical estimate
# 🟠 model approximation
# 🔴 hardcoded assumption
_BADGE = {
    "published":     "[published]",
    "estimate":      "[statistical estimate]",
    "approximation": "[approximation]",
    "assumption":    "[hardcoded assumption]",
}


def _box_open(title: str, badge_key: str = "approximation") -> str:
    badge = _BADGE.get(badge_key, "")
    return f"**{title}**  ·  *{badge}*"


# =====================================================================
# Loan-Book methodology · for the Bank Portfolio page
# =====================================================================
def render_loan_methodology(*, kappa_lgd: float = 0.30) -> None:
    """Four collapsible boxes: PD, LGD, EAD, EL — Loan Book context.

    Maps directly to Critique 2 of the supervising professor:
    'Herleitung von PD, LGD, EAD und EL ist nicht ausreichend dokumentiert'.
    """
    st.markdown(
        '<span class="mc-eyebrow-inline">'
        'Methodik · PD · LGD · EAD · EL'
        '</span>',
        unsafe_allow_html=True,
    )
    with st.expander(_box_open("Probability of Default (PD)", "approximation"),
                     expanded=False):
        st.markdown(r"""
**Berechnung**
$$\text{PD}_{b,c} = \frac{\text{Defaulted Exposure}}{\text{Original Exposure}}
= \frac{\text{EBA Item 2520512 (Status = 2)}}{\text{EBA Item 2520502}}$$

**Datengrundlage:** EBA EU-wide Transparency Exercise 2025, Reporting-Stichtag
Juni 2025. Aggregation pro Bank × Vasicek-Exposure-Class.

**Approximations-Charakter:** Diese PD ist die *beobachtete* Default-Quote
zum Stichtag (Stock-Größe, backward-looking) — **keine** forward-looking
1-Jahres-PD im Sinne von CRR Art. 178/180. Im Stress-Test wird sie über
die Vasicek-Conditional-PD-Funktion in eine bedingte Forward-PD
überführt:

$$\text{PD}(M) = N\!\left(\frac{N^{-1}(\text{PD}) - \sqrt{\rho}\,M}{\sqrt{1-\rho}}\right)$$

**Floor / Cap:** 3 bp (Basel-Sovereign-Floor) und 50% für numerische
Stabilität in der Vasicek-Engine.
""")

    with st.expander(_box_open("Loss Given Default (LGD)", "assumption"),
                     expanded=False):
        st.markdown(r"""
**Baseline:** Basel-F-IRB-Standard-LGDs pro Vasicek-Class:

| Klasse | LGD |
|---|---|
| Corporate / Bank / Sovereign / SME | 45% |
| Residential Mortgage | 20% |
| Qualifying Revolving Retail (QRRE) | 65% |
| Other Retail | 45% |

**Stress:** Downturn-LGD nach EBA-2023-Stresstest-Konvention
(CRR Art. 181, EBA GL 14):

$$\text{LGD}_{\text{stress}} = \text{LGD}_{\text{base}} \cdot \bigl(1 + \kappa \cdot \max(-M, 0)\bigr)$$
""")
        st.markdown(
            f"mit **κ = {kappa_lgd:.2f}**, gecapped bei 100%. "
            f"Bei M = −2.5 (EBA-2025-Adverse-Anker) entspricht das einer "
            f"LGD-Erhöhung um **{kappa_lgd*2.5*100:.0f}%** relativ zur Baseline."
        )
        st.markdown(
            "**Approximations-Charakter:** Bank-spezifische A-IRB-LGDs sind in "
            "der EBA-Public-Disclosure **nicht enthalten**. F-IRB-Defaults "
            "dienen als konservativer regulatorischer Proxy. Sektor-"
            "spezifisches κ (z.B. niedriger für Mortgage, höher für Retail) "
            "wird in einer V2-Erweiterung ergänzt."
        )

    with st.expander(_box_open("Exposure at Default (EAD)", "approximation"),
                     expanded=False):
        st.markdown(r"""
**Berechnung:** Direkter Pull aus **EBA Item 2520522** (Exposure Value),
das bereits CCF-adjustiert ist (post-Conversion-Factor, Basel-konform).

**Datengrundlage:** Aggregation pro Bank × Vasicek-Class über alle
Counterparty-Countries und Maturities.

**Approximations-Charakter (V1):**
- EAD bleibt **statisch unter Stress** — das ist eine Vereinfachung.
- **Drawdown-Risk** auf Off-Balance-Linien (CCF kann von 50% auf 75%
  steigen unter Stress) ist in V1 nicht modelliert.
- **FX-Effekte** sind in der EBA-Disclosure bereits EUR-konsolidiert →
  Currency-Channel nicht trennbar.

**Geplante V2-Erweiterung:** stress-elastische CCF-Funktion
$\text{CCF}_{\text{stress}} = \text{CCF}_{\text{base}} \cdot (1 + \theta \cdot \max(-M, 0))$.
""")

    with st.expander(_box_open("Expected Loss (EL)", "published"),
                     expanded=False):
        st.markdown(r"""
**Definition:** Pro Segment

$$\text{EL}_{b,c} = \text{PD}_{b,c} \cdot \text{LGD}_{b,c} \cdot \text{EAD}_{b,c}$$

**Stress-EL:**

$$\text{EL}^{*}_{b,c} = \text{PD}(M)_{b,c} \cdot \text{LGD}(M)_{b,c} \cdot \text{EAD}_{b,c}$$

**Decomposition (sequential):** ΔEL wird in zwei additive Beiträge zerlegt
(siehe Capital-Bridge-Sektion):

$$\Delta\text{EL} = \underbrace{(\text{PD}^*\text{LGD} - \text{PD}\,\text{LGD})\,\text{EAD}}_{\Delta\text{EL aus PD-Shift}}
+ \underbrace{(\text{PD}^*\text{LGD}^* - \text{PD}^*\text{LGD})\,\text{EAD}}_{\Delta\text{EL aus LGD-Shift}}$$

Diese Decomposition ist **exakt additiv** und antwortet direkt auf die
Forderung "transparente Wirkungskette" — siehe Capital-Bridge-Tab unten.

**Verhältnis zur IRB-Capital-Charge:** Während EL als "Erwartungswert"
in die Risiko-Vorsorge (Provisions) fließt, ist die IRB-Capital-Charge
$K = (L_\alpha - \text{EL}) \cdot \text{MA}(M_{\text{eff}})$ der
*Unexpected Loss* bei 99.9%-Konfidenz. Beide werden in der
Capital-Bridge nebeneinander gezeigt.
""")


# =====================================================================
# Sovereign-Book methodology · for the Sovereign Risk page
# =====================================================================
def render_sovereign_methodology() -> None:
    """Three collapsible boxes: Exposure measurement, Duration, P&L."""
    st.markdown(
        '<span class="mc-eyebrow-inline">'
        'Methodik · Sovereign Exposure · Duration · MtM-P&L'
        '</span>',
        unsafe_allow_html=True,
    )
    with st.expander(_box_open("Sovereign Exposure", "published"),
                     expanded=False):
        st.markdown(r"""
**Berechnung:** Aggregation aus **EBA Item 2520810** (On-balance Gross
Carrying Amount der nicht-derivativen Finanzaktiva), pro Bank ×
Counterparty-Country × Maturity-Bucket × Accounting-Portfolio.

**Datengrundlage:** EBA-Transparency-2025, Stichtag Juni 2025, Total
über alle Maturity-Buckets (für Konzentration) und alle Country-Codes
(für Maturity-Ladder).

**Approximations-Charakter:**
- Off-Balance-Derivative-Engagements (Items 2520816–2520819) sind in V1
  *nicht* eingerechnet — würden die Doom-Loop-Story nicht qualitativ
  ändern, kosten aber Komplexität.
- Off-Balance-Nominal-Commitments (Item 2520820) ebenfalls ausgeschlossen.
""")

    with st.expander(_box_open("Modified-Duration-Approximation", "approximation"),
                     expanded=False):
        st.markdown(r"""
**Methodik:** Bucket-Midpoint-Approximation der Macaulay-Duration
(angenommen ≈ Modified Duration für Sovereign-Bonds nahe par):

| Maturity-Bucket | Approx. D (Jahre) |
|---|---|
| < 3M | 0.125 |
| 3M – 1Y | 0.625 |
| 1 – 2Y | 1.5 |
| 2 – 3Y | 2.5 |
| 3 – 5Y | 4.0 |
| 5 – 10Y | 7.5 |
| > 10Y | 15.0 |

**Approximations-Charakter:** Annahmen "Bullet-Bond at par",
Coupon-Effekte und Konvexität ignoriert. Im Mittel ±10–15% Abweichung
zur tatsächlichen Cashflow-basierten Duration. Konfigurierbar in
`eba_loader.DURATION_BY_BUCKET`.
""")

    with st.expander(_box_open("Mark-to-Market P&L unter Rate-Shock", "approximation"),
                     expanded=False):
        st.markdown(r"""
**Funktionale Form:**

$$\Delta P_b = -\sum_{m \in \text{buckets}} D_m \cdot \Delta y \cdot E_{b,m}, \qquad \Delta y = \frac{\Delta r_{10y\,\text{pp}}}{100}$$

**Pro Bank:** Σ über alle Maturity-Buckets ergibt den Mark-to-Market-
Verlust unter parallelem Yield-Shift, in EUR.

**Limitationen:**
- **Nur Parallel-Shift** angenommen — Slope/Curvature-Stress (Δβ₁/β₂/β₃ aus
  Svensson) wirkt aktuell nicht direkt auf das Sovereign-P&L (würde
  Bucket-spezifische Yields erfordern).
- **Kein Credit-Spread-Risiko** — Italian-vs-Bund-Spread-Stress fehlt.
- **Kein Hedging** — Swaps/Futures-Hedges aus EBA-Public-Disclosure
  nicht rekonstruierbar.
- **Buchwert vs. Fair Value:** HtM-Bestände nehmen MtM-Verluste **nicht**
  durchs P&L auf; das Modell aggregiert über alle Accounting-Portfolios
  (HfT/FVTPL/FVOCI/AC/HtM) und überschätzt damit die P&L-Wirkung
  konservativ.
""")
