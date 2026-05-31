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
    """Drei kollabierbare Erklär-Boxen: Exposure, Duration, Mark-to-Market.

    Bewusst einfach gehalten — Zielgruppe: Leser ohne Quant-Vorwissen.
    Jede Box folgt dem Muster: Was ist das? → Wie messen wir es? →
    Warum ist es für unser Stress-Modell wichtig?
    """
    st.markdown(
        '<span class="mc-eyebrow-inline">'
        'Methodik · Was sind Sovereign Exposures · Duration · Mark-to-Market'
        '</span>',
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------
    with st.expander("Was sind Sovereign Exposures · und woher kommen sie?",
                     expanded=False):
        st.markdown("""
**Was ist das?**
Sovereign Exposures sind alle Bestände einer Bank an Staatsanleihen
(oder Forderungen gegen Staaten). Beispiel: Wenn die Deutsche Bank
italienische Staatsanleihen für 30 Mrd. € hält, sind das 30 Mrd. €
Sovereign Exposure gegenüber Italien.

**Woher haben wir die Zahlen?**
Direkt aus dem EBA-Transparency-2025-Datensatz (`tr_sov.csv`,
Stichtag Juni 2025). Jede Bank meldet pro Land und Restlaufzeit, wie
viel sie an Staatspapieren hält. Wir aggregieren auf Bank × Land ×
Maturity-Bucket.

**Warum ist das für unser Modell wichtig?**
Ein steigender 10y-Zins lässt den Marktwert der Anleihen sinken
(klassischer Bond-MtM-Verlust). Je mehr eine Bank hält und je länger
die Restlaufzeit, desto stärker schlägt das auf die CET1-Quote durch
— das ist der **zweite** der drei Stress-Kanäle unseres Modells
(Kreditbuch · Marktbuch · Trading-Book).

**Was lassen wir bewusst weg?**
- Derivative-Engagements auf Staatsanleihen (z.B. Sovereign-CDS)
- Off-Balance-Sheet-Commitments
- Hedges via Swaps/Futures — sind im öffentlichen EBA-Datensatz
  nicht rekonstruierbar
""")

    # ----------------------------------------------------------------
    with st.expander("Was ist Modified Duration · und warum brauchen wir sie?",
                     expanded=False):
        st.markdown("""
**Was ist das?**
Die **Modified Duration** ist eine Zahl (in Jahren), die misst, **wie
stark der Marktwert eines Bonds fällt, wenn der Zins um 1 Prozentpunkt
steigt**.

> Faustformel: Δ Marktwert ≈ −Duration × Δ Zins × Exposure

Eine Bundesanleihe mit 10 Jahren Restlaufzeit hat z.B. eine Duration
von ca. 7.5 Jahren. Steigt der 10y-Zins um 1pp, fällt der Marktwert
um ca. 7.5%.

**Wie schätzen wir sie?**
Die EBA gibt für jede Position nur den **Laufzeit-Bucket** an
(z.B. "5-10 Jahre"), nicht das exakte Restalter. Wir nehmen den
Mittelpunkt jedes Buckets als Approximation:
""")
        st.dataframe(
            {
                "Laufzeit-Bucket": ["< 3 Monate", "3 Monate – 1 Jahr",
                                    "1 – 2 Jahre", "2 – 3 Jahre",
                                    "3 – 5 Jahre", "5 – 10 Jahre",
                                    "> 10 Jahre"],
                "Angenommene Duration": ["0.125 Jahre", "0.625 Jahre",
                                          "1.5 Jahre", "2.5 Jahre",
                                          "4.0 Jahre", "7.5 Jahre",
                                          "15.0 Jahre"],
            },
            hide_index=True, use_container_width=True,
        )
        st.markdown("""
**Warum ist das für unser Modell wichtig?**
Ohne Duration könnten wir den Zinsschock nicht in einen
EUR-Verlust übersetzen. Genau hier verbindet das Modell den
Makro-Schock (Δr_10y aus der Sidebar) mit einer konkreten
Bilanzwirkung pro Bank.

**Vereinfachung, die wir eingehen:**
- "Bullet-Bond at par" — Coupon-Effekte und Konvexität (= die
  zweite Ableitung) werden ignoriert. Im Mittel führt das zu
  ±10-15% Abweichung gegenüber einer exakten Cashflow-Rechnung —
  vertretbar für ein Stress-Frame.
""")

    # ----------------------------------------------------------------
    with st.expander("Wie berechnen wir den Mark-to-Market-Verlust unter Zinsschock?",
                     expanded=False):
        st.markdown("""
**Was ist das?**
Ein "Mark-to-Market-Verlust" ist der Wertverlust, der entsteht, wenn
sich der Marktzins ändert und die Bank ihre Bonds zum neuen
(niedrigeren) Marktpreis bilanziert. Es ist ein **rein rechnerischer
Verlust** — die Bank verliert nichts, wenn sie den Bond bis zur
Endfälligkeit hält. Aber unter IFRS-9 muss sie ihn (je nach
Klassifizierung) sofort buchen.

**Die Formel.** Pro Bank summieren wir über alle Laufzeit-Buckets:
""")
        st.latex(r"""
\Delta P_b = -\sum_{m \in \text{buckets}}
              D_m \cdot \Delta y \cdot E_{b,m}
""")
        st.markdown("""
mit:
- D_m = Duration des Buckets (Tabelle oben)
- Δy = Zinsänderung in Dezimal (z.B. +1pp → Δy = 0.01)
- E_{b,m} = Sovereign-Exposure von Bank b im Bucket m

**Warum ist das für unser Modell wichtig?**
Dies ist der konkrete EUR-Verlust, der direkt auf die CET1-Quote
durchschlägt — aber **nicht für alle Bonds gleich** (siehe IFRS-9-
Klassen-Erklärung weiter unten in diesem Sub-Tab).

**Was wir bewusst nicht abbilden:**
- **Slope-/Curvature-Schocks** (kurze vs. lange Zinsen unterschiedlich)
  — wir nehmen einen **Parallel-Shift** an
- **Credit-Spread-Risiko** (Italien-Bund-Spread) — out of scope
- **Hedging** — Swap-/Future-Positionen sind nicht öffentlich
  ausgewiesen
""")
