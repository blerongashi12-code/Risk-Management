"""
============================================================================
 two_factor_stress.py · 2-Faktor-Stress-Modell (Brent + Δr_10y)
============================================================================

Ersetzt das frühere Vasicek/ASRF-Single-Factor-Modell durch eine direkte,
sektor-differenzierte 2-Faktor-Stress-Transmission:

    ΔDR_class = β_oil_class · ΔBrent_log + β_rate_class · Δr_10y_pp
    ΔLGD_class = γ_oil_class · ΔBrent_log + γ_rate_class · Δr_10y_pp

Vorteile gegenüber Vasicek/M:
- Adressiert direkt die Professor-Kritik (Punkt 5-8) zur Faktor-
  Aggregation, fehlenden Ursache-Wirkungs-Kette und zu vereinfachten
  Zinslogik.
- Sektor-Differenzierung via klassen-spezifische β-Koeffizienten —
  steigende Zinsen sind NICHT mehr pauschal "schlecht": z.B. Bank-Klasse
  hat β_rate < 0 (NIM-Uplift).
- Transparente lineare Form ohne Conditional-PD-Inverse-Normal-Tricks.

β-Koeffizienten · Herkunft (Recalibration Juni 2026, Modell-Stichtag 31.12.2024)
-------------------------------------------------------------------------
Die β/γ sind als "sectoral sensitivities applied to portfolio-level
projections" im Sinne der EBA-Methodik §2.4.2 ¶123 zu verstehen —
genau dieser Ansatz ist aufsichtsrechtlich vorgesehen (die EBA gibt
KEINE fertigen β-Tabellen vor, sondern verlangt, dass die projizierten
Parameter in *Richtung und Größenordnung* mit den Szenario-Schocks
konsistent sind). Pro Klasse + Faktor + Risk-Parameter gibt es einen
β-Wert mit explizit zitierter, AKTUELLER Quelle. Es sind "defensible
defaults", via Sidebar-Override veränderbar.

Wichtig zur Faktor-Struktur:
- ZINS-Kanal (β_rate): direkt auf aktuelle, segment-spezifische Quellen
  und den Modell-Stichtag 31.12.2024 kalibrierbar — EBA-2025-Adverse-Szenario
  (10y-Pfad) + ECB-Satellitenmodelle.
- ÖL-Kanal (β_oil): Brent ist hier ein Proxy für den Energie-/
  Angebotsschock. Die Quellen ab 2024 liefern keine fertige Öl-Beta.
  Die Logik ist: höhere Energiepreise erhöhen Kosten, drücken Margen,
  verschlechtern Cashflows und können dadurch Ausfallwahrscheinlichkeiten
  erhöhen. Der Öl-Kanal ist also eine plausible Übersetzung des Szenario-
  Auslösers, nicht eine separat publizierte Öl-Ausfall-Elastizität.

Quellen (aktuell)
-----------------
- EBA (2024). *2025 EU-wide Stress Test — Methodological Note* (11 Nov
  2024), Kap. 2 Credit risk, §2.4.2 "Projected point-in-time parameters
  (a hierarchy of approaches)", ¶122–130 (¶123 sectoral sensitivities,
  ¶130 LGD reflektiert Sicherheiten-FV-Verfall).
- EBA/ESRB (2025). *2025 EU-wide Stress Test — Macro-financial
  scenario* (Jan 2025), §4.1.6 Long-term rates (adverse 10y-Pfad,
  Startpunkt Ø Dez-2024): Euroraum 2,73 → 4,63 % (+1,9 pp), DE +1,3,
  FR +2,1, IT +2,9, ES +2,8 pp.
- Lo Duca, Moccero & Parlapiano (2024). *The impact of macroeconomic
  and monetary policy shocks on credit risk in the euro area corporate
  sector*, ECB WP 2897 — Ausfallwahrscheinlichkeit nichtfinanzieller
  Unternehmen; kleine und mittlere Unternehmen reagieren deutlich stärker
  als große Unternehmen.
  → Corporate/KMU, beide Faktoren (Angebotsschock ≈ Öl-Proxy, Zinskanal).
- Konietschke, Metzler & Ponte Marques (2026). *A quantile probability
  model for sectoral corporate defaults in Europe*, ECB WP 3207 —
  Quantil-PD über 5 Mio. Firmen/9 Länder; Tail-Sensitivität 3–5× Median,
  ausgeprägte Sektor-Heterogenität (Bau, Handel, Hospitality, Immobilien).
  → Zusatzanker für Corporate/KMU-Heterogenität, nicht als direkte Öl-β-Tabelle.
- Bandoni, Fourné & Jarmulska (2025). *Mortgage loan rates and the
  defaults of variable rate mortgages*, ECB WP 3112 — Mortgage-Default ↔
  Zins, nichtlinear & asymmetrisch (Zins-Anstiege treiben Default stark).
  → Mortgage, Zins-Kanal.
- ECB (2024). *Financial Stability Review, Mai 2024* — Energiepreis-/
  Lebenshaltungskosten-Schock → Belastung der Haushalte (untere Einkommens-
  quintile); unterschiedliche erwartete Ausfallraten nach Segment.
  → Mortgage/QRRE/Other-Retail, Öl- und Zins-Kanal.
- EBA (2025). *2025 EU-wide Stress Test — Results* (Aug 2025), Fig. 22
  „Projected credit risk losses by portfolio" — Verlustquote je Portfolio
  (Startpunkt end-2024); Retail mit höchster projizierter Verlustquote,
  Public Sector am unteren Ende. All-Segment-Querverankerung der relativen
  Sensitivitäts-Struktur.
- EBA (2025). *Report on the 2024 Credit Risk Benchmarking Exercise*
  (März 2025) — aktuelle PD/LGD-Niveaus je IRB-Asset-Klasse.

Hinweis: PD/LGD-Baselines = 31.12.2024, Schock-Größe = EBA-2025-Adverse-
Szenario. Die Quellen ab 2024 liefern keine fertige aufsichtliche Beta-
Tabelle, sondern Richtung, relative Stärke und Plausibilitätsanker.

API
---
- get_beta(class_name, factor, parameter) -> float
- stress_pd(pd_base_dec, d_brent_log, d_r_10y_pp, class_name) -> float
- stress_lgd(lgd_base_dec, d_brent_log, d_r_10y_pp, class_name) -> float
- stress_segment(seg_dict, d_brent, d_r, override_betas=None) -> dict
- VASICEK_CLASSES (constant tuple)
- SENSITIVITY_MATRIX (frozen dict for documentation)
============================================================================
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ============================================================================
# 1. Konfiguration · die zentrale Sensitivitäts-Matrix
# ============================================================================
# Format:  {class: {'pd_oil': β, 'pd_rate': β, 'lgd_oil': γ, 'lgd_rate': γ,
#                   'source_pd': str, 'source_lgd': str, 'economic_logic': str}}
#
# Interpretation der β:
# - pd_oil:  ΔPD in Prozentpunkten pro 1.0 (= 100%) Brent-Log-Return
#            Beispiel: β_oil = 0.30 → +30% Brent erhöht PD um +0.30·0.30 =
#            0.09 Prozentpunkte
# - pd_rate: ΔPD in Prozentpunkten pro 1.0 Prozentpunkt Δr_10y
#            Beispiel: β_rate = 0.25 → +200bp Δr erhöht PD um +0.25·2 =
#            0.50 Prozentpunkte
# - lgd_oil / lgd_rate analog für LGD-Veränderung in Prozentpunkten

SENSITIVITY_MATRIX = {
    "corporate": {
        "pd_oil":   0.30,  # Angebotsschock-Kanal (ECB WP 2897)
        "pd_rate":  0.20,  # +1pp rate → +20 bp PD (Refi-Kosten, WP 2897)
        "lgd_oil":  0.50,  # Sicherheiten-Werte mild öl-sensitiv
        "lgd_rate": 1.00,  # Sicherheiten-FV-Verfall bei +Δr (EBA §2.4.2 ¶130)
        "source_pd":  "ECB WP 2897 (2024) — Ausfallwahrscheinlichkeit "
                      "nichtfinanzieller Unternehmen auf Angebots- und "
                      "Zinsschocks; ECB WP 3207 (2026) — sektorale "
                      "Unternehmens-Ausfälle im Stress-Test-Kontext; "
                      "EBA Methodology §2.4.2 ¶123",
        "source_lgd": "EBA 2025 Methodology §2.4.2 ¶130 (LGD spiegelt "
                      "Sicherheiten-FV-Verfall); EBA Credit Risk "
                      "Benchmarking 2024 (LGD-Niveau Corporate)",
        "economic_logic":
            "Corporate ist der Anker: Der Öl-Kanal ist ein Proxy für den "
            "Energie-/Angebotsschock, nicht eine veröffentlichte Öl-Beta. "
            "Höhere Energiepreise erhöhen Kosten und drücken Margen; daher "
            "β_oil = 0.30. Höhere Zinsen verteuern Refinanzierung; daher "
            "β_rate = 0.20. LGD reagiert stärker auf Zins, weil ein "
            "Zinsschock den Fair Value von Sicherheiten senkt (EBA ¶130).",
    },
    "sme_corporate": {
        "pd_oil":   0.60,  # ≈2× Large-Corp (ECB WP 2897 Fig. 5, Size)
        "pd_rate":  0.40,  # ≈2× Large-Corp; Geldpolitik trifft KMU härter
        "lgd_oil":  0.45,
        "lgd_rate": 1.10,
        "source_pd":  "ECB WP 2897 (2024) — kleine und mittlere Unternehmen "
                      "reagieren deutlich stärker als große Unternehmen; "
                      "ECB WP 3207 (2026) — sektorale Unterschiede im Stress.",
        "source_lgd": "EBA 2025 Methodology §2.4.2 ¶130; weniger liquide "
                      "KMU-Sicherheiten → LGD leicht über Large-Corp",
        "economic_logic":
            "KMU haben geringere Diversifikation, dünnere Liquiditäts-"
            "puffer und schwächere Preissetzungsmacht — Energie- und "
            "Zinsschocks schlagen härter durch. ECB WP 2897 quantifiziert "
            "diese Größen-Heterogenität direkt: ein Angebotsschock hebt "
            "die Mikro-/KMU-PD rund doppelt, ein Geldpolitik-Schock rund "
            "dreifach gegenüber Large-Corporates. Daher sind die β exakt "
            "2× Corporate und damit am konservativen unteren Rand der Quelle. "
            "LGD höher, weil Sicherheiten weniger liquide sind.",
    },
    "mortgage": {
        "pd_oil":   0.05,  # nur indirekt via Haushaltseinkommen/Inflation
        "pd_rate":  0.30,  # Affordability + Hauspreis-Kanal (ECB WP 3112)
        "lgd_oil":  0.10,
        "lgd_rate": 1.50,  # Wohnimmobilien-Preis sinkt mit Zinsschock
        "source_pd":  "Zins: ECB WP 3112 (2025) Abstract/Introduction — "
                      "Defaults variabel verzinster Euro-Area-Hypotheken; "
                      "Zinsanstiege wirken nichtlinear/asymmetrisch, wiederholte "
                      "Anstiege ca. 3× stärker; Öl: ECB FSR Mai 2024 "
                      "(Energiekosten → Haushaltsbudgets)",
        "source_lgd": "EBA 2025 Methodology §2.4.2 ¶130 (Immobilien-"
                      "Preisschock → Wohnsicherheiten-FV); EBA-2025-"
                      "Szenario unterstellt starken RRE-Preisrückgang",
        "economic_logic":
            "Hypotheken sind das archetypische zins-sensitive Asset: der "
            "Floating-Rate-Anteil belastet die Affordability sofort, und "
            "der Zinsschock drückt die Wohnimmobilien-Preise und damit "
            "die Recovery (höchstes γ_rate, EBA ¶130). Brent wirkt nur "
            "indirekt über das verfügbare Haushaltseinkommen.",
    },
    "qrre": {
        "pd_oil":   0.40,  # Energie-Inflation belastet Konsumenten direkt
        "pd_rate":  0.15,  # geringe Zins-Sensitivität (kurze Laufzeiten)
        "lgd_oil":  0.30,
        "lgd_rate": 0.50,
        "source_pd":  "ECB FSR Mai 2024 (Haushalts-Schuldendienst und "
                      "Lebenshaltungskosten-/Energiepreisrisiken); EBA 2025 Results "
                      "Fig. 22 (Retail: höchste projizierte Verlustquote; "
                      "<10% Exposure, fast 30% projizierte Kreditrisikoverluste)",
        "source_lgd": "Basel III F-IRB-Baseline + EBA Credit Risk "
                      "Benchmarking 2024 (QRRE-LGD-Niveau)",
        "economic_logic":
            "QRRE (Kreditkarten, Revolving) reagiert primär auf das "
            "verfügbare Haushaltseinkommen — Energiepreise belasten es "
            "über die Inflation (ECB FSR 2024). Die Zins-Sensitivität ist "
            "niedrig, weil QRRE-Sätze ohnehin variabel und hoch sind "
            "(oft 15-20 %) und die Laufzeiten kurz.",
    },
    "other_retail": {
        "pd_oil":   0.30,
        "pd_rate":  0.25,
        "lgd_oil":  0.25,
        "lgd_rate": 0.80,
        "source_pd":  "ECB FSR Mai 2024 (Haushalte); EBA 2025 Results "
                      "Fig. 22 (Retail-Portfolio als Verlustquoten-Plausibilisierung); "
                      "Mischprofil zwischen Mortgage und QRRE",
        "source_lgd": "EBA 2025 Methodology §2.4.2 ¶130; EBA Credit Risk "
                      "Benchmarking 2024 (Other-Retail-LGD-Niveau)",
        "economic_logic":
            "Konsumkredite, Auto-Finanzierung, Personal Loans: mittlere "
            "Sensitivität sowohl auf Energie-Inflation als auch auf "
            "Refi-Kosten — ein Mischprofil zwischen besichertem Mortgage "
            "und unbesichertem QRRE.",
    },
    "bank": {
        "pd_oil":   0.05,  # nur indirekt
        "pd_rate": -0.05,  # NEGATIV: NIM-Uplift kompensiert Kreditrisiko
        "lgd_oil":  0.10,
        "lgd_rate": 0.50,
        "source_pd":  "WP 2897/3207 nicht passend (nichtfinanzielle "
                      "Unternehmen). EBA 2025 Methodology Kap. 4 "
                      "(Net Interest Income = Zinsüberschuss) + EBA Results "
                      "2025; β_rate<0 ist kleine Expertenannahme, keine "
                      "publizierte Bank-Ausfall-Beta.",
        "source_lgd": "Basel III F-IRB-LGD 45 % Baseline (Senior-"
                      "Unsecured Bank-Exposure)",
        "economic_logic":
            "Bank-Counterparties können von steigenden Zinsen über NIM "
            "profitieren, gleichzeitig steigen Funding- und Asset-Quality-"
            "Risiken. Daher nur ein kleines negatives β_rate (-0.05) als "
            "transparente Expertenannahme; Öl wirkt nur indirekt.",
    },
    "sovereign": {
        "pd_oil":   0.00,
        "pd_rate":  0.00,  # Sovereign-Default ist macro-orthogonal
        "lgd_oil":  0.00,
        "lgd_rate": 0.00,
        "source_pd":  "EBA 2025 Methodology §2.4.2 ¶154 — Sovereign-Default-"
                      "und Impairment-Flows separat; Zins wirkt im Modell "
                      "über Marktbuch-Kurswert/Duration. EBA Results Fig. 22: "
                      "Public Sector mit niedrigen Verlustquoten im Vergleich "
                      "der Exposure-Klassen.",
        "source_lgd": "F-IRB-LGD 45 % (Sovereign-Recovery ~ Senior-"
                      "Unsecured)",
        "economic_logic":
            "Sovereign-Default ist primär fiskalisch determiniert "
            "(Schuldenstand, Steuereinnahmen, politische Stabilität) und "
            "NICHT direkt durch Brent oder Δr_10y. Der Zinsschock wirkt "
            "stattdessen über den Sovereign-Kurswert-/Duration-Kanal (separate "
            "Modellebene, Tab Marktbuch).",
    },
}

VASICEK_CLASSES = tuple(SENSITIVITY_MATRIX.keys())


# ============================================================================
# 2. Lookup & validation helpers
# ============================================================================
def get_beta(class_name: str, factor: str, parameter: str) -> float:
    """Liefert β für (class, factor='oil'|'rate', parameter='pd'|'lgd').

    Returns 0.0 wenn die Klasse nicht in der Matrix vorkommt (defensive
    default für unbekannte Klassen-Strings).
    """
    if class_name not in SENSITIVITY_MATRIX:
        return 0.0
    key = f"{parameter}_{factor}"
    return float(SENSITIVITY_MATRIX[class_name].get(key, 0.0))


def get_economic_logic(class_name: str) -> str:
    """Liefert die ökonomische Begründung als Klartext-String."""
    if class_name not in SENSITIVITY_MATRIX:
        return "Klasse unbekannt — keine ökonomische Logik hinterlegt."
    return SENSITIVITY_MATRIX[class_name].get("economic_logic", "")


def get_sources(class_name: str) -> dict:
    """Liefert die Quellen-Zitierung für PD- und LGD-Sensitivität."""
    if class_name not in SENSITIVITY_MATRIX:
        return {"pd": "n/a", "lgd": "n/a"}
    return {
        "pd":  SENSITIVITY_MATRIX[class_name].get("source_pd", ""),
        "lgd": SENSITIVITY_MATRIX[class_name].get("source_lgd", ""),
    }


# ============================================================================
# 3. Stress-Anwendung
# ============================================================================
def stress_pd(
    pd_base_dec: float,
    d_brent_log: float,
    d_r_10y_pp: float,
    class_name: str,
    *,
    override_betas: Optional[dict] = None,
    pd_floor: float = 0.0003,    # 3 bp Basel sovereign floor
    pd_cap: float = 0.50,         # 50% numerische Stabilität
) -> float:
    """Wende den 2-Faktor-Stress auf eine baseline PD (in Dezimal) an.

    ΔPD_pp = β_oil · ΔBrent_log + β_rate · Δr_10y_pp
    PD_stressed = max(min(PD_base + ΔPD_pp/100, pd_cap), pd_floor)

    Returns: PD_stressed als Dezimal.
    """
    b_oil  = (override_betas or {}).get("pd_oil",
                                         get_beta(class_name, "oil",  "pd"))
    b_rate = (override_betas or {}).get("pd_rate",
                                         get_beta(class_name, "rate", "pd"))
    delta_pd_pp = b_oil * d_brent_log + b_rate * d_r_10y_pp
    pd_stress = pd_base_dec + delta_pd_pp / 100.0
    return float(np.clip(pd_stress, pd_floor, pd_cap))


def stress_lgd(
    lgd_base_dec: float,
    d_brent_log: float,
    d_r_10y_pp: float,
    class_name: str,
    *,
    override_betas: Optional[dict] = None,
    lgd_floor: float = 0.05,     # 5% sanity floor
    lgd_cap: float = 1.0,         # 100% cap
) -> float:
    """Wende den 2-Faktor-Stress auf eine baseline LGD (in Dezimal) an.

    ΔLGD_pp = γ_oil · ΔBrent_log + γ_rate · Δr_10y_pp
    LGD_stressed = clip(LGD_base + ΔLGD_pp/100, lgd_floor, lgd_cap)
    """
    g_oil  = (override_betas or {}).get("lgd_oil",
                                         get_beta(class_name, "oil",  "lgd"))
    g_rate = (override_betas or {}).get("lgd_rate",
                                         get_beta(class_name, "rate", "lgd"))
    delta_lgd_pp = g_oil * d_brent_log + g_rate * d_r_10y_pp
    lgd_stress = lgd_base_dec + delta_lgd_pp / 100.0
    return float(np.clip(lgd_stress, lgd_floor, lgd_cap))


def stress_segment(
    pd_base_dec: float,
    lgd_base_dec: float,
    ead_eur: float,
    class_name: str,
    *,
    d_brent_log: float,
    d_r_10y_pp: float,
    override_betas: Optional[dict] = None,
) -> dict:
    """Komplettes Stress-Ergebnis für ein Segment.

    Returns dict mit:
      pd_base, pd_stress, d_pd_pp,
      lgd_base, lgd_stress, d_lgd_pp,
      ead, el_base, el_stress, d_el,
      class_name
    """
    pd_s  = stress_pd(pd_base_dec,  d_brent_log, d_r_10y_pp,
                       class_name, override_betas=override_betas)
    lgd_s = stress_lgd(lgd_base_dec, d_brent_log, d_r_10y_pp,
                        class_name, override_betas=override_betas)
    el_base   = pd_base_dec  * lgd_base_dec  * ead_eur
    el_stress = pd_s         * lgd_s         * ead_eur
    return {
        "class_name":  class_name,
        "pd_base":     pd_base_dec,
        "pd_stress":   pd_s,
        "d_pd_pp":     (pd_s - pd_base_dec) * 100,
        "lgd_base":    lgd_base_dec,
        "lgd_stress":  lgd_s,
        "d_lgd_pp":    (lgd_s - lgd_base_dec) * 100,
        "ead":         ead_eur,
        "el_base":     el_base,
        "el_stress":   el_stress,
        "d_el":        el_stress - el_base,
    }


def apply_stress_to_universe(
    universe,
    d_brent_log: float,
    d_r_10y_pp: float,
    *,
    override_betas: Optional[dict] = None,
) -> None:
    """Wendet 2-Faktor-Stress in-place auf alle Segmente eines Universums an.

    Pro Segment:
      - Speichert die baseline-PD und -LGD als `_pd_base` / `_lgd_base`
        (idempotent — bei wiederholtem Aufruf werden die ursprünglichen
        Werte aus den Attributen rekonstruiert)
      - Überschreibt `segment.pd` und `segment.lgd` mit den
        2-Faktor-gestressten Werten

    Nach diesem Aufruf können sämtliche existierenden Vasicek-Funktionen
    (stressed_kpis, capital_bridge etc.) mit z_factor=0 aufgerufen werden
    — sie nehmen die nun bereits gestressten PD/LGD als Input und liefern
    konsistente Capital-Math, ohne weitere M-Shifts.

    Side effect: modifiziert universe.banks in-place.
    """
    for bank_name, portfolio in universe.banks.items():
        for s in portfolio.segments:
            # Idempotent: erste Anwendung speichert baseline ab
            if not hasattr(s, "_pd_base"):
                s._pd_base  = s.pd
                s._lgd_base = s.lgd

            # 2-Faktor-Stress auf gespeicherten baseline anwenden
            class_overrides = (
                override_betas.get(s.exposure_class)
                if isinstance(override_betas, dict) else None
            )
            s.pd  = stress_pd(s._pd_base,  d_brent_log, d_r_10y_pp,
                              s.exposure_class,
                              override_betas=class_overrides)
            s.lgd = stress_lgd(s._lgd_base, d_brent_log, d_r_10y_pp,
                               s.exposure_class,
                               override_betas=class_overrides)


def reset_universe_to_baseline(universe) -> None:
    """Stellt die baseline-PD/LGD-Werte aller Segmente wieder her."""
    for bank_name, portfolio in universe.banks.items():
        for s in portfolio.segments:
            if hasattr(s, "_pd_base"):
                s.pd  = s._pd_base
                s.lgd = s._lgd_base


def capital_bridge_2factor(
    portfolio,
    d_brent_log: float,
    d_r_10y_pp: float,
    *,
    confidence: float = 0.999,
    rho_multiplier: float = 1.0,
    lgd_calibration: float = 1.0,
    override_betas: Optional[dict] = None,
) -> dict:
    """Sequentielle ΔK/ΔRWA/ΔEL-Decomposition auf Basis des 2-Faktor-Modells.

    Ersatz für die alte ``BankPortfolio.capital_bridge(z_factor=...)``-Methode,
    die intern ``conditional_pd(s.pd, rho, z=0)`` aufrief — was am Median des
    Systemfaktors NICHT die Identität ist, sondern die PD um typisch 30–50 %
    reduziert. Das führte zu „EL sinkt egal in welche Richtung der Stress
    geht" — siehe Diskussion ChangeLog 2026-05-29.

    Diese Funktion umgeht das Problem, indem sie die Stages direkt aus den
    2-Faktor-gestressten PD/LGD-Werten ableitet, ohne weitere Vasicek-
    Transformation. Sie verlangt KEINEN vorherigen ``apply_stress_to_universe``-
    Aufruf; sie verwendet pro Segment ``s.pd``/``s.lgd`` als Baseline und
    berechnet das Stress-Niveau selbst über ``stress_pd``/``stress_lgd``.

    Stages
    ------
      Stage 0:  (PD_base,    LGD_base)    → K0, RWA0, EL0
      Stage 1:  (PD_stress,  LGD_base)    → K1, RWA1, EL1  (PD-Channel)
      Stage 2:  (PD_stress,  LGD_stress)  → K2, RWA2, EL2  (PD+LGD)

    Returns dict mit identischer Struktur wie ``BankPortfolio.capital_bridge``
    (K_base/pd_only/stress, delta_K_pd/lgd, RWA_*, EL_*, per_segment).
    """
    # Local import to avoid circulars at module load
    from vasicek import irb_capital_requirement  # type: ignore

    rmult = float(rho_multiplier)
    lgd_cal = float(lgd_calibration)
    rows = []
    for s in portfolio.segments:
        # Baseline werte direkt aus dem Segment (was, falls bereits gestresst,
        # via _pd_base wieder ausgelesen wird)
        pd_b  = float(getattr(s, "_pd_base",  s.pd))
        lgd_b = float(getattr(s, "_lgd_base", s.lgd))

        class_overrides = (
            override_betas.get(s.exposure_class)
            if isinstance(override_betas, dict) else None
        )
        pd_s_val  = stress_pd(pd_b,  d_brent_log, d_r_10y_pp,
                              s.exposure_class,
                              override_betas=class_overrides)
        lgd_s_val = stress_lgd(lgd_b, d_brent_log, d_r_10y_pp,
                               s.exposure_class,
                               override_betas=class_overrides)

        lgd_b_cal = float(np.clip(lgd_b * lgd_cal, 1e-4, 1.0))
        lgd_s_cal = float(np.clip(lgd_s_val * lgd_cal, 1e-4, 1.0))

        def _metrics(pd_v: float, lgd_v: float) -> tuple:
            m = irb_capital_requirement(
                pd_v, lgd_v, s.exposure_class,
                maturity_years=s.maturity_years,
                sales_m_eur=s.sales_m_eur,
                confidence=confidence,
                rho_multiplier=rmult,
                lgd_calibration=1.0,  # already calibrated above
            )
            K   = float(m["K"])           * s.ead
            RWA = float(m["rwa_density"]) * s.ead
            EL  = float(m["EL"])          * s.ead
            return K, RWA, EL

        K0, RWA0, EL0 = _metrics(pd_b,    lgd_b_cal)
        K1, RWA1, EL1 = _metrics(pd_s_val, lgd_b_cal)
        K2, RWA2, EL2 = _metrics(pd_s_val, lgd_s_cal)

        rows.append({
            "name":           s.name,
            "exposure_class": s.exposure_class,
            "ead":            s.ead,
            "pd_base":        pd_b,
            "pd_stress":      pd_s_val,
            "lgd_base":       lgd_b,
            "lgd_stress":     lgd_s_val,
            "K_base":         K0,
            "K_pd_only":      K1,
            "K_stress":       K2,
            "dK_pd":          K1 - K0,
            "dK_lgd":         K2 - K1,
            "RWA_base":       RWA0,
            "RWA_pd_only":    RWA1,
            "RWA_stress":     RWA2,
            "dRWA_pd":        RWA1 - RWA0,
            "dRWA_lgd":       RWA2 - RWA1,
            "EL_base":        EL0,
            "EL_pd_only":     EL1,
            "EL_stress":      EL2,
            "dEL_pd":         EL1 - EL0,
            "dEL_lgd":        EL2 - EL1,
        })

    import pandas as pd  # local import
    seg_df = pd.DataFrame(rows)

    if seg_df.empty:
        return {
            "K_base": 0.0, "K_pd_only": 0.0, "K_stress": 0.0,
            "delta_K_pd": 0.0, "delta_K_lgd": 0.0, "delta_K": 0.0,
            "rwa_base": 0.0, "rwa_pd_only": 0.0, "rwa_stress": 0.0,
            "delta_rwa_pd": 0.0, "delta_rwa_lgd": 0.0, "delta_rwa": 0.0,
            "el_base": 0.0, "el_pd_only": 0.0, "el_stress": 0.0,
            "delta_el_pd": 0.0, "delta_el_lgd": 0.0, "delta_el": 0.0,
            "per_segment": seg_df,
        }

    return {
        "K_base":        float(seg_df["K_base"].sum()),
        "K_pd_only":     float(seg_df["K_pd_only"].sum()),
        "K_stress":      float(seg_df["K_stress"].sum()),
        "delta_K_pd":    float(seg_df["dK_pd"].sum()),
        "delta_K_lgd":   float(seg_df["dK_lgd"].sum()),
        "delta_K":       float(seg_df["K_stress"].sum() - seg_df["K_base"].sum()),
        "rwa_base":      float(seg_df["RWA_base"].sum()),
        "rwa_pd_only":   float(seg_df["RWA_pd_only"].sum()),
        "rwa_stress":    float(seg_df["RWA_stress"].sum()),
        "delta_rwa_pd":  float(seg_df["dRWA_pd"].sum()),
        "delta_rwa_lgd": float(seg_df["dRWA_lgd"].sum()),
        "delta_rwa":     float(seg_df["RWA_stress"].sum() - seg_df["RWA_base"].sum()),
        "el_base":       float(seg_df["EL_base"].sum()),
        "el_pd_only":    float(seg_df["EL_pd_only"].sum()),
        "el_stress":     float(seg_df["EL_stress"].sum()),
        "delta_el_pd":   float(seg_df["dEL_pd"].sum()),
        "delta_el_lgd":  float(seg_df["dEL_lgd"].sum()),
        "delta_el":      float(seg_df["EL_stress"].sum() - seg_df["EL_base"].sum()),
        "per_segment":   seg_df,
    }


def stress_rwa(
    rwa_base_eur: float,
    pd_base_dec: float,
    pd_stress_dec: float,
    *,
    lgd_base_dec: float = 0.45,
    lgd_stress_dec: float = 0.45,
) -> float:
    """Lineare RWA-Skalierung unter 2-Faktor-Stress.

    Vereinfachung statt voller IRB-Formel: RWA skaliert proportional zur
    PD- und LGD-Veränderung. Dokumentation: dieser linearer Ansatz ersetzt
    die frühere Vasicek-ASRF-Schließungs-Formel.

        RWA_stress / RWA_base ≈ (PD_stress · LGD_stress) / (PD_base · LGD_base)
    """
    base_el  = pd_base_dec   * lgd_base_dec
    stress_el = pd_stress_dec * lgd_stress_dec
    if base_el <= 0:
        return rwa_base_eur
    return float(rwa_base_eur * stress_el / base_el)


# ============================================================================
# 4. Tests
# ============================================================================
def _test_classes_complete():
    expected = {"corporate", "sme_corporate", "mortgage", "qrre",
                "other_retail", "bank", "sovereign"}
    assert expected <= set(VASICEK_CLASSES), \
        f"Fehlende Klassen: {expected - set(VASICEK_CLASSES)}"


def _test_bank_class_has_negative_rate_beta():
    """Critique Punkt 7: Bank-Klasse muss negative Rate-Sensitivität haben."""
    assert get_beta("bank", "rate", "pd") < 0, \
        "Bank-Klasse muss β_rate < 0 haben (NIM-Effekt)"


def _test_stress_application():
    # Baseline PD 1%, +50% Brent, +200bp Rate, Corporate-Klasse:
    # ΔPD = 0.30·0.5 + 0.20·2.0 = 0.55pp → PD_stress = 1.55%
    p = stress_pd(0.01, 0.5, 2.0, "corporate")
    assert abs(p - 0.0155) < 1e-4, f"Expected 1.55%, got {p*100:.3f}%"


def _test_mortgage_rate_dominant():
    """Mortgage muss stärker auf Rate als auf Oil reagieren."""
    assert get_beta("mortgage", "rate", "pd") > \
           get_beta("mortgage", "oil", "pd")


def _test_sovereign_inert():
    """Sovereign muss auf Macro-Schocks 0 sein (separater Channel)."""
    p = stress_pd(0.001, 1.0, 5.0, "sovereign")
    assert abs(p - 0.001) < 1e-6


def _test_sme_amplifies_corporate():
    """ECB WP 2897 (2024): KMU-PD reagiert ≈2× so stark wie Large-Corp.

    Recalibration Juni 2026 — die SME-β müssen die in WP 2897 gemessene
    Größen-Heterogenität (≈2× für Angebots-, ≈3× für Geldpolitik-Schock)
    abbilden, also klar über den Corporate-β liegen.
    """
    for factor in ("oil", "rate"):
        b_sme = get_beta("sme_corporate", factor, "pd")
        b_cor = get_beta("corporate",     factor, "pd")
        assert b_sme >= 1.8 * b_cor - 1e-9, (
            f"SME β_{factor} ({b_sme}) sollte ≈2× Corporate ({b_cor}) sein")


def _test_sources_current_no_stale_refs():
    """Keine falschen EBA-Abschnittsnummern mehr (Recalibration Juni 2026).

    §5.3.2 = Conduct risk, §6.1/§6.2 = Non-interest income im echten
    EBA-2025-Methodology-Note. Kreditrisiko-Projektion ist §2.4.2. Diese
    Stale-Referenzen dürfen NICHT mehr in den Quellen stehen.
    """
    stale = ("5.3.2", "5.3.4", "Sec. 6.1", "Sec. 6.2", "§3.4", "§3.5")
    for cls in VASICEK_CLASSES:
        src = get_sources(cls)
        blob = (src["pd"] + " " + src["lgd"])
        for bad in stale:
            assert bad not in blob, \
                f"{cls}: veraltete Quellen-Referenz '{bad}' in {blob!r}"
        # jede Nicht-Sovereign-Klasse muss einen aktuellen (2024/2025)
        # Quellen-Anker zitieren
        if cls != "sovereign":
            assert any(k in blob for k in
                       ("WP 2897", "WP 3112", "FSR Mai 2024", "FSR 2024",
                        "§2.4.2", "Benchmarking 2024", "Zinsüberschuss",
                        "EBA-2025-Results")), \
                f"{cls}: kein aktueller Quellen-Anker in {blob!r}"


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("two_factor_stress.py · Tests")
    print("=" * 60)
    for label, fn in [
        ("Klassen vollständig",              _test_classes_complete),
        ("Bank β_rate negativ (NIM)",        _test_bank_class_has_negative_rate_beta),
        ("Stress-Anwendung Corporate",       _test_stress_application),
        ("Mortgage Rate-dominant",           _test_mortgage_rate_dominant),
        ("Sovereign macro-inert",            _test_sovereign_inert),
        ("SME ≈2× Corporate (WP 2897)",      _test_sme_amplifies_corporate),
        ("Quellen aktuell (keine §5.3.2)",   _test_sources_current_no_stale_refs),
    ]:
        try:
            fn()
            print(f"  [PASS]  {label}")
        except AssertionError as e:
            print(f"  [FAIL]  {label}: {e}")
            raise
    print()
    print("Sensitivität-Matrix-Zusammenfassung:")
    for c in VASICEK_CLASSES:
        m = SENSITIVITY_MATRIX[c]
        print(f"  {c:15s} β_pd: oil={m['pd_oil']:+.2f} rate={m['pd_rate']:+.2f}  "
              f"γ_lgd: oil={m['lgd_oil']:+.2f} rate={m['lgd_rate']:+.2f}")
    print()
    print("[PASS] All tests passed.")
