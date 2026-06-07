"""
============================================================================
 vasicek.py · ASRF Single-Factor Credit Portfolio Model (Basel III IRB)
============================================================================

Implementiert den Vasicek (1991/2002) Asymptotic-Single-Risk-Factor-Modell
in der genauen Form, wie er in der Basel-III-IRB-Capital-Formula
(BCBS §272 ff.) verwendet wird. Geschlossene Form, vektorisiert.

Kern-Beziehung
--------------
Asset-Return einer Counterparty:
    A_i = √ρ · M + √(1-ρ) · ε_i,        M, ε_i ~ N(0,1) iid
Default tritt ein, wenn A_i < N⁻¹(PD_i).

Bedingte PD gegeben Systemfaktor M = z:
    PD(z) = N( (N⁻¹(PD) - √ρ · z) / √(1-ρ) )

ASRF-Loss-Quantil (Vasicek 2002):
    L_α = LGD · N( (N⁻¹(PD) + √ρ · N⁻¹(α)) / √(1-ρ) )

Asset-Korrelation ρ (Basel III)
-------------------------------
Corporate / Bank / Sovereign:
    ρ = 0.12·(1-e⁻⁵⁰ᴾᴰ)/(1-e⁻⁵⁰) + 0.24·(1 - (1-e⁻⁵⁰ᴾᴰ)/(1-e⁻⁵⁰))

SME-Adjustment (Umsatz S ∈ [5,50] m EUR):
    ρ_SME = ρ_corp − 0.04·(1 − (S-5)/45)

Residential Mortgage:   ρ = 0.15
QRRE (Qualifying Revolving Retail): ρ = 0.04
Other Retail:
    ρ = 0.03·(1-e⁻³⁵ᴾᴰ)/(1-e⁻³⁵) + 0.16·(1 - (1-e⁻³⁵ᴾᴰ)/(1-e⁻³⁵))

Maturity-Adjustment (Corporate only)
------------------------------------
    b(PD) = (0.11852 − 0.05478 · ln PD)²
    MA(M) = (1 + (M-2.5)·b) / (1 − 1.5·b)

IRB-Capital-Charge K
--------------------
    K = [LGD · N(…) − PD · LGD] · MA(M)
    RWA = K · 12.5 · EAD                    (Basel-Faktor 12.5 = 1/8% Mindestkap.)

Quellen
-------
- Vasicek, O. (1991). *Limiting Loan Loss Probability Distribution.* KMV Working Paper.
- Vasicek, O. (2002). *Loan Portfolio Value.* Risk Magazine, December.
- BCBS (2017). *Basel III: Finalising post-crisis reforms.*
- Gordy, M. (2003). *A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules.*
============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================================
# 1. Asset-Korrelation ρ (Basel III)
# ============================================================================
ExposureClass = Literal[
    "corporate",
    "sme_corporate",
    "bank",
    "sovereign",
    "mortgage",
    "qrre",
    "other_retail",
]


def _is_scalar(x) -> bool:
    """True iff x is a Python scalar or a 0-d numpy array."""
    return np.ndim(x) == 0


def rho_corporate(pd_value: np.ndarray | float) -> np.ndarray | float:
    """Asset-Korrelation für Corporate / Bank / Sovereign Exposure.

    BCBS §272(a):
        ρ(PD) = 0.12·R(PD) + 0.24·(1 − R(PD))
        R(PD) = (1-e⁻⁵⁰ᴾᴰ)/(1-e⁻⁵⁰)

    Monoton fallend in PD von 0.24 → 0.12.
    """
    scalar = _is_scalar(pd_value)
    pd_arr = np.clip(np.asarray(pd_value, dtype=float), 1e-12, 1.0)
    r = (1.0 - np.exp(-50.0 * pd_arr)) / (1.0 - np.exp(-50.0))
    rho = 0.12 * r + 0.24 * (1.0 - r)
    return float(rho) if scalar else rho


def rho_sme(pd_value: np.ndarray | float, sales_m_eur: float) -> np.ndarray | float:
    """SME-Korrelations-Adjustment. `sales_m_eur` = Umsatz in m EUR."""
    s = float(np.clip(sales_m_eur, 5.0, 50.0))
    return rho_corporate(pd_value) - 0.04 * (1.0 - (s - 5.0) / 45.0)


def rho_mortgage(pd_value: np.ndarray | float) -> np.ndarray | float:
    """Residential Mortgage: konstant 0.15."""
    if _is_scalar(pd_value):
        return 0.15
    return np.full_like(np.asarray(pd_value, dtype=float), 0.15)


def rho_qrre(pd_value: np.ndarray | float) -> np.ndarray | float:
    """Qualifying Revolving Retail: konstant 0.04."""
    if _is_scalar(pd_value):
        return 0.04
    return np.full_like(np.asarray(pd_value, dtype=float), 0.04)


def rho_other_retail(pd_value: np.ndarray | float) -> np.ndarray | float:
    """Other-Retail Korrelation (BCBS §330)."""
    scalar = _is_scalar(pd_value)
    pd_arr = np.clip(np.asarray(pd_value, dtype=float), 1e-12, 1.0)
    r = (1.0 - np.exp(-35.0 * pd_arr)) / (1.0 - np.exp(-35.0))
    rho = 0.03 * r + 0.16 * (1.0 - r)
    return float(rho) if scalar else rho


def asset_correlation(
    pd_value: np.ndarray | float,
    exposure_class: ExposureClass,
    sales_m_eur: float | None = None,
) -> np.ndarray | float:
    """Dispatcher — einheitliche API für alle Exposure-Classes."""
    if exposure_class in ("corporate", "bank", "sovereign"):
        return rho_corporate(pd_value)
    if exposure_class == "sme_corporate":
        if sales_m_eur is None:
            raise ValueError("sme_corporate requires `sales_m_eur` argument.")
        return rho_sme(pd_value, sales_m_eur)
    if exposure_class == "mortgage":
        return rho_mortgage(pd_value)
    if exposure_class == "qrre":
        return rho_qrre(pd_value)
    if exposure_class == "other_retail":
        return rho_other_retail(pd_value)
    raise ValueError(f"Unknown exposure class: {exposure_class!r}")


# ============================================================================
# 2. Maturity Adjustment (Corporate)
# ============================================================================
def maturity_b(pd_value: np.ndarray | float) -> np.ndarray | float:
    """Smoothed-PD-Slope b(PD) aus BCBS §272."""
    scalar = _is_scalar(pd_value)
    pd_arr = np.clip(np.asarray(pd_value, dtype=float), 1e-12, 1.0)
    b = (0.11852 - 0.05478 * np.log(pd_arr)) ** 2
    return float(b) if scalar else b


def maturity_adjustment(
    pd_value: np.ndarray | float, maturity_years: float = 2.5,
) -> np.ndarray | float:
    """Effective-Maturity-Adjustment (Corporate only).

    MA(M) = (1 + (M − 2.5) · b(PD)) / (1 − 1.5 · b(PD))

    Bei M = 2.5 → MA = 1 / (1 − 1.5·b). Basel cap: M ∈ [1.0, 5.0].
    """
    m = float(np.clip(maturity_years, 1.0, 5.0))
    b = maturity_b(pd_value)
    return (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)


# ============================================================================
# 3. Conditional PD und Loss-Quantile
# ============================================================================
def conditional_pd(
    pd_value: np.ndarray | float,
    rho: np.ndarray | float,
    z_factor: float,
) -> np.ndarray | float:
    """PD bedingt auf Systemfaktor M = z_factor.

    PD(z) = N( (N⁻¹(PD) − √ρ · z) / √(1−ρ) )

    Sign-Konvention: positives z = günstiger Marktzustand → niedrigere PD.
    Adverse Stress entspricht z < 0, z.B. z = N⁻¹(0.001) ≈ −3.09.
    """
    scalar = _is_scalar(pd_value) and _is_scalar(rho)
    pd_arr = np.clip(np.asarray(pd_value, dtype=float), 1e-12, 1.0 - 1e-12)
    rho_arr = np.clip(np.asarray(rho, dtype=float), 1e-12, 1.0 - 1e-12)

    num = norm.ppf(pd_arr) - np.sqrt(rho_arr) * z_factor
    cond = norm.cdf(num / np.sqrt(1.0 - rho_arr))
    return float(cond) if scalar else cond


def asrf_loss_quantile(
    pd_value: np.ndarray | float,
    lgd: np.ndarray | float,
    rho: np.ndarray | float,
    confidence: float = 0.999,
) -> np.ndarray | float:
    """ASRF-Loss-Quantil bei Confidence α.

    L_α = LGD · N( (N⁻¹(PD) + √ρ · N⁻¹(α)) / √(1−ρ) )

    Das ist die *unbedingte* α-Quantile-Loss-Rate, ausgedrückt als Anteil
    an EAD. Multiplikation mit EAD liefert den Loss in EUR.
    """
    z_alpha = norm.ppf(confidence)
    # conditional_pd nutzt M=z mit Sign-Konvention "z<0 = stress"
    # Hier: stress entspricht z = -z_alpha (worst case M)
    cond = conditional_pd(pd_value, rho, -z_alpha)
    return np.asarray(lgd) * cond


# ============================================================================
# 3a. Downturn-LGD (stress-elastische LGD)
# ============================================================================
def downturn_lgd(
    lgd_base: float | np.ndarray,
    m_factor: float,
    kappa: float = 0.3,
    *,
    exponent: float = 1.0,
    cap_mode: str = "hard",
) -> float | np.ndarray:
    """Stress-elastische LGD mit V2-Erweiterungen (Non-Linearität + Logit-Cap).

    Funktionale Form V1 (linear, hard-cap):
        LGD_stress = min(LGD_base · (1 + κ · |M|), 1.0),  M < 0

    Funktionale Form V2 (Non-Linearität via exponent ≠ 1, Logit-Cap optional):
        stress_signal = κ · |M|^exponent
        cap_mode = "hard"  : LGD = min(LGD_base · (1 + stress_signal), 1.0)
        cap_mode = "logit" : LGD = σ(logit(LGD_base) + stress_signal),
                              asymptotisch gegen 1.0 ohne mathematische
                              Varianz-Abschnürung an der 100%-Grenze.

    Adressiert vier methodische Kritikpunkte:

      1. Lineare Skalierung vs. Realität (Tail-Risk) — `exponent > 1`
         erlaubt überproportionalen LGD-Anstieg bei extremem M.
         Beispiel exponent = 1.5: bei |M| = 2.5 wird der Multiplikator
         κ·2.5^1.5 = 0.30·3.95 = 1.19 statt 0.75 (lineare V1).

      2. Hard-Cap bei 100% — bei QRRE (Baseline 65%) sättigt die V1 schon
         bei |M| ≈ 1.79 (= (1/0.65 - 1) / 0.30). `cap_mode = "logit"`
         entfernt diese Diskontinuität: die Sigmoid-Transformation nähert
         sich 100% asymptotisch ohne Informationsverlust im Extrembereich.

      3. Sektor-Granularität (κ pro Class) — wird durch die Aufrufer
         (BankPortfolio.*) erreicht; downturn_lgd selbst bleibt skalar-κ.

      4. F-IRB vs. A-IRB Kalibrierung — wird im `lgd_calibration`-Parameter
         der Aufrufer (siehe irb_capital_requirement) abgebildet, NICHT
         hier. Eingang `lgd_base` ist bereits kalibriert.

    Konsequenz für V1-Default (exponent=1.0, cap_mode="hard"): identische
    Werte wie zuvor — keine Verhaltensänderung ohne explizite V2-Aktivierung.
    """
    if m_factor >= 0:
        return lgd_base

    stress_signal = float(kappa) * (abs(float(m_factor)) ** float(exponent))

    if cap_mode == "logit":
        eps = 1e-9
        if isinstance(lgd_base, np.ndarray):
            x = np.clip(lgd_base, eps, 1.0 - eps)
            logit_base = np.log(x / (1.0 - x))
            return 1.0 / (1.0 + np.exp(-(logit_base + stress_signal)))
        x = float(np.clip(lgd_base, eps, 1.0 - eps))
        logit_base = float(np.log(x / (1.0 - x)))
        return float(1.0 / (1.0 + np.exp(-(logit_base + stress_signal))))

    # Default: hard cap at 100%
    multiplier = 1.0 + stress_signal
    if isinstance(lgd_base, np.ndarray):
        return np.minimum(lgd_base * multiplier, 1.0)
    return min(float(lgd_base) * multiplier, 1.0)


def irb_capital_requirement(
    pd_value: np.ndarray | float,
    lgd: np.ndarray | float,
    exposure_class: ExposureClass,
    *,
    maturity_years: float = 2.5,
    sales_m_eur: float | None = None,
    confidence: float = 0.999,
    rho_multiplier: float = 1.0,
    lgd_calibration: float = 1.0,
) -> dict:
    """Vollständige IRB-Capital-Berechnung pro Exposure.

    Parameters
    ----------
    rho_multiplier : float, default 1.0
        Modell-Risiko-Override auf die Basel-formelmäßige Asset-Korrelation
        ρ. Default 1.0 = Basel-Standard. Werte > 1 simulieren höhere
        Korrelation (Crisis-Stress), < 1 niedrigere. Wirkt sowohl im
        ASRF-Loss-Quantil als auch in der maturity-adjustment-Anwendung.

    Returns
    -------
    dict mit Keys:
        rho   : Asset-Korrelation (nach Multiplikator)
        K     : Kapital-Anforderung pro EAD-Einheit (Anteil)
        UL    : Unexpected-Loss-Rate = K (synonym)
        EL    : Expected-Loss-Rate = PD · LGD
        L_a   : ASRF-Loss-Quantil bei `confidence`
        rwa_density : K · 12.5 (RWA pro EAD-Einheit)
        ma    : Maturity-Adjustment (1 für Retail/Mortgage)
    """
    rho_base = asset_correlation(pd_value, exposure_class, sales_m_eur)
    rho = np.clip(rho_base * float(rho_multiplier), 1e-4, 0.999)
    # F-IRB → A-IRB-Proxy: lgd_calibration < 1 simuliert empirisch
    # niedrigere realized LGDs (Sicherheiten, Collateral) wie sie A-IRB-
    # Banken intern schätzen. lgd_calibration > 1 wäre Stress-Up auf
    # die regulatorischen Defaults.
    lgd_eff = np.clip(np.asarray(lgd) * float(lgd_calibration), 1e-4, 1.0)
    L_a = asrf_loss_quantile(pd_value, lgd_eff, rho, confidence=confidence)
    el = np.asarray(pd_value) * lgd_eff

    if exposure_class in ("corporate", "sme_corporate", "bank", "sovereign"):
        ma = maturity_adjustment(pd_value, maturity_years)
    else:
        ma = 1.0 if _is_scalar(pd_value) else np.ones_like(np.asarray(pd_value, dtype=float))

    K = (np.asarray(L_a) - el) * np.asarray(ma)
    K = np.maximum(K, 0.0)
    rwa_density = K * 12.5

    return {
        "rho":          rho,
        "K":            K,
        "UL":           K,
        "EL":           el,
        "L_a":          L_a,
        "rwa_density":  rwa_density,
        "ma":           ma,
    }


# ============================================================================
# 4. Portfolio-Aggregation
# ============================================================================
@dataclass
class PortfolioSegment:
    """Ein Exposure-Class-Segment in einem Bank-Portfolio."""

    name: str
    exposure_class: ExposureClass
    ead: float                  # EUR
    pd: float                   # 1Y-Default-Probability (Dezimal)
    lgd: float                  # Loss-Given-Default (Dezimal)
    maturity_years: float = 2.5
    sales_m_eur: float | None = None    # nur bei sme_corporate genutzt

    def basel_metrics(self, confidence: float = 0.999,
                      *, rho_multiplier: float = 1.0,
                      lgd_calibration: float = 1.0) -> dict:
        """IRB-Metriken für das Segment."""
        m = irb_capital_requirement(
            self.pd, self.lgd, self.exposure_class,
            maturity_years=self.maturity_years,
            sales_m_eur=self.sales_m_eur,
            confidence=confidence,
            rho_multiplier=rho_multiplier,
            lgd_calibration=lgd_calibration,
        )
        return {
            "name":         self.name,
            "exposure_class": self.exposure_class,
            "ead":          self.ead,
            "pd":           self.pd,
            "lgd":          self.lgd,
            "rho":          float(m["rho"]),
            "ma":           float(m["ma"]),
            "el_pct":       float(m["EL"]),
            "el_eur":       float(m["EL"]) * self.ead,
            "ul_pct":       float(m["K"]),
            "ul_eur":       float(m["K"]) * self.ead,
            "rwa":          float(m["rwa_density"]) * self.ead,
            "loss_p999":    float(m["L_a"]) * self.ead,
        }


@dataclass
class BankPortfolio:
    """Aggregat-Bank-Portfolio über mehrere Exposure-Class-Segmente."""

    name: str
    segments: list[PortfolioSegment] = field(default_factory=list)
    lei: str = ""   # offizieller LEI-Code — robuster Join-Key zur Pillar-3-CSV

    def add(self, *segments: PortfolioSegment) -> "BankPortfolio":
        self.segments.extend(segments)
        return self

    @property
    def total_ead(self) -> float:
        return float(sum(s.ead for s in self.segments))

    def baseline_metrics(self, confidence: float = 0.999,
                         *, rho_multiplier: float = 1.0,
                         lgd_calibration: float = 1.0) -> pd.DataFrame:
        rows = [s.basel_metrics(confidence,
                                 rho_multiplier=rho_multiplier,
                                 lgd_calibration=lgd_calibration)
                for s in self.segments]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["ead_share"] = df["ead"] / df["ead"].sum()
        return df

    def portfolio_kpis(self, confidence: float = 0.999,
                       *, rho_multiplier: float = 1.0,
                       lgd_calibration: float = 1.0) -> dict:
        df = self.baseline_metrics(confidence,
                                     rho_multiplier=rho_multiplier,
                                     lgd_calibration=lgd_calibration)
        if df.empty:
            return {}
        return {
            "name":          self.name,
            "n_segments":    len(self.segments),
            "total_ead":     float(df["ead"].sum()),
            "el_eur":        float(df["el_eur"].sum()),
            "ul_eur":        float(df["ul_eur"].sum()),
            "rwa":           float(df["rwa"].sum()),
            "loss_p999":     float(df["loss_p999"].sum()),
            "el_pct":        float(df["el_eur"].sum() / df["ead"].sum()),
            "ul_pct":        float(df["ul_eur"].sum() / df["ead"].sum()),
            "rwa_density":   float(df["rwa"].sum() / df["ead"].sum()),
            "capital_ratio_at_999": float(df["loss_p999"].sum() / df["ead"].sum()),
        }

    def stressed_metrics(
        self, z_factor: float,
        *,
        confidence: float = 0.999,
        kappa_lgd: float = 0.3,
        rho_multiplier: float = 1.0,
        lgd_calibration: float = 1.0,
        stress_exponent: float = 1.0,
        cap_mode: str = "hard",
    ) -> pd.DataFrame:
        """Portfolio-Metriken unter Systemfaktor-Schock M = z_factor.

        Beide Risiko-Parameter werden gestresst:
          - PD via Vasicek conditional PD (siehe §11.1)
          - LGD via downturn-LGD-Funktion (siehe §11.x):
              LGD_stress = LGD_base · (1 + κ · max(-M, 0))

        ``rho_multiplier`` skaliert die Basel-ρ konsistent in Conditional-
        PD und Capital-Charge (Modell-Risiko-Override).
        """
        rows = []
        rmult = float(rho_multiplier)
        lgd_cal = float(lgd_calibration)
        for s in self.segments:
            rho_basel = asset_correlation(s.pd, s.exposure_class, s.sales_m_eur)
            rho = float(np.clip(rho_basel * rmult, 1e-4, 0.999))
            # Apply lgd_calibration BEFORE downturn-LGD (A-IRB-Proxy first,
            # then stress on top) — methodisch konsistent mit "echtem"
            # bank-internen LGD-Niveau.
            lgd_calibrated = float(np.clip(s.lgd * lgd_cal, 1e-4, 1.0))
            stressed_pd = float(conditional_pd(s.pd, rho, z_factor))
            stressed_lgd = float(downturn_lgd(
                lgd_calibrated, z_factor, kappa=kappa_lgd,
                exponent=stress_exponent, cap_mode=cap_mode,
            ))
            stressed_seg = PortfolioSegment(
                name=s.name, exposure_class=s.exposure_class,
                ead=s.ead, pd=stressed_pd, lgd=stressed_lgd,
                maturity_years=s.maturity_years, sales_m_eur=s.sales_m_eur,
            )
            # NB: pass lgd_calibration=1.0 hier — die Kalibrierung wurde
            # bereits oben in s.lgd → lgd_calibrated angewendet.
            row = stressed_seg.basel_metrics(confidence, rho_multiplier=rmult,
                                              lgd_calibration=1.0)
            row["pd_baseline"]  = s.pd
            row["pd_stressed"]  = stressed_pd
            row["delta_pd"]     = stressed_pd - s.pd
            row["lgd_baseline"] = s.lgd
            row["lgd_stressed"] = stressed_lgd
            row["delta_lgd"]    = stressed_lgd - s.lgd
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["ead_share"] = df["ead"] / df["ead"].sum()
        return df

    def stressed_kpis(
        self, z_factor: float,
        *,
        confidence: float = 0.999,
        kappa_lgd: float = 0.3,
        rho_multiplier: float = 1.0,
        lgd_calibration: float = 1.0,
        stress_exponent: float = 1.0,
        cap_mode: str = "hard",
    ) -> dict:
        df = self.stressed_metrics(
            z_factor, confidence=confidence, kappa_lgd=kappa_lgd,
            rho_multiplier=rho_multiplier,
            lgd_calibration=lgd_calibration,
            stress_exponent=stress_exponent, cap_mode=cap_mode,
        )
        if df.empty:
            return {}
        baseline_kpis = self.portfolio_kpis(
            confidence, rho_multiplier=rho_multiplier,
            lgd_calibration=lgd_calibration,
        )
        return {
            "name":          self.name,
            "z_factor":      z_factor,
            "kappa_lgd":     kappa_lgd,
            "n_segments":    len(self.segments),
            "total_ead":     float(df["ead"].sum()),
            "el_eur":        float(df["el_eur"].sum()),
            "ul_eur":        float(df["ul_eur"].sum()),
            "rwa":           float(df["rwa"].sum()),
            "loss_p999":     float(df["loss_p999"].sum()),
            "el_pct":        float(df["el_eur"].sum() / df["ead"].sum()),
            "delta_el_eur":  float(df["el_eur"].sum() - baseline_kpis["el_eur"]),
            "delta_rwa":     float(df["rwa"].sum() - baseline_kpis["rwa"]),
        }

    # ------------------------------------------------------------------
    # Capital Bridge · sequential decomposition of ΔK and ΔRWA
    # ------------------------------------------------------------------
    def capital_bridge(
        self, z_factor: float,
        *,
        confidence: float = 0.999,
        kappa_lgd: float = 0.3,
        rho_multiplier: float = 1.0,
        lgd_calibration: float = 1.0,
        stress_exponent: float = 1.0,
        cap_mode: str = "hard",
    ) -> dict:
        """Sequentielle Aktivierung der zwei Stress-Channels.

        Zerlegt ΔK = K_stress − K_base in additive Komponenten durch
        sequenzielles Anschalten der Stresses:

          Stage 0:  (PD_base, LGD_base)        → K_base
          Stage 1:  (PD(M),   LGD_base)        → K_pd_only
          Stage 2:  (PD(M),   LGD(M))          → K_stress

        Dann gilt:
          ΔK_PD  = K_pd_only - K_base   ← Beitrag PD-Shift (LGD fix)
          ΔK_LGD = K_stress  - K_pd_only ← Beitrag LGD-Shift (PD bereits stress)
          ΔK     = ΔK_PD + ΔK_LGD        ← exakt additiv

        Returns
        -------
        dict mit
          K_base, K_pd_only, K_stress      (jeweils EUR, gewichtet mit EAD)
          delta_K_pd, delta_K_lgd, delta_K (EUR)
          rwa_base, rwa_pd_only, rwa_stress, delta_rwa_*  (EUR)
          el_base, el_pd_only, el_stress, delta_el_*       (EUR)
          per_segment: DataFrame mit Stage-by-Stage Werten pro Segment
        """
        rows = []
        rmult   = float(rho_multiplier)
        lgd_cal = float(lgd_calibration)
        expn    = float(stress_exponent)
        cmode   = str(cap_mode)
        for s in self.segments:
            rho_basel = asset_correlation(s.pd, s.exposure_class, s.sales_m_eur)
            rho = float(np.clip(rho_basel * rmult, 1e-4, 0.999))
            # Apply lgd_calibration (A-IRB proxy) first, then downturn-LGD
            lgd_calibrated = float(np.clip(s.lgd * lgd_cal, 1e-4, 1.0))
            pd_stress  = float(conditional_pd(s.pd, rho, z_factor))
            lgd_stress = float(downturn_lgd(
                lgd_calibrated, z_factor, kappa=kappa_lgd,
                exponent=expn, cap_mode=cmode,
            ))

            def _metrics(pd_v: float, lgd_v: float) -> tuple[float, float, float]:
                """Return (K, RWA, EL) in EUR for this segment.

                NB: lgd_calibration ist BEREITS in pd_v/lgd_v eingepreist,
                deshalb hier lgd_calibration=1.0 weiterreichen.
                """
                m = irb_capital_requirement(
                    pd_v, lgd_v, s.exposure_class,
                    maturity_years=s.maturity_years,
                    sales_m_eur=s.sales_m_eur,
                    confidence=confidence,
                    rho_multiplier=rmult,
                    lgd_calibration=1.0,
                )
                K   = float(m["K"])           * s.ead
                RWA = float(m["rwa_density"]) * s.ead
                EL  = float(m["EL"])          * s.ead
                return K, RWA, EL

            # Stage 0 uses calibrated LGD as the BASELINE — d.h. wenn der
            # User lgd_calibration ändert, verschiebt sich die Baseline mit.
            # Andernfalls würde der LGD-Bridge-Beitrag verzerrt aussehen.
            K0, RWA0, EL0 = _metrics(s.pd,        lgd_calibrated)
            K1, RWA1, EL1 = _metrics(pd_stress,   lgd_calibrated)
            K2, RWA2, EL2 = _metrics(pd_stress,   lgd_stress)

            rows.append({
                "name":           s.name,
                "exposure_class": s.exposure_class,
                "ead":            s.ead,
                "pd_base":        s.pd,
                "pd_stress":      pd_stress,
                "lgd_base":       s.lgd,
                "lgd_stress":     lgd_stress,
                # K
                "K_base":         K0,
                "K_pd_only":      K1,
                "K_stress":       K2,
                "dK_pd":          K1 - K0,
                "dK_lgd":         K2 - K1,
                # RWA
                "RWA_base":       RWA0,
                "RWA_pd_only":    RWA1,
                "RWA_stress":     RWA2,
                "dRWA_pd":        RWA1 - RWA0,
                "dRWA_lgd":       RWA2 - RWA1,
                # EL
                "EL_base":        EL0,
                "EL_pd_only":     EL1,
                "EL_stress":      EL2,
                "dEL_pd":         EL1 - EL0,
                "dEL_lgd":        EL2 - EL1,
            })

        seg_df = pd.DataFrame(rows)

        # Totals across all segments
        return {
            "z_factor":      z_factor,
            "kappa_lgd":     kappa_lgd,
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


# ============================================================================
# 5. Validation tests
# ============================================================================
def _test_rho_monotonicity():
    """ρ_corporate monoton fallend in PD von 0.24 → 0.12."""
    pds = np.array([1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5])
    rhos = rho_corporate(pds)
    assert np.all(np.diff(rhos) < 0), "ρ should be monotonically decreasing in PD"
    assert abs(rhos[0] - 0.24) < 1e-3, f"ρ(PD→0) ≈ 0.24, got {rhos[0]:.4f}"
    assert abs(rhos[-1] - 0.12) < 1e-3, f"ρ(PD→1) ≈ 0.12, got {rhos[-1]:.4f}"


def _test_basel_irb_reference():
    """Reproduziere ein BCBS-Beispiel — Corporate, PD=1%, LGD=45%, M=2.5y.

    Erwartete Capital-Charge laut Basel-III-Beispielrechnungen:
        K ≈ 8.4% · EAD bei PD=1%, LGD=45%, M=2.5y
    """
    res = irb_capital_requirement(0.01, 0.45, "corporate", maturity_years=2.5)
    K = float(res["K"])
    assert 0.07 < K < 0.10, f"K={K:.4f}, expected ≈ 0.084 for PD=1%/LGD=45%/M=2.5"


def _test_conditional_pd_monotonic_in_z():
    """Bedingte PD monoton fallend in z (positives z = besserer Zustand)."""
    pd0 = 0.02
    rho = float(rho_corporate(pd0))
    zs = np.linspace(-3, 3, 7)
    pds = np.array([float(conditional_pd(pd0, rho, z)) for z in zs])
    assert np.all(np.diff(pds) < 0), "conditional_pd should decrease in z"
    # Bei z=0 muss conditional_pd ≥ pd0 sein? Nein — bei z=0 ist es genau N(N⁻¹(PD)/√(1-ρ)) > PD weil √(1-ρ) < 1
    # OK das stimmt: bei z=0 wird PD verstärkt durch den ρ-Faktor.
    # Stattdessen: bei z=0 muss die unbedingte PD via Integral wieder pd0 ergeben — das testen wir hier nicht.


def _test_unconditional_via_integral():
    """E_M[PD(M)] muss = unbedingte PD ergeben (Konsistenz-Check)."""
    pd0 = 0.05
    rho = float(rho_corporate(pd0))
    # Numerische Integration über M ~ N(0,1)
    zs = np.linspace(-6, 6, 2001)
    weights = norm.pdf(zs) * (zs[1] - zs[0])
    weights /= weights.sum()
    cond_pds = np.array([float(conditional_pd(pd0, rho, z)) for z in zs])
    expected_pd = float(np.sum(cond_pds * weights))
    assert abs(expected_pd - pd0) < 1e-3, \
        f"E[PD(M)] = {expected_pd:.6f}, expected {pd0:.6f}"


def _test_portfolio_aggregation():
    """Sanity: Portfolio mit 3 Segmenten — KPI-Summen stimmen."""
    p = BankPortfolio("TestBank")
    p.add(
        PortfolioSegment("Large Corp", "corporate", ead=100e9, pd=0.012, lgd=0.45),
        PortfolioSegment("SME",        "sme_corporate", ead=40e9, pd=0.025,
                         lgd=0.50, sales_m_eur=20.0),
        PortfolioSegment("Retail Mtg", "mortgage", ead=80e9, pd=0.008, lgd=0.20),
    )
    df = p.baseline_metrics()
    assert len(df) == 3
    kpi = p.portfolio_kpis()
    assert abs(kpi["total_ead"] - 220e9) < 1.0
    assert kpi["el_eur"] > 0
    assert kpi["ul_eur"] > 0
    assert kpi["rwa"] > kpi["ul_eur"], "RWA = K·12.5·EAD must exceed K·EAD"


def _test_stress_increases_pd():
    """Adverse Stress (z < 0) erhöht PD, EL und RWA gegenüber Baseline."""
    p = BankPortfolio("TestBank").add(
        PortfolioSegment("Corp", "corporate", ead=100e9, pd=0.015, lgd=0.45)
    )
    base = p.portfolio_kpis()
    stress = p.stressed_kpis(z_factor=-2.0)
    assert stress["el_eur"] > base["el_eur"], "Stress must increase EL"
    assert stress["rwa"] > base["rwa"], "Stress must increase RWA"
    assert stress["delta_el_eur"] > 0


def _test_downturn_lgd():
    """Downturn-LGD: linear in |M|, no effect for benign M, capped at 1."""
    # Benign / no stress
    assert downturn_lgd(0.45, m_factor=0.0) == 0.45
    assert downturn_lgd(0.45, m_factor=+1.0) == 0.45
    # Adverse stress
    assert abs(downturn_lgd(0.45, m_factor=-1.0, kappa=0.3) - 0.45*1.30) < 1e-9
    assert abs(downturn_lgd(0.45, m_factor=-2.5, kappa=0.3) - 0.45*1.75) < 1e-9
    # Cap at 1.0
    assert downturn_lgd(0.95, m_factor=-3.0, kappa=0.5) == 1.0
    # Linearity in |M|
    a = downturn_lgd(0.45, m_factor=-1.0, kappa=0.3)
    b = downturn_lgd(0.45, m_factor=-2.0, kappa=0.3)
    assert abs((b - 0.45) - 2 * (a - 0.45)) < 1e-9, "Should be linear in |M|"


def _test_capital_bridge_additivity():
    """Capital Bridge: ΔK_PD + ΔK_LGD = ΔK_total exakt (sequenzielle Aktiv.)."""
    p = BankPortfolio("TestBank").add(
        PortfolioSegment("Corp",  "corporate",  ead=100e9, pd=0.015, lgd=0.45),
        PortfolioSegment("Mortg", "mortgage",   ead=80e9,  pd=0.008, lgd=0.20),
        PortfolioSegment("QRRE",  "qrre",       ead=20e9,  pd=0.030, lgd=0.65),
    )
    bridge = p.capital_bridge(z_factor=-2.0, kappa_lgd=0.3)
    # Additivity check
    sum_decomp = bridge["delta_K_pd"] + bridge["delta_K_lgd"]
    assert abs(sum_decomp - bridge["delta_K"]) < 1.0, \
        f"Bridge not additive: {sum_decomp:.2f} vs {bridge['delta_K']:.2f}"
    # Same for RWA, EL
    assert abs((bridge["delta_rwa_pd"] + bridge["delta_rwa_lgd"]) - bridge["delta_rwa"]) < 1.0
    assert abs((bridge["delta_el_pd"]  + bridge["delta_el_lgd"])  - bridge["delta_el"])  < 1.0
    # Both contributions positive under adverse stress
    assert bridge["delta_K_pd"] > 0,  "PD shift should raise K"
    assert bridge["delta_K_lgd"] > 0, "LGD shift should raise K"


def _test_capital_bridge_zero_at_baseline():
    """Bei z_factor=0 sollte ΔK ≈ 0 sein (downturn-LGD nicht aktiv, conditional PD aber > base)."""
    p = BankPortfolio("TestBank").add(
        PortfolioSegment("Corp", "corporate", ead=100e9, pd=0.015, lgd=0.45),
    )
    bridge = p.capital_bridge(z_factor=0.0, kappa_lgd=0.3)
    # At z=0: LGD unchanged, but conditional_pd(PD, ρ, 0) > PD because √(1−ρ)<1
    # So ΔK_LGD must be exactly 0 (no LGD movement at z=0)
    assert abs(bridge["delta_K_lgd"]) < 1e-6, \
        f"LGD-channel must be silent at z=0, got {bridge['delta_K_lgd']}"


if __name__ == "__main__":
    import sys
    # Force UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("vasicek.py - ASRF / Basel III IRB - Validation")
    print("=" * 60)
    tests = [
        ("Monotonicity of rho_corporate(PD)", _test_rho_monotonicity),
        ("Basel IRB reference (PD=1%, LGD=45%, M=2.5y)", _test_basel_irb_reference),
        ("Conditional PD monotone in z", _test_conditional_pd_monotonic_in_z),
        ("E_M[PD(M)] = unconditional PD", _test_unconditional_via_integral),
        ("Portfolio aggregation sanity", _test_portfolio_aggregation),
        ("Adverse stress increases EL & RWA", _test_stress_increases_pd),
        ("Downturn-LGD function", _test_downturn_lgd),
        ("Capital Bridge additivity", _test_capital_bridge_additivity),
        ("Capital Bridge silent at z=0 LGD-channel", _test_capital_bridge_zero_at_baseline),
    ]
    for label, fn in tests:
        try:
            fn()
            print(f"  [PASS]  {label}")
        except AssertionError as e:
            print(f"  [FAIL]  {label}: {e}")
            raise

    print("\n--- Demo: EU Reference Bank Portfolio (Baseline + Stress) ---\n")
    p = BankPortfolio("EU Reference Bank").add(
        PortfolioSegment("Large Corp",     "corporate",     ead=180e9, pd=0.012, lgd=0.45, maturity_years=3.0),
        PortfolioSegment("SME Corp",       "sme_corporate", ead=70e9,  pd=0.025, lgd=0.50, maturity_years=2.5, sales_m_eur=20.0),
        PortfolioSegment("Sovereign",      "sovereign",     ead=120e9, pd=0.001, lgd=0.45, maturity_years=5.0),
        PortfolioSegment("Retail Mtg",     "mortgage",      ead=200e9, pd=0.008, lgd=0.20),
        PortfolioSegment("QRRE",           "qrre",          ead=30e9,  pd=0.030, lgd=0.65),
        PortfolioSegment("Other Retail",   "other_retail",  ead=40e9,  pd=0.020, lgd=0.45),
    )
    base_df = p.baseline_metrics()
    print(base_df[["name", "exposure_class", "ead", "pd", "rho",
                   "el_eur", "ul_eur", "rwa"]].to_string(index=False))
    print()
    base_kpi = p.portfolio_kpis()
    print(f"  Total EAD:     {base_kpi['total_ead']/1e9:>8.1f} bn EUR")
    print(f"  EL:            {base_kpi['el_eur']/1e9:>8.2f} bn EUR  ({base_kpi['el_pct']*100:.3f}%)")
    print(f"  UL (capital):  {base_kpi['ul_eur']/1e9:>8.2f} bn EUR  ({base_kpi['ul_pct']*100:.3f}%)")
    print(f"  RWA:           {base_kpi['rwa']/1e9:>8.1f} bn EUR  (density {base_kpi['rwa_density']*100:.1f}%)")
    print(f"  Loss @ p99.9:  {base_kpi['loss_p999']/1e9:>8.2f} bn EUR")

    print("\n--- Stress: z = -2.5 (approx EBA 2025 adverse) ---\n")
    stress_df = p.stressed_metrics(z_factor=-2.5)
    print(stress_df[["name", "pd_baseline", "pd_stressed",
                     "el_eur", "rwa"]].to_string(index=False))
    s_kpi = p.stressed_kpis(z_factor=-2.5)
    print(f"\n  Delta EL:   {s_kpi['delta_el_eur']/1e9:+.2f} bn EUR")
    print(f"  Delta RWA:  {s_kpi['delta_rwa']/1e9:+.1f} bn EUR")

    print("\n[PASS] All Vasicek tests passed.")
