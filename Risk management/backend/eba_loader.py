"""
============================================================================
 eba_loader.py · EBA Transparency & Stress-Test Data Ingestion
============================================================================

Ladelogik für die zwei zentralen EBA-Datenquellen:

  1. EBA EU-Wide Transparency Exercise (jährlich, ~70 Banken)
     URL-Muster: eba.europa.eu/risk-and-data-analysis/eu-wide-transparency-exercise
     Inhalt: PD, LGD, EAD, RWA pro Bank x Country x Exposure Class

  2. EBA EU-Wide Stress Test (alle 2 Jahre, letzte: 2025, davor 2023)
     URL-Muster: eba.europa.eu/risk-and-data-analysis/stress-tests
     Inhalt: Baseline + Adverse Szenario, projizierte PDs/LGDs/Capital

----------------------------------------------------------------------------
 SCAFFOLD-MODUS
----------------------------------------------------------------------------
 Dieses Modul liefert eine **funktionsfähige Synthetik-Replik** der EBA-
 Daten, basierend auf öffentlichen Stress-Test-Ergebnissen 2023/2025 plus
 reasonable Defaults für Konsistenz-Checks. Sobald die echten EBA-Files
 lokal in `data/eba/` liegen, schaltet `from_real_files()` auf echte
 Daten um. Die API zu Streamlit / `vasicek.py` bleibt identisch.

 Synthetische Anker (gerundete Public-Disclosures aus EBA 2025):
   - Top-10-EU-Banken nach RWA
   - Sechs Exposure-Classes pro Bank (Corporate, SME-Corp, Sovereign,
     Mortgage, QRRE, Other Retail)
   - Baseline-PDs aus jüngsten Transparency-Releases
   - Adverse-Stress-Anker aus EBA 2025 (Brent-Pfad, 10y-Yield-Pfad,
     PD-Migration)
============================================================================
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Project root (directory above `backend/`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vasicek import BankPortfolio, PortfolioSegment   # noqa: E402


# ============================================================================
# 0a. Derived-cache fallback
# ============================================================================
# Die EBA-Rohdateien (tr_cre.csv, tr_sov.csv, tr_oth.csv, TR_Metadata.xlsx)
# sind zu groß für Git und fehlen daher in der Cloud-Deployment-Umgebung.
# `backend/precompute_eba_cache.py` schreibt einmalig die gefilterten/
# geparsten Ergebnisse als kleine Parquet-Dateien nach `data/derived/`.
# Fehlt eine Rohdatei, greifen die Loader transparent auf diese committeten
# Parquets zurück — die öffentliche API bleibt identisch.
_DERIVED_DIR = _ROOT / "data" / "derived"


def _derived(name: str) -> Path:
    return _DERIVED_DIR / name


# ============================================================================
# 0. EBA Schema Constants
# ============================================================================

# IRB Item codes (from SDD.xlsx, Template = Credit Risk_IRB_a/b)
ITEM_ORIGINAL_EXPOSURE       = 2520502   # Original Exposure (SA_and_IRB)
ITEM_DEFAULTED_EXPOSURE      = 2520512   # Original Exposure of which DEFAULTED
ITEM_EXPOSURE_VALUE          = 2520522   # Exposure value (SA_and_IRB) ≈ EAD
ITEM_RWA                     = 2520532   # Risk Exposure Amount (RWA)
ITEM_VALUE_ADJUSTMENTS       = 2520552   # Value adjustments and provisions

# Portfolio dimension codes (TR_Metadata, sheet "Portfolio")
PORTFOLIO_TOTAL = 0
PORTFOLIO_SA    = 1
PORTFOLIO_IRB   = 2
PORTFOLIO_FIRB  = 3
PORTFOLIO_AIRB  = 4

# EBA Exposure code → Vasicek exposure class mapping.
# Code references TR_Metadata "Exposure" sheet. Codes outside this map are
# treated as ancillary and excluded from the Vasicek portfolio (e.g. 601
# Exposures-in-default flag, 604-607 securitisations / equity).
EXPOSURE_TO_VASICEK_CLASS: dict[int, str] = {
    # Sovereign-like
    103: "sovereign",     # Central governments / central banks
    104: "sovereign",     # Regional governments / local authorities
    105: "sovereign",     # Public sector entities
    106: "sovereign",     # Multilateral Development Banks
    107: "sovereign",     # International Organisations
    # Banks
    203: "bank",          # Institutions
    204: "bank",          # Institutions w/o short-term credit assessment
    603: "bank",          # Covered bonds (treated as bank exposure)
    # Corporates
    303: "corporate",     # Corporates
    302: "sme_corporate", # Corporates - SME
    311: "sme_corporate", # Non-financial - SME
    304: "corporate",     # Corporates - Specialised Lending
    305: "corporate",     # Corporates other than specialised lending
    307: "corporate",     # Institutions+Corporates with short-term assessment
    202: "corporate",     # Financial corporations other than CIs
    301: "corporate",     # Non-financial corporations
    # Mortgages
    402: "mortgage",      # Real estate. Residential
    406: "mortgage",      # Retail – Secured by real estate
    407: "mortgage",      # Retail – Secured by RE - SME
    408: "mortgage",      # Retail – Secured by RE - non SME
    501: "mortgage",      # Secured by mortgages on immovable property
    502: "mortgage",      # Secured by mortgages - SME
    312: "mortgage",      # Non-financial - Collateralised by commercial RE
    308: "mortgage",      # Corporates - Real estate. Commercial
    431: "mortgage",      # Households of which: Collat. by residential RE
    # Retail revolving (QRRE)
    409: "qrre",          # Retail – Qualifying Revolving
    # Other retail
    404: "other_retail",  # Retail (default bucket)
    405: "other_retail",  # Retail - SME
    410: "other_retail",  # Retail – Other Retail
    411: "other_retail",  # Retail – Other Retail - SME
    412: "other_retail",  # Retail – Other Retail - non SME
    401: "other_retail",  # Households (default bucket)
    403: "other_retail",  # Credit for consumption
    432: "other_retail",  # Households - of which Credit for consumption
}

# Standard LGD assumptions per Vasicek class (Basel F-IRB defaults where
# applicable, else regulator-cited proxies). Banks using A-IRB report
# their own LGDs but those are not in the Transparency disclosure, so we
# fall back to these published assumptions. Documented in MODEL_ASSUMPTIONS.md §11.
LGD_BY_VASICEK_CLASS: dict[str, float] = {
    "sovereign":     0.45,    # Basel F-IRB senior unsecured
    "bank":          0.45,    # Basel F-IRB senior unsecured
    "corporate":     0.45,    # Basel F-IRB senior unsecured
    "sme_corporate": 0.45,    # Basel F-IRB senior unsecured
    "mortgage":      0.20,    # Residential mortgage LGD floor (Basel)
    "qrre":          0.65,    # Unsecured revolving — published EBA/regulator range
    "other_retail":  0.45,    # Other retail unsecured
}

# Standard maturity (years) — Basel F-IRB cap [1.0, 5.0], standard 2.5
MATURITY_BY_VASICEK_CLASS: dict[str, float] = {
    "sovereign":     5.0,
    "bank":          3.0,
    "corporate":     2.5,
    "sme_corporate": 2.5,
    "mortgage":      2.5,    # MA not applied for mortgage
    "qrre":          2.5,    # MA not applied for retail
    "other_retail":  2.5,    # MA not applied for retail
}


# ============================================================================
# 0b. Sovereign Schema (tr_sov.csv)
# ============================================================================
# Items from SDD.xlsx, Template = "Sovereign"
ITEM_SOV_GROSS_ON_BS    = 2520810   # Gross carrying amount, on-balance
ITEM_SOV_NET_ON_BS      = 2520811   # Net carrying amount, on-balance
ITEM_SOV_HFT            = 2520812   # of which: Held for Trading (P&L)
ITEM_SOV_FVTPL          = 2520813   # of which: Designated FVTPL (P&L)
ITEM_SOV_FVOCI          = 2520814   # of which: FVOCI / AfS (OCI)
ITEM_SOV_AC             = 2520815   # of which: Amortised Cost (no MtM)
ITEM_SOV_RWA            = 2520822   # RWA on sovereign exposures

# Sovereign accounting-item to CET1-transmission channel.
#   "P&L"  → ΔFV durchläuft P&L → CET1 retained earnings
#   "OCI"  → ΔFV via Other Comprehensive Income → CET1 (FVOCI/AfS)
#   "none" → bei Buchwert (HtM/AC), kein direkter CET1-Effekt unter Stress
SOV_ACCOUNTING_ITEMS = {
    ITEM_SOV_HFT:   {"label": "HfT",   "channel": "P&L"},
    ITEM_SOV_FVTPL: {"label": "FVTPL", "channel": "P&L"},
    ITEM_SOV_FVOCI: {"label": "FVOCI", "channel": "OCI"},
    ITEM_SOV_AC:    {"label": "AC",    "channel": "none"},
}

# Maturity dimension (TR_Metadata sheet "Maturity")
MATURITY_BUCKETS: dict[int, str] = {
    1: "[ 0 - 3M [",
    2: "[ 3M - 1Y [",
    3: "[ 1Y - 2Y [",
    4: "[ 2Y - 3Y [",
    5: "[ 3Y - 5Y [",
    6: "[ 5Y - 10Y [",
    7: "[ 10Y - more",
    8: "Total",
}

# Approximate modified duration per maturity bucket (years).
# Uses bucket midpoint as a first-cut approximation. For >10Y bucket we
# assume an effective ~15Y duration (typical sovereign 30Y bond at par
# gives ~17, but the EBA bucket aggregates everything 10Y+).
DURATION_BY_BUCKET: dict[int, float] = {
    1: 0.125,   # < 3M  → midpoint ~6 weeks
    2: 0.625,   # 3M–1Y → midpoint ~7.5 months
    3: 1.5,     # 1–2Y
    4: 2.5,     # 2–3Y
    5: 4.0,     # 3–5Y
    6: 7.5,     # 5–10Y
    7: 15.0,    # 10Y+
}


# ============================================================================
# 1. Synthetic Top-10 EU Bank Anchors
# ============================================================================
# Größenordnungen sind aus EBA-2025-Transparency-Disclosure abgeleitet
# (gerundet, keine Tagesgenauigkeit erforderlich für Stress-Demo).
# EAD in bn EUR, PD/LGD in Dezimal.
SYNTHETIC_BANKS = {
    "BNP Paribas": {
        "country": "FR",
        "lei": "R0MUWSFPU8MPRO8K5P83",
        "rwa_total_bn": 720,
        "segments": [
            ("Large Corp",      "corporate",     480, 0.014, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate", 180, 0.024, 0.50, 2.5, 22.0),
            ("Sovereign",       "sovereign",     250, 0.002, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      390, 0.009, 0.18, 0.0, None),
            ("QRRE",            "qrre",           70, 0.034, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   85, 0.022, 0.45, 0.0, None),
        ],
    },
    "Groupe Crédit Agricole": {
        "country": "FR",
        "lei": "FR969500TJ5KRTCJQWXH",
        "rwa_total_bn": 380,
        "segments": [
            ("Large Corp",      "corporate",     290, 0.013, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate", 160, 0.026, 0.50, 2.5, 18.0),
            ("Sovereign",       "sovereign",     180, 0.002, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      330, 0.009, 0.18, 0.0, None),
            ("QRRE",            "qrre",           45, 0.035, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   70, 0.024, 0.45, 0.0, None),
        ],
    },
    "Banco Santander": {
        "country": "ES",
        "lei": "5493006QMFDDMYWIAM13",
        "rwa_total_bn": 620,
        "segments": [
            ("Large Corp",      "corporate",     310, 0.018, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate", 145, 0.034, 0.50, 2.5, 15.0),
            ("Sovereign",       "sovereign",     130, 0.004, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      350, 0.012, 0.20, 0.0, None),
            ("QRRE",            "qrre",           55, 0.045, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   95, 0.030, 0.45, 0.0, None),
        ],
    },
    "Deutsche Bank": {
        "country": "DE",
        "lei": "7LTWFZYICNSX8D621K86",
        "rwa_total_bn": 360,
        "segments": [
            ("Large Corp",      "corporate",     230, 0.011, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate",  90, 0.022, 0.50, 2.5, 25.0),
            ("Sovereign",       "sovereign",     180, 0.001, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      170, 0.008, 0.18, 0.0, None),
            ("QRRE",            "qrre",           20, 0.030, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   30, 0.020, 0.45, 0.0, None),
        ],
    },
    "ING Groep N.V.": {
        "country": "NL",
        "lei": "549300NYKK9MWM7GGW15",
        "rwa_total_bn": 330,
        "segments": [
            ("Large Corp",      "corporate",     250, 0.012, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate", 100, 0.024, 0.50, 2.5, 20.0),
            ("Sovereign",       "sovereign",     110, 0.002, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      290, 0.007, 0.16, 0.0, None),
            ("QRRE",            "qrre",           25, 0.030, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   30, 0.020, 0.45, 0.0, None),
        ],
    },
    "UniCredit": {
        "country": "IT",
        "lei": "549300TRUWO2CD2G5692",
        "rwa_total_bn": 320,
        "segments": [
            ("Large Corp",      "corporate",     220, 0.020, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate", 120, 0.038, 0.50, 2.5, 14.0),
            ("Sovereign",       "sovereign",     180, 0.005, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      200, 0.013, 0.22, 0.0, None),
            ("QRRE",            "qrre",           30, 0.045, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   45, 0.030, 0.45, 0.0, None),
        ],
    },
    "Groupe BPCE": {
        "country": "FR",
        "lei": "FR9695005MSX1OYEMGDF",
        "rwa_total_bn": 290,
        "segments": [
            ("Large Corp",      "corporate",     200, 0.022, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate", 140, 0.040, 0.50, 2.5, 12.0),
            ("Sovereign",       "sovereign",     220, 0.005, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      180, 0.015, 0.22, 0.0, None),
            ("QRRE",            "qrre",           28, 0.045, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   42, 0.032, 0.45, 0.0, None),
        ],
    },
    "Confédération Nationale du Crédit Mutuel": {
        "country": "FR",
        "lei": "9695000CG7B84NLR5984",
        "rwa_total_bn": 360,
        "segments": [
            ("Large Corp",      "corporate",     180, 0.020, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate",  90, 0.038, 0.50, 2.5, 14.0),
            ("Sovereign",       "sovereign",     100, 0.004, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      210, 0.013, 0.20, 0.0, None),
            ("QRRE",            "qrre",           40, 0.044, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   55, 0.028, 0.45, 0.0, None),
        ],
    },
    "Société générale S.A.": {
        "country": "FR",
        "lei": "O2RNE8IBXP4R0TD8PU41",
        "rwa_total_bn": 380,
        "segments": [
            ("Large Corp",      "corporate",     260, 0.015, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate", 110, 0.027, 0.50, 2.5, 19.0),
            ("Sovereign",       "sovereign",     130, 0.002, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",      220, 0.010, 0.18, 0.0, None),
            ("QRRE",            "qrre",           35, 0.036, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   50, 0.024, 0.45, 0.0, None),
        ],
    },
    "Coöperatieve Rabobank": {
        "country": "NL",
        "lei": "DG3RU1DBUFHT4ZF9WN62",
        "rwa_total_bn": 165,
        "segments": [
            ("Large Corp",      "corporate",     115, 0.013, 0.45, 3.0, None),
            ("SME Corp",        "sme_corporate",  80, 0.025, 0.50, 2.5, 20.0),
            ("Sovereign",       "sovereign",      90, 0.001, 0.45, 5.0, None),
            ("Retail Mortgage", "mortgage",       95, 0.008, 0.18, 0.0, None),
            ("QRRE",            "qrre",           12, 0.030, 0.65, 0.0, None),
            ("Other Retail",    "other_retail",   18, 0.022, 0.45, 0.0, None),
        ],
    },
}


# ============================================================================
# 2. EBA 2025 Stress-Test Anker
# ============================================================================
# Aus EBA-Stress-Test-2025-Methodology-Note (Macro-Pfad 2025-2027 adverse):
#   - Brent kumulativ: ~ +60% peak shock (log ≈ +0.47)
#   - 10y EU AAA Yield: +200 bp im adverse-Szenario
#   - GDP-Drop: −6 pp kumuliert
#   - Implied systematic factor M ≈ -2.5σ (worst-case 1% tail)
EBA_2025_ADVERSE_ANCHOR = {
    "vintage":              "2025",
    "publication_date":     "2025-08-01",
    "brent_log_shock":      0.47,    # log-Return über 3y Horizon, peak
    "rate_10y_pp_shock":    2.00,    # +200 bp 10y AAA Yield
    "gdp_pp_shock":        -6.0,     # kumulierter GDP-Verlust pp
    "z_factor_implied":    -2.5,     # implizierter Vasicek-Systemfaktor
    "horizon_years":        3.0,
    "source": "EBA Stress Test 2025 — Methodology Note + Macroeconomic Scenarios",
    "url":    "https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/stress-tests",
    "status": "synthetic anchor — replace with parsed EBA Excel files when available",
}

EBA_2023_ADVERSE_ANCHOR = {
    "vintage":              "2023",
    "publication_date":     "2023-07-28",
    "brent_log_shock":      0.55,
    "rate_10y_pp_shock":    2.50,
    "gdp_pp_shock":        -6.0,
    "z_factor_implied":    -2.7,
    "horizon_years":        3.0,
    "source": "EBA Stress Test 2023 — Macro-Adverse Scenario",
    "url":    "https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/stress-tests",
    "status": "synthetic anchor — replace with parsed EBA Excel files when available",
}


# ============================================================================
# 3. Public API
# ============================================================================
@dataclass
class EbaUniverse:
    """Wrapper für die geladene EBA-Datenwelt."""

    banks: dict[str, BankPortfolio] = field(default_factory=dict)
    adverse_anchor: dict = field(default_factory=dict)
    source: str = "synthetic"

    @property
    def n_banks(self) -> int:
        return len(self.banks)

    @property
    def total_ead_eur(self) -> float:
        return sum(b.total_ead for b in self.banks.values())

    def aggregated_portfolio(self, name: str = "EU Aggregate") -> BankPortfolio:
        """Konsolidiert alle Banken zu einem Aggregat-Portfolio."""
        agg = BankPortfolio(name)
        for b in self.banks.values():
            agg.segments.extend(b.segments)
        return agg

    def summary_table(self, confidence: float = 0.999) -> pd.DataFrame:
        """Bank-by-bank KPI-Tabelle."""
        rows = []
        for name, p in self.banks.items():
            kpi = p.portfolio_kpis(confidence)
            rows.append({
                "Bank":          name,
                "Total EAD bn":  kpi["total_ead"] / 1e9,
                "EL bn":         kpi["el_eur"] / 1e9,
                "UL bn":         kpi["ul_eur"] / 1e9,
                "RWA bn":        kpi["rwa"] / 1e9,
                "RWA density":   kpi["rwa_density"],
                "EL %":          kpi["el_pct"],
            })
        return pd.DataFrame(rows)


def from_synthetic(vintage: str = "2025") -> EbaUniverse:
    """Baut die Synthetic-Replik der EBA-Daten auf.

    Wird nur als Fallback genutzt, wenn weder EBA-Rohdaten noch der
    Derived-Cache vorliegen (im ausgelieferten ZIP feuert er nicht, da der
    Derived-Cache mitgeliefert wird). Die Bank-Identitäten (Name + LEI) sind
    identisch mit der kuratierten Top-10-Welt (`top10_irb_banks.csv`), sodass
    auch der Fallback exakt dieselben 10 Banken adressiert. Die PD/LGD- und
    EAD-Werte bleiben hier bewusst synthetisch (gerundete Größenordnungen);
    den Live-Pfad reichert `filter_universe_to_top10` mit den echten
    Pillar-3-Werten an.
    """
    universe = EbaUniverse(source=f"synthetic ({vintage})")
    for bank_name, spec in SYNTHETIC_BANKS.items():
        portfolio = BankPortfolio(bank_name, lei=spec.get("lei", ""))
        for seg_name, exp_class, ead_bn, pd_val, lgd, mat, sales in spec["segments"]:
            portfolio.add(PortfolioSegment(
                name=seg_name,
                exposure_class=exp_class,
                ead=ead_bn * 1e9,
                pd=pd_val,
                lgd=lgd,
                maturity_years=mat if mat > 0 else 2.5,
                sales_m_eur=sales,
            ))
        universe.banks[bank_name] = portfolio

    universe.adverse_anchor = (
        EBA_2025_ADVERSE_ANCHOR if vintage == "2025" else EBA_2023_ADVERSE_ANCHOR
    )
    return universe


def load_bank_directory(metadata_path: Path) -> pd.DataFrame:
    """Lädt die EBA Bank-Liste (LEI -> Name + Country) aus TR_Metadata.xlsx."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        fb = _derived("bank_dir.parquet")
        if fb.exists():
            return pd.read_parquet(fb)
        raise FileNotFoundError(
            f"TR_Metadata.xlsx nicht gefunden ({metadata_path}) und kein "
            f"Derived-Fallback unter {fb}. `python backend/"
            f"precompute_eba_cache.py` ausführen."
        )
    df = pd.read_excel(metadata_path, sheet_name="List of Institutions",
                       header=1)
    df = df.rename(columns={
        "Country": "country",
        "Desc_country": "country_name",
        "LEI_Code": "lei",
        "Name": "bank_name",
    })
    return df[["lei", "bank_name", "country", "country_name"]].dropna(
        subset=["lei", "bank_name"]
    ).copy()


def parse_credit_risk_csv(
    csv_path: Path,
    *,
    period: int | None = None,
    items: tuple[int, ...] = (
        ITEM_ORIGINAL_EXPOSURE, ITEM_DEFAULTED_EXPOSURE,
        ITEM_EXPOSURE_VALUE, ITEM_RWA, ITEM_VALUE_ADJUSTMENTS,
    ),
    portfolio: int = PORTFOLIO_IRB,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Streamt tr_cre.csv und filtert auf relevante Zeilen.

    Filterlogik:
      - Item ∈ items
      - Portfolio == portfolio (Default: 2 = IRB)
      - Country == 0 (kein Counterparty-Country-Breakdown — Aggregat über alle)
      - Status ∈ {0, 2} (Total-Stock + Defaulted-Subset)
      - Perf_Status == 0
      - optional Period == period

    Returns
    -------
    DataFrame: lei × period × item × exposure × status → amount
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        fb = _derived("cre_raw.parquet")
        if fb.exists():
            df = pd.read_parquet(fb)
            if period is not None:
                df = df[df["Period"] == period]
            return df.reset_index(drop=True)
        raise FileNotFoundError(
            f"tr_cre.csv nicht gefunden ({csv_path}) und kein Derived-"
            f"Fallback unter {fb}. `python backend/precompute_eba_cache.py` "
            f"ausführen."
        )
    keep_items = set(items)
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        sub = chunk[
            chunk["Item"].isin(keep_items)
            & (chunk["Portfolio"] == portfolio)
            & (chunk["Country"] == 0)
            & chunk["Status"].isin([0, 2])
            & (chunk["Perf_Status"] == 0)
        ]
        if period is not None:
            sub = sub[sub["Period"] == period]
        if len(sub):
            chunks.append(sub[[
                "LEI_Code", "Period", "Item", "Exposure", "Status", "Amount",
            ]])
    if not chunks:
        return pd.DataFrame(columns=[
            "LEI_Code", "Period", "Item", "Exposure", "Status", "Amount",
        ])
    return pd.concat(chunks, ignore_index=True)


def latest_period(df: pd.DataFrame) -> int:
    return int(df["Period"].max())


def aggregate_to_vasicek_segments(
    cre_df: pd.DataFrame,
    *,
    period: int,
) -> pd.DataFrame:
    """Aggregiert die Long-Form-Tabelle pro Bank × Vasicek-Class.

    Returns
    -------
    DataFrame mit Spalten:
        lei, vasicek_class, ead_eur, original_exposure_eur, rwa_eur,
        defaulted_eur, value_adj_eur, implied_pd
    Werte in EUR (Input ist m EUR → ×1e6).
    """
    df = cre_df[cre_df["Period"] == period].copy()
    df["v_class"] = df["Exposure"].map(EXPOSURE_TO_VASICEK_CLASS)
    df = df.dropna(subset=["v_class"])

    # EAD = Item 2520522 (Exposure Value), Status=0 (total)
    ead = (df[(df["Item"] == ITEM_EXPOSURE_VALUE) & (df["Status"] == 0)]
           .groupby(["LEI_Code", "v_class"], as_index=False)["Amount"].sum()
           .rename(columns={"Amount": "ead_m_eur"}))

    # Original Exposure = Item 2520502, Status=0 (total)
    oe = (df[(df["Item"] == ITEM_ORIGINAL_EXPOSURE) & (df["Status"] == 0)]
          .groupby(["LEI_Code", "v_class"], as_index=False)["Amount"].sum()
          .rename(columns={"Amount": "oe_m_eur"}))

    # Defaulted = Item 2520512 reported under Status=2 ("Defaulted assets").
    # NOT Status=0 — defaulted-of-which is only published under Status=2.
    defaulted = (df[(df["Item"] == ITEM_DEFAULTED_EXPOSURE) & (df["Status"] == 2)]
                 .groupby(["LEI_Code", "v_class"], as_index=False)["Amount"].sum()
                 .rename(columns={"Amount": "defaulted_m_eur"}))

    # RWA = Item 2520532, Status=0
    rwa = (df[(df["Item"] == ITEM_RWA) & (df["Status"] == 0)]
           .groupby(["LEI_Code", "v_class"], as_index=False)["Amount"].sum()
           .rename(columns={"Amount": "rwa_m_eur"}))

    # Provisions = Item 2520552
    prov = (df[(df["Item"] == ITEM_VALUE_ADJUSTMENTS) & (df["Status"] == 0)]
            .groupby(["LEI_Code", "v_class"], as_index=False)["Amount"].sum()
            .rename(columns={"Amount": "value_adj_m_eur"}))

    # Outer-merge so we keep classes even if some metrics are missing
    out = ead.merge(oe, on=["LEI_Code", "v_class"], how="outer")
    out = out.merge(defaulted, on=["LEI_Code", "v_class"], how="outer")
    out = out.merge(rwa, on=["LEI_Code", "v_class"], how="outer")
    out = out.merge(prov, on=["LEI_Code", "v_class"], how="outer")
    out = out.fillna(0.0)

    # Implied PD = defaulted / original exposure (observed default ratio).
    # Floor at 3 bp (Basel sovereign floor) to keep Vasicek formulas stable;
    # cap at 50% to keep Basel formulas in well-tested ranges.
    import numpy as np
    safe_oe = out["oe_m_eur"].where(out["oe_m_eur"] > 0, np.nan)
    raw_pd = out["defaulted_m_eur"] / safe_oe
    out["implied_pd"] = raw_pd.fillna(0.0).clip(lower=3e-4, upper=0.50)

    # Convert to EUR
    for col in ("ead_m_eur", "oe_m_eur", "defaulted_m_eur", "rwa_m_eur",
                "value_adj_m_eur"):
        out[col.replace("_m_eur", "_eur")] = out[col] * 1e6

    return out[["LEI_Code", "v_class",
                "ead_eur", "oe_eur", "defaulted_eur", "rwa_eur",
                "value_adj_eur", "implied_pd"]]


# ============================================================================
# 4b. Banking-Book Bond-Class Breakdown (for the Bonds page)
# ============================================================================
# Map an EBA Exposure-Code to a "Bond category" suitable for the Bonds tab.
# These categories overlap with the Vasicek classes but slice differently —
# they keep Covered Bonds (603) separate from regular Banks (203/204), and
# they group Corporate codes that are otherwise split into "corporate" and
# "sme_corporate" in the Vasicek mapping.
EXPOSURE_TO_BOND_CATEGORY: dict[int, str] = {
    # Financials = banks + institutions (excludes covered bonds)
    202: "Financials", 203: "Financials", 204: "Financials",
    # Corporates (incl. SME, Specialised Lending, etc.)
    301: "Corporates", 302: "Corporates", 303: "Corporates", 304: "Corporates",
    305: "Corporates", 306: "Corporates", 307: "Corporates", 311: "Corporates",
    312: "Corporates", 308: "Corporates",
    # Covered Bonds (kept separate)
    603: "Covered Bonds",
}


def loan_book_class_breakdown(
    cre_df: pd.DataFrame, *, period: int,
) -> pd.DataFrame:
    """Banking-Book Exposures aufgeschlüsselt nach Bond-Kategorie pro Bank.

    Aggregiert Item 2520522 (Exposure Value, post-CCF) aus Items des Loan-
    Book-IRB pro (Bank × Bond-Category × {Financials, Corporates, Covered}).
    Das ist die Datenbasis für den "Banking Book Bonds"-Sub-Tab.

    Wichtig: Die EBA-Disclosure unterscheidet **nicht** zwischen Bonds und
    Loans innerhalb einer Exposure-Class. Das Aggregat enthält beides.
    Disclaimer im Frontend.

    Filter:
      - Item = 2520522 (Exposure Value)
      - Portfolio = 2 (IRB)
      - Country = 0 (Aggregat)
      - Status = 0, Perf_Status = 0
      - Period = period
      - Exposure-Code in EXPOSURE_TO_BOND_CATEGORY

    Returns
    -------
    DataFrame [LEI_Code, bond_category, ead_eur, rwa_eur, defaulted_eur,
               implied_pd]
    """
    df = cre_df[
        (cre_df["Period"] == period)
        & cre_df["Exposure"].isin(EXPOSURE_TO_BOND_CATEGORY.keys())
    ].copy()
    if df.empty:
        return pd.DataFrame()
    df["bond_category"] = df["Exposure"].map(EXPOSURE_TO_BOND_CATEGORY)

    # EAD (Item 2520522, Status=0)
    ead = (df[(df["Item"] == ITEM_EXPOSURE_VALUE) & (df["Status"] == 0)]
           .groupby(["LEI_Code", "bond_category"], as_index=False)
           ["Amount"].sum().rename(columns={"Amount": "ead_m_eur"}))
    # Original Exposure (for default ratio denominator)
    oe = (df[(df["Item"] == ITEM_ORIGINAL_EXPOSURE) & (df["Status"] == 0)]
          .groupby(["LEI_Code", "bond_category"], as_index=False)
          ["Amount"].sum().rename(columns={"Amount": "oe_m_eur"}))
    # Defaulted (Status=2)
    deflt = (df[(df["Item"] == ITEM_DEFAULTED_EXPOSURE) & (df["Status"] == 2)]
             .groupby(["LEI_Code", "bond_category"], as_index=False)
             ["Amount"].sum().rename(columns={"Amount": "defaulted_m_eur"}))
    # RWA (Item 2520532, Status=0)
    rwa = (df[(df["Item"] == ITEM_RWA) & (df["Status"] == 0)]
           .groupby(["LEI_Code", "bond_category"], as_index=False)
           ["Amount"].sum().rename(columns={"Amount": "rwa_m_eur"}))

    out = ead.merge(oe, on=["LEI_Code", "bond_category"], how="outer")
    out = out.merge(deflt, on=["LEI_Code", "bond_category"], how="outer")
    out = out.merge(rwa, on=["LEI_Code", "bond_category"], how="outer")
    out = out.fillna(0.0)

    # Implied PD = defaulted / original_exposure (gleiche Konvention wie
    # aggregate_to_vasicek_segments). Floor 3 bp, Cap 50%.
    safe_oe = out["oe_m_eur"].where(out["oe_m_eur"] > 0, np.nan)
    raw_pd = out["defaulted_m_eur"] / safe_oe
    out["implied_pd"] = raw_pd.fillna(0.0).clip(lower=3e-4, upper=0.50)

    # Convert to EUR
    for c in ("ead_m_eur", "oe_m_eur", "defaulted_m_eur", "rwa_m_eur"):
        out[c.replace("_m_eur", "_eur")] = out[c] * 1e6

    return out[["LEI_Code", "bond_category",
                "ead_eur", "oe_eur", "defaulted_eur", "rwa_eur",
                "implied_pd"]]


def build_bank_portfolios(
    seg_df: pd.DataFrame, bank_dir: pd.DataFrame,
) -> dict[str, BankPortfolio]:
    """Konvertiert Aggregat-DataFrame in Vasicek-BankPortfolio-Objekte."""
    lei_to_name = dict(zip(bank_dir["lei"], bank_dir["bank_name"]))
    out: dict[str, BankPortfolio] = {}

    for lei, group in seg_df.groupby("LEI_Code"):
        bank_name = lei_to_name.get(lei, f"LEI {lei[:8]}…")
        portfolio = BankPortfolio(bank_name, lei=str(lei))
        for _, row in group.iterrows():
            v_class = row["v_class"]
            if row["ead_eur"] <= 0:
                continue
            portfolio.add(PortfolioSegment(
                name=v_class.replace("_", " ").title(),
                exposure_class=v_class,
                ead=float(row["ead_eur"]),
                pd=float(row["implied_pd"]),
                lgd=LGD_BY_VASICEK_CLASS[v_class],
                maturity_years=MATURITY_BY_VASICEK_CLASS[v_class],
                sales_m_eur=20.0 if v_class == "sme_corporate" else None,
            ))
        if portfolio.segments:
            out[bank_name] = portfolio
    return out


def from_real_files(
    eba_dir: Path | str | None = None,
    *,
    vintage: str = "2025",
    period: int | None = None,
    top_n: int | None = None,
) -> EbaUniverse:
    """Lädt das echte EBA-Universum aus `eba_dir/tr_cre.csv` + Metadata.

    Parameters
    ----------
    eba_dir : Verzeichnis mit tr_cre.csv + TR_Metadata.xlsx; default = config.EBA_RAW_DIR
    vintage : nur für Adverse-Anker-Auswahl ("2025" | "2023")
    period : EBA-Period-Filter (YYYYMM int). None = jüngster verfügbarer Stichtag.
    top_n : optional Filter auf die Top-N-Banken nach EAD.
    """
    if eba_dir is None:
        from config import EBA_RAW_DIR
        eba_dir = EBA_RAW_DIR
    eba_dir = Path(eba_dir)

    cre_path = eba_dir / "tr_cre.csv"
    meta_path = eba_dir / "TR_Metadata.xlsx"
    # Akzeptiere entweder die Rohdateien ODER den committeten Derived-Cache
    # (Cloud-Deployment ohne die großen Rohdaten). load_bank_directory /
    # parse_credit_risk_csv fallen intern auf data/derived/ zurück.
    have_raw = cre_path.exists() and meta_path.exists()
    have_derived = (_derived("cre_raw.parquet").exists()
                    and _derived("bank_dir.parquet").exists())
    if not have_raw and not have_derived:
        raise FileNotFoundError(
            f"Weder EBA-Rohdateien in {eba_dir} noch Derived-Cache in "
            f"{_DERIVED_DIR}. `python backend/precompute_eba_cache.py` "
            f"ausführen oder Rohdaten bereitstellen."
        )

    bank_dir = load_bank_directory(meta_path)
    cre_df = parse_credit_risk_csv(cre_path, period=period,
                                   portfolio=PORTFOLIO_IRB)
    if cre_df.empty:
        raise RuntimeError(
            f"No matching IRB rows in {cre_path}. Check period filter or "
            f"item codes."
        )

    use_period = period if period is not None else latest_period(cre_df)
    seg_df = aggregate_to_vasicek_segments(cre_df, period=use_period)
    portfolios = build_bank_portfolios(seg_df, bank_dir)

    # Optional Top-N Filter
    if top_n is not None and top_n < len(portfolios):
        ranked = sorted(portfolios.items(), key=lambda kv: -kv[1].total_ead)
        portfolios = dict(ranked[:top_n])

    universe = EbaUniverse(
        banks=portfolios,
        adverse_anchor=(EBA_2025_ADVERSE_ANCHOR if vintage == "2025"
                        else EBA_2023_ADVERSE_ANCHOR),
        source=f"EBA Transparency {vintage} (period {use_period}, IRB)",
    )
    return universe


# ============================================================================
# 5. Sovereign Loader (tr_sov.csv)
# ============================================================================
def load_country_dim(metadata_path: Path) -> pd.DataFrame:
    """Lädt Country-Dim mit Code, Name, ISO-Code."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        fb = _derived("country_dim.parquet")
        if fb.exists():
            return pd.read_parquet(fb)
        raise FileNotFoundError(
            f"TR_Metadata.xlsx nicht gefunden ({metadata_path}) und kein "
            f"Derived-Fallback unter {fb}."
        )
    df = pd.read_excel(metadata_path, sheet_name="Country", header=1)
    df.columns = ["code", "country_name", "iso"]
    df = df.dropna(subset=["code"])
    df["code"] = df["code"].astype(int)
    return df


def parse_sovereign_csv(
    csv_path: Path,
    *,
    period: int | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Streamt tr_sov.csv und filtert auf relevante Items.

    Items:
      2520810 — On-balance gross carrying amount (primary exposure measure)
      2520822 — RWA on sovereign exposures
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        fb = _derived("sov_raw.parquet")
        if fb.exists():
            df = pd.read_parquet(fb)
            if period is not None:
                df = df[df["Period"] == period]
            return df.reset_index(drop=True)
        raise FileNotFoundError(
            f"tr_sov.csv nicht gefunden ({csv_path}) und kein Derived-"
            f"Fallback unter {fb}. `python backend/precompute_eba_cache.py` "
            f"ausführen."
        )
    keep_items = {
        ITEM_SOV_GROSS_ON_BS, ITEM_SOV_RWA,
        ITEM_SOV_HFT, ITEM_SOV_FVTPL, ITEM_SOV_FVOCI, ITEM_SOV_AC,
    }
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        sub = chunk[chunk["Item"].isin(keep_items)]
        if period is not None:
            sub = sub[sub["Period"] == period]
        if len(sub):
            chunks.append(sub[[
                "LEI_Code", "Period", "Item", "Country",
                "Accounting_portfolio", "Maturity", "Amount",
            ]])
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def sovereign_concentration(
    sov_df: pd.DataFrame, period: int,
) -> pd.DataFrame:
    """Bank × Counterparty-Country exposure (gross, all maturities, all books).

    Filter:
      Item = 2520810 (on-balance gross)
      Maturity = 8 (Total over all maturities)
      Accounting_portfolio = 0 (no breakdown)
      Country > 0 (specific countries, not Total)

    Returns: DataFrame with columns [LEI_Code, Country, exposure_eur].
    """
    df = sov_df[
        (sov_df["Item"] == ITEM_SOV_GROSS_ON_BS)
        & (sov_df["Period"] == period)
        & (sov_df["Maturity"] == 8)
        & (sov_df["Accounting_portfolio"] == 0)
        & (sov_df["Country"] > 0)
    ].copy()
    out = (df.groupby(["LEI_Code", "Country"], as_index=False)["Amount"].sum()
             .rename(columns={"Amount": "exposure_m_eur"}))
    out["exposure_eur"] = out["exposure_m_eur"] * 1e6
    return out[["LEI_Code", "Country", "exposure_eur"]]


def sovereign_maturity_ladder(
    sov_df: pd.DataFrame, period: int,
) -> pd.DataFrame:
    """Bank × Maturity-Bucket exposure (gross, summed over all counterparty
    countries and all accounting portfolios).

    Filter:
      Item = 2520810
      Country > 0 (specific counterparty countries — Country=0 not reported)
      Accounting_portfolio = 0 (total over books — only level reported)
      Maturity ∈ [1, 7] (specific buckets, not Total=8)

    Aggregation: Σ_country Amount → per-bank maturity ladder.

    Returns: DataFrame with columns [LEI_Code, Maturity, exposure_eur,
                                     duration_years, label].
    """
    df = sov_df[
        (sov_df["Item"] == ITEM_SOV_GROSS_ON_BS)
        & (sov_df["Period"] == period)
        & (sov_df["Country"] > 0)
        & (sov_df["Accounting_portfolio"] == 0)
        & (sov_df["Maturity"].between(1, 7))
    ].copy()
    out = (df.groupby(["LEI_Code", "Maturity"], as_index=False)["Amount"].sum()
             .rename(columns={"Amount": "exposure_m_eur"}))
    out["exposure_eur"] = out["exposure_m_eur"] * 1e6
    out["duration_years"] = out["Maturity"].map(DURATION_BY_BUCKET)
    out["label"] = out["Maturity"].map(MATURITY_BUCKETS)
    return out[["LEI_Code", "Maturity", "label", "exposure_eur",
                "duration_years"]]


# ============================================================================
# 5b1. Sovereign Accounting-Portfolio Split (Items 2520812-2520815)
# ============================================================================
def sovereign_by_accounting_class(
    sov_df: pd.DataFrame, period: int,
) -> pd.DataFrame:
    """Sovereign-Bestände pro Bank × Accounting-Class × Maturity-Bucket.

    Unterscheidet die vier IFRS-9-Accounting-Categorien (HfT, FVTPL, FVOCI,
    AC) — entscheidend für die CET1-Wirkungskette:
      - HfT/FVTPL : ΔFV durchläuft P&L → CET1 (durchschlagend)
      - FVOCI     : ΔFV via OCI → CET1 (durchschlagend)
      - AC        : zu Buchwert, kein P&L-Effekt unter Rate-Stress

    Filter:
      - Item ∈ {2520812, 2520813, 2520814, 2520815}
      - Country > 0 (specific countries — Country=0 nicht reportiert)
      - Maturity ∈ [1, 7] (specific buckets, nicht Total=8)
      - Accounting_portfolio = 0 (Items kodieren bereits die Accounting-Class)

    Returns
    -------
    DataFrame mit Spalten [LEI_Code, accounting_class, channel, Maturity,
    label, exposure_eur, duration_years]. exposure_eur ist die Summe
    über alle Counterparty-Countries innerhalb der gegebenen
    Bank/Class/Maturity-Kombination.
    """
    df = sov_df[
        sov_df["Item"].isin(SOV_ACCOUNTING_ITEMS.keys())
        & (sov_df["Period"] == period)
        & (sov_df["Country"] > 0)
        & (sov_df["Accounting_portfolio"] == 0)
        & (sov_df["Maturity"].between(1, 7))
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # Sum across counterparty-countries
    out = (df.groupby(["LEI_Code", "Item", "Maturity"], as_index=False)
             ["Amount"].sum()
             .rename(columns={"Amount": "exposure_m_eur"}))
    out["exposure_eur"]   = out["exposure_m_eur"] * 1e6
    out["duration_years"] = out["Maturity"].map(DURATION_BY_BUCKET)
    out["label"]          = out["Maturity"].map(MATURITY_BUCKETS)
    out["accounting_class"] = out["Item"].map(
        lambda x: SOV_ACCOUNTING_ITEMS[x]["label"]
    )
    out["channel"] = out["Item"].map(
        lambda x: SOV_ACCOUNTING_ITEMS[x]["channel"]
    )
    return out[["LEI_Code", "accounting_class", "channel", "Maturity",
                "label", "exposure_eur", "duration_years"]]


def sovereign_cet1_impact(
    sov_acct_df: pd.DataFrame, delta_r_pp: float,
) -> pd.DataFrame:
    """Pro Bank: CET1-Impact aus Sovereign-Bonds unter Δr-Shock, gespalten
    nach Accounting-Class.

    Formel:  ΔFV_bucket = -D_bucket * (Δr_pp/100) * exposure_eur
    CET1-Wirkung: nur HfT/FVTPL/FVOCI durchschlagend; AC bleibt konstant.

    Returns
    -------
    DataFrame [LEI_Code, accounting_class, channel, fair_value_eur,
               delta_fv_eur, cet1_impact_eur]
    """
    if sov_acct_df.empty:
        return pd.DataFrame(columns=["LEI_Code", "accounting_class", "channel",
                                     "fair_value_eur", "delta_fv_eur",
                                     "cet1_impact_eur"])
    delta_y = delta_r_pp / 100.0
    df = sov_acct_df.copy()
    df["delta_fv_eur"] = -df["duration_years"] * delta_y * df["exposure_eur"]

    # Aggregate per bank × accounting_class
    agg = (df.groupby(["LEI_Code", "accounting_class", "channel"],
                      as_index=False)
             .agg(fair_value_eur=("exposure_eur", "sum"),
                  delta_fv_eur=("delta_fv_eur", "sum")))
    # CET1 impact: P&L and OCI channels durchschlagend; AC nicht
    agg["cet1_impact_eur"] = agg.apply(
        lambda r: r["delta_fv_eur"] if r["channel"] in ("P&L", "OCI") else 0.0,
        axis=1,
    )
    return agg


def sovereign_cet1_pnl_lookup(
    sov_acct_df: pd.DataFrame, delta_r_pp: float,
) -> dict[str, float]:
    """Pro Bank: CET1-wirksamer Sovereign-MtM unter Δr-Schock (EUR, signiert).

    Single Source für den Sovereign-Kanal der CET1-Bridge. Nutzt den
    bank-individuell GEMELDETEN IFRS-9-Split aus der EBA Transparency
    (tr_sov.csv, Items 2520812 HfT / 2520813 FVTPL / 2520814 FVOCI /
    2520815 AC) — KEINE Stylized-Fact-Annahme (frühere 60/40-Aufteilung
    ersetzt, ebenso die V1-Vereinfachung "gesamte Ladder FVOCI-ähnlich").

    Ökonomik: Nur zum Marktwert geführte Bestände (HfT/FVTPL via GuV,
    FVOCI via OCI-Rücklage) schlagen auf die CET1-Quote durch; AC bleibt
    zu fortgeführten Anschaffungskosten (latenter, nicht CET1-wirksamer
    Verlust — vgl. SVB 2023, Jiang et al. 2023 NBER WP 31048).
    Mathematik: ΔFV = −D_bucket · Δy · Exposure pro Laufzeit-Bucket
    (Tuckman/Serrat 2012, Kap. 4), summiert über die CET1-wirksamen
    Klassen je Bank.

    Returns: dict {LEI_Code: Σ cet1_impact_eur} (negativ = Verlust).
    """
    imp = sovereign_cet1_impact(sov_acct_df, delta_r_pp=delta_r_pp)
    if imp.empty:
        return {}
    return imp.groupby("LEI_Code")["cet1_impact_eur"].sum().to_dict()


def sovereign_kpis_per_bank(
    conc_df: pd.DataFrame, bank_dir: pd.DataFrame,
) -> pd.DataFrame:
    """Pro Bank: Total, Domestic-Share, Top-1-Land, HHI auf Länder-Verteilung."""
    bank_meta = bank_dir.set_index("lei")[["bank_name", "country"]]

    # Map ISO → Country dim code is annoying — bank_dir has ISO ('DE'),
    # conc_df has integer code (10 = Germany). Use bank_country directly
    # by joining with the country dim externally.
    rows = []
    totals = conc_df.groupby("LEI_Code")["exposure_eur"].sum()

    for lei, group in conc_df.groupby("LEI_Code"):
        if lei not in bank_meta.index:
            continue
        meta = bank_meta.loc[lei]
        bank_name = meta["bank_name"]
        bank_country_iso = meta["country"]

        total = float(group["exposure_eur"].sum())
        if total <= 0:
            continue

        shares = group["exposure_eur"] / total
        # Top-1
        top_idx = group["exposure_eur"].idxmax()
        top_country_code = int(group.loc[top_idx, "Country"])
        top_share = float(shares.loc[top_idx])
        # HHI
        hhi = float((shares ** 2).sum())
        # Top-3 share
        top3_share = float(shares.nlargest(3).sum())

        rows.append({
            "lei":             lei,
            "bank_name":       bank_name,
            "bank_country":    bank_country_iso,
            "total_eur":       total,
            "top_country_code": top_country_code,
            "top_share":       top_share,
            "top3_share":      top3_share,
            "hhi":             hhi,
            "n_countries":     int(len(group)),
        })

    return pd.DataFrame(rows)


def attach_country_names(
    conc_df: pd.DataFrame, country_dim: pd.DataFrame,
) -> pd.DataFrame:
    """Mappt numerischen Country-Code → Name + ISO an."""
    cd = country_dim.set_index("code")[["country_name", "iso"]]
    out = conc_df.copy()
    out["country_name"] = out["Country"].map(cd["country_name"])
    out["country_iso"]  = out["Country"].map(cd["iso"])
    return out


def domestic_share_per_bank(
    conc_df: pd.DataFrame, bank_dir: pd.DataFrame,
    country_dim: pd.DataFrame,
) -> pd.DataFrame:
    """Pro Bank: Anteil der Sovereign-Exposure im Heimatland.

    Verbindet conc_df (Bank × Country-Code → Exposure) mit dem Country-Dim
    (ISO ↔ Code) und vergleicht mit der Bank-Land-ISO aus bank_dir.
    """
    iso_to_code = country_dim.set_index("iso")["code"].to_dict()
    bank_meta = bank_dir[["lei", "bank_name", "country"]].copy()
    bank_meta["home_code"] = bank_meta["country"].map(iso_to_code)

    rows = []
    for lei, group in conc_df.groupby("LEI_Code"):
        meta_match = bank_meta[bank_meta["lei"] == lei]
        if meta_match.empty:
            continue
        meta = meta_match.iloc[0]
        home_code = meta["home_code"]
        total = float(group["exposure_eur"].sum())
        domestic = float(
            group[group["Country"] == home_code]["exposure_eur"].sum()
        ) if pd.notna(home_code) else 0.0
        rows.append({
            "lei":            lei,
            "bank_name":      meta["bank_name"],
            "bank_country":   meta["country"],
            "total_eur":      total,
            "domestic_eur":   domestic,
            "domestic_share": domestic / total if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ============================================================================
# 5b. Rate-Shock P&L (Modified-Duration Approximation)
# ============================================================================
def rate_shock_pnl(
    maturity_df: pd.DataFrame, delta_r_pp: float,
) -> pd.DataFrame:
    """Per-Bank Mark-to-Market P&L unter parallelem Rate-Shock.

    Modified-Duration Approximation: ΔP/P ≈ −D · Δy
    wobei Δy in Dezimal (Δr_pp / 100).

    Per Bucket: ΔP_bucket = −D_bucket · Δy · Exposure_bucket
    Per Bank:   ΔP_bank   = Σ_bucket ΔP_bucket
    """
    delta_y_dec = delta_r_pp / 100.0   # pp → decimal
    df = maturity_df.copy()
    df["delta_pnl_eur"] = (
        -df["duration_years"] * delta_y_dec * df["exposure_eur"]
    )
    return (df.groupby("LEI_Code", as_index=False)
              .agg(total_eur=("exposure_eur", "sum"),
                   delta_pnl_eur=("delta_pnl_eur", "sum")))


def rate_shock_pnl_per_bucket(
    maturity_df: pd.DataFrame, delta_r_pp: float,
) -> pd.DataFrame:
    """Per-Bank-per-Bucket P&L Decomposition."""
    delta_y_dec = delta_r_pp / 100.0
    df = maturity_df.copy()
    df["delta_y_decimal"] = delta_y_dec
    df["delta_pnl_eur"] = (
        -df["duration_years"] * delta_y_dec * df["exposure_eur"]
    )
    return df


# ============================================================================
# 5c. CET1 Capital + RWA Loader (tr_oth.csv)
# ============================================================================
# Item codes (TR_2025 — current vintage). Phase 5 backtest will resolve
# vintage-specific codes via the SDD translation table.
ITEM_CET1_CAPITAL    = 2520102   # Common Equity Tier 1 Capital
ITEM_OCI             = 2520105   # Accumulated Other Comprehensive Income
ITEM_CR_RWA          = 2520201   # Credit Risk RWA (excl. CCR & Securitisations)
ITEM_SECURITISATION_RWA = 2520209  # Securitisation exposures RWA (banking book)
ITEM_MR_RWA          = 2520210   # Market Risk RWA (Position, FX, Commodities)
ITEM_OP_RWA          = 2520215   # Operational Risk RWA
ITEM_TOTAL_RWA       = 2520220   # Total Risk Exposure Amount
ITEM_TB_PNL          = 2520311   # Gains/losses on financial assets HFT (P&L)


def parse_capital_overview(
    csv_path: Path,
    *,
    period: int = 202506,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Streamt tr_oth.csv und extrahiert per-Bank Capital + RWA-Breakdown.

    Output (eine Zeile pro Bank):
      LEI_Code, cet1_eur, oci_eur, rwa_credit_eur, rwa_market_eur,
      rwa_operational_eur, rwa_total_eur, tb_pnl_eur

    Werte in EUR (Input ist m EUR → ×1e6).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        fb = _derived(f"cap_df_{period}.parquet")
        if fb.exists():
            return pd.read_parquet(fb)
        raise FileNotFoundError(
            f"tr_oth.csv nicht gefunden ({csv_path}) und kein Derived-"
            f"Fallback unter {fb}. `python backend/precompute_eba_cache.py` "
            f"ausführen."
        )
    keep_items = {
        ITEM_CET1_CAPITAL, ITEM_OCI, ITEM_CR_RWA, ITEM_SECURITISATION_RWA,
        ITEM_MR_RWA, ITEM_OP_RWA, ITEM_TOTAL_RWA, ITEM_TB_PNL,
    }
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        sub = chunk[
            (chunk["Period"] == period)
            & chunk["Item"].isin(keep_items)
        ]
        if len(sub):
            chunks.append(sub[["LEI_Code", "Item", "Amount"]])
    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)
    # Pivot: rows = bank, columns = item
    pivot = raw.pivot_table(
        index="LEI_Code", columns="Item", values="Amount",
        aggfunc="first", fill_value=0.0,
    )

    # Map item codes to readable column names
    rename_map = {
        ITEM_CET1_CAPITAL:        "cet1_m_eur",
        ITEM_OCI:                 "oci_m_eur",
        ITEM_CR_RWA:              "rwa_credit_m_eur",
        ITEM_SECURITISATION_RWA:  "rwa_securitisation_m_eur",
        ITEM_MR_RWA:              "rwa_market_m_eur",
        ITEM_OP_RWA:              "rwa_operational_m_eur",
        ITEM_TOTAL_RWA:           "rwa_total_m_eur",
        ITEM_TB_PNL:              "tb_pnl_m_eur",
    }
    pivot = pivot.rename(columns=rename_map).reset_index()

    # Ensure all expected columns exist
    for col in rename_map.values():
        if col not in pivot.columns:
            pivot[col] = 0.0

    # Convert to EUR
    for col_m in rename_map.values():
        col_eur = col_m.replace("_m_eur", "_eur")
        pivot[col_eur] = pivot[col_m] * 1e6

    out_cols = ["LEI_Code"] + [c.replace("_m_eur", "_eur") for c in rename_map.values()]
    return pivot[out_cols]


# ============================================================================
# 5d. Trading Book Market-Risk Stress Channel
# ============================================================================
def trading_book_stress(
    capital_df: pd.DataFrame,
    *,
    m_factor: float,
    mr_rwa_uplift_at_m_minus_2_5: float = 0.30,
    tb_pnl_haircut_at_m_minus_2_5: float = 0.50,
) -> pd.DataFrame:
    """Stress-elastische Market-Risk-RWA und Trading-Book-P&L.

    Zwei Effekte unter adversem Stress (M < 0):

    (a) Market-Risk-RWA wächst, weil VaR/SVaR-Multiplikatoren in volatileren
        Märkten steigen (FRTB-konsistent):
            RWA_MR_stress = RWA_MR_base · (1 + λ_RWA · max(-M, 0))
        Default λ_RWA = 0.30/2.5 = 0.12 — d.h. +30% MR-RWA bei M = -2.5.

    (b) Trading-Book-P&L wird gekürzt — die Annahme: laufende Q-Erträge
        aus HFT/FVTPL fallen unter Stress aus (Liquiditäts-/Spread-Schocks):
            TB_PnL_stress = TB_PnL_base · (1 - λ_PnL · max(-M, 0))
        Default λ_PnL = 0.50/2.5 = 0.20 — d.h. P&L halbiert bei M = -2.5.

    Beide λ sind hardcoded V1-Parameter (Klasse 'assumption' im
    Approximations-Inventar A-04 erweitert) und in V2 kalibrierbar gegen
    historische EBA-Stresstest-Auswirkungen.

    Returns
    -------
    DataFrame mit zusätzlichen Spalten:
        rwa_market_eur_stress, delta_rwa_market_eur,
        tb_pnl_eur_stress,    delta_tb_pnl_eur
    """
    out = capital_df.copy()
    if m_factor < 0:
        m_abs = abs(m_factor)
        rwa_mul = 1.0 + (mr_rwa_uplift_at_m_minus_2_5 / 2.5) * m_abs
        pnl_mul = 1.0 - (tb_pnl_haircut_at_m_minus_2_5 / 2.5) * m_abs
        # Cap pnl_mul at 0 (no negative scaling)
        pnl_mul = max(pnl_mul, 0.0)
    else:
        rwa_mul = 1.0
        pnl_mul = 1.0

    out["rwa_market_eur_stress"] = out["rwa_market_eur"] * rwa_mul
    out["delta_rwa_market_eur"]  = out["rwa_market_eur_stress"] - out["rwa_market_eur"]
    out["tb_pnl_eur_stress"]     = out["tb_pnl_eur"] * pnl_mul
    out["delta_tb_pnl_eur"]      = out["tb_pnl_eur_stress"] - out["tb_pnl_eur"]
    return out


# ============================================================================
# 5e. CET1-Ratio Bridge (3-channel decomposition)
# ============================================================================
# ============================================================================
# 5f. Vintage-aware loader (for backtesting · Phase 5)
# ============================================================================
def get_sdd_translation_table(metadata_dir: Path) -> pd.DataFrame:
    """Lädt die SDD-Item-Translation-Tabelle (TR_2020 ... TR_2025).

    Returns
    -------
    DataFrame mit Spalten ['Item', 'Item_TR_2024', 'Item_TR_2023',
    'Item_TR_2022', 'Item_TR_2021', 'Item_TR_2020A', 'Label', 'CSV',
    'Template'].
    """
    sdd_path = Path(metadata_dir) / "SDD.xlsx"
    if not sdd_path.exists():
        raise FileNotFoundError(f"SDD.xlsx not found in {metadata_dir}")
    return pd.read_excel(sdd_path, sheet_name="SDD", header=1)


def resolve_item_for_vintage(
    item_2025: int, vintage: str, sdd: pd.DataFrame,
) -> int | None:
    """Mappt einen 2025-Item-Code auf den entsprechenden vintage-spezifischen Code.

    EBA Items shiften pro Jahr (2x20yyy mit x = vintage decade digit):
        2020102 (TR_2020) → 2120102 (TR_2021) → ... → 2520102 (TR_2025)
    """
    col_map = {
        "2020": "Item_TR_2020A",   # Spring 2020 vintage
        "2020S": "Item_TR_2020S",
        "2021": "Item_TR_2021",
        "2022": "Item_TR_2022",
        "2023": "Item_TR_2023",
        "2024": "Item_TR_2024",
        "2025": "Item",            # Item column itself = TR_2025
    }
    col = col_map.get(vintage)
    if col is None or col not in sdd.columns:
        return None
    rows = sdd[sdd["Item"] == item_2025]
    if len(rows) == 0:
        return None
    val = rows[col].iloc[0]
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _read_csv_resilient(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with utf-8 → latin-1 fallback (older EBA vintages need latin-1)."""
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def parse_capital_overview_vintage(
    eba_dir: Path | str,
    *,
    vintage: str,
    items_2025: tuple[int, ...] = (
        ITEM_CET1_CAPITAL, ITEM_OCI, ITEM_CR_RWA, ITEM_MR_RWA,
        ITEM_OP_RWA, ITEM_TOTAL_RWA, ITEM_TB_PNL,
    ),
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Lädt Capital + RWA aus einem historischen Vintage.

    Erkennt Pfade automatisch:
      - vintage = "2025" → eba_dir / "tr_oth.csv"  (current snapshot)
      - vintage = "2024" → eba_dir / "transparency_2024" / "tr_oth.csv"
      - usw.

    Mappt 2025-Item-Codes auf vintage-spezifische Codes via SDD-Translation.

    Returns long-format DataFrame [LEI_Code, Period, item_label, Amount_eur].
    """
    eba_dir = Path(eba_dir)
    if vintage == "2025":
        oth_path = eba_dir / "tr_oth.csv"
        sdd_dir = eba_dir
    else:
        oth_path = eba_dir / f"transparency_{vintage}" / "tr_oth.csv"
        sdd_dir = eba_dir / f"transparency_{vintage}"
    if not oth_path.exists():
        raise FileNotFoundError(f"tr_oth.csv not found at {oth_path}")

    # Load SDD for this vintage (or fall back to 2025 SDD which has all
    # historical mappings).
    sdd = get_sdd_translation_table(eba_dir)

    # Resolve items for this vintage
    item_resolution = {}
    for item_2025 in items_2025:
        resolved = resolve_item_for_vintage(item_2025, vintage, sdd)
        if resolved is not None:
            item_resolution[resolved] = item_2025

    if not item_resolution:
        return pd.DataFrame()

    # Stream and filter
    chunks = []
    for chunk in _read_csv_resilient(oth_path, chunksize=chunksize, low_memory=False):
        sub = chunk[chunk["Item"].isin(item_resolution.keys())]
        if len(sub):
            chunks.append(sub[["LEI_Code", "Period", "Item", "Amount"]])
    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)
    # Translate vintage-Item back to canonical 2025-label
    raw["item_2025"] = raw["Item"].map(item_resolution)
    raw["item_label"] = raw["item_2025"].map({
        ITEM_CET1_CAPITAL:   "cet1",
        ITEM_OCI:            "oci",
        ITEM_CR_RWA:         "rwa_credit",
        ITEM_MR_RWA:         "rwa_market",
        ITEM_OP_RWA:         "rwa_operational",
        ITEM_TOTAL_RWA:      "rwa_total",
        ITEM_TB_PNL:         "tb_pnl",
    })
    raw["Amount_eur"] = raw["Amount"] * 1e6
    raw["vintage"] = vintage
    return raw[["LEI_Code", "Period", "item_label", "Amount_eur", "vintage"]]


def load_historical_capital_panel(
    eba_dir: Path | str,
    vintages: tuple[str, ...] = ("2020", "2021", "2022", "2023", "2024", "2025"),
) -> pd.DataFrame:
    """Konkateniert Capital-Panels über alle Vintages zu einer langen Zeitreihe.

    Output schema:
      LEI_Code (str) · Period (int YYYYMM) · item_label (str) · Amount_eur (float) · vintage (str)

    Bei überlappenden Periods (z.B. Q3 2020 in 2021-Vintage UND 2020-Autumn-Vintage)
    wird die *neuere* Vintage bevorzugt — die EBA korrigiert in späteren
    Releases gelegentlich publizierte Daten der Vorperiode.
    """
    eba_dir = Path(eba_dir)
    # Cloud-Fallback: ohne Roh-tr_oth.csv die committete Panel-Parquet nutzen.
    if not (eba_dir / "tr_oth.csv").exists():
        fb = _derived("hist_capital_panel.parquet")
        if fb.exists():
            return pd.read_parquet(fb)
    panels = []
    for v in vintages:
        try:
            p = parse_capital_overview_vintage(eba_dir, vintage=v)
            if not p.empty:
                panels.append(p)
        except FileNotFoundError:
            continue
    if not panels:
        return pd.DataFrame()

    full = pd.concat(panels, ignore_index=True)
    # When same (LEI, Period, label) appears in multiple vintages, keep the latest vintage
    full = (full.sort_values(["LEI_Code", "Period", "item_label", "vintage"])
                .drop_duplicates(subset=["LEI_Code", "Period", "item_label"],
                                 keep="last"))
    return full


def panel_to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivots the long panel to wide: rows = (LEI, Period), columns = item_label."""
    if panel.empty:
        return pd.DataFrame()
    wide = (panel.pivot_table(
                index=["LEI_Code", "Period"],
                columns="item_label", values="Amount_eur",
                aggfunc="first",
            ).reset_index())
    if "rwa_total" in wide.columns and "cet1" in wide.columns:
        # Compute CET1 ratio inline
        wide["cet1_ratio"] = wide["cet1"] / wide["rwa_total"].replace(0, pd.NA)
    return wide


def cet1_ratio_bridge(
    capital_df: pd.DataFrame,
    loan_book_bridge_per_bank: dict[str, dict],
    sovereign_pnl_per_bank: dict[str, float],
    tb_stress_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Zwei-Kanal-CET1-Ratio-Decomposition pro Bank.

    V1 modelliert zwei Stress-Kanäle auf die CET1-Quote — den Kreditbuch-
    und den Sovereign-Kanal. Der frühere Trading-Book-Kanal wurde entfernt
    (die Handels­bücher der zehn überwiegend Retail-/Corporate-lastigen
    Banken sind klein, und eine belastbare FRTB-Sensitivität ließe sich aus
    den EBA-Aggregaten nicht sauber kalibrieren). Das Markt-Risiko bleibt im
    Marktbuch-Tab rein deskriptiv. `tb_stress_df` ist nur noch ein optionaler
    Legacy-Parameter und wird, falls übergeben, ignoriert.

    Architektur:
      Numerator (CET1):
        CET1_stress = CET1_base
                      − ΔEL_loan_book          (Loan-Book Provisions-Hit)
                      + Δ_sovereign_MtM         (Sovereign FVOCI/AfS via OCI, signiert)

      Denominator (Total RWA):
        RWA_stress = RWA_base
                     + ΔRWA_credit_loan_book   (from loan_book bridge)

    Parameters
    ----------
    capital_df : Output von parse_capital_overview
                 (per-bank CET1, RWA-Breakdown, OCI)
    loan_book_bridge_per_bank : dict {LEI: bridge dict from
                 BankPortfolio.capital_bridge}
    sovereign_pnl_per_bank : dict {LEI: signed P&L EUR from
                 rate_shock_pnl}
    tb_stress_df : deprecated/ignored (Trading-Book-Kanal entfernt)

    Returns
    -------
    DataFrame mit Spalten:
        LEI_Code, cet1_base, cet1_stress, cet1_ratio_base, cet1_ratio_stress,
        delta_cet1_loan, delta_cet1_sovereign, delta_cet1_tb (=0),
        delta_rwa_credit, delta_rwa_market (=0), rwa_total_base, rwa_total_stress
    """
    df = capital_df

    rows = []
    for _, r in df.iterrows():
        lei = r["LEI_Code"]
        cet1_base = float(r["cet1_eur"])
        rwa_base  = float(r["rwa_total_eur"])

        # Channel 1 — Loan Book (ΔRWA_credit + ΔEL into provisions)
        lb = loan_book_bridge_per_bank.get(lei)
        if lb is None:
            d_rwa_credit = 0.0
            d_el_loan    = 0.0
        else:
            d_rwa_credit = float(lb["delta_rwa"])
            d_el_loan    = float(lb["delta_el"])

        # Channel 2 — Sovereign MtM (negative = loss; flow through OCI)
        d_sov_mtm = float(sovereign_pnl_per_bank.get(lei, 0.0))

        # Trading-Book-Kanal entfernt (V1) — Beiträge fest 0.
        d_tb_pnl     = 0.0
        d_rwa_market = 0.0

        # Numerator effect (signiert: d_sov_mtm < 0 unter Zins-up)
        cet1_stress = cet1_base - d_el_loan + d_sov_mtm

        # Denominator effect
        rwa_total_stress = rwa_base + d_rwa_credit

        ratio_base   = cet1_base / rwa_base       if rwa_base > 0 else float("nan")
        ratio_stress = cet1_stress / rwa_total_stress if rwa_total_stress > 0 else float("nan")

        rows.append({
            "LEI_Code":           lei,
            "cet1_base":          cet1_base,
            "delta_cet1_loan":    -d_el_loan,
            "delta_cet1_sovereign": d_sov_mtm,
            "delta_cet1_tb":      d_tb_pnl,
            "cet1_stress":        cet1_stress,
            "rwa_total_base":     rwa_base,
            "delta_rwa_credit":   d_rwa_credit,
            "delta_rwa_market":   d_rwa_market,
            "rwa_total_stress":   rwa_total_stress,
            "cet1_ratio_base":    ratio_base,
            "cet1_ratio_stress":  ratio_stress,
            "delta_cet1_ratio_pp": (ratio_stress - ratio_base) * 100,
        })
    return pd.DataFrame(rows)


# ============================================================================
# 6. Public API (unchanged)
# ============================================================================
def load_eba_universe(
    *,
    vintage: str = "2025",
    real_files_dir: Path | str | None = None,
    period: int | None = None,
    top_n: int | None = None,
    prefer_real: bool = True,
) -> EbaUniverse:
    """Top-Level Entry-Point. Versucht echte Files; fällt auf Synthetik zurück."""
    if prefer_real:
        try:
            return from_real_files(eba_dir=real_files_dir, vintage=vintage,
                                   period=period, top_n=top_n)
        except (FileNotFoundError, RuntimeError):
            pass
    return from_synthetic(vintage=vintage)


# ============================================================================
# 4. Validation
# ============================================================================
def _test_synthetic_universe_loads():
    u = from_synthetic("2025")
    assert u.n_banks == 10, f"Expected 10 banks, got {u.n_banks}"
    assert u.total_ead_eur > 5e12, "Aggregate EAD should be > 5 trillion EUR"


def _test_summary_table_complete():
    u = from_synthetic("2025")
    df = u.summary_table()
    assert len(df) == 10
    assert df["Total EAD bn"].sum() > 5000
    assert (df["RWA density"] > 0.2).all() and (df["RWA density"] < 1.5).all()


def _test_aggregated_portfolio_consistency():
    u = from_synthetic("2025")
    agg = u.aggregated_portfolio()
    assert len(agg.segments) == 60   # 10 banks x 6 segments
    sum_eads = sum(b.total_ead for b in u.banks.values())
    assert abs(agg.total_ead - sum_eads) < 1.0


def _test_adverse_anchor_present():
    u = from_synthetic("2025")
    assert "z_factor_implied" in u.adverse_anchor
    assert u.adverse_anchor["z_factor_implied"] < 0


def _test_real_loader_if_files_present():
    """Smoke-Test für den echten Loader — überspringt wenn Files fehlen."""
    from config import EBA_RAW_DIR
    if not (EBA_RAW_DIR / "tr_cre.csv").exists():
        return  # Skip silently
    u = from_real_files(EBA_RAW_DIR, top_n=10)
    assert u.n_banks > 0, "Real loader returned 0 banks"
    assert u.total_ead_eur > 100e9, f"Real EAD too small: {u.total_ead_eur/1e9:.1f}"
    df = u.summary_table()
    assert (df["RWA density"] > 0.05).all(), "Some banks show implausibly low RWA density"
    assert (df["RWA density"] < 1.5).all(), "Some banks show implausibly high RWA density"


def _test_exposure_mapping_complete():
    """Sicherheits-Check: alle gemappten Vasicek-Klassen haben LGD und Maturity."""
    classes = set(EXPOSURE_TO_VASICEK_CLASS.values())
    for c in classes:
        assert c in LGD_BY_VASICEK_CLASS, f"LGD missing for class {c}"
        assert c in MATURITY_BY_VASICEK_CLASS, f"Maturity missing for class {c}"


def _test_filter_keeps_all_10_curated_banks():
    """Regression: filter_universe_to_top10 darf KEINE kuratierte Bank
    verlieren. Früher fiel Société Générale wegen Akzent-Mismatch
    ("Societe Generale" vs. "Société générale S.A.") still aus → 9 statt 10
    Banken. Jetzt LEI-First-Matching + Guard.

    Testet den realen Pfad, falls Daten/Derived-Cache vorhanden — sonst
    den synthetischen Fallback (beide tragen jetzt alle 10 LEIs).
    """
    import warnings as _w
    from eba_pd_loader import filter_universe_to_top10
    try:
        u = load_eba_universe(vintage="2025", top_n=10)
    except Exception:
        u = from_synthetic("2025")
    # Der eingebaute Guard warnt, falls eine kuratierte Bank nicht gematcht
    # wird; als Error behandelt fängt das die SocGen-Regression hart ab.
    # (CA/BPCE werden über den Namens-Fallback gebrückt, da EBA-Metadaten
    #  FR-präfixierte LEIs führen, die Pillar-3-CSV die kanonischen.)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        filter_universe_to_top10(u)
    assert u.n_banks == 10, f"Erwartet 10 gefilterte Banken, gefunden {u.n_banks}"


def _test_santander_sovereign_excluded_from_irb_book():
    """Santander Sovereign ist 100 % Standardansatz und darf daher nicht
    als künstliches IRB-Segment in EL/RWA eingehen. Der separate
    Sovereign-Marktwertkanal bleibt davon unberührt."""
    from eba_pd_loader import filter_universe_to_top10
    try:
        u = load_eba_universe(vintage="2025", top_n=10)
    except Exception:
        u = from_synthetic("2025")
    filter_universe_to_top10(u)
    santander = next(
        pf for pf in u.banks.values()
        if (getattr(pf, "lei", "") or "") == "5493006QMFDDMYWIAM13"
        or "santander" in pf.name.lower()
    )
    classes = {seg.exposure_class for seg in santander.segments}
    assert "sovereign" not in classes, (
        "Santander Sovereign darf nicht im IRB-Kreditbuch enthalten sein"
    )


def _test_curated_leis_in_bank_dir():
    """Alle 10 kuratierten LEIs müssen EXAKT im EBA-Bank-Directory liegen.
    Sonst fallen Banken aus den cap_df-/Sovereign-/Walk-Forward-Joins
    (gefiltert per get_top10_leis) → ein Tab zeigt z. B. 8 statt 10 Banken.
    Historischer Fall: CA/BPCE hatten kanonische LEIs in der CSV, die
    EBA-Daten aber FR-präfixierte. Überspringt ohne EBA-Daten/Metadata.
    """
    from config import EBA_RAW_DIR
    meta = EBA_RAW_DIR / "TR_Metadata.xlsx"
    if not meta.exists() and not _derived("bank_dir.parquet").exists():
        return
    from eba_pd_loader import get_top10_leis
    eba_leis = set(load_bank_directory(meta)["lei"])
    missing = get_top10_leis() - eba_leis
    assert not missing, (
        f"Kuratierte LEIs fehlen im EBA-Directory: {missing}. "
        f"pillar3_bank_pd_lgd.csv muss dieselben LEI-Codes wie die "
        f"EBA-Transparency-Daten verwenden (gemeinsamer Join-Key)."
    )


def _test_sovereign_effective_lt_gross():
    """CET1-wirksamer Sovereign-MtM (echter IFRS-9-Split) muss betraglich
    unter dem Brutto-MtM der vollen Ladder liegen (AC-Anteil > 0) und
    alle 10 kuratierten Banken abdecken."""
    from config import EBA_RAW_DIR as _RAW
    if not (_RAW / "tr_sov.csv").exists():
        return  # Daten nicht vorhanden — Skip (wie Real-loader-Smoke)
    from eba_pd_loader import get_top10_leis
    leis = set(get_top10_leis())
    sov = parse_sovereign_csv(_RAW / "tr_sov.csv", period=202506)
    acct = sovereign_by_accounting_class(sov, period=202506)
    acct = acct[acct["LEI_Code"].isin(leis)]
    mat = sovereign_maturity_ladder(sov, period=202506)
    mat = mat[mat["LEI_Code"].isin(leis)]
    eff = sovereign_cet1_pnl_lookup(acct, delta_r_pp=2.0)
    gross = dict(zip(*[rate_shock_pnl(mat, delta_r_pp=2.0)[c]
                       for c in ("LEI_Code", "delta_pnl_eur")]))
    assert set(eff.keys()) == leis, \
        f"Effective lookup deckt {len(eff)}/10 Banken ab"
    tot_eff, tot_gross = sum(eff.values()), sum(gross.values())
    assert abs(tot_eff) < abs(tot_gross), \
        f"effective {tot_eff:.3e} nicht < gross {tot_gross:.3e}"
    assert 0.30 < abs(tot_eff / tot_gross) < 0.90, \
        f"effective/gross-Ratio {tot_eff/tot_gross:.2f} unplausibel"


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("eba_loader.py - EBA Universe Loader - Validation")
    print("=" * 60)
    tests = [
        ("Synthetic universe loads",          _test_synthetic_universe_loads),
        ("Summary table complete",            _test_summary_table_complete),
        ("Aggregated portfolio consistency",  _test_aggregated_portfolio_consistency),
        ("Adverse anchor present",            _test_adverse_anchor_present),
        ("Exposure mapping complete",         _test_exposure_mapping_complete),
        ("Real loader smoke (if files present)", _test_real_loader_if_files_present),
        ("Filter keeps all 10 curated banks", _test_filter_keeps_all_10_curated_banks),
        ("Santander Sovereign excluded from IRB", _test_santander_sovereign_excluded_from_irb_book),
        ("Curated LEIs match EBA directory",  _test_curated_leis_in_bank_dir),
        ("Sovereign effective < gross MtM",   _test_sovereign_effective_lt_gross),
    ]
    for label, fn in tests:
        try:
            fn()
            print(f"  [PASS]  {label}")
        except AssertionError as e:
            print(f"  [FAIL]  {label}: {e}")
            raise

    print("\n--- EU Top-10 Bank Universe (Baseline) ---\n")
    u = from_synthetic("2025")
    df = u.summary_table()
    df_disp = df.copy()
    df_disp["Total EAD bn"] = df_disp["Total EAD bn"].round(0).astype(int)
    df_disp["EL bn"]        = df_disp["EL bn"].round(2)
    df_disp["UL bn"]        = df_disp["UL bn"].round(2)
    df_disp["RWA bn"]       = df_disp["RWA bn"].round(0).astype(int)
    df_disp["RWA density"]  = (df_disp["RWA density"] * 100).round(1).astype(str) + "%"
    df_disp["EL %"]         = (df_disp["EL %"] * 100).map(lambda v: f"{v:.3f}%")
    print(df_disp.to_string(index=False))

    print("\nTotals:")
    print(f"  Σ EAD:  {u.total_ead_eur/1e9:>8.0f} bn EUR")
    print(f"  Σ EL:   {df['EL bn'].sum():>8.1f} bn EUR")
    print(f"  Σ UL:   {df['UL bn'].sum():>8.1f} bn EUR")
    print(f"  Σ RWA:  {df['RWA bn'].sum():>8.0f} bn EUR")

    anchor = u.adverse_anchor
    print(f"\nAdverse anchor (vintage {anchor['vintage']}):")
    print(f"  Brent log-shock: {anchor['brent_log_shock']:+.2f}")
    print(f"  rate_10y shock:  {anchor['rate_10y_pp_shock']*100:+.0f} bp ({anchor['rate_10y_pp_shock']:+.2f} pp)")
    print(f"  GDP shock:       {anchor['gdp_pp_shock']:+.1f} pp")
    print(f"  Implied M (z):   {anchor['z_factor_implied']:+.2f}")
    print(f"  Status:          {anchor['status']}")

    print("\n[PASS] All EBA loader tests passed.")
