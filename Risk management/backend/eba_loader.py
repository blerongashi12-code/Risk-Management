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

import pandas as pd

# Project root (directory above `backend/`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vasicek import BankPortfolio, PortfolioSegment   # noqa: E402


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
ITEM_SOV_RWA            = 2520822   # RWA on sovereign exposures

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
    "Credit Agricole": {
        "country": "FR",
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
    "ING Group": {
        "country": "NL",
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
    "Intesa Sanpaolo": {
        "country": "IT",
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
    "BBVA": {
        "country": "ES",
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
    "Societe Generale": {
        "country": "FR",
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
    "Commerzbank": {
        "country": "DE",
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
    """Baut die Synthetic-Replik der EBA-Daten auf."""
    universe = EbaUniverse(source=f"synthetic ({vintage})")
    for bank_name, spec in SYNTHETIC_BANKS.items():
        portfolio = BankPortfolio(bank_name)
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


def build_bank_portfolios(
    seg_df: pd.DataFrame, bank_dir: pd.DataFrame,
) -> dict[str, BankPortfolio]:
    """Konvertiert Aggregat-DataFrame in Vasicek-BankPortfolio-Objekte."""
    lei_to_name = dict(zip(bank_dir["lei"], bank_dir["bank_name"]))
    out: dict[str, BankPortfolio] = {}

    for lei, group in seg_df.groupby("LEI_Code"):
        bank_name = lei_to_name.get(lei, f"LEI {lei[:8]}…")
        portfolio = BankPortfolio(bank_name)
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
    if not cre_path.exists():
        raise FileNotFoundError(
            f"tr_cre.csv not found in {eba_dir}. "
            f"Place EBA Transparency files there or pass an explicit path."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"TR_Metadata.xlsx not found in {eba_dir}."
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
    keep_items = {ITEM_SOV_GROSS_ON_BS, ITEM_SOV_RWA}
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
