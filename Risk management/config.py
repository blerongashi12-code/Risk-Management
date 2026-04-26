"""
============================================================================
 config.py  ·  Zentrale Konfiguration für das DAX Credit Stress Modell
============================================================================
 Enthält:
   - Pfade zu Excel-Workbook & Data-Cache
   - DAX-40 Ticker-Mapping
   - Datenquellen-Konstanten (Start-Datum, Proxy-Ticker)
   - Modellparameter (Horizon, LGD, etc.)
============================================================================
"""

from pathlib import Path
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# 1. PFADE
# ----------------------------------------------------------------------
# Basis-Verzeichnis = Ordner, in dem dieses config.py liegt
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Unterordner automatisch anlegen
DATA_DIR       = BASE_DIR / "data"
CACHE_DIR      = DATA_DIR / "cache"
OUTPUT_DIR     = BASE_DIR / "output"
for p in (DATA_DIR, CACHE_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. DAX-40 TICKER
# ----------------------------------------------------------------------
DAX40 = {
    "ADS.DE":  "Adidas",
    "AIR.DE":  "Airbus",
    "ALV.DE":  "Allianz",
    "BAS.DE":  "BASF",
    "BAYN.DE": "Bayer",
    "BMW.DE":  "BMW",
    "BNR.DE":  "Brenntag",
    "CBK.DE":  "Commerzbank",
    "CON.DE":  "Continental",
    "1COV.DE": "Covestro",
    "DHL.DE":  "DHL Group",
    "DB1.DE":  "Deutsche Börse",
    "DBK.DE":  "Deutsche Bank",
    "DTE.DE":  "Deutsche Telekom",
    "EOAN.DE": "E.ON",
    "FRE.DE":  "Fresenius",
    "HNR1.DE": "Hannover Rück",
    "HEI.DE":  "Heidelberg Materials",
    "HEN3.DE": "Henkel",
    "IFX.DE":  "Infineon",
    "MBG.DE":  "Mercedes-Benz",
    "MRK.DE":  "Merck",
    "MTX.DE":  "MTU Aero",
    "MUV2.DE": "Münchener Rück",
    "P911.DE": "Porsche AG",
    "PAH3.DE": "Porsche SE",
    "QIA.DE":  "Qiagen",
    "RHM.DE":  "Rheinmetall",
    "RWE.DE":  "RWE",
    "SAP.DE":  "SAP",
    "SRT3.DE": "Sartorius",
    "SIE.DE":  "Siemens",
    "ENR.DE":  "Siemens Energy",
    "SHL.DE":  "Siemens Healthineers",
    "SY1.DE":  "Symrise",
    "VOW3.DE": "Volkswagen",
    "VNA.DE":  "Vonovia",
    "ZAL.DE":  "Zalando",
}

DAX40_TICKERS = list(DAX40.keys())


# ----------------------------------------------------------------------
# 3. MARKTDATEN-PROXIES
# ----------------------------------------------------------------------
ENERGY_PROXY = "BZ=F"      # ICE Brent Crude Futures (USD/bbl)
FX_PROXY     = "EURUSD=X"  # für Umrechnung Brent USD → EUR falls gewünscht
MARKET_PROXY = "^GDAXI"    # DAX Performance Index (für Markt-Beta)


# ----------------------------------------------------------------------
# 4. DATEN-ZEITRAUM
# ----------------------------------------------------------------------
# 6 Jahre Historie, passt zum Svensson-Zeitraum
LOOKBACK_YEARS = 6
START_DATE = (datetime.today() - timedelta(days=int(LOOKBACK_YEARS * 365.25))).strftime("%Y-%m-%d")
END_DATE   = datetime.today().strftime("%Y-%m-%d")


# ----------------------------------------------------------------------
# 5. MODELLPARAMETER
# ----------------------------------------------------------------------
DEFAULT_HORIZON   = 1.0    # Jahre — für Merton PD
DEFAULT_LGD       = 0.45   # Loss Given Default (Basel-Standard)
DEFAULT_N_SIMS    = 10_000 # Monte Carlo Pfade


# ----------------------------------------------------------------------
# 5a. CREDIT-MODELL-PRÄMISSEN  (siehe MODEL_ASSUMPTIONS.md)
# ----------------------------------------------------------------------
# Default Point (DPT) nach Moody's KMV / Bharath-Shumway (2008):
#     DPT = ShortTermDebt + DPT_LTD_WEIGHT · LongTermDebt
# Begründung: Langfristige Schulden lösen nicht akut einen Default aus —
# nur ein Anteil davon zählt zur Default-Schwelle. α=0.5 ist
# akademischer Industrie-Standard.
DPT_LTD_WEIGHT = 0.5

# Sektor-spezifischer σ_V-Multiplier (post-KMV).
# Standard-Merton untertreibt PDs für stark geleveragte Sektoren
# (Banken, REITs), weil Equity-Vola strukturell gedämpft ist
# (Einlagensicherung, Liquiditäts-Backstop, regulatorischer Floor).
# Quellen: Hovakimian/Kane/Laeven (2012); Moody's KMV CreditEdge-Doku.
# Werte > 1.0 erhöhen die effektive Asset-Vola → höhere PD.
SECTOR_VOL_MULTIPLIER = {
    "Financial Services": 1.5,  # Banken + Versicherungen
    "Real Estate":        1.2,  # ähnliche Bilanz-Hebelung
}
DEFAULT_SECTOR_VOL_MULTIPLIER = 1.0  # alle anderen Sektoren


# ----------------------------------------------------------------------
# 5b. FAKTOR-MODELL · ENERGY-BETA-MULTIPLIER  (siehe MODEL_ASSUMPTIONS.md §4.5)
# ----------------------------------------------------------------------
# Methodik (fachlich fundiert):
#     EnergyMul_sector = clip( (E/U)_sector / (E/U)_Industrials, 0.1, 4.0 )
# wobei E/U = Energieaufwand / Umsatz beim DAX-Benchmark-Unternehmen.
# Industrials (Siemens) dient als Referenz-Sektor mit Mul=1.0.
#
# Werte sind First-Cut basierend auf öffentlich bekannten Sektorstudien
# und Geschäftsberichten. Spätere Kalibrierung über direkte GB-Recherche
# ist explizit vorgesehen — alle Werte sind hier zentral änderbar.
#
# Anwendung in factor_model.run_dax40:
#     beta_brent_adjusted = beta_brent_raw * EnergyMul[sector]
SECTOR_ENERGY_MUL = {
    "Utilities":              4.0,   # Benchmark RWE,  E/U ~35%, gecapped
    "Energy":                 4.0,   # Branchenstandard ~50% E/U, gecapped
    "Basic Materials":        2.4,   # Benchmark BASF, E/U ~12% (VCI)
    "Industrials":            1.0,   # Benchmark Siemens, E/U ~5% (Referenz)
    "Consumer Cyclical":      0.8,   # Benchmark BMW,  E/U ~4%
    "Communication Services": 0.6,   # Benchmark DT,   E/U ~3% (RZ-Strom)
    "Healthcare":             0.5,   # Benchmark Merck, E/U ~2.5%
    "Real Estate":            0.4,   # Benchmark Vonovia, E/U ~2%
    "Consumer Defensive":     0.4,   # Benchmark Henkel, E/U ~2%
    "Technology":             0.3,   # Benchmark SAP,  E/U ~1.5% (RZ-Strom)
    "Financial Services":     0.1,   # Benchmark DB,   E/U ~0.5%
}
DEFAULT_SECTOR_ENERGY_MUL = 1.0      # Fallback für unbekannte Sektoren

# Faktor-Modell: Lookback (gleich wie Merton, Konsistenz)
FACTOR_LOOKBACK_DAYS = 252
FACTOR_MIN_OBS       = 60   # darunter wird eine Firma als "nicht schätzbar" markiert
FACTOR_MATURITY      = 10.0 # Δr-Faktor: Δ Svensson-Rate bei 10y


# ----------------------------------------------------------------------
# 6. BUNDESBANK CSV-STRUKTUR
# ----------------------------------------------------------------------
# Spaltenordnung im offiziellen Bundesbank-Export
BBK_COLUMNS = ["Date", "Beta0", "Beta1", "Beta2", "Beta3", "Tau1", "Tau2"]
BBK_RAW_INDICES = [0, 1, 3, 5, 7, 9, 11]   # Spalten-Indizes (Rest sind Flag-Spalten)


def show_config():
    """Hilfsfunktion: zeigt aktuelle Konfiguration."""
    print("=" * 60)
    print("DAX Credit Stress · Konfiguration")
    print("=" * 60)
    print(f"  Base Dir:       {BASE_DIR}")
    print(f"  Data Dir:       {DATA_DIR}")
    print(f"  Cache Dir:      {CACHE_DIR}")
    print(f"  DAX-40 Tickers: {len(DAX40_TICKERS)}")
    print(f"  Zeitraum:       {START_DATE} bis {END_DATE}")
    print(f"  Horizon:        {DEFAULT_HORIZON} Jahre")
    print(f"  MC Paths:       {DEFAULT_N_SIMS:,}")
    print("=" * 60)


if __name__ == "__main__":
    show_config()
