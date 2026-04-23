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

# Master-Workbook Pfad (wird vom Benutzer gesetzt/überschrieben)
WORKBOOK_PATH = BASE_DIR / "DAX_Credit_Stress_Master.xlsx"


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
    print(f"  Workbook:       {WORKBOOK_PATH}")
    print(f"  DAX-40 Tickers: {len(DAX40_TICKERS)}")
    print(f"  Zeitraum:       {START_DATE} bis {END_DATE}")
    print(f"  Horizon:        {DEFAULT_HORIZON} Jahre")
    print(f"  MC Paths:       {DEFAULT_N_SIMS:,}")
    print("=" * 60)


if __name__ == "__main__":
    show_config()
