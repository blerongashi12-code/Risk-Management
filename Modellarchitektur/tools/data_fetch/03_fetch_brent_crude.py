"""
============================================================================
 03_fetch_brent_crude.py
 ---------------------------------------------------------------------------
 Zieht ICE Brent Crude Oil Futures Preise (BZ=F) als Energy-Faktor.
 Dient als Input für Faktor-Beta-Schätzung und Monte-Carlo Shock-Kalibrierung.

 Output:
   - DataFrame `brent_prices`:
       Index = Date
       Columns = [Close_USD, Return_log, Volatility_30d_ann]
   - Cache: data/cache/brent_crude.parquet

 Warum Brent und nicht TTF Gas?
   - TTF (Niederlande Gas Spot) ist nicht via yfinance verfügbar
   - Brent ist globaler Energiepreis-Proxy, hoch korreliert mit europ. Energie
   - Für 2022-Energiekrise zeigt Brent den Shock ebenfalls deutlich
============================================================================
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf

# Pfad-Bootstrap: Skript liegt in tools/data_fetch/ — config.py am App-Root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import ENERGY_PROXY, START_DATE, END_DATE, CACHE_DIR


def fetch_brent_prices(
    start: str = None,
    end: str = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 12,
) -> pd.DataFrame:
    """
    Lädt Brent-Preis-Historie und berechnet Returns + rolling Vola.

    Returns
    -------
    pd.DataFrame mit Spalten:
        - Close_USD:           Schlusskurs in USD
        - Return_log:          Tägliche Log-Rendite
        - Volatility_30d_ann:  30-Tage rollende annualisierte Volatilität
    """
    start = start or START_DATE
    end = end or END_DATE

    cache_file = CACHE_DIR / "brent_crude.parquet"

    if use_cache and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < cache_max_age_hours:
            print(f"[CACHE] Lade Brent aus Cache ({age_h:.1f}h alt)")
            return pd.read_parquet(cache_file)

    print(f"[FETCH] Brent Crude ({ENERGY_PROXY})  {start} bis {end}")

    # --- Retry Download ------------------------------------------------
    raw = None
    for attempt in range(1, 4):
        try:
            raw = yf.download(
                ENERGY_PROXY,
                start=start, end=end,
                auto_adjust=True, progress=False,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            print(f"   ! Versuch {attempt}: {e}")
        time.sleep(2 ** attempt)

    if raw is None or raw.empty:
        raise RuntimeError("Brent-Daten nicht abrufbar")

    # --- DataFrame zurechtschneiden -----------------------------------
    # yfinance kann MultiIndex oder Flat columns zurückgeben
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][ENERGY_PROXY] if ENERGY_PROXY in raw["Close"].columns else raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    df = pd.DataFrame({"Close_USD": close}).dropna()

    # Returns & Volatilität
    df["Return_log"] = np.log(df["Close_USD"] / df["Close_USD"].shift(1))
    df["Volatility_30d_ann"] = df["Return_log"].rolling(30).std() * np.sqrt(252)

    print(f"[DONE]  {len(df):,} Handelstage")
    print(f"        Mean Return:    {df['Return_log'].mean()*252*100:+.2f}% p.a.")
    print(f"        Vol (gesamt):   {df['Return_log'].std()*np.sqrt(252)*100:.2f}% p.a.")
    print(f"        Latest Price:   ${df['Close_USD'].iloc[-1]:.2f}")

    df.to_parquet(cache_file)
    print(f"[CACHE] Gespeichert nach {cache_file}")

    return df


# ======================================================================
if __name__ == "__main__":
    brent_prices = fetch_brent_prices()

    print("\n[PREVIEW] Letzte 10 Handelstage:")
    print(brent_prices.tail(10).round(4))

    # Krisenmomente identifizieren
    print("\n[ANALYSE] Top-5 tägliche Spikes (absolute Returns):")
    spikes = brent_prices["Return_log"].abs().nlargest(5)
    for d, v in spikes.items():
        price = brent_prices.loc[d, "Close_USD"]
        ret = brent_prices.loc[d, "Return_log"]
        print(f"   {d.date()}  Ret={ret*100:+6.2f}%  Price=${price:.2f}")
