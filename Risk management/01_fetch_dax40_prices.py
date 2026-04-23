"""
============================================================================
 01_fetch_dax40_prices.py
 ---------------------------------------------------------------------------
 Zieht tägliche Adjusted-Close-Kurse für alle DAX-40 Mitglieder via yfinance.
 Mit Retry-Logik, Einzelabruf-Fallback, Parquet-Caching.

 Output:
   - DataFrame `dax_prices`: Spalten = Tickers, Index = Datum, Values = Close
   - Cache-Datei: data/cache/dax40_prices.parquet
   - Info-Log: welche Tickers erfolgreich gezogen, welche fehlgeschlagen
============================================================================
"""

import time
import pandas as pd
import yfinance as yf
from pathlib import Path

from config import (
    DAX40_TICKERS,
    START_DATE,
    END_DATE,
    CACHE_DIR,
)


def fetch_dax40_prices(
    tickers: list = None,
    start: str = None,
    end: str = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 12,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Lädt tägliche Kurs-Historie für alle DAX-40 Mitglieder.

    Parameters
    ----------
    tickers : list, optional
        Liste der Ticker. Default: config.DAX40_TICKERS
    start, end : str
        Datum im Format 'YYYY-MM-DD'
    use_cache : bool
        Falls True, nutzt Parquet-Cache wenn jünger als cache_max_age_hours
    cache_max_age_hours : int
        Maximales Alter des Cache in Stunden
    max_retries : int
        Max. Retry-Versuche bei Batch-Download

    Returns
    -------
    pd.DataFrame  (Index=Date, Columns=Tickers, Values=Adj Close)
    """
    tickers = tickers or DAX40_TICKERS
    start = start or START_DATE
    end = end or END_DATE

    cache_file = CACHE_DIR / "dax40_prices.parquet"

    # --- Cache Check --------------------------------------------------
    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_max_age_hours:
            print(f"[CACHE] Lade DAX-40 Kurse aus Cache ({age_hours:.1f}h alt)")
            return pd.read_parquet(cache_file)

    print(f"[FETCH] Lade Kurse für {len(tickers)} Tickers  ({start} bis {end})")

    # --- Batch-Download mit Retries -----------------------------------
    closes_df = None
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                tickers=tickers,
                start=start, end=end,
                auto_adjust=True, progress=False,
                group_by="ticker", threads=False,
            )
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    closes = {}
                    for t in tickers:
                        if t in data.columns.get_level_values(0):
                            sub = data[t]
                            if "Close" in sub.columns:
                                closes[t] = sub["Close"]
                    if closes:
                        closes_df = pd.DataFrame(closes).dropna(how="all")
                else:
                    closes_df = data[["Close"]].rename(columns={"Close": tickers[0]})
                break
        except Exception as e:
            print(f"   ! Batch-Versuch {attempt} fehlgeschlagen: {e}")
        time.sleep(2 ** attempt)

    # --- Fallback: Einzelabruf ----------------------------------------
    if closes_df is None or closes_df.empty or closes_df.shape[1] < len(tickers) * 0.5:
        print("   → Fallback: Einzelabruf pro Ticker...")
        singles = {}
        for t in tickers:
            try:
                df = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
                if not df.empty:
                    singles[t] = df["Close"]
                time.sleep(0.2)
            except Exception as e:
                print(f"   ! {t}: {e}")
        closes_df = pd.DataFrame(singles).dropna(how="all") if singles else pd.DataFrame()

    if closes_df.empty:
        raise RuntimeError("Keine Kursdaten abrufbar — yfinance blockiert?")

    # --- Qualitäts-Report ---------------------------------------------
    n_days = len(closes_df)
    n_tickers = closes_df.shape[1]
    missing = [t for t in tickers if t not in closes_df.columns]
    print(f"[DONE]  {n_days:,} Handelstage × {n_tickers} Ticker geladen")
    if missing:
        print(f"   Fehlend ({len(missing)}): {', '.join(missing)}")

    # --- Cache speichern ----------------------------------------------
    closes_df.to_parquet(cache_file)
    print(f"[CACHE] Gespeichert nach {cache_file}")

    return closes_df


# ======================================================================
# MAIN — Skript eigenständig laufen lassen
# ======================================================================
if __name__ == "__main__":
    dax_prices = fetch_dax40_prices()

    print("\n[PREVIEW] Erste 5 Zeilen × erste 6 Ticker:")
    print(dax_prices.iloc[:5, :6].round(2))

    print("\n[PREVIEW] Letzte 5 Zeilen × erste 6 Ticker:")
    print(dax_prices.iloc[-5:, :6].round(2))

    print(f"\n[INFO] Shape: {dax_prices.shape}")
    print(f"       Von:   {dax_prices.index.min().date()}")
    print(f"       Bis:   {dax_prices.index.max().date()}")
