"""
============================================================================
 05_fetch_market_proxy.py
 ---------------------------------------------------------------------------
 Zieht DAX-Performance-Index (^GDAXI) als Markt-Proxy für Faktor-Beta
 Schätzung der Aktienrenditen (Marktfaktor im CAPM-/Mehrfaktor-Modell).

 Output:
   - DataFrame `market_prices`:
       Index   = Date
       Columns = [Close, Return_log, Volatility_30d_ann]
   - Cache: data/cache/market_proxy.parquet

 Hinweis:
   - Ursprünglich war MSCI World (MXWO) in deiner BB_Betas Vorarbeit dabei.
   - Für ein reines DAX-Kreditmodell ist ^GDAXI (der DAX selbst) natürlicher.
   - Falls du MSCI World bevorzugst: einfach MARKET_PROXY in config.py ändern.
============================================================================
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf

from config import MARKET_PROXY, START_DATE, END_DATE, CACHE_DIR


def fetch_market_proxy(
    ticker: str = None,
    start: str = None,
    end: str = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 12,
) -> pd.DataFrame:
    """Lädt Markt-Index mit Returns und rollender Vola."""
    ticker = ticker or MARKET_PROXY
    start = start or START_DATE
    end = end or END_DATE

    cache_file = CACHE_DIR / "market_proxy.parquet"

    if use_cache and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < cache_max_age_hours:
            print(f"[CACHE] Lade Market-Proxy aus Cache ({age_h:.1f}h alt)")
            return pd.read_parquet(cache_file)

    print(f"[FETCH] Market Proxy ({ticker})  {start} bis {end}")

    raw = None
    for attempt in range(1, 4):
        try:
            raw = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            print(f"   ! Versuch {attempt}: {e}")
        time.sleep(2 ** attempt)

    if raw is None or raw.empty:
        raise RuntimeError(f"Market-Proxy {ticker} nicht abrufbar")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][ticker] if ticker in raw["Close"].columns else raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    df = pd.DataFrame({"Close": close}).dropna()
    df["Return_log"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_30d_ann"] = df["Return_log"].rolling(30).std() * np.sqrt(252)

    print(f"[DONE]  {len(df):,} Handelstage")
    print(f"        Mean Return: {df['Return_log'].mean()*252*100:+.2f}% p.a.")
    print(f"        Vol:         {df['Return_log'].std()*np.sqrt(252)*100:.2f}% p.a.")
    print(f"        Latest:      {df['Close'].iloc[-1]:,.2f}")

    df.to_parquet(cache_file)
    print(f"[CACHE] Gespeichert nach {cache_file}")

    return df


# ======================================================================
if __name__ == "__main__":
    market_prices = fetch_market_proxy()
    print("\n[PREVIEW] Letzte 10 Handelstage:")
    print(market_prices.tail(10).round(4))
