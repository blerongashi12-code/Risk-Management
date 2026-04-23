"""
============================================================================
 02_fetch_dax40_fundamentals.py
 ---------------------------------------------------------------------------
 Zieht Bilanzdaten (Market Cap, Total Debt, Debt-Maturity-Buckets) für
 alle DAX-40 Mitglieder via yfinance.

 Output:
   - DataFrame `dax_fundamentals`:
       Index = Ticker
       Columns = [Name, MarketCap_EURbn, TotalDebt_EURbn, ShortTermDebt_EURbn,
                  LongTermDebt_EURbn, Sector, Currency, ReportDate]
   - Cache: data/cache/dax40_fundamentals.parquet

 Hinweis:
   - yfinance-Bilanzdaten sind in Unternehmenswährung (meist EUR für DAX,
     aber Qiagen=USD, Airbus=EUR etc.). Wir dokumentieren die Währung.
   - Märkte in Mrd € (bn), Schulden ebenfalls.
   - Bei fehlenden Daten: NaN, keine Exception.
============================================================================
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf

from config import DAX40, DAX40_TICKERS, CACHE_DIR


# Yahoo-Finance Bilanz-Schlüssel (variieren über Tickers)
DEBT_KEYS_PRIORITY = [
    "Total Debt",
    "TotalDebt",
    "Long Term Debt And Capital Lease Obligation",
    "Long Term Debt",
    "LongTermDebt",
]
ST_DEBT_KEYS = ["Current Debt", "Short Long Term Debt", "CurrentDebt"]
LT_DEBT_KEYS = ["Long Term Debt", "LongTermDebt",
                "Long Term Debt And Capital Lease Obligation"]


def _find_first_match(bs: pd.DataFrame, keys: list):
    """Sucht nach dem ersten vorhandenen Key in der Bilanz."""
    if bs is None or bs.empty:
        return np.nan
    for k in keys:
        if k in bs.index:
            val = bs.loc[k].iloc[0]
            if pd.notna(val):
                return float(val)
    return np.nan


def fetch_one_firm(ticker: str, retries: int = 2) -> dict:
    """Holt Bilanzdaten und Marktkapitalisierung für eine einzelne Firma."""
    for attempt in range(1, retries + 1):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            bs = tk.balance_sheet

            mcap = info.get("marketCap", np.nan)
            currency = info.get("financialCurrency", info.get("currency", "EUR"))
            sector = info.get("sector", "n/a")

            total_debt = _find_first_match(bs, DEBT_KEYS_PRIORITY)
            st_debt    = _find_first_match(bs, ST_DEBT_KEYS)
            lt_debt    = _find_first_match(bs, LT_DEBT_KEYS)

            # Falls Total Debt fehlt, aus Komponenten rekonstruieren
            if pd.isna(total_debt):
                if pd.notna(st_debt) and pd.notna(lt_debt):
                    total_debt = st_debt + lt_debt
                elif pd.notna(lt_debt):
                    total_debt = lt_debt

            # Bilanzstichtag
            report_date = bs.columns[0] if bs is not None and not bs.empty else None

            return {
                "Ticker":              ticker,
                "Name":                DAX40.get(ticker, info.get("longName", ticker)),
                "Sector_Yahoo":        sector,
                "Currency":            currency,
                "MarketCap":           mcap,
                "TotalDebt":           total_debt,
                "ShortTermDebt":       st_debt,
                "LongTermDebt":        lt_debt,
                "ReportDate":          report_date,
            }
        except Exception as e:
            if attempt == retries:
                return {
                    "Ticker": ticker,
                    "Name":   DAX40.get(ticker, ticker),
                    "Error":  str(e),
                }
            time.sleep(1.5)


def fetch_dax40_fundamentals(
    tickers: list = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
) -> pd.DataFrame:
    """Zentrale Funktion: Bilanz- und Marktdaten für alle DAX-40."""
    tickers = tickers or DAX40_TICKERS
    cache_file = CACHE_DIR / "dax40_fundamentals.parquet"

    # --- Cache Check ---
    if use_cache and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < cache_max_age_hours:
            print(f"[CACHE] Lade Fundamentaldaten aus Cache ({age_h:.1f}h alt)")
            return pd.read_parquet(cache_file)

    print(f"[FETCH] Bilanzdaten für {len(tickers)} Tickers...")

    results = []
    for i, t in enumerate(tickers, 1):
        print(f"   [{i:>2}/{len(tickers)}]  {t:<10}  ", end="", flush=True)
        rec = fetch_one_firm(t)
        if "Error" in rec:
            print(f"  ! FEHLER: {rec['Error'][:50]}")
        else:
            mcap_bn = rec.get("MarketCap", np.nan)
            debt_bn = rec.get("TotalDebt", np.nan)
            print(
                f"  MCap={mcap_bn/1e9:>7.1f} bn  "
                f"Debt={debt_bn/1e9 if pd.notna(debt_bn) else float('nan'):>7.1f} bn  "
                f"[{rec.get('Currency','?')}]"
            )
        results.append(rec)
        time.sleep(0.3)

    df = pd.DataFrame(results).set_index("Ticker")

    # In Mrd umrechnen (besser lesbar)
    for col in ("MarketCap", "TotalDebt", "ShortTermDebt", "LongTermDebt"):
        if col in df.columns:
            df[col + "_bn"] = df[col] / 1e9

    # --- Qualitäts-Report ---
    ok = df["TotalDebt"].notna().sum() if "TotalDebt" in df else 0
    print(f"\n[DONE]  Erfolgreich: {ok}/{len(tickers)} mit Total Debt verfügbar")

    df.to_parquet(cache_file)
    print(f"[CACHE] Gespeichert nach {cache_file}")

    return df


# ======================================================================
if __name__ == "__main__":
    dax_fundamentals = fetch_dax40_fundamentals()

    print("\n[PREVIEW]  (Werte in Mrd)")
    preview_cols = ["Name", "MarketCap_bn", "TotalDebt_bn",
                    "ShortTermDebt_bn", "LongTermDebt_bn", "Currency"]
    cols = [c for c in preview_cols if c in dax_fundamentals.columns]
    print(dax_fundamentals[cols].head(10).round(2))

    print(f"\n[INFO] Shape: {dax_fundamentals.shape}")
