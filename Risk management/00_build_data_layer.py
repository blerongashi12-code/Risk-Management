"""
============================================================================
 00_build_data_layer.py  ·  MASTER ORCHESTRATOR
 ---------------------------------------------------------------------------
 Ruft alle einzelnen Fetch-Module auf und konsolidiert die Rohdaten in
 einem einzigen Dictionary `data_layer`, das du in weiteren Notebook-Zellen
 verwenden kannst.

 Output:
   data_layer = {
       "dax_prices":        DataFrame,   # aus 01
       "dax_fundamentals":  DataFrame,   # aus 02
       "brent":             DataFrame,   # aus 03
       "svensson":          DataFrame,   # aus 04
       "svensson_stats":    dict,        # historische Stats
       "market":            DataFrame,   # aus 05
       "meta": {
           "build_timestamp": ...,
           "n_firms":         ...,
           "n_trading_days":  ...,
       }
   }

 Verwendung im Jupyter Notebook:
   %run 00_build_data_layer.py
   → data_layer liegt im Namespace

 Oder als Import:
   from build_data_layer import build_all
   data_layer = build_all()
============================================================================
"""

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# Helper: Modul-Import aus nummerierter Datei
# ----------------------------------------------------------------------
def _import_by_path(module_name: str, file_path: Path):
    """Importiert eine .py-Datei, deren Name mit einer Ziffer beginnt
    (Standard-`import` funktioniert dort nicht)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def build_all(use_cache: bool = True, skip_bundesbank_if_missing: bool = False) -> dict:
    """
    Führt alle Fetch-Module aus und konsolidiert die Ergebnisse.

    Parameters
    ----------
    use_cache : bool
        Ob Parquet-Cache genutzt werden soll
    skip_bundesbank_if_missing : bool
        Falls True und Bundesbank-CSV fehlt → weitermachen ohne Abbruch
    """
    print("=" * 70)
    print("DATA LAYER BUILD · DAX Credit Stress Model")
    print("=" * 70)

    data = {"meta": {"build_timestamp": datetime.now()}}

    # ------ 01 · DAX-40 Kurse -----------------------------------------
    print("\n[1/5] DAX-40 PRICES")
    print("-" * 70)
    m1 = _import_by_path("fetch_dax40_prices", BASE / "01_fetch_dax40_prices.py")
    data["dax_prices"] = m1.fetch_dax40_prices(use_cache=use_cache)

    # ------ 02 · Fundamentals -----------------------------------------
    print("\n[2/5] DAX-40 FUNDAMENTALS")
    print("-" * 70)
    m2 = _import_by_path("fetch_dax40_fundamentals",
                         BASE / "02_fetch_dax40_fundamentals.py")
    data["dax_fundamentals"] = m2.fetch_dax40_fundamentals(use_cache=use_cache)

    # ------ 03 · Brent -------------------------------------------------
    print("\n[3/5] BRENT CRUDE")
    print("-" * 70)
    m3 = _import_by_path("fetch_brent_crude", BASE / "03_fetch_brent_crude.py")
    data["brent"] = m3.fetch_brent_prices(use_cache=use_cache)

    # ------ 04 · Svensson ---------------------------------------------
    print("\n[4/5] BUNDESBANK SVENSSON PARAMETERS")
    print("-" * 70)
    m4 = _import_by_path("fetch_bundesbank_svensson",
                         BASE / "04_fetch_bundesbank_svensson.py")
    try:
        data["svensson"] = m4.fetch_svensson_params(use_cache=use_cache)
        data["svensson_stats"] = m4.compute_historical_stats(data["svensson"])
    except FileNotFoundError as e:
        if skip_bundesbank_if_missing:
            print(f"⚠️  Übersprungen: {e}")
            data["svensson"] = None
            data["svensson_stats"] = None
        else:
            raise

    # ------ 05 · Market Proxy -----------------------------------------
    print("\n[5/5] MARKET PROXY")
    print("-" * 70)
    m5 = _import_by_path("fetch_market_proxy", BASE / "05_fetch_market_proxy.py")
    data["market"] = m5.fetch_market_proxy(use_cache=use_cache)

    # ------ Meta-Info -------------------------------------------------
    data["meta"]["n_firms"] = data["dax_prices"].shape[1]
    data["meta"]["n_trading_days"] = len(data["dax_prices"])
    data["meta"]["date_range"] = (data["dax_prices"].index.min(),
                                  data["dax_prices"].index.max())

    # ------ Summary ---------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"  DAX-40 Prices:      {data['dax_prices'].shape}")
    print(f"  DAX-40 Fundamentals:{data['dax_fundamentals'].shape}")
    print(f"  Brent:              {data['brent'].shape}")
    if data["svensson"] is not None:
        print(f"  Svensson:           {data['svensson'].shape}")
    print(f"  Market:             {data['market'].shape}")
    print(f"  Timestamp:          {data['meta']['build_timestamp']:%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    return data


# ======================================================================
if __name__ == "__main__":
    data_layer = build_all()

    print("\n▶ data_layer Keys:")
    for k, v in data_layer.items():
        if isinstance(v, pd.DataFrame):
            print(f"   {k:<22}  DataFrame {v.shape}")
        elif isinstance(v, dict):
            print(f"   {k:<22}  dict (len={len(v)})")
        else:
            print(f"   {k:<22}  {type(v).__name__}")
