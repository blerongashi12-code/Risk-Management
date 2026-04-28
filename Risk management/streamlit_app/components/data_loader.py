"""Cached Wrapper rund um die Backend-Module.

Streamlit ruft diese Funktionen bei jedem Re-Run auf; @st.cache_data
verhindert Doppel-Berechnung. TTL = 24h, weil Daten täglich aktualisiert
werden (Bundesbank + yfinance).
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from components.backend_path import setup
setup()

from config import CACHE_DIR  # type: ignore


# ----------------------------------------------------------------------
# 1. Statische Daten aus dem Cache
# ----------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner="Lade Daten-Layer …")
def load_data_layer() -> dict:
    """Liest alle Parquet-Files aus data/cache/ und gibt sie als Dict zurück.

    Keys: 'prices', 'brent', 'svensson', 'fundamentals', 'market'.
    """
    out = {}
    needed = {
        "prices":       "dax40_prices.parquet",
        "brent":        "brent_crude.parquet",
        "svensson":     "svensson_params.parquet",
        "fundamentals": "dax40_fundamentals.parquet",
        "market":       "market_proxy.parquet",
    }
    for key, fname in needed.items():
        path = CACHE_DIR / fname
        if path.exists():
            out[key] = pd.read_parquet(path)
        else:
            out[key] = None
    return out


# ----------------------------------------------------------------------
# 2. Baseline (Merton + Faktor-Modell)
# ----------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner="Berechne Baseline-Merton …")
def run_baseline() -> dict:
    """Lädt Daten und führt Merton + Faktor-Modell aus.

    Returns
    -------
    dict mit
        merton_df : Output von merton.run_dax40
        factor_df : Output von factor_model.run_dax40
        as_of     : letzter Preis-Tag (Timestamp)
    """
    from merton import run_dax40 as run_merton  # type: ignore
    from factor_model import run_dax40 as run_factor  # type: ignore

    data = load_data_layer()
    if any(data[k] is None for k in ("prices", "brent", "svensson", "fundamentals")):
        return {"merton_df": None, "factor_df": None, "as_of": None}

    merton_df = run_merton(data["prices"], data["fundamentals"], data["svensson"])
    factor_df = run_factor(data["prices"], data["brent"], data["svensson"],
                           data["fundamentals"])
    as_of = data["prices"].index.max()
    return {"merton_df": merton_df, "factor_df": factor_df, "as_of": as_of}


# ----------------------------------------------------------------------
# 3. Szenario-Library (cached, da deterministisch)
# ----------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner="Berechne Szenario-Library …")
def run_scenarios() -> dict:
    from scenarios import run_scenario_library  # type: ignore
    data = load_data_layer()
    return run_scenario_library(
        data["prices"], data["brent"], data["svensson"], data["fundamentals"],
    )


# ----------------------------------------------------------------------
# 4. Δr-10y aus Svensson-β-Shifts
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
def delta_r_from_beta_shifts(
    d_beta0: float, d_beta1: float, d_beta2: float, d_beta3: float,
    *, maturity: float = 10.0,
) -> float:
    """Berechnet Δr_10y in Prozentpunkten aus den vier β-Shifts.

    Nutzt den letzten verfügbaren Svensson-Tag als Basis.
    """
    from svensson import (historical_curve, shift_curve, zero_rate)  # type: ignore

    data = load_data_layer()
    if data["svensson"] is None:
        return 0.0

    base_params = historical_curve(data["svensson"].index[-1], data["svensson"])
    shifted = shift_curve(
        base_params,
        dlevel=d_beta0, dslope=d_beta1, dcurv1=d_beta2, dcurv2=d_beta3,
    )
    r_base = zero_rate(maturity, base_params)
    r_new  = zero_rate(maturity, shifted)
    return float(r_new - r_base)


@st.cache_data(ttl=3600)
def baseline_yield_curve(maturities: tuple = None) -> pd.DataFrame:
    """Liefert die aktuelle Bundesbank-Zinskurve über ein Maturity-Grid."""
    from svensson import historical_curve, curve_grid  # type: ignore

    data = load_data_layer()
    if data["svensson"] is None:
        return pd.DataFrame()
    base_params = historical_curve(data["svensson"].index[-1], data["svensson"])
    mats = np.array(maturities) if maturities else np.arange(0.25, 30.25, 0.25)
    rates = curve_grid(base_params, maturities=mats)
    return pd.DataFrame({"maturity": rates.index, "rate_pct": rates.values})
