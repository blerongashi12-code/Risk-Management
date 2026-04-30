"""Cached wrappers around the data layer.

Slim version after the Tier-1 cleanup — only the macro-factor cache
(Brent + Bundesbank-Svensson) and the Svensson curve helpers remain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components.backend_path import setup
setup()

from config import CACHE_DIR  # type: ignore


# ----------------------------------------------------------------------
# 1. Static cache (Brent + Svensson only)
# ----------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner="Loading data layer …")
def load_data_layer() -> dict:
    """Reads Brent + Svensson parquet files from data/cache/."""
    out = {}
    needed = {
        "brent":    "brent_crude.parquet",
        "svensson": "svensson_params.parquet",
    }
    for key, fname in needed.items():
        path = CACHE_DIR / fname
        out[key] = pd.read_parquet(path) if path.exists() else None
    return out


# ----------------------------------------------------------------------
# 2. Δr_10y from Svensson β-shifts
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
def delta_r_from_beta_shifts(
    d_beta0: float, d_beta1: float, d_beta2: float, d_beta3: float,
    *, maturity: float = 10.0,
) -> float:
    """Computes Δr_10y in pp from the four β-shifts using the latest curve."""
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
    """Current Bundesbank zero curve over a maturity grid."""
    from svensson import historical_curve, curve_grid  # type: ignore

    data = load_data_layer()
    if data["svensson"] is None:
        return pd.DataFrame()
    base_params = historical_curve(data["svensson"].index[-1], data["svensson"])
    mats = np.array(maturities) if maturities else np.arange(0.25, 30.25, 0.25)
    rates = curve_grid(base_params, maturities=mats)
    return pd.DataFrame({"maturity": rates.index, "rate_pct": rates.values})
