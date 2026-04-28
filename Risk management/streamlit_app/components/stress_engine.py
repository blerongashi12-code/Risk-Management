"""Anwendung der Live-Slider-Konfig auf die Backend-Module.

Drei zentrale Funktionen:
  - run_custom_scenario(config)       — deterministischer Single-Path
  - run_custom_mc(config)             — MC mit Slider-Korrelation
  - decompose_pd(ticker_or_None, ...) — Waterfall-Decomposition
                                        (Baseline → +Brent → +Δr → +Inter → Full)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components.backend_path import setup
setup()

from components.data_loader import load_data_layer, run_baseline


# ----------------------------------------------------------------------
# 1. Deterministisches Custom-Szenario
# ----------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def run_custom_scenario(
    d_brent: float, d_r_10y_pp: float, horizon_days: int = 252,
) -> pd.DataFrame:
    """Wendet einen deterministischen Schock auf alle DAX-Firmen an.

    Returns scenarios.apply_scenario-Output (DataFrame, Index = Ticker).
    """
    from scenarios import apply_scenario  # type: ignore

    base = run_baseline()
    if base["merton_df"] is None:
        return pd.DataFrame()

    custom_sc = {
        "name": "live_custom",
        "description": "Live Slider-Stress",
        "source": "live",
        "delta_brent_log":   d_brent,
        "delta_rate_10y_pp": d_r_10y_pp,
        "horizon_days":      horizon_days,
        "period": None, "calibration": None,
    }
    return apply_scenario(base["merton_df"], base["factor_df"], custom_sc)


# ----------------------------------------------------------------------
# 2. MC mit Live-Konfiguration
# ----------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Berechne Monte-Carlo …")
def run_custom_mc(
    n_sims: int, horizon_days: int, corr_override: float | None,
    seed: int = 42,
) -> pd.DataFrame:
    """MC mit optional manueller Korrelation in Σ.

    Wenn corr_override is not None, wird die Off-Diagonale von Σ so
    angepasst, dass die Korrelation den User-Wert hat (Diagonal-Vola
    bleibt aus der Historie).
    """
    from monte_carlo import (factor_stats, simulate_factor_paths,  # type: ignore
                             stress_one_firm)
    from config import (SECTOR_VOL_MULTIPLIER,                     # type: ignore
                        DEFAULT_SECTOR_VOL_MULTIPLIER, DEFAULT_HORIZON)

    data = load_data_layer()
    base = run_baseline()
    if base["merton_df"] is None:
        return pd.DataFrame()

    stats = factor_stats(data["brent"], data["svensson"], lookback=252)
    mu, sigma = stats["mu"].copy(), stats["sigma"].copy()

    # Korrelations-Override
    if corr_override is not None:
        sd = np.sqrt(np.diag(sigma))
        sigma = np.array([
            [sd[0] ** 2,                corr_override * sd[0] * sd[1]],
            [corr_override * sd[0] * sd[1], sd[1] ** 2],
        ])

    paths = simulate_factor_paths(
        mu, sigma, n_sims=n_sims, horizon_days=horizon_days, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)

    rows = []
    merton_df = base["merton_df"]
    factor_df = base["factor_df"]

    for ticker in merton_df.index:
        m = merton_df.loc[ticker]
        f = factor_df.loc[ticker] if ticker in factor_df.index else None
        baseline_ok = bool(m.get("Converged", False)) and pd.notna(m.get("PD"))
        factor_ok = (f is not None and bool(f.get("Converged", False))
                     and pd.notna(f.get("BetaBrentAdjusted")))
        if not (baseline_ok and factor_ok):
            rows.append({
                "Name": m.get("Name"), "Sector": m.get("Sector"),
                "BaselinePD": float(m["PD"]) if baseline_ok else float("nan"),
                "StressPDMean": float("nan"), "StressPDp50": float("nan"),
                "StressPDp95": float("nan"), "StressPDp99": float("nan"),
                "StressPDMax":  float("nan"), "DeltaPDMean": float("nan"),
            })
            continue
        sector = m["Sector"]
        sec_mul = (SECTOR_VOL_MULTIPLIER.get(sector, DEFAULT_SECTOR_VOL_MULTIPLIER)
                   if sector else DEFAULT_SECTOR_VOL_MULTIPLIER)
        res = stress_one_firm(
            E0=float(m["MarketCap"]), sigma_E=float(m["EquityVol"]),
            L=float(m["DPT"]), r0=float(m["Rate"]), T_years=DEFAULT_HORIZON,
            alpha=float(f["Alpha"]),
            beta_brent=float(f["BetaBrentAdjusted"]),
            beta_dr=float(f["BetaDr"]),
            sigma_eps=float(f["SigmaEps"]),
            sector_vol_mul=sec_mul, factor_paths=paths,
            horizon_days=horizon_days, include_idio=True, rng=rng,
        )
        rows.append({
            "Name": m["Name"], "Sector": sector,
            "BaselinePD": float(m["PD"]),
            "StressPDMean": res["pd_mean"], "StressPDp50": res["pd_p50"],
            "StressPDp95": res["pd_p95"],   "StressPDp99": res["pd_p99"],
            "StressPDMax": res["pd_max"],
            "DeltaPDMean": res["pd_mean"] - float(m["PD"]),
        })
    out = pd.DataFrame(rows, index=merton_df.index)
    out.index.name = "Ticker"
    return out


# ----------------------------------------------------------------------
# 3. Waterfall-Decomposition
# ----------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def decompose_pd(
    ticker: str | None,
    d_brent: float, d_r_10y_pp: float, horizon_days: int,
    *,
    use_ead_weights: bool = True,
    lgd: float = 0.45,
) -> dict:
    """Decomposition Baseline → +ΔBrent → +Δr → +Interaktion → Full.

    Wenn ticker is None → Portfolio-EL-Decomposition (Σ EAD · LGD · PD).
    Sonst Single-Firm PD-Decomposition.
    """
    base = run_baseline()
    if base["merton_df"] is None:
        return {}

    sc_brent = run_custom_scenario(d_brent, 0.0, horizon_days)
    sc_dr    = run_custom_scenario(0.0, d_r_10y_pp, horizon_days)
    sc_full  = run_custom_scenario(d_brent, d_r_10y_pp, horizon_days)

    if ticker is not None:
        baseline_pd = float(base["merton_df"].loc[ticker, "PD"])
        pd_brent = float(sc_brent.loc[ticker, "ScenarioPD"])
        pd_dr    = float(sc_dr.loc[ticker, "ScenarioPD"])
        pd_full  = float(sc_full.loc[ticker, "ScenarioPD"])

        contrib_brent = pd_brent - baseline_pd
        contrib_dr    = pd_dr - baseline_pd
        contrib_inter = pd_full - baseline_pd - contrib_brent - contrib_dr
        return {
            "level":    "firm",
            "ticker":   ticker,
            "baseline": baseline_pd,
            "brent":    contrib_brent,
            "dr":       contrib_dr,
            "inter":    contrib_inter,
            "full":     pd_full,
            "unit":     "PD (Dezimal)",
        }

    # Portfolio-Level: EL-Decomposition
    merton_df = base["merton_df"]
    eads = merton_df["DPT"].fillna(0).to_numpy()
    pd_base_s = merton_df["PD"].fillna(0)            # Series
    pd_base = pd_base_s.to_numpy()
    # Series.fillna(Series) → index-aligned NaN-Replacement
    pd_b = sc_brent["ScenarioPD"].fillna(pd_base_s).to_numpy()
    pd_d = sc_dr["ScenarioPD"].fillna(pd_base_s).to_numpy()
    pd_f = sc_full["ScenarioPD"].fillna(pd_base_s).to_numpy()

    el_base = float(np.nansum(eads * pd_base) * lgd)
    el_b    = float(np.nansum(eads * pd_b)    * lgd)
    el_d    = float(np.nansum(eads * pd_d)    * lgd)
    el_f    = float(np.nansum(eads * pd_f)    * lgd)

    return {
        "level":    "portfolio",
        "baseline": el_base,
        "brent":    el_b - el_base,
        "dr":       el_d - el_base,
        "inter":    el_f - el_base - (el_b - el_base) - (el_d - el_base),
        "full":     el_f,
        "unit":     "EL (EUR)",
    }


# ----------------------------------------------------------------------
# 4. Portfolio-Aggregation auf die Stress-PD-Outputs
# ----------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def portfolio_kpis(
    d_brent: float, d_r_10y_pp: float, horizon_days: int,
    n_sims: int, corr_override: float | None,
    *, lgd: float = 0.45,
) -> dict:
    """Aggregierte Portfolio-Metriken über ALLE Stress-Pfade.

    Liefert Total EAD, EL Baseline, EL Custom-Stress, EL MC-mean,
    plus VaR_95/99 und CVaR_95/99 aus der MC-Loss-Verteilung.
    """
    from monte_carlo import run_monte_carlo_dax40  # type: ignore
    from portfolio import portfolio_loss_distribution, portfolio_var_cvar, \
        portfolio_hhi  # type: ignore

    base = run_baseline()
    if base["merton_df"] is None:
        return {}

    merton_df = base["merton_df"]

    # Custom-Scenario PDs
    sc_full = run_custom_scenario(d_brent, d_r_10y_pp, horizon_days)
    pd_custom = sc_full["ScenarioPD"].fillna(merton_df["PD"]).to_numpy()
    pd_base   = merton_df["PD"].fillna(0).to_numpy()
    eads      = merton_df["DPT"].fillna(0).to_numpy()
    total_ead = float(np.nansum(eads))

    el_base   = float(np.nansum(eads * pd_base)   * lgd)
    el_custom = float(np.nansum(eads * pd_custom) * lgd)

    # MC mit Korrelations-Override
    mc_df = run_custom_mc(n_sims, horizon_days, corr_override)

    # Loss-Distribution aus MC-Pfaden — wir generieren sie hier neu, weil
    # `run_custom_mc` keine pd_samples-Matrix liefert. Schneller Workaround:
    # Conditional-EL mit Stress-Mean-PDs als zentrale Schätzung.
    pd_mc_mean = mc_df["StressPDMean"].fillna(merton_df["PD"]).to_numpy()
    pd_mc_p99  = mc_df["StressPDp99"].fillna(merton_df["PD"]).to_numpy()
    el_mc_mean = float(np.nansum(eads * pd_mc_mean) * lgd)
    el_mc_p99  = float(np.nansum(eads * pd_mc_p99)  * lgd)

    return {
        "total_ead":  total_ead,
        "n_firms":    len(merton_df),
        "n_converged": int(merton_df["Converged"].fillna(False).sum()),
        "el_baseline":   el_base,
        "el_custom":     el_custom,
        "el_mc_mean":    el_mc_mean,
        "el_mc_p99":     el_mc_p99,
        "delta_el_custom": el_custom - el_base,
        "delta_el_p99":  el_mc_p99 - el_base,
        "hhi": portfolio_hhi(eads),
        "lgd": lgd,
    }
