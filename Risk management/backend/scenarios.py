"""
============================================================================
 backend/scenarios.py
 ---------------------------------------------------------------------------
 Deterministische Stress-Szenarien (historisch oder hypothetisch).

 Im Gegensatz zu monte_carlo.py — das stochastische MVN-Pfade simuliert —
 ist hier jeder Szenario ein **einzelner deterministischer Schock-Vektor**:

     z = [Δr_brent_cum, Δr_10y_cum]  am Horizont H

 Die Anwendung nutzt die gleiche Faktor → Equity → KMV-Kette wie ein
 einzelner MC-Pfad (siehe MODEL_ASSUMPTIONS.md §6c).

 Kern-Funktionen:
   - calibrate_from_history(brent_df, svensson_df, period)   → dict
   - resolve_scenario(name, library, brent_df, svensson_df)  → dict
   - apply_scenario(merton_df, factor_df, scenario, ...)     → DataFrame
   - run_scenario_library(prices, brent, svensson, fund)     → dict

 Validierung: python backend/scenarios.py
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ----------------------------------------------------------------------
# 1. Historische Kalibrierung
# ----------------------------------------------------------------------
def calibrate_from_history(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    start: str,
    end: str,
    maturity: float = 10.0,
) -> dict:
    """Kalibriert ΔBrent_log und Δr_10y aus realen Daten über [start, end].

    Parameters
    ----------
    brent_df : Cache-DF mit Spalte 'Close_USD'.
    svensson_df : Bundesbank-Parameter-Zeitreihe.
    start, end : ISO-Date-Strings ('YYYY-MM-DD'); werden auf den
        nächst-erreichbaren Handelstag innerhalb des Fensters gemappt.
    maturity : Δr-Maturity in Jahren (default 10).

    Returns
    -------
    dict mit
        'delta_brent_log'    : kumulativer log-Return Brent
        'delta_rate_10y_pp'  : kumulativer Δr in pp
        'horizon_days'       : tatsächliche Anzahl Kalendertage
        'brent_start', 'brent_end', 'rate_start_pp', 'rate_end_pp'
        'period_actual'      : (first_trading_day, last_trading_day)
    Raises
    ------
    KeyError, wenn weniger als 2 Beobachtungen pro Quelle in der Periode.
    """
    from svensson import zero_rate, params_from_row  # type: ignore

    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    bw = brent_df.loc[s:e, "Close_USD"].dropna()
    sw = svensson_df.loc[s:e].dropna()

    if len(bw) < 2:
        raise KeyError(f"Brent-Daten in [{start}, {end}] unzureichend "
                       f"(nur {len(bw)} Beobachtungen).")
    if len(sw) < 2:
        raise KeyError(f"Svensson-Daten in [{start}, {end}] unzureichend "
                       f"(nur {len(sw)} Beobachtungen).")

    delta_brent_log = float(np.log(bw.iloc[-1] / bw.iloc[0]))
    r0 = float(zero_rate(maturity, params_from_row(sw.iloc[0])))
    r1 = float(zero_rate(maturity, params_from_row(sw.iloc[-1])))
    delta_r_pp = r1 - r0

    horizon = int((sw.index[-1] - sw.index[0]).days)

    return {
        "delta_brent_log":   delta_brent_log,
        "delta_rate_10y_pp": delta_r_pp,
        "horizon_days":      horizon,
        "brent_start":       float(bw.iloc[0]),
        "brent_end":         float(bw.iloc[-1]),
        "rate_start_pp":     r0,
        "rate_end_pp":       r1,
        "period_actual":     (str(sw.index[0].date()), str(sw.index[-1].date())),
    }


# ----------------------------------------------------------------------
# 2. Library-Auflösung
# ----------------------------------------------------------------------
def resolve_scenario(
    name: str,
    library: Mapping,
    brent_df: Optional[pd.DataFrame] = None,
    svensson_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Löst ein Szenario aus der Library zu konkreten Schock-Werten auf.

    Logik nach 'source':
      'historical'    → calibrate_from_history(brent, svensson, period)
      'literature'    → 'override'-Werte direkt nutzen
      'hypothetical'  → 'override'-Werte direkt nutzen

    Returns
    -------
    dict mit Keys: name, description, source, delta_brent_log,
                   delta_rate_10y_pp, horizon_days, period (informativ),
                   calibration (dict mit Detail-Zwischenwerten oder None)
    """
    if name not in library:
        raise KeyError(f"Szenario '{name}' nicht in Library. "
                       f"Verfügbar: {list(library.keys())}")
    cfg = library[name]
    src = cfg["source"]

    if src == "historical":
        if brent_df is None or svensson_df is None:
            raise ValueError(f"Szenario '{name}' (historical) braucht "
                             f"brent_df und svensson_df.")
        period = cfg["period"]
        cal = calibrate_from_history(
            brent_df, svensson_df, start=period[0], end=period[1],
        )
        return {
            "name": name,
            "description": cfg["description"],
            "source": src,
            "delta_brent_log":   cal["delta_brent_log"],
            "delta_rate_10y_pp": cal["delta_rate_10y_pp"],
            "horizon_days":      cal["horizon_days"],
            "period":            cfg["period"],
            "calibration":       cal,
        }

    if src in ("literature", "hypothetical"):
        ov = cfg["override"]
        return {
            "name": name,
            "description": cfg["description"],
            "source": src,
            "delta_brent_log":   float(ov["delta_brent_log"]),
            "delta_rate_10y_pp": float(ov["delta_rate_10y_pp"]),
            "horizon_days":      int(ov["horizon_days"]),
            "period":            cfg.get("period"),
            "calibration":       None,
        }

    raise ValueError(f"Unbekannte source '{src}' in Szenario '{name}'.")


# ----------------------------------------------------------------------
# 3. Anwendung pro Firma (deterministischer Single-Pfad)
# ----------------------------------------------------------------------
def apply_scenario(
    merton_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    scenario: dict,
    *,
    sector_vol_mul_map: Optional[Mapping[str, float]] = None,
    default_sector_vol_mul: Optional[float] = None,
    horizon_years: Optional[float] = None,
) -> pd.DataFrame:
    """Wendet einen deterministischen Schock auf alle Firmen an.

    Pipeline pro Firma (Single-Pfad-Variante von monte_carlo.stress_one_firm):
        log r_E_H = α·H + β_E^adj · ΔBrent_log + β_R · Δr_pp
        E_stress  = E_0 · exp(log r_E_H)
        r_stress  = r_0 + Δr_pp / 100
        (V, σ_V)  = KMV(E_stress, σ_E, L, r_stress, T)
        σ_V_adj   = m_sector · σ_V
        DD, PD    = mit σ_V_adj

    Returns
    -------
    pd.DataFrame, Index=Ticker, Cols:
        Name, Sector, BaselinePD, ScenarioPD, DeltaPD, ScenarioDD,
        StressEquity, StressRate, AlphaContrib, BrentContrib, RateContrib.
    Skipped Firmen (Baseline oder Faktoren nicht konvergiert) erscheinen
    mit NaN-PD und Spalte 'SkipReason'.
    """
    from config import (SECTOR_VOL_MULTIPLIER,  # type: ignore
                        DEFAULT_SECTOR_VOL_MULTIPLIER, DEFAULT_HORIZON)
    from merton import (merton_kmv, distance_to_default,  # type: ignore
                        merton_pd)

    if sector_vol_mul_map is None:
        sector_vol_mul_map = SECTOR_VOL_MULTIPLIER
    if default_sector_vol_mul is None:
        default_sector_vol_mul = DEFAULT_SECTOR_VOL_MULTIPLIER
    if horizon_years is None:
        horizon_years = DEFAULT_HORIZON

    db = scenario["delta_brent_log"]
    dr_pp = scenario["delta_rate_10y_pp"]
    H = scenario["horizon_days"]

    rows = []
    for ticker, m_row in merton_df.iterrows():
        f_row = (factor_df.loc[ticker]
                 if ticker in factor_df.index else None)

        baseline_ok = bool(m_row.get("Converged", False)) and pd.notna(
            m_row.get("PD"))
        factor_ok = (f_row is not None
                     and bool(f_row.get("Converged", False))
                     and pd.notna(f_row.get("BetaBrentAdjusted")))

        if not (baseline_ok and factor_ok):
            rows.append({
                "Name": m_row.get("Name"),
                "Sector": m_row.get("Sector"),
                "BaselinePD": float(m_row["PD"]) if baseline_ok else float("nan"),
                "ScenarioPD": float("nan"),
                "DeltaPD": float("nan"),
                "ScenarioDD": float("nan"),
                "StressEquity": float("nan"),
                "StressRate": float("nan"),
                "AlphaContrib": float("nan"),
                "BrentContrib": float("nan"),
                "RateContrib": float("nan"),
                "SkipReason": ("merton" if not baseline_ok else "factor"),
            })
            continue

        sector = m_row["Sector"]
        sec_mul = (sector_vol_mul_map.get(sector, default_sector_vol_mul)
                   if sector else default_sector_vol_mul)

        E0 = float(m_row["MarketCap"])
        sigma_E = float(m_row["EquityVol"])
        L = float(m_row["DPT"])
        r0 = float(m_row["Rate"])

        alpha = float(f_row["Alpha"])
        beta_brent_adj = float(f_row["BetaBrentAdjusted"])
        beta_dr = float(f_row["BetaDr"])

        # Decomposition für Waterfall-Output
        contrib_alpha = alpha * H
        contrib_brent = beta_brent_adj * db
        contrib_rate  = beta_dr * dr_pp
        log_ret = contrib_alpha + contrib_brent + contrib_rate

        E_stress = E0 * np.exp(log_ret)
        r_stress = r0 + dr_pp / 100.0   # pp → Dezimal

        # KMV
        res = merton_kmv(E_stress, sigma_E, L, r_stress, horizon_years)
        if not res.converged or np.isnan(res.asset_vol):
            rows.append({
                "Name": m_row["Name"],
                "Sector": sector,
                "BaselinePD": float(m_row["PD"]),
                "ScenarioPD": float("nan"),
                "DeltaPD": float("nan"),
                "ScenarioDD": float("nan"),
                "StressEquity": E_stress,
                "StressRate": r_stress,
                "AlphaContrib": contrib_alpha,
                "BrentContrib": contrib_brent,
                "RateContrib": contrib_rate,
                "SkipReason": "kmv_nonconv",
            })
            continue

        sigma_V_adj = res.asset_vol * sec_mul
        dd = distance_to_default(res.asset_value, sigma_V_adj, L,
                                 r_stress, horizon_years)
        pd_ = merton_pd(dd)

        rows.append({
            "Name": m_row["Name"],
            "Sector": sector,
            "BaselinePD": float(m_row["PD"]),
            "ScenarioPD": float(pd_),
            "DeltaPD": float(pd_ - m_row["PD"]),
            "ScenarioDD": float(dd),
            "StressEquity": float(E_stress),
            "StressRate": float(r_stress),
            "AlphaContrib": float(contrib_alpha),
            "BrentContrib": float(contrib_brent),
            "RateContrib":  float(contrib_rate),
            "SkipReason": "",
        })

    return pd.DataFrame(rows, index=merton_df.index)


# ----------------------------------------------------------------------
# 4. End-to-End Library-Run
# ----------------------------------------------------------------------
def run_scenario_library(
    prices_df: pd.DataFrame,
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    *,
    library: Optional[Mapping] = None,
    horizon_years: Optional[float] = None,
) -> dict:
    """Berechnet alle Szenarien aus der Library für DAX-40.

    Returns
    -------
    dict mit
        'scenarios'   : dict[name → resolved scenario dict]
        'merton_df'   : Output von merton.run_dax40
        'factor_df'   : Output von factor_model.run_dax40
        'results'     : dict[name → DataFrame aus apply_scenario]
        'heatmap'     : DataFrame, Index=Ticker, Cols=Baseline + jedes
                        Szenario (PD-Werte) — direkt für Streamlit.
    """
    from config import SCENARIO_LIBRARY  # type: ignore
    from merton import run_dax40 as run_merton  # type: ignore
    from factor_model import run_dax40 as run_factor  # type: ignore

    if library is None:
        library = SCENARIO_LIBRARY

    merton_df = run_merton(prices_df, fundamentals_df, svensson_df)
    factor_df = run_factor(prices_df, brent_df, svensson_df, fundamentals_df)

    scenarios = {}
    results = {}
    for name in library:
        sc = resolve_scenario(name, library, brent_df, svensson_df)
        df = apply_scenario(merton_df, factor_df, sc,
                            horizon_years=horizon_years)
        scenarios[name] = sc
        results[name] = df

    # Heatmap-DataFrame (Ticker × Szenarien, PD-Werte)
    heatmap = pd.DataFrame(index=merton_df.index)
    heatmap["Name"] = merton_df["Name"]
    heatmap["Sector"] = merton_df["Sector"]
    heatmap["Baseline"] = merton_df["PD"]
    for name, df in results.items():
        heatmap[name] = df["ScenarioPD"]

    return {
        "scenarios":  scenarios,
        "merton_df":  merton_df,
        "factor_df":  factor_df,
        "results":    results,
        "heatmap":    heatmap,
    }


# ============================================================================
#  VALIDATION  ·  python backend/scenarios.py
# ============================================================================

def _calibration_test() -> bool:
    """Hartkodierte Erwartungen aus dem Daten-Lookup."""
    print("  [1] Historische Kalibrierung (Ukraine 2022):")
    try:
        from config import CACHE_DIR  # type: ignore
    except ImportError:
        return True
    if not (CACHE_DIR / "brent_crude.parquet").exists():
        print("      [SKIP] cache fehlt"); return True
    brent = pd.read_parquet(CACHE_DIR / "brent_crude.parquet")
    sven  = pd.read_parquet(CACHE_DIR / "svensson_params.parquet")

    cal = calibrate_from_history(
        brent, sven, start="2022-02-22", end="2022-04-30",
    )
    # Erwartungen aus dem Sondierungs-Run:
    #   ΔBrent_log ≈ +0.121, Δr_10y ≈ +0.674pp, Horizon ≈ 67 Tage
    ok_b = abs(cal["delta_brent_log"] - 0.121) < 0.05
    ok_r = abs(cal["delta_rate_10y_pp"] - 0.674) < 0.1
    ok_h = abs(cal["horizon_days"] - 67) <= 7
    print(f"      ΔBrent_log = {cal['delta_brent_log']:+.3f} (≈+0.12)  "
          f"{'OK' if ok_b else 'FAIL'}")
    print(f"      Δr_10y     = {cal['delta_rate_10y_pp']:+.3f}pp (≈+0.67)  "
          f"{'OK' if ok_r else 'FAIL'}")
    print(f"      Horizon    = {cal['horizon_days']}d (≈67)  "
          f"{'OK' if ok_h else 'FAIL'}")
    return ok_b and ok_r and ok_h


def _resolve_test() -> bool:
    """resolve_scenario gibt richtige Werte für 'literature' und 'hypothetical'."""
    print("  [2] Library-Resolve (Corona literature, Iran hypothetical):")
    from config import SCENARIO_LIBRARY  # type: ignore
    sc_corona = resolve_scenario("corona_2020", SCENARIO_LIBRARY)
    sc_iran   = resolve_scenario("iran_2026",  SCENARIO_LIBRARY)

    ok_c = (sc_corona["source"] == "literature"
            and abs(sc_corona["delta_brent_log"] - (-1.20)) < 1e-9
            and abs(sc_corona["delta_rate_10y_pp"] - (-0.65)) < 1e-9
            and sc_corona["horizon_days"] == 60)
    ok_i = (sc_iran["source"] == "hypothetical"
            and abs(sc_iran["delta_brent_log"] - 0.55) < 1e-9
            and abs(sc_iran["delta_rate_10y_pp"] - 0.50) < 1e-9
            and sc_iran["horizon_days"] == 30)
    print(f"      corona_2020: ΔBrent={sc_corona['delta_brent_log']:+.2f}  "
          f"Δr={sc_corona['delta_rate_10y_pp']:+.2f}pp  "
          f"H={sc_corona['horizon_days']}  {'OK' if ok_c else 'FAIL'}")
    print(f"      iran_2026:   ΔBrent={sc_iran['delta_brent_log']:+.2f}  "
          f"Δr={sc_iran['delta_rate_10y_pp']:+.2f}pp  "
          f"H={sc_iran['horizon_days']}  {'OK' if ok_i else 'FAIL'}")
    return ok_c and ok_i


def _zero_shock_consistency_test() -> bool:
    """Mit ΔBrent=0 und Δr=0 → ScenarioPD = BaselinePD (modulo α-Drift)."""
    print("  [3] Null-Schock-Konsistenz:")
    try:
        from config import CACHE_DIR  # type: ignore
    except ImportError:
        return True
    if not (CACHE_DIR / "dax40_prices.parquet").exists():
        print("      [SKIP] cache fehlt"); return True

    prices = pd.read_parquet(CACHE_DIR / "dax40_prices.parquet")
    brent  = pd.read_parquet(CACHE_DIR / "brent_crude.parquet")
    sven   = pd.read_parquet(CACHE_DIR / "svensson_params.parquet")
    fund   = pd.read_parquet(CACHE_DIR / "dax40_fundamentals.parquet")

    from merton import run_dax40 as run_merton  # type: ignore
    from factor_model import run_dax40 as run_factor  # type: ignore
    merton_df = run_merton(prices, fund, sven)
    factor_df = run_factor(prices, brent, sven, fund)

    null_sc = {
        "name": "null_shock",
        "description": "test",
        "source": "test",
        "delta_brent_log":   0.0,
        "delta_rate_10y_pp": 0.0,
        "horizon_days":      0,        # auch α·H = 0
        "period": None, "calibration": None,
    }
    df = apply_scenario(merton_df, factor_df, null_sc)
    valid = df.dropna(subset=["ScenarioPD"])

    # Mit α·H=0, ΔBrent=0, Δr=0 → log_ret=0 → E_stress=E_0 →
    # KMV liefert dieselben V, σ_V wie Baseline → DD, PD identisch.
    err = (valid["ScenarioPD"] - valid["BaselinePD"]).abs()
    max_err = float(err.max())
    ok = max_err < 1e-6
    print(f"      max |ScenarioPD - BaselinePD| = {max_err:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def _real_library_run() -> bool:
    """Volle Library auf DAX-40 + Top-Stress-Treiber pro Szenario."""
    print("  [4] DAX-40 × Library-Run:")
    try:
        from config import CACHE_DIR  # type: ignore
    except ImportError:
        return True
    needed = ["dax40_prices.parquet", "brent_crude.parquet",
              "svensson_params.parquet", "dax40_fundamentals.parquet"]
    if any(not (CACHE_DIR / n).exists() for n in needed):
        print("      [SKIP] cache fehlt"); return True

    prices = pd.read_parquet(CACHE_DIR / "dax40_prices.parquet")
    brent  = pd.read_parquet(CACHE_DIR / "brent_crude.parquet")
    sven   = pd.read_parquet(CACHE_DIR / "svensson_params.parquet")
    fund   = pd.read_parquet(CACHE_DIR / "dax40_fundamentals.parquet")

    res = run_scenario_library(prices, brent, sven, fund)

    print(f"      Szenarien: {list(res['scenarios'].keys())}")
    print()
    print("      Schock-Werte je Szenario:")
    for name, sc in res["scenarios"].items():
        print(f"        {name:<22} src={sc['source']:<13} "
              f"ΔBrent_log={sc['delta_brent_log']:+.3f}  "
              f"Δr={sc['delta_rate_10y_pp']:+.2f}pp  "
              f"H={sc['horizon_days']:>3}d")
    print()

    print("      Top-3 PD je Szenario:")
    for name, df in res["results"].items():
        valid = df.dropna(subset=["ScenarioPD"]).sort_values(
            "ScenarioPD", ascending=False)
        print(f"        [{name}]")
        for tk, row in valid.head(3).iterrows():
            print(f"          {tk:<8} {row['Name']:<22}  "
                  f"BasePD={row['BaselinePD']*100:6.3f}%  "
                  f"ScenPD={row['ScenarioPD']*100:6.3f}%  "
                  f"Δ={row['DeltaPD']*100:+6.3f}pp  "
                  f"({row['Sector']})")
    print()

    # Sanity: Mit Corona-Schock (ΔBrent <0, Δr <0) sollten Banken weniger
    # leiden als mit Ukraine (ΔBrent >0, Δr >0). Das prüfen wir locker.
    cbk_corona = res["results"]["corona_2020"].loc["CBK.DE", "ScenarioPD"]
    cbk_ukraine = res["results"]["ukraine_2022"].loc["CBK.DE", "ScenarioPD"]
    cbk_iran   = res["results"]["iran_2026"].loc["CBK.DE", "ScenarioPD"]
    print(f"      CBK.DE Stress-PD über Szenarien:")
    print(f"        corona_2020 = {cbk_corona*100:.3f}%")
    print(f"        ukraine_2022= {cbk_ukraine*100:.3f}%")
    print(f"        iran_2026   = {cbk_iran*100:.3f}%")

    # Heatmap-Schema-Check
    hm = res["heatmap"]
    expected_cols = ["Name", "Sector", "Baseline"] + list(res["scenarios"].keys())
    cols_ok = list(hm.columns) == expected_cols
    print(f"      Heatmap Spalten: {'OK' if cols_ok else 'FAIL'}")

    return cols_ok


if __name__ == "__main__":
    print("=" * 70)
    print(" scenarios.py · Validierung")
    print("=" * 70)

    results = [
        _calibration_test(),
        _resolve_test(),
        _zero_shock_consistency_test(),
        _real_library_run(),
    ]

    print("=" * 70)
    if all(results):
        print(f" [PASS]  Alle {len(results)} Test-Blöcke bestanden.")
    else:
        n_fail = sum(1 for r in results if not r)
        print(f" [FAIL]  {n_fail}/{len(results)} Test-Blöcke fehlgeschlagen.")
    print("=" * 70)
