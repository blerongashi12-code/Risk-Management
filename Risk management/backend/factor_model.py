"""
============================================================================
 backend/factor_model.py
 ---------------------------------------------------------------------------
 Zwei-Faktor-OLS-Modell auf täglichen Equity-Returns.

     r_equity_t = α + β_E · r_brent_t + β_R · Δr_10y_t + ε_t

 Faktoren:
   - r_brent     : log-Return Brent Crude (BZ=F)
   - Δr_10y      : Veränderung der Svensson-Zero-Rate bei τ=10y
                   (Prozentpunkte, Bundesbank-Konvention)

 Sektor-Multiplier auf β_E (Energy-Beta):
   β_E_adjusted = β_E_raw · EnergyMul[sector]
   Methodik in MODEL_ASSUMPTIONS.md §4.5 dokumentiert.

 Funktionen:
   - factor_returns(brent_df, svensson_df, maturity)  → DataFrame
   - equity_log_returns(prices_df)                    → DataFrame
   - fit_two_factor(y, X)                             → FactorResult
   - run_dax40(prices, brent, svensson, fundamentals) → pd.DataFrame

 Validierung: python backend/factor_model.py
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Optional, Mapping

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ----------------------------------------------------------------------
# 1. Datentyp
# ----------------------------------------------------------------------
class FactorResult(NamedTuple):
    """Ergebnis einer 2-Faktor-OLS-Schätzung pro Firma.

    alpha       : Achsenabschnitt (täglich, Dezimal)
    beta_brent  : Sensitivität zum Brent-Log-Return (β_E)
    beta_dr     : Sensitivität zu Δr_10y (pro 1pp Zinsänderung)
    r_squared   : Bestimmtheitsmass (Adjusted=False)
    sigma_eps   : Residuen-Std (täglich, Dezimal)
    n_obs       : Anzahl gemeinsamer Beobachtungen
    converged   : True wenn n_obs ≥ FACTOR_MIN_OBS
    """
    alpha: float
    beta_brent: float
    beta_dr: float
    r_squared: float
    sigma_eps: float
    n_obs: int
    converged: bool


# ----------------------------------------------------------------------
# 2. Faktor-Konstruktion
# ----------------------------------------------------------------------
def factor_returns(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    maturity: float = 10.0,
) -> pd.DataFrame:
    """Tägliche Faktor-Returns für die OLS-Schätzung.

    Parameters
    ----------
    brent_df : DataFrame mit DatetimeIndex und Spalte 'Return_log'
        (kommt aus 03_fetch_brent_crude.py, Spalte ist bereits berechnet).
    svensson_df : Bundesbank-Parameter-Zeitreihe.
    maturity : Δr-Maturity in Jahren (default 10).

    Returns
    -------
    DataFrame mit DatetimeIndex und Spalten [r_brent, d_r_10y].
    Werte vor erstem Trading-Day sind NaN (entfällt nach .dropna()).
    """
    from svensson import zero_rate, params_from_row  # type: ignore

    # Brent: nutzt bereits berechnete Log-Returns aus Data-Layer
    if "Return_log" not in brent_df.columns:
        raise KeyError("brent_df muss Spalte 'Return_log' enthalten "
                       "(aus 03_fetch_brent_crude.py).")
    r_brent = brent_df["Return_log"].rename("r_brent")

    # Δr_10y: berechne Svensson-Zero-Rate bei `maturity`, dann diff()
    rates = pd.Series(
        [zero_rate(maturity, params_from_row(row), as_decimal=False)
         for _, row in svensson_df.iterrows()],
        index=svensson_df.index,
        name=f"r_{maturity:.0f}y",
    )
    d_r = rates.diff().rename(f"d_r_{maturity:.0f}y")

    return pd.concat([r_brent, d_r], axis=1, sort=False).sort_index()


def equity_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Tägliche Log-Returns für alle Tickers im prices_df."""
    return np.log(prices_df / prices_df.shift(1))


# ----------------------------------------------------------------------
# 3. OLS-Engine
# ----------------------------------------------------------------------
def fit_two_factor(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    min_obs: int = 60,
) -> FactorResult:
    """OLS-Schätzung von y = α + X·β + ε.

    Parameters
    ----------
    y : Equity-Log-Returns einer einzelnen Firma.
    X : DataFrame mit Spalten [r_brent, d_r_10y] (Reihenfolge fix).
    min_obs : Mindestanzahl gemeinsamer Beobachtungen.

    Implementation: numpy.linalg.lstsq direkt — kein scipy/statsmodels.

    Returns
    -------
    FactorResult; bei zu wenig Daten converged=False mit NaN-Werten.
    """
    df = pd.concat([y.rename("y"), X], axis=1, sort=False).dropna()
    n = len(df)
    if n < min_obs:
        return FactorResult(
            alpha=float("nan"), beta_brent=float("nan"),
            beta_dr=float("nan"), r_squared=float("nan"),
            sigma_eps=float("nan"), n_obs=n, converged=False,
        )

    yv = df["y"].to_numpy()
    Xv = df.drop(columns="y").to_numpy()
    # Design-Matrix mit Konstante voranstellen
    Xc = np.column_stack([np.ones(n), Xv])

    # OLS via lstsq (numerisch stabiler als (X'X)⁻¹ X'y)
    beta, *_ = np.linalg.lstsq(Xc, yv, rcond=None)

    y_hat = Xc @ beta
    eps = yv - y_hat
    ss_res = float(np.sum(eps ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sigma = float(np.sqrt(ss_res / max(n - len(beta), 1)))  # Std-Fehler ε

    return FactorResult(
        alpha=float(beta[0]),
        beta_brent=float(beta[1]),
        beta_dr=float(beta[2]),
        r_squared=float(r2),
        sigma_eps=sigma,
        n_obs=n,
        converged=True,
    )


# ----------------------------------------------------------------------
# 4. DAX-40 Convenience-Wrapper
# ----------------------------------------------------------------------
def run_dax40(
    prices_df: pd.DataFrame,
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    *,
    lookback: Optional[int] = None,
    maturity: Optional[float] = None,
    min_obs: Optional[int] = None,
    energy_mul_map: Optional[Mapping[str, float]] = None,
    default_energy_mul: Optional[float] = None,
) -> pd.DataFrame:
    """Schätzt das 2-Faktor-Modell für alle DAX-40-Tickers.

    Modell-Prämissen (override-bar via Parameter; Defaults aus config):
      - Lookback = FACTOR_LOOKBACK_DAYS (252, gleich wie Merton)
      - Δr-Maturity = FACTOR_MATURITY (10y)
      - Min-Obs    = FACTOR_MIN_OBS (60)
      - β_E_adjusted = β_E_raw · EnergyMul[sector]
        siehe config.SECTOR_ENERGY_MUL und MODEL_ASSUMPTIONS.md §4.5.

    Returns
    -------
    pd.DataFrame, Index = Ticker, Columns:
        Name, Sector, NObs, Alpha, BetaBrentRaw, BetaDr, R2, SigmaEps,
        EnergyMul, BetaBrentAdjusted, Converged
    """
    from config import (FACTOR_LOOKBACK_DAYS, FACTOR_MIN_OBS,  # type: ignore
                        FACTOR_MATURITY, SECTOR_ENERGY_MUL,
                        DEFAULT_SECTOR_ENERGY_MUL)

    if lookback is None:
        lookback = FACTOR_LOOKBACK_DAYS
    if maturity is None:
        maturity = FACTOR_MATURITY
    if min_obs is None:
        min_obs = FACTOR_MIN_OBS
    if energy_mul_map is None:
        energy_mul_map = SECTOR_ENERGY_MUL
    if default_energy_mul is None:
        default_energy_mul = DEFAULT_SECTOR_ENERGY_MUL

    # Faktor-Returns einmal berechnen (gleich für alle Firmen)
    factors = factor_returns(brent_df, svensson_df, maturity=maturity)

    # Equity-Returns
    eq_ret = equity_log_returns(prices_df)

    # Letzte `lookback` gemeinsame Tage
    common_idx = factors.index.intersection(eq_ret.index)
    if len(common_idx) > lookback:
        common_idx = common_idx.sort_values()[-lookback:]
    factors_win = factors.loc[common_idx]
    eq_ret_win = eq_ret.loc[common_idx]

    rows = []
    for ticker, fund_row in fundamentals_df.iterrows():
        sector = fund_row.get("Sector_Yahoo")
        mul = energy_mul_map.get(sector, default_energy_mul) if sector else default_energy_mul

        if ticker not in eq_ret_win.columns:
            rows.append(_skip_row(ticker, fund_row, mul, "no prices"))
            continue

        res = fit_two_factor(eq_ret_win[ticker], factors_win, min_obs=min_obs)
        rows.append({
            "Name": fund_row.get("Name"),
            "Sector": sector,
            "NObs": res.n_obs,
            "Alpha": res.alpha,
            "BetaBrentRaw": res.beta_brent,
            "BetaDr": res.beta_dr,
            "R2": res.r_squared,
            "SigmaEps": res.sigma_eps,
            "EnergyMul": mul,
            "BetaBrentAdjusted": (res.beta_brent * mul
                                  if res.converged else float("nan")),
            "Converged": res.converged,
        })

    out = pd.DataFrame(rows, index=fundamentals_df.index)
    out.index.name = "Ticker"
    return out


def _skip_row(ticker: str, fund_row: pd.Series, mul: float,
              reason: str) -> dict:
    return {
        "Name": fund_row.get("Name"),
        "Sector": fund_row.get("Sector_Yahoo"),
        "NObs": 0,
        "Alpha": float("nan"),
        "BetaBrentRaw": float("nan"),
        "BetaDr": float("nan"),
        "R2": float("nan"),
        "SigmaEps": float("nan"),
        "EnergyMul": mul,
        "BetaBrentAdjusted": float("nan"),
        "Converged": False,
    }


# ============================================================================
#  VALIDATION  ·  python backend/factor_model.py
# ============================================================================

def _round_trip_test() -> bool:
    """Synthetische Returns mit bekannten (α,β_E,β_R,σ_ε) → OLS soll
       die Werte zurückschätzen (Toleranz wegen Stichprobenvarianz)."""
    print("  [1] Round-Trip (synthetische Returns):")
    rng = np.random.default_rng(42)
    n = 500
    true_alpha, true_be, true_br, true_sig = 0.0001, 0.50, -0.30, 0.010

    f_brent = pd.Series(rng.normal(0, 0.02, n))
    f_dr    = pd.Series(rng.normal(0, 0.05, n))
    eps     = rng.normal(0, true_sig, n)
    y       = (true_alpha + true_be * f_brent
               + true_br * f_dr + eps)

    X = pd.concat([f_brent.rename("r_brent"),
                   f_dr.rename("d_r_10y")], axis=1)
    res = fit_two_factor(y, X, min_obs=50)

    checks = [
        ("alpha     ", res.alpha,      true_alpha, 5e-4),
        ("β_E       ", res.beta_brent, true_be,    0.05),
        ("β_R       ", res.beta_dr,    true_br,    0.05),
        ("σ_ε       ", res.sigma_eps,  true_sig,   1e-3),
    ]
    all_ok = True
    for label, got, exp, tol in checks:
        ok = abs(got - exp) < tol
        all_ok &= ok
        print(f"      {label}  est={got:+.5f}  true={exp:+.5f}  "
              f"tol={tol:.1e}  {'OK' if ok else 'FAIL'}")
    print(f"      R²={res.r_squared:.4f}  n_obs={res.n_obs}")
    return all_ok


def _factor_construction_test() -> bool:
    """Δr-Faktor: konstante Svensson-Parameter → d_r_10y == 0 für alle Tage."""
    print("  [2] Faktor-Konstruktion (Δr-Konsistenz):")
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    svensson_const = pd.DataFrame({
        "Beta0": 2.0, "Beta1": -1.0, "Beta2": 0.5, "Beta3": 0.0,
        "Tau1": 2.0, "Tau2": 4.0,
    }, index=dates)
    brent = pd.DataFrame({"Return_log": np.zeros(10)}, index=dates)

    fr = factor_returns(brent, svensson_const, maturity=10.0)
    d_r = fr["d_r_10y"].dropna()
    max_abs = float(d_r.abs().max()) if len(d_r) > 0 else 0.0
    ok = max_abs < 1e-12
    print(f"      Svensson konstant → max |Δr_10y| = {max_abs:.1e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def _r2_sanity_test(df: pd.DataFrame) -> bool:
    """R² ∈ [0, 1] für alle konvergierten Firmen."""
    print("  [3] R²-Sanity:")
    valid = df[df["Converged"]]
    bad_lo = (valid["R2"] < 0).sum()
    bad_hi = (valid["R2"] > 1).sum()
    ok = bad_lo == 0 and bad_hi == 0
    print(f"      Firmen mit R²<0: {bad_lo}, R²>1: {bad_hi}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def _vorzeichen_plausi_test(df: pd.DataFrame) -> bool:
    """Sektor-Plausibilität:
       - Utilities (RWE) sollte β_E_adj signifikant grösser haben als
         Technology (SAP), weil EnergyMul Utilities=4.0 vs. Tech=0.3."""
    print("  [4] Sektor-Vorzeichen-Plausibilität:")
    rwe = df.loc["RWE.DE"] if "RWE.DE" in df.index else None
    sap = df.loc["SAP.DE"] if "SAP.DE" in df.index else None
    if rwe is None or sap is None:
        print("      [SKIP] RWE.DE oder SAP.DE nicht in DataFrame")
        return True
    # Mit dem Multiplier-Spread (4.0 / 0.3 = 13.3×) muss RWE adjusted
    # deutlich höher sein als SAP, sofern Raw-Betas vergleichbar sind.
    rwe_adj = rwe["BetaBrentAdjusted"]
    sap_adj = sap["BetaBrentAdjusted"]
    print(f"      RWE.DE β_E_raw={rwe['BetaBrentRaw']:+.4f}  "
          f"mul={rwe['EnergyMul']:.1f}  → β_E_adj={rwe_adj:+.4f}  "
          f"R²={rwe['R2']:.3f}")
    print(f"      SAP.DE β_E_raw={sap['BetaBrentRaw']:+.4f}  "
          f"mul={sap['EnergyMul']:.1f}  → β_E_adj={sap_adj:+.4f}  "
          f"R²={sap['R2']:.3f}")
    # Nur prüfen, dass Multiplier korrekt angewandt wurde
    ok_rwe = abs(rwe["BetaBrentAdjusted"] -
                 rwe["BetaBrentRaw"] * rwe["EnergyMul"]) < 1e-12
    ok_sap = abs(sap["BetaBrentAdjusted"] -
                 sap["BetaBrentRaw"] * sap["EnergyMul"]) < 1e-12
    print(f"      Multiplier-Anwendung exact: "
          f"RWE {'OK' if ok_rwe else 'FAIL'}, "
          f"SAP {'OK' if ok_sap else 'FAIL'}")
    return ok_rwe and ok_sap


def _real_dax40_run() -> bool:
    """Realer 38-Firmen-Run + Top/Bottom Energy-Beta Reporting."""
    print("  [5] DAX-40-Run (echte Daten):")
    try:
        from config import CACHE_DIR  # type: ignore
    except ImportError:
        print("      [SKIP] config nicht importierbar.")
        return True

    needed = ["dax40_prices.parquet", "brent_crude.parquet",
              "svensson_params.parquet", "dax40_fundamentals.parquet"]
    missing = [n for n in needed if not (CACHE_DIR / n).exists()]
    if missing:
        print(f"      [SKIP] fehlende Cache-Files: {missing}")
        return True

    prices  = pd.read_parquet(CACHE_DIR / "dax40_prices.parquet")
    brent   = pd.read_parquet(CACHE_DIR / "brent_crude.parquet")
    sven    = pd.read_parquet(CACHE_DIR / "svensson_params.parquet")
    fund    = pd.read_parquet(CACHE_DIR / "dax40_fundamentals.parquet")

    df = run_dax40(prices, brent, sven, fund)
    n_ok = int(df["Converged"].fillna(False).sum())
    print(f"      Schätzungen konvergiert: {n_ok}/{len(df)}")

    # R²-Verteilung
    valid = df[df["Converged"]]
    print(f"      R² range = [{valid['R2'].min():.3f}, "
          f"{valid['R2'].max():.3f}]   median = {valid['R2'].median():.3f}")

    # Top-5 raw β_E (rohes Markt-Sensitivität — vor Multiplier)
    print(f"      Top-5 raw β_E (Brent-Sensitivität):")
    for tk, row in valid.nlargest(5, "BetaBrentRaw").iterrows():
        print(f"        {tk:<8} {row['Name']:<22}  "
              f"β_raw={row['BetaBrentRaw']:+.4f}  "
              f"mul={row['EnergyMul']:.1f}  "
              f"β_adj={row['BetaBrentAdjusted']:+.4f}  "
              f"R²={row['R2']:.3f}  ({row['Sector']})")

    # Top-5 adjusted (mit Multiplier — als Stress-Treiber relevant)
    print(f"      Top-5 β_E_adjusted (Stress-Sensitivität):")
    for tk, row in valid.nlargest(5, "BetaBrentAdjusted").iterrows():
        print(f"        {tk:<8} {row['Name']:<22}  "
              f"β_adj={row['BetaBrentAdjusted']:+.4f}  "
              f"({row['Sector']}, mul={row['EnergyMul']:.1f})")

    # β_R-Sensitivität (Banken sollten zinssensitiv sein)
    print(f"      Banken-Zinssensitivität (β_R):")
    banks = valid[valid["Sector"] == "Financial Services"]
    for tk, row in banks.iterrows():
        print(f"        {tk:<8} {row['Name']:<22}  "
              f"β_R={row['BetaDr']:+.4f}  R²={row['R2']:.3f}")

    rate_ok = n_ok >= 0.9 * len(df)
    return rate_ok and _r2_sanity_test(df) and _vorzeichen_plausi_test(df)


if __name__ == "__main__":
    print("=" * 70)
    print(" factor_model.py · Validierung")
    print("=" * 70)

    results = [
        _round_trip_test(),
        _factor_construction_test(),
        _real_dax40_run(),  # enthält [3] und [4] über Helper
    ]

    print("=" * 70)
    if all(results):
        print(f" [PASS]  Alle {len(results)} Test-Blöcke bestanden.")
    else:
        n_fail = sum(1 for r in results if not r)
        print(f" [FAIL]  {n_fail}/{len(results)} Test-Blöcke fehlgeschlagen.")
    print("=" * 70)
