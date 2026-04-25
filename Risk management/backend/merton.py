"""
============================================================================
 backend/merton.py
 ---------------------------------------------------------------------------
 Strukturmodell nach Merton (1974) mit KMV-Iteration.

 Ein Unternehmen mit Equity E (beobachtbar) und Liabilities L (Bilanz)
 wird als Call-Option auf den Firmenwert V interpretiert. KMV löst das
 implizite System für (V, σ_V) aus den Equity-Beobachtungen (E, σ_E).

     E         = V·N(d1) − L·exp(−rT)·N(d2)
     σ_E · E   = σ_V · V · N(d1)
     d1        = [ln(V/L) + (r + ½σ_V²)·T] / (σ_V·√T)
     d2        = d1 − σ_V·√T

 Output:
     DD = [ln(V/L) + (r − ½σ_V²)·T] / (σ_V·√T)
     PD = N(−DD)

 Funktionen:
     - merton_kmv(equity, equity_vol, debt, r, T)        → MertonResult
     - distance_to_default(V, σ_V, L, r, T)              → float
     - merton_pd(dd)                                     → float
     - equity_vol_from_prices(prices, lookback=252)      → float
     - run_dax40(prices_df, fundamentals_df, svensson_df) → pd.DataFrame

 Validierung: Run als Skript:  python backend/merton.py
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# Pfad-Setup für `from config import …`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ----------------------------------------------------------------------
# 1. Datentypen
# ----------------------------------------------------------------------
class MertonResult(NamedTuple):
    """Ergebnis einer KMV-Lösung für eine Firma.

    Felder
    ------
    asset_value : V (gleiche Einheit wie equity / debt)
    asset_vol   : σ_V (Dezimal, annualisiert)
    distance_to_default : DD
    pd          : N(−DD)
    iterations  : Anzahl KMV-Schritte bis Konvergenz
    converged   : True wenn Iter unter tol konvergiert ist
    equity, debt, rate, horizon : Inputs (gespiegelt für Nachvollziehbarkeit)
    """
    asset_value: float
    asset_vol: float
    distance_to_default: float
    pd: float
    iterations: int
    converged: bool
    equity: float
    debt: float
    rate: float
    horizon: float


# ----------------------------------------------------------------------
# 2. Closed-form Helper
# ----------------------------------------------------------------------
def _d1_d2(V: float, sigma_V: float, L: float, r: float, T: float
           ) -> tuple[float, float]:
    """Black-Scholes d1, d2 für Asset-Wert V vs. Default-Schwelle L."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(V / L) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrtT)
    d2 = d1 - sigma_V * sqrtT
    return d1, d2


def equity_value_from_v(V: float, sigma_V: float, L: float,
                        r: float, T: float) -> float:
    """E = V·N(d1) − L·exp(−rT)·N(d2). Verifikations-Helper."""
    d1, d2 = _d1_d2(V, sigma_V, L, r, T)
    return V * norm.cdf(d1) - L * np.exp(-r * T) * norm.cdf(d2)


def equity_vol_from_v(V: float, sigma_V: float, L: float,
                      r: float, T: float, E: float) -> float:
    """σ_E = σ_V · V · N(d1) / E. Verifikations-Helper."""
    d1, _ = _d1_d2(V, sigma_V, L, r, T)
    return sigma_V * V * norm.cdf(d1) / E


def distance_to_default(V: float, sigma_V: float, L: float,
                        r: float, T: float) -> float:
    """DD = [ln(V/L) + (r − ½σ_V²)·T] / (σ_V·√T)."""
    return (np.log(V / L) + (r - 0.5 * sigma_V ** 2) * T) / (
        sigma_V * np.sqrt(T)
    )


def merton_pd(dd: float) -> float:
    """PD = N(−DD)."""
    return float(norm.cdf(-dd))


# ----------------------------------------------------------------------
# 3. KMV-Iteration
# ----------------------------------------------------------------------
def merton_kmv(
    equity: float,
    equity_vol: float,
    debt: float,
    r: float,
    T: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> MertonResult:
    """Löst das Merton/KMV-System für (V, σ_V) per Fixpunkt-Iteration.

    Parameters
    ----------
    equity : E (z.B. MarketCap, lokale Währung)
    equity_vol : σ_E, annualisiert, Dezimal (z.B. 0.32 für 32%)
    debt : L (z.B. TotalDebt, gleiche Währung wie equity)
    r : Risk-free rate, Dezimal annualisiert (NICHT Prozent)
    T : Horizon in Jahren
    tol : relative Konvergenz-Toleranz auf V und σ_V
    max_iter : Abbruch nach so vielen Schritten

    Iteration
    ---------
        V_neu     = (E + L·exp(−rT)·N(d2)) / N(d1)
        σ_V_neu   = σ_E · E / (V_neu · N(d1))

    Liefert MertonResult; bei Nicht-Konvergenz `converged=False`,
    Felder enthalten letzten Iterations-Stand.
    """
    if equity <= 0 or debt <= 0 or equity_vol <= 0 or T <= 0:
        return MertonResult(
            asset_value=np.nan, asset_vol=np.nan,
            distance_to_default=np.nan, pd=np.nan,
            iterations=0, converged=False,
            equity=equity, debt=debt, rate=r, horizon=T,
        )

    # Init
    V = equity + debt
    sigma_V = equity_vol * equity / V

    converged = False
    for k in range(1, max_iter + 1):
        d1, d2 = _d1_d2(V, sigma_V, debt, r, T)
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)

        if Nd1 < 1e-12:
            break  # Numerisch entartet

        V_new = (equity + debt * np.exp(-r * T) * Nd2) / Nd1
        sigma_V_new = equity_vol * equity / (V_new * Nd1)

        rel_err = max(
            abs(V_new - V) / max(abs(V), 1e-12),
            abs(sigma_V_new - sigma_V) / max(abs(sigma_V), 1e-12),
        )
        V, sigma_V = V_new, sigma_V_new
        if rel_err < tol:
            converged = True
            break

    dd = distance_to_default(V, sigma_V, debt, r, T)
    pd_ = merton_pd(dd)

    return MertonResult(
        asset_value=float(V),
        asset_vol=float(sigma_V),
        distance_to_default=float(dd),
        pd=float(pd_),
        iterations=k,
        converged=converged,
        equity=float(equity),
        debt=float(debt),
        rate=float(r),
        horizon=float(T),
    )


# ----------------------------------------------------------------------
# 4. Equity-Vola aus Preisreihe
# ----------------------------------------------------------------------
def equity_vol_from_prices(
    prices: pd.Series,
    *,
    lookback: int = 252,
    trading_days_per_year: int = 252,
) -> float:
    """Annualisierte Vola aus Log-Returns der letzten `lookback` Tage.

    Parameters
    ----------
    prices : pd.Series
        Zeitreihe der Aktienkurse (ungestrippt von NaN).
    lookback : int
        Anzahl der letzten Returns für die Stichproben-Vola.
    trading_days_per_year : int
        Annualisierungsfaktor (default 252).

    Returns
    -------
    σ_E (Dezimal). NaN, wenn nicht genug Daten.
    """
    s = prices.dropna()
    if len(s) < lookback + 1:
        return float("nan")
    log_ret = np.log(s.iloc[-(lookback + 1):] / s.iloc[-(lookback + 1):].shift(1)).dropna()
    return float(log_ret.std(ddof=1) * np.sqrt(trading_days_per_year))


# ----------------------------------------------------------------------
# 5. DAX-40 Convenience-Wrapper
# ----------------------------------------------------------------------
def _resolve_debt(row: pd.Series) -> tuple[float, str]:
    """Liefert (debt_value, source_tag).

    Quellen-Hierarchie:
        1. TotalDebt
        2. ShortTermDebt + LongTermDebt
        3. NaN, source = 'missing'
    """
    td = row.get("TotalDebt")
    if pd.notna(td) and td > 0:
        return float(td), "TotalDebt"
    st = row.get("ShortTermDebt")
    lt = row.get("LongTermDebt")
    if pd.notna(st) and pd.notna(lt) and (st + lt) > 0:
        return float(st) + float(lt), "ST+LT"
    return float("nan"), "missing"


def run_dax40(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    horizon: float = 1.0,
    lookback: int = 252,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Wendet Merton/KMV auf alle Tickers in `fundamentals_df` an.

    Parameters
    ----------
    prices_df : DatetimeIndex × Tickers (Aktienkurse)
    fundamentals_df : Index = Ticker, mit MarketCap, TotalDebt, ShortTermDebt,
        LongTermDebt, Sector_Yahoo, Currency, ReportDate, Name
    svensson_df : Bundesbank-Parameter-Zeitreihe
    horizon : T in Jahren (default 1)
    lookback : Lookback für Equity-Vola (default 252)
    as_of : Stichtag für Risk-free rate; default = letzter Preis-Tag

    Returns
    -------
    pd.DataFrame, Index = Ticker, Columns:
        Name, Sector, MarketCap, Debt, DebtSource, EquityVol,
        Rate, AssetValue, AssetVol, DD, PD, Iterations, Converged
    """
    from svensson import historical_curve, zero_rate  # type: ignore

    if as_of is None:
        as_of = prices_df.index.max()

    params = historical_curve(as_of, svensson_df)
    r = zero_rate(horizon, params, as_decimal=True)

    rows = []
    for ticker, fund_row in fundamentals_df.iterrows():
        # Equity
        E = fund_row.get("MarketCap")
        if pd.isna(E) or E <= 0:
            rows.append(_skip_row(ticker, fund_row, reason="no MarketCap"))
            continue

        # Debt
        L, debt_src = _resolve_debt(fund_row)
        if pd.isna(L):
            rows.append(_skip_row(ticker, fund_row, reason="no debt"))
            continue

        # Equity-Vola
        if ticker not in prices_df.columns:
            rows.append(_skip_row(ticker, fund_row, reason="no prices"))
            continue
        sigma_E = equity_vol_from_prices(prices_df[ticker], lookback=lookback)
        if np.isnan(sigma_E):
            rows.append(_skip_row(ticker, fund_row, reason="vola NaN"))
            continue

        res = merton_kmv(E, sigma_E, L, r, horizon)
        rows.append({
            "Name": fund_row.get("Name"),
            "Sector": fund_row.get("Sector_Yahoo"),
            "MarketCap": E,
            "Debt": L,
            "DebtSource": debt_src,
            "EquityVol": sigma_E,
            "Rate": r,
            "AssetValue": res.asset_value,
            "AssetVol": res.asset_vol,
            "DD": res.distance_to_default,
            "PD": res.pd,
            "Iterations": res.iterations,
            "Converged": res.converged,
        })

    out = pd.DataFrame(rows, index=fundamentals_df.index)
    out.index.name = "Ticker"
    return out


def _skip_row(ticker: str, fund_row: pd.Series, *, reason: str) -> dict:
    """Leere Result-Zeile mit dokumentiertem Skip-Grund."""
    return {
        "Name": fund_row.get("Name"),
        "Sector": fund_row.get("Sector_Yahoo"),
        "MarketCap": fund_row.get("MarketCap"),
        "Debt": np.nan,
        "DebtSource": f"skipped: {reason}",
        "EquityVol": np.nan,
        "Rate": np.nan,
        "AssetValue": np.nan,
        "AssetVol": np.nan,
        "DD": np.nan,
        "PD": np.nan,
        "Iterations": 0,
        "Converged": False,
    }


# ============================================================================
#  VALIDATION  ·  python backend/merton.py
# ============================================================================

def _round_trip_test() -> bool:
    """Stärkster Test: Forward (V,σ_V → E,σ_E) und KMV-Rückrichtung
    müssen sich exakt schliessen."""
    print("  [1] Round-Trip (V,σ_V → E,σ_E → KMV → V,σ_V):")
    cases = [
        ("Toy-Firma A",  {"V": 100.0, "σV": 0.25, "L": 60.0, "r": 0.03, "T": 1.0}),
        ("Toy-Firma B",  {"V":  50.0, "σV": 0.40, "L": 40.0, "r": 0.02, "T": 2.0}),
        ("High-Lev",     {"V": 100.0, "σV": 0.20, "L": 90.0, "r": 0.04, "T": 1.0}),
        ("Low-Lev",      {"V": 100.0, "σV": 0.30, "L": 20.0, "r": 0.03, "T": 1.0}),
    ]
    all_ok = True
    for name, c in cases:
        E = equity_value_from_v(c["V"], c["σV"], c["L"], c["r"], c["T"])
        sE = equity_vol_from_v(c["V"], c["σV"], c["L"], c["r"], c["T"], E)
        res = merton_kmv(E, sE, c["L"], c["r"], c["T"])
        err_V = abs(res.asset_value - c["V"]) / c["V"]
        err_s = abs(res.asset_vol - c["σV"]) / c["σV"]
        ok = res.converged and err_V < 1e-7 and err_s < 1e-7
        all_ok &= ok
        print(f"      {name:<14}  V err={err_V:.1e}  σ_V err={err_s:.1e}  "
              f"iter={res.iterations:3d}  {'OK' if ok else 'FAIL'}")
    return all_ok


def _hull_spot_test() -> bool:
    """Hull, Risk Mgmt & Financial Inst.: E=3, L=10, σ_E=0.80, r=0.05, T=1.
    Erwartung: V≈12.40, PD ≈ 12-13%."""
    print("  [2] Hull-Spot-Test (E=3, L=10, σ_E=0.80, r=0.05, T=1):")
    res = merton_kmv(equity=3.0, equity_vol=0.80,
                     debt=10.0, r=0.05, T=1.0)
    # Sanity: V im erwarteten Range, PD im erwarteten Range
    ok_v = 12.0 < res.asset_value < 13.0
    ok_pd = 0.10 < res.pd < 0.15
    ok_conv = res.converged
    print(f"      V={res.asset_value:.4f}  σ_V={res.asset_vol:.4f}  "
          f"DD={res.distance_to_default:.4f}  PD={res.pd*100:.2f}%  "
          f"iter={res.iterations}")
    print(f"      converged={ok_conv}  V in (12,13)={ok_v}  "
          f"PD in (10%,15%)={ok_pd}")
    return ok_v and ok_pd and ok_conv


def _monotonicity_test() -> bool:
    """PD-Sensitivitäten: ↑L, ↑σ → PD↑;  ↑r, ↑V/L → PD↓."""
    print("  [3] Sanity-Monotonie:")
    base = dict(equity=100.0, equity_vol=0.30, debt=80.0, r=0.03, T=1.0)
    pd_base = merton_kmv(**base).pd

    pd_more_debt = merton_kmv(**{**base, "debt": 95.0}).pd
    pd_more_vol = merton_kmv(**{**base, "equity_vol": 0.50}).pd
    pd_more_r = merton_kmv(**{**base, "r": 0.08}).pd
    pd_more_T = merton_kmv(**{**base, "T": 3.0}).pd

    checks = [
        ("Debt 80 → 95",      pd_more_debt > pd_base, pd_base, pd_more_debt),
        ("Vol 0.30 → 0.50",   pd_more_vol  > pd_base, pd_base, pd_more_vol),
        ("Rate 0.03 → 0.08",  pd_more_r    < pd_base, pd_base, pd_more_r),
        ("Horizon 1y → 3y",   pd_more_T    > pd_base, pd_base, pd_more_T),
    ]
    all_ok = True
    for label, ok, pd_a, pd_b in checks:
        all_ok &= ok
        print(f"      {label:<22}  PD: {pd_a*100:6.3f}% → {pd_b*100:6.3f}%  "
              f"{'OK' if ok else 'FAIL'}")
    return all_ok


def _boundary_test() -> bool:
    """V/L → ∞: PD → 0;  KMV-Konvergenz in <50 iter."""
    print("  [4] Boundary:")
    # V/L sehr gross
    res_big = merton_kmv(equity=1000.0, equity_vol=0.10,
                         debt=10.0, r=0.03, T=1.0)
    ok_zero = res_big.pd < 1e-10
    print(f"      Equity≫Debt → PD={res_big.pd:.2e}  "
          f"{'OK' if ok_zero else 'FAIL'}")
    # Iter < 50 für realistische Inputs
    ok_iter = res_big.iterations < 50
    print(f"      KMV-Iter (Low-Lev) = {res_big.iterations}  "
          f"{'OK (<50)' if ok_iter else 'FAIL'}")
    return ok_zero and ok_iter


def _real_dax40_run() -> bool:
    """Echter DAX-40-Run: alle 38 Firmen über die Pipeline."""
    print("  [5] DAX-40 Run (echte Daten):")
    try:
        from config import CACHE_DIR  # type: ignore
    except ImportError:
        print("      [SKIP] config nicht importierbar.")
        return True

    needed = ["dax40_prices.parquet", "dax40_fundamentals.parquet",
              "svensson_params.parquet"]
    missing = [n for n in needed if not (CACHE_DIR / n).exists()]
    if missing:
        print(f"      [SKIP] fehlende Cache-Files: {missing}")
        return True

    prices = pd.read_parquet(CACHE_DIR / "dax40_prices.parquet")
    fund = pd.read_parquet(CACHE_DIR / "dax40_fundamentals.parquet")
    svensson = pd.read_parquet(CACHE_DIR / "svensson_params.parquet")

    df = run_dax40(prices, fund, svensson)
    n = len(df)
    n_ok = int(df["Converged"].fillna(False).sum())
    n_skipped = int(df["DebtSource"].fillna("").str.startswith("skipped").sum())

    pd_min = df["PD"].min(skipna=True) * 100
    pd_max = df["PD"].max(skipna=True) * 100
    pd_med = df["PD"].median(skipna=True) * 100

    print(f"      Firmen total = {n}, KMV converged = {n_ok}, skipped = {n_skipped}")
    print(f"      PD range     = [{pd_min:.4f}%, {pd_max:.4f}%]   median = {pd_med:.4f}%")

    # Top-3 PDs (ausser skipped)
    valid = df[df["Converged"]].sort_values("PD", ascending=False)
    print(f"      Top-3 riskiest:")
    for tk, row in valid.head(3).iterrows():
        print(f"        {tk:<8} {row['Name']:<22}  "
              f"DD={row['DD']:5.2f}  PD={row['PD']*100:6.3f}%  "
              f"({row['Sector']})")
    print(f"      Bottom-3 (lowest PD):")
    for tk, row in valid.tail(3).iterrows():
        print(f"        {tk:<8} {row['Name']:<22}  "
              f"DD={row['DD']:5.2f}  PD={row['PD']*100:.2e}%  "
              f"({row['Sector']})")

    # Akzeptanz-Kriterien: ≥80% konvergiert, alle PDs in [0,1]
    rate_ok = n_ok >= 0.8 * (n - n_skipped)
    range_ok = (df["PD"].dropna() >= 0).all() and (df["PD"].dropna() <= 1).all()
    print(f"      ≥80% converged: {'OK' if rate_ok else 'FAIL'}    "
          f"PD ∈ [0,1]: {'OK' if range_ok else 'FAIL'}")
    return rate_ok and range_ok


if __name__ == "__main__":
    print("=" * 70)
    print(" merton.py · Validierung")
    print("=" * 70)

    results = [
        _round_trip_test(),
        _hull_spot_test(),
        _monotonicity_test(),
        _boundary_test(),
        _real_dax40_run(),
    ]

    print("=" * 70)
    if all(results):
        print(f" [PASS]  Alle {len(results)} Test-Blöcke bestanden.")
    else:
        n_fail = sum(1 for r in results if not r)
        print(f" [FAIL]  {n_fail}/{len(results)} Test-Blöcke fehlgeschlagen.")
    print("=" * 70)
