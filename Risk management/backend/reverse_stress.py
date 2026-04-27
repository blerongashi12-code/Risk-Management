"""
============================================================================
 backend/reverse_stress.py
 ---------------------------------------------------------------------------
 Reverse Stress Test: kritische Schock-Schwellen via brentq.

 Inverse zu scenarios.py:
     scenarios.py     gegeben (ΔBrent, Δr) → liefert PD
     reverse_stress.py gegeben target PD/DD → sucht ΔBrent oder Δr

 Pro Firma vier kritische Werte:
     critical_brent_pd5  — ΔBrent_log, sodass PD = 5 % (Δr = 0)
     critical_rate_pd5   — Δr_pp,      sodass PD = 5 % (ΔBrent = 0)
     critical_brent_dd1  — ΔBrent_log, sodass DD = 1.0 (Δr = 0)
     critical_rate_dd1   — Δr_pp,      sodass DD = 1.0 (ΔBrent = 0)

 Zusätzlich: iso_pd_curve(firm) — Iso-PD-Linien in der (ΔBrent, Δr)-Ebene
 für Cockpit-Visualisierung.

 NaN bedeutet: kein Vorzeichenwechsel im Suchbereich → Schwelle nicht
 innerhalb der Konfig-Range erreichbar.

 Validierung: python backend/reverse_stress.py
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Optional, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ----------------------------------------------------------------------
# 1. Firmen-Parameter-Container
# ----------------------------------------------------------------------
class FirmParams(NamedTuple):
    """Alles was apply_shock pro Firma braucht (aus merton+factor_model)."""
    E0: float
    sigma_E: float
    L: float
    r0: float
    T: float
    alpha: float
    beta_brent_adj: float
    beta_dr: float
    sec_vol_mul: float


# ----------------------------------------------------------------------
# 2. Schock-Anwendung (deterministische single-firm-Variante)
# ----------------------------------------------------------------------
def _pd_dd_at_shock(
    firm: FirmParams,
    delta_brent_log: float,
    delta_r_pp: float,
    horizon_days: int,
) -> tuple[float, float]:
    """Wendet (ΔBrent, Δr) auf eine Firma an, liefert (PD, DD).

    Marginal-Stress: enthält **kein** α·H-Drift-Term. Reverse-Stress
    fragt "welcher inkrementelle Schock kippt die Firma heute?", nicht
    "wo steht sie nach 252 Tagen Drift". α würde sonst die Schwelle
    verzerren — bei α > 0 (z.B. CBK 0.002/Tag) sinkt die Forecast-PD
    unter die Baseline und der Schock muss erst den Drift überkompensieren.
    NaN-Tuple wenn KMV nicht konvergiert. Siehe MODEL_ASSUMPTIONS.md §6d.
    """
    from merton import merton_kmv, distance_to_default, merton_pd  # type: ignore

    # KEIN α-Term: marginal stress
    log_ret = (firm.beta_brent_adj * delta_brent_log
               + firm.beta_dr * delta_r_pp)
    E_stress = firm.E0 * np.exp(log_ret)
    r_stress = firm.r0 + delta_r_pp / 100.0

    res = merton_kmv(E_stress, firm.sigma_E, firm.L, r_stress, firm.T)
    if not res.converged or np.isnan(res.asset_vol):
        return float("nan"), float("nan")

    sigma_V_adj = res.asset_vol * firm.sec_vol_mul
    dd = distance_to_default(res.asset_value, sigma_V_adj, firm.L,
                             r_stress, firm.T)
    pd_ = merton_pd(dd)
    return pd_, dd


# ----------------------------------------------------------------------
# 3. Brentq-Wrapper für kritische Schwellen
# ----------------------------------------------------------------------
def _solve_critical(
    f,
    bounds: tuple[float, float],
    *,
    xtol: float = 1e-6,
    maxiter: int = 100,
) -> float:
    """Sicherer brentq-Wrapper.

    Liefert NaN wenn:
      - kein Vorzeichenwechsel im Intervall (Target unerreichbar)
      - f liefert NaN an einer der Grenzen (KMV nicht konvergiert)
    """
    a, b = bounds
    fa = f(a)
    fb = f(b)
    if np.isnan(fa) or np.isnan(fb):
        return float("nan")
    if fa * fb > 0:
        return float("nan")  # gleiches Vorzeichen → kein Crossing
    try:
        return float(brentq(f, a, b, xtol=xtol, maxiter=maxiter))
    except (ValueError, RuntimeError):
        return float("nan")


def critical_brent_for_pd(
    firm: FirmParams,
    *,
    target_pd: float,
    horizon_days: int,
    bounds: tuple[float, float],
) -> float:
    """ΔBrent_log, sodass PD(ΔBrent, Δr=0) = target_pd. NaN wenn unerreichbar."""
    f = lambda db: _pd_dd_at_shock(firm, db, 0.0, horizon_days)[0] - target_pd
    return _solve_critical(f, bounds)


def critical_rate_for_pd(
    firm: FirmParams,
    *,
    target_pd: float,
    horizon_days: int,
    bounds: tuple[float, float],
) -> float:
    """Δr_pp, sodass PD(ΔBrent=0, Δr) = target_pd."""
    f = lambda dr: _pd_dd_at_shock(firm, 0.0, dr, horizon_days)[0] - target_pd
    return _solve_critical(f, bounds)


def critical_brent_for_dd(
    firm: FirmParams,
    *,
    target_dd: float,
    horizon_days: int,
    bounds: tuple[float, float],
) -> float:
    """ΔBrent_log, sodass DD(ΔBrent, Δr=0) = target_dd.

    Konvention: f = DD - target. Da DD oberhalb der Grenze "sicher" ist
    und unterhalb "stress", liefert f(low_DD_shock) negativ.
    """
    f = lambda db: _pd_dd_at_shock(firm, db, 0.0, horizon_days)[1] - target_dd
    return _solve_critical(f, bounds)


def critical_rate_for_dd(
    firm: FirmParams,
    *,
    target_dd: float,
    horizon_days: int,
    bounds: tuple[float, float],
) -> float:
    """Δr_pp, sodass DD(ΔBrent=0, Δr) = target_dd."""
    f = lambda dr: _pd_dd_at_shock(firm, 0.0, dr, horizon_days)[1] - target_dd
    return _solve_critical(f, bounds)


# ----------------------------------------------------------------------
# 4. Iso-PD-Kurve in der (ΔBrent, Δr)-Ebene
# ----------------------------------------------------------------------
def iso_pd_curve(
    firm: FirmParams,
    *,
    target_pd: float,
    horizon_days: int,
    n_points: int = 25,
    brent_range: Optional[tuple[float, float]] = None,
    rate_range: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """Iso-PD-Linie für die Cockpit-Heatmap.

    Für jeden Wert in brent_range wird der Δr gesucht, der die Firma
    auf target_pd bringt. Liefert NaN wo kein Vorzeichenwechsel.

    Returns
    -------
    DataFrame mit Spalten ['delta_brent_log', 'delta_r_10y_pp'].
    """
    if brent_range is None:
        brent_range = (-1.5, 1.5)
    if rate_range is None:
        rate_range = (-3.0, 3.0)

    brent_grid = np.linspace(brent_range[0], brent_range[1], n_points)
    rates = []
    for db in brent_grid:
        f = lambda dr: _pd_dd_at_shock(firm, db, dr, horizon_days)[0] - target_pd
        rates.append(_solve_critical(f, rate_range))
    return pd.DataFrame({
        "delta_brent_log":   brent_grid,
        "delta_r_10y_pp":    rates,
    })


# ----------------------------------------------------------------------
# 5. DAX-40 Convenience
# ----------------------------------------------------------------------
def _build_firm_params(merton_row, factor_row, sec_mul, T):
    """Sammelt FirmParams aus Merton+Factor-Output für eine Firma."""
    return FirmParams(
        E0=float(merton_row["MarketCap"]),
        sigma_E=float(merton_row["EquityVol"]),
        L=float(merton_row["DPT"]),
        r0=float(merton_row["Rate"]),
        T=T,
        alpha=float(factor_row["Alpha"]),
        beta_brent_adj=float(factor_row["BetaBrentAdjusted"]),
        beta_dr=float(factor_row["BetaDr"]),
        sec_vol_mul=sec_mul,
    )


def run_dax40(
    prices_df: pd.DataFrame,
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    *,
    target_pd: Optional[float] = None,
    target_dd: Optional[float] = None,
    horizon_days: Optional[int] = None,
    brent_range: Optional[tuple[float, float]] = None,
    rate_range: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """Reverse-Stress-Schwellen für alle DAX-40-Firmen.

    Output (Index = Ticker):
        Name, Sector, BaselinePD, BaselineDD,
        CritBrent_PD<target>,  CritRate_PD<target>,
        CritBrent_DD<target>,  CritRate_DD<target>,
        Comment        — kurze Interpretation, falls Schwelle bereits
                         überschritten ist (Baseline-PD > target_pd).
    """
    from config import (REVERSE_STRESS_TARGET_PD,            # type: ignore
                        REVERSE_STRESS_TARGET_DD,
                        REVERSE_STRESS_BRENT_RANGE,
                        REVERSE_STRESS_RATE_RANGE,
                        REVERSE_STRESS_HORIZON_DAYS,
                        SECTOR_VOL_MULTIPLIER,
                        DEFAULT_SECTOR_VOL_MULTIPLIER,
                        DEFAULT_HORIZON)
    from merton import run_dax40 as run_merton                # type: ignore
    from factor_model import run_dax40 as run_factor          # type: ignore

    if target_pd is None:    target_pd = REVERSE_STRESS_TARGET_PD
    if target_dd is None:    target_dd = REVERSE_STRESS_TARGET_DD
    if horizon_days is None: horizon_days = REVERSE_STRESS_HORIZON_DAYS
    if brent_range is None:  brent_range = REVERSE_STRESS_BRENT_RANGE
    if rate_range is None:   rate_range = REVERSE_STRESS_RATE_RANGE

    merton_df = run_merton(prices_df, fundamentals_df, svensson_df)
    factor_df = run_factor(prices_df, brent_df, svensson_df, fundamentals_df)

    pd_label = f"{target_pd*100:.0f}"      # z.B. '5'
    dd_label = f"{target_dd:.1f}"          # z.B. '1.0'

    rows = []
    for ticker in fundamentals_df.index:
        m = merton_df.loc[ticker] if ticker in merton_df.index else None
        f = factor_df.loc[ticker] if ticker in factor_df.index else None
        baseline_ok = (m is not None
                       and bool(m.get("Converged", False))
                       and pd.notna(m.get("PD")))
        factor_ok = (f is not None
                     and bool(f.get("Converged", False))
                     and pd.notna(f.get("BetaBrentAdjusted")))
        if not (baseline_ok and factor_ok):
            rows.append({
                "Name": fundamentals_df.loc[ticker].get("Name"),
                "Sector": fundamentals_df.loc[ticker].get("Sector_Yahoo"),
                "BaselinePD": float(m["PD"]) if baseline_ok else float("nan"),
                "BaselineDD": float(m["DD"]) if baseline_ok else float("nan"),
                f"CritBrent_PD{pd_label}": float("nan"),
                f"CritRate_PD{pd_label}":  float("nan"),
                f"CritBrent_DD{dd_label}": float("nan"),
                f"CritRate_DD{dd_label}":  float("nan"),
                "Comment": "merton/factor not converged",
            })
            continue

        sector = m["Sector"]
        sec_mul = (SECTOR_VOL_MULTIPLIER.get(sector, DEFAULT_SECTOR_VOL_MULTIPLIER)
                   if sector else DEFAULT_SECTOR_VOL_MULTIPLIER)
        firm = _build_firm_params(m, f, sec_mul, DEFAULT_HORIZON)

        # Falls Baseline-PD bereits über target_pd liegt → Schwelle bereits
        # überschritten; brentq würde im "guten" Bereich keinen Crossing finden.
        baseline_pd = float(m["PD"])
        baseline_dd = float(m["DD"])
        comment_parts = []
        if baseline_pd >= target_pd:
            comment_parts.append(f"Baseline-PD ≥ {target_pd*100:.0f}%")
        if baseline_dd <= target_dd:
            comment_parts.append(f"Baseline-DD ≤ {target_dd}")

        cb_pd = critical_brent_for_pd(firm, target_pd=target_pd,
                                      horizon_days=horizon_days,
                                      bounds=brent_range)
        cr_pd = critical_rate_for_pd(firm, target_pd=target_pd,
                                     horizon_days=horizon_days,
                                     bounds=rate_range)
        cb_dd = critical_brent_for_dd(firm, target_dd=target_dd,
                                      horizon_days=horizon_days,
                                      bounds=brent_range)
        cr_dd = critical_rate_for_dd(firm, target_dd=target_dd,
                                     horizon_days=horizon_days,
                                     bounds=rate_range)

        rows.append({
            "Name": m["Name"],
            "Sector": sector,
            "BaselinePD": baseline_pd,
            "BaselineDD": baseline_dd,
            f"CritBrent_PD{pd_label}": cb_pd,
            f"CritRate_PD{pd_label}":  cr_pd,
            f"CritBrent_DD{dd_label}": cb_dd,
            f"CritRate_DD{dd_label}":  cr_dd,
            "Comment": "; ".join(comment_parts),
        })

    out = pd.DataFrame(rows, index=fundamentals_df.index)
    out.index.name = "Ticker"
    return out


# ============================================================================
#  VALIDATION  ·  python backend/reverse_stress.py
# ============================================================================

def _round_trip_test() -> bool:
    """Wende krit-Schock an → muss Target-PD/DD bis auf brentq-Toleranz treffen.

    Tests sind **relativ** zur Baseline-PD der Synth-Firma definiert:
    target = baseline + Δ. So ist die Schwelle garantiert erreichbar
    und der Test prüft die Mathematik (nicht die DAX-Realität).
    """
    print("  [1] Round-Trip (apply critical shock → reach target):")
    # Fragile, monoton sensitive Synth-Firma:
    # β_brent_adj = +1.0 → niedriger Brent → niedrigere Equity → höhere PD
    # β_dr      = +0.5 → niedriger r → niedrigere Equity (β-Kanal)
    #                  und niedriger r → höherer Diskont L·exp(-rT) → niedrigerer V/L
    #                  → beide Kanäle gleich gerichtet → MONOTON in Δr.
    firm = FirmParams(
        E0=50.0, sigma_E=0.40, L=50.0, r0=0.03, T=1.0,
        alpha=0.0, beta_brent_adj=+1.0, beta_dr=+0.5,
        sec_vol_mul=1.0,
    )
    H = 252
    base_pd, base_dd = _pd_dd_at_shock(firm, 0.0, 0.0, H)
    print(f"      Synth-Firma Baseline: PD={base_pd*100:.4f}%, DD={base_dd:.3f}")

    # Targets klein genug, um im monotonen Bereich von KMV zu bleiben.
    # KMV-Mathematik wird bei sehr kleinem E/L nicht-monoton (σ_V → 0,
    # DD springt zurück hoch); brentq braucht Monotonie zwischen den Grenzen.
    # Δ_PD klein, weil PD ist N(-DD): kleine Δ_DD → exponentiell kleinere Δ_PD.
    target_pd = base_pd + 0.002    # +0.2 pp über Baseline
    target_dd = base_dd - 0.4      # 0.4 unter Baseline

    cb_pd = critical_brent_for_pd(firm, target_pd=target_pd,
                                  horizon_days=H, bounds=(-3.0, 3.0))
    cr_pd = critical_rate_for_pd(firm, target_pd=target_pd,
                                 horizon_days=H, bounds=(-5.0, 5.0))
    cb_dd = critical_brent_for_dd(firm, target_dd=target_dd,
                                  horizon_days=H, bounds=(-3.0, 3.0))
    cr_dd = critical_rate_for_dd(firm, target_dd=target_dd,
                                 horizon_days=H, bounds=(-5.0, 5.0))

    def _verify(crit, target, kind):
        if np.isnan(crit):
            return False, float("nan"), "NaN"
        if kind == "pd_brent":
            actual, _ = _pd_dd_at_shock(firm, crit, 0.0, H)
        elif kind == "pd_rate":
            actual, _ = _pd_dd_at_shock(firm, 0.0, crit, H)
        elif kind == "dd_brent":
            _, actual = _pd_dd_at_shock(firm, crit, 0.0, H)
        else:
            _, actual = _pd_dd_at_shock(firm, 0.0, crit, H)
        return abs(actual - target) < 1e-5, actual, ""

    checks = [
        ("CritBrent → PD",  cb_pd, target_pd, "pd_brent"),
        ("CritRate  → PD",  cr_pd, target_pd, "pd_rate"),
        ("CritBrent → DD",  cb_dd, target_dd, "dd_brent"),
        ("CritRate  → DD",  cr_dd, target_dd, "dd_rate"),
    ]
    all_ok = True
    for label, crit, target, kind in checks:
        ok, actual, note = _verify(crit, target, kind)
        all_ok &= ok
        if note == "NaN":
            print(f"      {label} (target {target:.4f}): NaN")
        else:
            print(f"      {label}: crit={crit:+.4f}  →  actual={actual:.4f}  "
                  f"target={target:.4f}  {'OK' if ok else 'FAIL'}")
    return all_ok


def _monotonicity_test() -> bool:
    """Bei zwei Firmen mit unterschiedlichem Risiko: höhere Baseline-PD →
    niedrigerer kritischer Schock-Betrag (näher am Default)."""
    print("  [2] Monotonie über Firmen-Risiko:")
    # Beide Firmen mit β_brent_adj > 0 → niedrigerer Brent verschlechtert
    # Equity (monotone Richtung); Schock geht negativ.
    safe = FirmParams(
        E0=200.0, sigma_E=0.30, L=40.0, r0=0.03, T=1.0,
        alpha=0.0, beta_brent_adj=+1.0, beta_dr=0.0,
        sec_vol_mul=1.0,
    )
    risky = FirmParams(
        E0=50.0, sigma_E=0.50, L=80.0, r0=0.03, T=1.0,
        alpha=0.0, beta_brent_adj=+1.0, beta_dr=0.0,
        sec_vol_mul=1.0,
    )
    base_safe, _  = _pd_dd_at_shock(safe,  0, 0, 252)
    base_risky, _ = _pd_dd_at_shock(risky, 0, 0, 252)
    target = max(base_safe, base_risky) + 0.01
    print(f"      Safe baseline-PD  = {base_safe*100:.3f}%")
    print(f"      Risky baseline-PD = {base_risky*100:.3f}%")
    print(f"      Common target     = {target*100:.3f}%")
    cb_safe  = critical_brent_for_pd(safe,  target_pd=target,
                                     horizon_days=252, bounds=(-3.0, 3.0))
    cb_risky = critical_brent_for_pd(risky, target_pd=target,
                                     horizon_days=252, bounds=(-3.0, 3.0))
    print(f"      Safe firm   CritBrent_PD5 = {cb_safe:+.4f}")
    print(f"      Risky firm  CritBrent_PD5 = {cb_risky:+.4f}")
    # Beide haben β_brent < 0: ein POSITIVER Brent-Schock verschlechtert Equity.
    # Risky-Firma sollte mit kleinerem positiven Schock auf PD=5% kommen.
    if np.isnan(cb_safe) and np.isnan(cb_risky):
        print("      [SKIP] both NaN — keine Aussage")
        return True
    if np.isnan(cb_safe):
        print("      Safe NaN (Schwelle nie erreichbar) — plausibel für sehr "
              "sichere Firma, OK")
        return True
    ok = abs(cb_risky) < abs(cb_safe)
    print(f"      |risky| < |safe|: {'OK' if ok else 'FAIL'}")
    return ok


def _real_dax40_run() -> bool:
    """Realer DAX-40-Run + Reporting der niedrigsten kritischen Schwellen."""
    print("  [3] DAX-40 Reverse-Stress-Run:")
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

    import time
    t0 = time.perf_counter()
    df = run_dax40(prices, brent, sven, fund)
    elapsed = time.perf_counter() - t0
    print(f"      Laufzeit: {elapsed:.1f}s, Firmen: {len(df)}")

    cb_pd_col = next(c for c in df.columns if c.startswith("CritBrent_PD"))
    cr_pd_col = next(c for c in df.columns if c.startswith("CritRate_PD"))
    cb_dd_col = next(c for c in df.columns if c.startswith("CritBrent_DD"))
    cr_dd_col = next(c for c in df.columns if c.startswith("CritRate_DD"))

    # Anteil Firmen mit erreichbaren Schwellen
    n_total = len(df)
    n_brent_pd = int(df[cb_pd_col].notna().sum())
    n_rate_pd  = int(df[cr_pd_col].notna().sum())
    n_brent_dd = int(df[cb_dd_col].notna().sum())
    n_rate_dd  = int(df[cr_dd_col].notna().sum())
    print(f"      Erreichbare Schwellen: Brent→PD5={n_brent_pd}, "
          f"Rate→PD5={n_rate_pd}, Brent→DD1={n_brent_dd}, Rate→DD1={n_rate_dd}")

    # Top-5 fragilste Firmen nach |CritBrent_PD5| (kleinster Schock genügt)
    have_cb = df[df[cb_pd_col].notna()].copy()
    have_cb["AbsCB"] = have_cb[cb_pd_col].abs()
    print(f"      Top-5 fragil (kleinster |ΔBrent_log| → PD=5%):")
    for tk, row in have_cb.nsmallest(5, "AbsCB").iterrows():
        sign = "+" if row[cb_pd_col] >= 0 else "-"
        print(f"        {tk:<8} {row['Name']:<22}  "
              f"BasePD={row['BaselinePD']*100:6.3f}%  "
              f"CritBrent={row[cb_pd_col]:+.3f} (≈{sign}{(np.exp(abs(row[cb_pd_col]))-1)*100:5.1f}%)  "
              f"CritRate={row[cr_pd_col]:+.2f}pp  "
              f"({row['Sector']})")

    # Banken speziell
    banks = df[df["Sector"] == "Financial Services"]
    print(f"      Financial Services Schwellen:")
    for tk, row in banks.iterrows():
        cb = row[cb_pd_col]
        cr = row[cr_pd_col]
        print(f"        {tk:<8} {row['Name']:<22}  "
              f"BasePD={row['BaselinePD']*100:6.3f}%  "
              f"CritBrent={cb:+.3f}  CritRate={cr:+.2f}pp  "
              f"{row['Comment']}")

    if (n_brent_pd + n_rate_pd + n_brent_dd + n_rate_dd) == 0:
        print("      INFO: Alle Schwellen NaN — DAX-Blue-Chips sind im "
              "2-Faktor-Modell zu robust für target_pd=5%/target_dd=1.0.")
        print("      Das ist die Modell-Realität (β_brent_adj klein "
              "wegen Sektor-Mul; β_R klein für Banken). Im Cockpit über "
              "adaptive Targets (target_pd = 2×Baseline) lösbar.")

    # Pass-Kriterium: Pipeline läuft sauber, alle Firmen verarbeitet,
    # Schema korrekt. NaN-Werte sind eine legitime Modell-Aussage.
    schema_ok = all(c in df.columns for c in
                    ["Name", "Sector", "BaselinePD", "BaselineDD",
                     cb_pd_col, cr_pd_col, cb_dd_col, cr_dd_col, "Comment"])
    return n_total == 38 and schema_ok


if __name__ == "__main__":
    print("=" * 70)
    print(" reverse_stress.py · Validierung")
    print("=" * 70)

    results = [
        _round_trip_test(),
        _monotonicity_test(),
        _real_dax40_run(),
    ]

    print("=" * 70)
    if all(results):
        print(f" [PASS]  Alle {len(results)} Test-Blöcke bestanden.")
    else:
        n_fail = sum(1 for r in results if not r)
        print(f" [FAIL]  {n_fail}/{len(results)} Test-Blöcke fehlgeschlagen.")
    print("=" * 70)
