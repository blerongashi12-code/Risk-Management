"""
============================================================================
 backend/monte_carlo.py
 ---------------------------------------------------------------------------
 Stress-Engine: 2-Faktor-MVN Pfade × Faktor-Modell × Stress-Merton/KMV.

 Schock-Vektor (täglich):  z_t = [r_brent_t, Δr_10y_t]
 Verteilung:               z_t ~ N(μ, Σ)
                           μ, Σ aus 252-Tage-Historie der gleichen
                           Faktoren wie in factor_model.factor_returns.
 Pfad-Generierung:         Vektorisiert via Cholesky-Decomposition.
 Translation auf Firma:    log r_E_H = α·H + β_E_adj·Z_brent
                                       + β_R·Z_Δr + ε·√H
 Stress-Anwendung:         E_stress = E_0 · exp(log r_E_H)
                           r_stress = r_baseline + Z_Δr / 100
                           KMV neu lösen + Sektor-σ_V-Multiplier.

 Modell-Konsistenz: BetaBrentAdjusted (Sektor-Energy-Mul) und
 Sektor-σ_V-Multiplier (Banken-Adj) werden identisch zur Baseline
 verwendet — siehe MODEL_ASSUMPTIONS.md §6a.

 Funktionen:
   - factor_stats(brent_df, svensson_df)             → dict(μ, Σ, ...)
   - simulate_factor_paths(μ, Σ, n_sims, horizon)    → ndarray (N, 2)
   - stress_one_firm(...)                            → dict (PD-Stats)
   - run_monte_carlo_dax40(...)                      → pd.DataFrame

 Validierung: python backend/monte_carlo.py
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
# 1. Historische Faktor-Statistiken
# ----------------------------------------------------------------------
def factor_stats(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    lookback: int = 252,
    maturity: float = 10.0,
) -> dict:
    """Berechnet μ und Σ der täglichen Faktor-Returns aus Historie.

    Parameters
    ----------
    brent_df : aus 03_fetch_brent_crude.py (Spalte 'Return_log').
    svensson_df : Bundesbank-Parameter-Zeitreihe.
    lookback : letzte N Handelstage.
    maturity : Δr-Maturity (default 10y).

    Returns
    -------
    dict mit
        'mu'    : ndarray (2,) — täglicher Mittelwert
        'sigma' : ndarray (2,2) — tägliche Kovarianzmatrix
        'corr'  : ndarray (2,2) — tägliche Korrelationsmatrix
        'n_obs' : int — Anzahl Beobachtungen
        'cols'  : Liste der Faktor-Namen
    """
    from factor_model import factor_returns  # type: ignore

    fr = factor_returns(brent_df, svensson_df, maturity=maturity).dropna()
    if len(fr) > lookback:
        fr = fr.iloc[-lookback:]

    return {
        "mu":    fr.mean().to_numpy(),
        "sigma": fr.cov().to_numpy(),
        "corr":  fr.corr().to_numpy(),
        "n_obs": int(len(fr)),
        "cols":  list(fr.columns),
    }


# ----------------------------------------------------------------------
# 2. Pfad-Simulation (vektorisiert)
# ----------------------------------------------------------------------
def simulate_factor_paths(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    n_sims: int,
    horizon_days: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Multivariate-Normal kumulative Schocks am Horizont.

    Pfad-Modell: tägliche Schocks z_t ~ N(μ, Σ), iid, summiert über H Tage.
    Cholesky-Decomposition → Schock = μ + L·u mit u ~ N(0, I).

    Parameters
    ----------
    mu : ndarray (d,)
    sigma : ndarray (d, d), positiv-semidefinit.
    n_sims : Anzahl Pfade.
    horizon_days : Horizont H in Handelstagen.
    seed : Reproduzierbarkeits-Seed.

    Returns
    -------
    ndarray (n_sims, d) — kumulierter Endzustand pro Pfad.
    Hinweis: Single-step Σ wird über H Tage summiert; das entspricht
    Var(Z_H) = H · Σ unter iid-Annahme.
    """
    rng = np.random.default_rng(seed)
    d = len(mu)

    # Cholesky; falls sigma exakt singulär (z.B. Σ=0 für Determinismus-Test),
    # füge winzigen Jitter an die Diagonale.
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        jitter = 1e-15 * np.eye(d)
        L = np.linalg.cholesky(sigma + jitter)

    # iid Standard-Normal (n_sims, horizon, d)
    u = rng.standard_normal(size=(n_sims, horizon_days, d))
    # Tägliche korrelierte Schocks: μ + u @ L.T
    daily = mu + u @ L.T
    # Kumulieren über Horizon-Achse
    return daily.sum(axis=1)


# ----------------------------------------------------------------------
# 3. Vektorisiertes KMV (intern, identisch zur scalar-Version in merton.py)
# ----------------------------------------------------------------------
def _kmv_vec(
    E: np.ndarray,
    sigma_E: float,
    L: float,
    r: np.ndarray,
    T: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Vektorisierte Fixpunkt-Iteration für N Pfade gleichzeitig.

    E, r können ndarrays sein; sigma_E, L, T sind scalars.
    Liefert (V, σ_V) ndarrays gleicher Shape wie E.
    Konsistent zur scalar-Version merton.merton_kmv (verifiziert via Test).
    """
    sqrtT = np.sqrt(T)
    V = E + L
    sigma_V = sigma_E * E / V

    for _ in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(V / L) + (r + 0.5 * sigma_V ** 2) * T) / (
                sigma_V * sqrtT
            )
            d2 = d1 - sigma_V * sqrtT
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)
            V_new = np.where(
                Nd1 > 1e-12,
                (E + L * np.exp(-r * T) * Nd2) / np.where(Nd1 > 1e-12, Nd1, 1.0),
                V,
            )
            sigma_V_new = np.where(
                (V_new * Nd1) > 1e-12,
                sigma_E * E / np.where(V_new * Nd1 > 1e-12, V_new * Nd1, 1.0),
                sigma_V,
            )

        max_rel = float(np.max(np.abs(V_new - V) / np.maximum(np.abs(V), 1e-12)))
        V, sigma_V = V_new, sigma_V_new
        if max_rel < tol:
            break

    return V, sigma_V


# ----------------------------------------------------------------------
# 4. Stress pro Firma
# ----------------------------------------------------------------------
def stress_one_firm(
    *,
    E0: float,
    sigma_E: float,
    L: float,
    r0: float,
    T_years: float,
    alpha: float,
    beta_brent: float,
    beta_dr: float,
    sigma_eps: float,
    sector_vol_mul: float,
    factor_paths: np.ndarray,
    horizon_days: int,
    include_idio: bool,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Berechnet Stress-PD-Distribution für eine Firma.

    Pipeline pro Pfad i ∈ 1..N:
        log_r_E_H_i = α·H + β_E·Z_brent_i + β_R·Z_Δr_i + ε_i·√H
        E_stress_i  = E_0 · exp(log_r_E_H_i)
        r_stress_i  = r_0 + Z_Δr_i / 100         (Δr in pp → Dezimal)
        (V_i, σ_V_i) = KMV(E_stress_i, σ_E, L, r_stress_i, T)
        σ_V_adj_i  = σ_V_i · sector_vol_mul
        DD_i = [ln(V_i/L) + (r_stress_i − ½σ_V_adj_i²)·T] / (σ_V_adj_i·√T)
        PD_i = N(−DD_i)

    Parameters
    ----------
    E0, sigma_E, L : Equity, Equity-Vola, Default Point (Baseline).
    r0 : Risk-free rate (Dezimal) der Baseline.
    T_years : Merton-Horizon in Jahren.
    alpha, beta_brent, beta_dr, sigma_eps : OLS-Parameter aus factor_model.
        beta_brent ist bereits BetaBrentAdjusted (Sektor-Energy-Mul drin).
    sector_vol_mul : σ_V-Multiplier (1.5 für Banken etc.).
    factor_paths : ndarray (N, 2), kumulierte Schocks [Z_brent, Z_Δr_pp].
    horizon_days : H, MC-Horizont in Tagen (NICHT zwingend identisch zu
        T_years·252; default ist beides 252 = 1 Jahr).
    include_idio : ε im Stress-Pfad mitziehen.
    rng : separater PRNG-Stream für ε; default = neuer Generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    Z_brent = factor_paths[:, 0]    # log-Returns Brent kumuliert
    Z_dr_pp = factor_paths[:, 1]    # Δr_10y kumuliert in Prozentpunkten

    # Cumulative log-equity-return über H Tage
    log_ret = (
        alpha * horizon_days
        + beta_brent * Z_brent
        + beta_dr * Z_dr_pp
    )
    if include_idio and sigma_eps > 0:
        log_ret = log_ret + rng.normal(0.0, sigma_eps * np.sqrt(horizon_days),
                                       size=log_ret.shape)

    E_stress = E0 * np.exp(log_ret)
    r_stress = r0 + Z_dr_pp / 100.0   # Δr pp → Dezimal-Anpassung an risk-free

    # Vektorisiertes KMV
    V_st, sigma_V_st = _kmv_vec(E_stress, sigma_E, L, r_stress, T_years)

    # Sektor-σ_V-Multiplier (post-KMV, identisch zur Baseline-Logik)
    sigma_V_adj = sigma_V_st * sector_vol_mul

    sqrtT = np.sqrt(T_years)
    with np.errstate(divide="ignore", invalid="ignore"):
        DD = (np.log(V_st / L) + (r_stress - 0.5 * sigma_V_adj ** 2) * T_years
              ) / (sigma_V_adj * sqrtT)
    PD = np.clip(norm.cdf(-DD), 0.0, 1.0)

    return {
        "pd_samples": PD,
        "pd_mean": float(np.nanmean(PD)),
        "pd_p50": float(np.nanpercentile(PD, 50)),
        "pd_p95": float(np.nanpercentile(PD, 95)),
        "pd_p99": float(np.nanpercentile(PD, 99)),
        "pd_max": float(np.nanmax(PD)),
        "dd_mean": float(np.nanmean(DD)),
        "log_ret_mean": float(np.nanmean(log_ret)),
        "log_ret_std":  float(np.nanstd(log_ret, ddof=1)),
    }


# ----------------------------------------------------------------------
# 5. DAX-40 MC-Wrapper
# ----------------------------------------------------------------------
def run_monte_carlo_dax40(
    prices_df: pd.DataFrame,
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    *,
    n_sims: Optional[int] = None,
    horizon_days: Optional[int] = None,
    seed: Optional[int] = None,
    include_idio: Optional[bool] = None,
    keep_samples: bool = False,
) -> pd.DataFrame:
    """Volle MC-Pipeline für alle DAX-40-Firmen.

    Pipeline:
      1. Baseline-Merton  (merton.run_dax40)
      2. Faktor-Modell    (factor_model.run_dax40)
      3. Faktor-Stats     (μ, Σ aus 252-Tage-Historie)
      4. Pfad-Simulation  (n_sims × horizon × 2 Faktoren)
      5. Pro Firma:        stress_one_firm

    Returns
    -------
    pd.DataFrame, Index = Ticker, Columns:
        Name, Sector, BaselinePD, StressPDMean, StressPDp50, StressPDp95,
        StressPDp99, StressPDMax, DeltaPDMean, Converged, SkipReason.
    Wenn keep_samples=True: zusätzlich 'StressPDSamples' (Liste).
    """
    from config import (MC_N_SIMS, MC_HORIZON_DAYS, MC_SEED,  # type: ignore
                        MC_INCLUDE_IDIO,
                        SECTOR_VOL_MULTIPLIER,
                        DEFAULT_SECTOR_VOL_MULTIPLIER,
                        DEFAULT_HORIZON)
    from merton import run_dax40 as run_merton  # type: ignore
    from factor_model import run_dax40 as run_factor  # type: ignore

    if n_sims is None:        n_sims = MC_N_SIMS
    if horizon_days is None:  horizon_days = MC_HORIZON_DAYS
    if seed is None:          seed = MC_SEED
    if include_idio is None:  include_idio = MC_INCLUDE_IDIO

    # 1. + 2. Baseline-Merton und Faktor-Modell
    merton_df  = run_merton(prices_df, fundamentals_df, svensson_df)
    factor_df  = run_factor(prices_df, brent_df, svensson_df, fundamentals_df)

    # 3. Faktor-Stats
    stats = factor_stats(brent_df, svensson_df, lookback=horizon_days)

    # 4. Pfade (einmal für alle Firmen)
    paths = simulate_factor_paths(
        stats["mu"], stats["sigma"],
        n_sims=n_sims, horizon_days=horizon_days, seed=seed,
    )

    # Separater PRNG für idio-Schocks (unabhängig von Faktor-Pfaden)
    rng = np.random.default_rng(seed + 1)

    rows = []
    for ticker in fundamentals_df.index:
        m_row = merton_df.loc[ticker] if ticker in merton_df.index else None
        f_row = factor_df.loc[ticker] if ticker in factor_df.index else None

        # Skip-Bedingung: Baseline oder Faktoren fehlen
        baseline_ok = (m_row is not None
                       and bool(m_row.get("Converged", False))
                       and pd.notna(m_row.get("PD")))
        factor_ok = (f_row is not None
                     and bool(f_row.get("Converged", False))
                     and pd.notna(f_row.get("BetaBrentAdjusted")))
        if not (baseline_ok and factor_ok):
            reason = ("merton not converged"
                      if not baseline_ok else "factor not converged")
            rows.append({
                "Name": fundamentals_df.loc[ticker].get("Name"),
                "Sector": fundamentals_df.loc[ticker].get("Sector_Yahoo"),
                "BaselinePD": (float(m_row["PD"])
                               if m_row is not None and pd.notna(m_row.get("PD"))
                               else float("nan")),
                "StressPDMean": float("nan"),
                "StressPDp50": float("nan"),
                "StressPDp95": float("nan"),
                "StressPDp99": float("nan"),
                "StressPDMax": float("nan"),
                "DeltaPDMean": float("nan"),
                "Converged": False,
                "SkipReason": reason,
            })
            continue

        sector = m_row["Sector"]
        sec_mul = (SECTOR_VOL_MULTIPLIER.get(sector, DEFAULT_SECTOR_VOL_MULTIPLIER)
                   if sector else DEFAULT_SECTOR_VOL_MULTIPLIER)

        result = stress_one_firm(
            E0=float(m_row["MarketCap"]),
            sigma_E=float(m_row["EquityVol"]),
            L=float(m_row["DPT"]),
            r0=float(m_row["Rate"]),
            T_years=DEFAULT_HORIZON,
            alpha=float(f_row["Alpha"]),
            beta_brent=float(f_row["BetaBrentAdjusted"]),
            beta_dr=float(f_row["BetaDr"]),
            sigma_eps=float(f_row["SigmaEps"]),
            sector_vol_mul=sec_mul,
            factor_paths=paths,
            horizon_days=horizon_days,
            include_idio=include_idio,
            rng=rng,
        )

        row = {
            "Name": m_row["Name"],
            "Sector": sector,
            "BaselinePD": float(m_row["PD"]),
            "StressPDMean": result["pd_mean"],
            "StressPDp50": result["pd_p50"],
            "StressPDp95": result["pd_p95"],
            "StressPDp99": result["pd_p99"],
            "StressPDMax": result["pd_max"],
            "DeltaPDMean": result["pd_mean"] - float(m_row["PD"]),
            "Converged": True,
            "SkipReason": "",
        }
        if keep_samples:
            row["StressPDSamples"] = result["pd_samples"]
        rows.append(row)

    out = pd.DataFrame(rows, index=fundamentals_df.index)
    out.index.name = "Ticker"
    return out


# ============================================================================
#  VALIDATION  ·  python backend/monte_carlo.py
# ============================================================================

def _determinism_test() -> bool:
    """Σ ≈ 0 → alle Pfade ≈ 0."""
    print("  [1] Determinismus (Σ ≈ 0):")
    paths = simulate_factor_paths(
        np.zeros(2), np.eye(2) * 1e-30,
        n_sims=100, horizon_days=10, seed=0,
    )
    max_abs = float(np.abs(paths).max())
    ok = max_abs < 1e-10
    print(f"      max|Z| = {max_abs:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def _distribution_test() -> bool:
    """Sim μ ≈ μ·H und Σ ≈ Σ·H (innerhalb Stichproben-Toleranz)."""
    print("  [2] Verteilung (Mittelwert + Kovarianz):")
    mu = np.array([0.001, 0.005])
    sig = np.array([[0.0001, 0.00005], [0.00005, 0.0002]])
    H = 100
    N = 50_000

    paths = simulate_factor_paths(mu, sig, n_sims=N, horizon_days=H, seed=7)
    sim_mu = paths.mean(axis=0)
    sim_cov = np.cov(paths.T)
    target_mu = mu * H
    target_cov = sig * H

    # Standard-Fehler des Stichproben-Mittels: sqrt(diag(Σ_H)/N)
    se_mu = np.sqrt(np.diag(target_cov) / N)
    err_mu = np.abs(sim_mu - target_mu)
    ok_mu = (err_mu < 4 * se_mu).all()  # 4σ-Toleranz

    rel_cov = np.abs(sim_cov - target_cov) / np.maximum(np.abs(target_cov), 1e-10)
    ok_cov = (rel_cov < 0.05).all()  # 5% relativer Fehler

    print(f"      sim μ = {sim_mu}, target = {target_mu}")
    print(f"      err_mu = {err_mu}, 4·SE = {4*se_mu}  {'OK' if ok_mu else 'FAIL'}")
    print(f"      max rel err Σ = {rel_cov.max():.3f}  {'OK' if ok_cov else 'FAIL'}")
    return ok_mu and ok_cov


def _reproducibility_test() -> bool:
    """Gleicher Seed → bit-identische Pfade."""
    print("  [3] Reproduzierbarkeit:")
    args = dict(mu=np.zeros(2), sigma=np.eye(2) * 0.01,
                n_sims=200, horizon_days=20, seed=99)
    p1 = simulate_factor_paths(**args)
    p2 = simulate_factor_paths(**args)
    ok = bool(np.array_equal(p1, p2))
    print(f"      identical: {'OK' if ok else 'FAIL'}")
    return ok


def _vec_kmv_consistency_test() -> bool:
    """Vektorisiertes KMV liefert gleiche Werte wie scalar merton.merton_kmv."""
    print("  [4] _kmv_vec ↔ merton.merton_kmv:")
    sys.path.insert(0, str(_ROOT / "backend"))
    from merton import merton_kmv  # type: ignore

    cases = [
        dict(equity=100.0, equity_vol=0.30, debt=60.0, r=0.03, T=1.0),
        dict(equity=50.0,  equity_vol=0.50, debt=40.0, r=0.04, T=2.0),
        dict(equity=200.0, equity_vol=0.20, debt=100.0, r=0.02, T=1.0),
        dict(equity=10.0,  equity_vol=0.40, debt=80.0, r=0.05, T=1.0),  # high lev
    ]
    all_ok = True
    for c in cases:
        scalar = merton_kmv(**c)
        V_v, s_v = _kmv_vec(
            np.array([c["equity"]]), c["equity_vol"], c["debt"],
            np.array([c["r"]]), c["T"],
        )
        err_V = abs(V_v[0] - scalar.asset_value) / max(abs(scalar.asset_value), 1e-12)
        err_s = abs(s_v[0] - scalar.asset_vol)   / max(abs(scalar.asset_vol),   1e-12)
        ok = err_V < 1e-5 and err_s < 1e-5
        all_ok &= ok
        print(f"      E={c['equity']:6.1f} σ_E={c['equity_vol']:.2f} L={c['debt']:5.1f}  "
              f"V err={err_V:.1e}  σ_V err={err_s:.1e}  "
              f"{'OK' if ok else 'FAIL'}")
    return all_ok


def _stress_baseline_consistency_test() -> bool:
    """Mit Σ → 0 und α=β_E=β_R=σ_ε=0 → Stress-PD ≡ Baseline-PD."""
    print("  [5] Baseline-Konsistenz (alle Faktoren neutralisiert):")
    paths = simulate_factor_paths(
        np.zeros(2), np.eye(2) * 1e-30,
        n_sims=500, horizon_days=252, seed=11,
    )
    res = stress_one_firm(
        E0=100.0, sigma_E=0.30, L=80.0, r0=0.03, T_years=1.0,
        alpha=0.0, beta_brent=0.0, beta_dr=0.0, sigma_eps=0.0,
        sector_vol_mul=1.5, factor_paths=paths, horizon_days=252,
        include_idio=False,
    )
    # Baseline mit gleichem Sektor-Multiplier
    from merton import merton_kmv, distance_to_default, merton_pd  # type: ignore
    base = merton_kmv(equity=100.0, equity_vol=0.30, debt=80.0, r=0.03, T=1.0)
    sigma_v_adj = base.asset_vol * 1.5
    base_dd = distance_to_default(base.asset_value, sigma_v_adj, 80.0, 0.03, 1.0)
    base_pd = merton_pd(base_dd)

    err = abs(res["pd_mean"] - base_pd)
    ok = err < 1e-6
    print(f"      Baseline-PD={base_pd*100:.6f}%  Stress-Mean={res['pd_mean']*100:.6f}%  "
          f"err={err:.1e}  {'OK' if ok else 'FAIL'}")
    return ok


def _real_dax40_run() -> bool:
    """Realer MC-Lauf für alle 38 Firmen + Top-Stress-Reporting."""
    print("  [6] DAX-40 MC-Run (echte Daten, n_sims=10k, H=252):")
    try:
        from config import CACHE_DIR  # type: ignore
    except ImportError:
        print("      [SKIP] config nicht importierbar.")
        return True

    needed = ["dax40_prices.parquet", "brent_crude.parquet",
              "svensson_params.parquet", "dax40_fundamentals.parquet"]
    missing = [n for n in needed if not (CACHE_DIR / n).exists()]
    if missing:
        print(f"      [SKIP] missing: {missing}")
        return True

    prices = pd.read_parquet(CACHE_DIR / "dax40_prices.parquet")
    brent  = pd.read_parquet(CACHE_DIR / "brent_crude.parquet")
    sven   = pd.read_parquet(CACHE_DIR / "svensson_params.parquet")
    fund   = pd.read_parquet(CACHE_DIR / "dax40_fundamentals.parquet")

    import time
    t0 = time.perf_counter()
    df = run_monte_carlo_dax40(prices, brent, sven, fund)
    elapsed = time.perf_counter() - t0

    n_ok = int(df["Converged"].fillna(False).sum())
    print(f"      {n_ok}/{len(df)} Firmen erfolgreich gestresst, "
          f"Laufzeit = {elapsed:.1f}s")

    valid = df[df["Converged"]].sort_values("StressPDMean", ascending=False)
    print(f"      Top-5 Stress-PD-Mean (1Y, mit Sektor-Multipliern):")
    for tk, row in valid.head(5).iterrows():
        print(f"        {tk:<8} {row['Name']:<22}  "
              f"Base={row['BaselinePD']*100:6.3f}%  "
              f"Stress μ={row['StressPDMean']*100:6.3f}%  "
              f"p95={row['StressPDp95']*100:6.3f}%  "
              f"p99={row['StressPDp99']*100:6.3f}%  "
              f"Δ={row['DeltaPDMean']*100:+6.3f}pp  "
              f"({row['Sector']})")

    # Plausi: ΔPD im Mittel ≥ 0 für die Top-Stress-Treiber
    top10_delta = float(valid.head(10)["DeltaPDMean"].mean())
    print(f"      Mean ΔPD across top-10 = {top10_delta*100:+.4f}pp  "
          f"{'OK' if top10_delta >= -0.001 else 'FAIL'}")

    rate_ok = n_ok >= 0.8 * len(df)
    return rate_ok and (top10_delta >= -0.001)


if __name__ == "__main__":
    print("=" * 70)
    print(" monte_carlo.py · Validierung")
    print("=" * 70)

    results = [
        _determinism_test(),
        _distribution_test(),
        _reproducibility_test(),
        _vec_kmv_consistency_test(),
        _stress_baseline_consistency_test(),
        _real_dax40_run(),
    ]

    print("=" * 70)
    if all(results):
        print(f" [PASS]  Alle {len(results)} Test-Blöcke bestanden.")
    else:
        n_fail = sum(1 for r in results if not r)
        print(f" [FAIL]  {n_fail}/{len(results)} Test-Blöcke fehlgeschlagen.")
    print("=" * 70)
