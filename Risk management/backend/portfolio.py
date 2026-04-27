"""
============================================================================
 backend/portfolio.py
 ---------------------------------------------------------------------------
 Aggregations-Layer: Einzelfirmen-PDs → Portfolio-Risiko-Metriken.

 Berechnete Metriken (siehe MODEL_ASSUMPTIONS.md §7a):
   - Expected Loss (EL)        EL = Σ EAD_i · LGD · PD_i
   - Stress-EL (Mean / p99)    EL über Stress-PD-Verteilung
   - Loss-Distribution         pro MC-Pfad: L_i = Σ EAD_j · LGD · PD_ij
   - VaR / CVaR                Quantile bzw. Tail-Mean der Loss-Verteilung
   - Konzentration (HHI)       Herfindahl-Hirschman-Index auf EAD-Anteilen
   - Sektor-Aggregation        Verlust-Beiträge je Yahoo-Sektor

 Konventionen:
   - EAD       = DPT (Default Point) je Firma  → ökonomische Default-Exposure
   - LGD       = config.DEFAULT_LGD = 0.45     → Basel-Standard
   - Korrel.   = via gemeinsame Faktoren in MC, nicht zusätzlich modelliert
   - Default   = Conditional EL pro Pfad (PD_ij · EAD_j · LGD), KEIN
                 zusätzliches Bernoulli-Sampling (Toggle in V2 möglich)

 Funktionen:
   - portfolio_expected_loss(pds, eads, lgd)           → float
   - portfolio_loss_distribution(pd_samples, eads, lgd) → ndarray
   - portfolio_var_cvar(losses, alpha)                  → (VaR, CVaR)
   - portfolio_hhi(eads)                                → float
   - effective_n(eads)                                  → float
   - aggregate(mc_df, merton_df, *, lgd, ead_col)       → dict
   - run_dax40(prices, brent, svensson, fundamentals)   → dict

 Validierung:  python backend/portfolio.py
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Mapping

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ----------------------------------------------------------------------
# 1. Atomic Metrics
# ----------------------------------------------------------------------
def portfolio_expected_loss(
    pds: np.ndarray,
    eads: np.ndarray,
    lgd: float,
) -> float:
    """EL = Σ EAD_i · LGD · PD_i.  Skalarer Erwartungswert."""
    pds = np.asarray(pds, dtype=float)
    eads = np.asarray(eads, dtype=float)
    return float(np.nansum(pds * eads) * lgd)


def portfolio_loss_distribution(
    pd_samples: np.ndarray,
    eads: np.ndarray,
    lgd: float,
    *,
    sample_defaults: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Loss pro Pfad über alle Firmen.

    Parameters
    ----------
    pd_samples : ndarray (n_sims, n_firms)
        PD-Realisation pro MC-Pfad und Firma. NaN-Werte werden als 0
        behandelt (Firma trägt nicht zum Verlust bei).
    eads : ndarray (n_firms,)
        Default-Exposures, gleiche Reihenfolge wie pd_samples-Spalten.
    lgd : float
        Loss Given Default (Dezimal).
    sample_defaults : bool, default False
        - False: Conditional Expected Loss pro Pfad
                 L_i = Σ_j EAD_j · LGD · PD_ij  (glatt, schnell)
        - True : Bernoulli-Sampling pro (Pfad, Firma)
                 D_ij ~ Bernoulli(PD_ij), L_i = Σ_j EAD_j · LGD · D_ij
                 (echte Loss-Distribution mit Default-Jumps; rauschiger)
    rng : np.random.Generator
        Erforderlich für sample_defaults=True (separater PRNG-Stream).

    Returns
    -------
    ndarray (n_sims,) — Verlust pro Pfad.
    """
    pd_samples = np.asarray(pd_samples, dtype=float)
    eads = np.asarray(eads, dtype=float)
    pds_safe = np.where(np.isnan(pd_samples), 0.0, pd_samples)

    if sample_defaults:
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(0.0, 1.0, size=pds_safe.shape)
        D = (u < pds_safe).astype(np.float64)
        contribs = D * eads * lgd
    else:
        contribs = pds_safe * eads * lgd

    return contribs.sum(axis=1)


def portfolio_var_cvar(
    losses: np.ndarray,
    *,
    alpha: float = 0.99,
) -> tuple[float, float]:
    """Value-at-Risk und Conditional VaR (Expected Shortfall).

    VaR_α   = α-Quantil der Loss-Verteilung
    CVaR_α  = E[ Loss | Loss ≥ VaR_α ]
    """
    losses = np.asarray(losses, dtype=float)
    losses = losses[~np.isnan(losses)]
    if len(losses) == 0:
        return float("nan"), float("nan")
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


def portfolio_hhi(eads: np.ndarray) -> float:
    """Herfindahl-Hirschman-Index auf EAD-Anteile (in [0, 1]).

    HHI = Σ w_i², mit w_i = EAD_i / Σ EAD_j.
    HHI = 1/N entspricht perfekter Gleichgewichtung,
    HHI = 1 entspricht 100% Konzentration auf eine Firma.
    """
    eads = np.asarray(eads, dtype=float)
    eads = eads[~np.isnan(eads)]
    total = eads.sum()
    if total <= 0:
        return float("nan")
    weights = eads / total
    return float(np.sum(weights ** 2))


def effective_n(eads: np.ndarray) -> float:
    """Effektive Anzahl Firmen = 1 / HHI.

    Maß für die Diversifikations-Tiefe: bei N gleich-gewichteten Firmen
    ist effective_n = N. Bei Konzentration sinkt sie.
    """
    hhi = portfolio_hhi(eads)
    return float(1.0 / hhi) if hhi > 0 else float("nan")


# ----------------------------------------------------------------------
# 2. Aggregation
# ----------------------------------------------------------------------
def aggregate(
    mc_df: pd.DataFrame,
    merton_df: pd.DataFrame,
    *,
    lgd: float,
    ead_col: str = "DPT",
    pd_samples_matrix: Optional[np.ndarray] = None,
    sample_defaults: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """Aggregiert Einzelfirmen-Risiken zu Portfolio-Metriken.

    Parameters
    ----------
    mc_df : output of monte_carlo.run_monte_carlo_dax40
        Muss enthalten: BaselinePD, StressPDMean, StressPDp99, Sector,
        Converged.
    merton_df : output of merton.run_dax40
        Muss enthalten: DPT (oder andere ead_col), Sector.
    lgd : Loss Given Default (Dezimal).
    ead_col : Spalte in merton_df, die als EAD genutzt wird.
    pd_samples_matrix : optional ndarray (n_sims, n_firms)
        Pfad-PDs pro Firma. Reihenfolge muss zu mc_df.index passen.
        Wenn None, werden VaR/CVaR und Loss-Distribution übersprungen.
    sample_defaults : Bernoulli-Variante für Loss-Distribution.
    seed : PRNG-Seed für sample_defaults.

    Returns
    -------
    dict mit
        n_firms, n_converged, total_ead,
        el_baseline, el_stress_mean, el_stress_p99,
        var_99, cvar_99, var_95, cvar_95,
        hhi, effective_n,
        loss_distribution (optional), sector_breakdown (DataFrame).
    """
    valid = mc_df[mc_df["Converged"].fillna(False)].copy()
    n_total = len(mc_df)
    n_ok = len(valid)

    # EAD-Spalte aus merton_df übernehmen (für die konvergierten Firmen)
    eads_full = merton_df.loc[valid.index, ead_col].astype(float)
    valid = valid.assign(EAD=eads_full)

    eads = eads_full.to_numpy()
    pd_base = valid["BaselinePD"].to_numpy()
    pd_stress_mean = valid["StressPDMean"].to_numpy()
    pd_stress_p99 = valid["StressPDp99"].to_numpy()

    out = {
        "n_firms": n_total,
        "n_converged": n_ok,
        "lgd": lgd,
        "ead_col": ead_col,
        "total_ead": float(np.nansum(eads)),
        "el_baseline":     portfolio_expected_loss(pd_base,        eads, lgd),
        "el_stress_mean":  portfolio_expected_loss(pd_stress_mean, eads, lgd),
        "el_stress_p99":   portfolio_expected_loss(pd_stress_p99,  eads, lgd),
        "hhi":          portfolio_hhi(eads),
        "effective_n":  effective_n(eads),
    }

    # Loss-Distribution & VaR/CVaR (nur wenn pd_samples_matrix vorhanden)
    if pd_samples_matrix is not None:
        rng = np.random.default_rng(seed) if seed is not None else None
        losses = portfolio_loss_distribution(
            pd_samples_matrix, eads, lgd,
            sample_defaults=sample_defaults, rng=rng,
        )
        var95, cvar95 = portfolio_var_cvar(losses, alpha=0.95)
        var99, cvar99 = portfolio_var_cvar(losses, alpha=0.99)
        out.update({
            "loss_distribution": losses,
            "var_95":  var95,  "cvar_95":  cvar95,
            "var_99":  var99,  "cvar_99":  cvar99,
            "loss_mean": float(np.nanmean(losses)),
            "loss_std":  float(np.nanstd(losses, ddof=1)),
        })

    # Sektor-Breakdown
    sector_rows = []
    for sector, grp in valid.groupby("Sector", dropna=False):
        eads_s = grp["EAD"].to_numpy()
        pd_b_s = grp["BaselinePD"].to_numpy()
        pd_s_s = grp["StressPDMean"].to_numpy()
        pd_99_s = grp["StressPDp99"].to_numpy()
        sector_rows.append({
            "Sector": sector if pd.notna(sector) else "(unknown)",
            "NFirms": len(grp),
            "EAD": float(np.nansum(eads_s)),
            "EADShare": float(np.nansum(eads_s) / out["total_ead"])
                if out["total_ead"] > 0 else float("nan"),
            "ELBaseline":   portfolio_expected_loss(pd_b_s,  eads_s, lgd),
            "ELStressMean": portfolio_expected_loss(pd_s_s,  eads_s, lgd),
            "ELStressP99":  portfolio_expected_loss(pd_99_s, eads_s, lgd),
        })
    sb = pd.DataFrame(sector_rows).sort_values("EAD", ascending=False)
    sb["ELShareBaseline"] = sb["ELBaseline"] / sb["ELBaseline"].sum() \
        if sb["ELBaseline"].sum() > 0 else np.nan
    out["sector_breakdown"] = sb.reset_index(drop=True)

    return out


# ----------------------------------------------------------------------
# 3. End-to-End Convenience
# ----------------------------------------------------------------------
def run_dax40(
    prices_df: pd.DataFrame,
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    *,
    lgd: Optional[float] = None,
    ead_col: str = "DPT",
    sample_defaults: bool = False,
    mc_kwargs: Optional[Mapping] = None,
) -> dict:
    """End-to-End: führt MC + Portfolio-Aggregation in einem Aufruf aus.

    Returns
    -------
    dict mit zusätzlichen Feldern:
        'merton_df'        : Output von merton.run_dax40
        'factor_df'        : Output von factor_model.run_dax40 (in mc_df enthalten)
        'mc_df'            : Output von monte_carlo.run_monte_carlo_dax40
        ...alle Felder von aggregate()
    """
    from config import DEFAULT_LGD  # type: ignore
    from monte_carlo import run_monte_carlo_dax40  # type: ignore
    from merton import run_dax40 as run_merton  # type: ignore

    if lgd is None:
        lgd = DEFAULT_LGD
    if mc_kwargs is None:
        mc_kwargs = {}

    # Baseline-Merton (separat — wird auch vom MC intern aufgerufen, aber
    # wir brauchen das Resultat hier explizit für DPT-EAD)
    merton_df = run_merton(prices_df, fundamentals_df, svensson_df)

    # MC mit pd_samples
    mc_df = run_monte_carlo_dax40(
        prices_df, brent_df, svensson_df, fundamentals_df,
        keep_samples=True, **mc_kwargs,
    )

    # PD-Samples-Matrix (n_sims × n_converged_firms)
    valid_mc = mc_df[mc_df["Converged"].fillna(False)]
    pd_matrix = np.column_stack([
        valid_mc.loc[tk, "StressPDSamples"]
        for tk in valid_mc.index
    ]) if len(valid_mc) > 0 else None

    seed = mc_kwargs.get("seed")
    seed_for_bernoulli = (seed + 2) if seed is not None else None

    agg = aggregate(
        mc_df, merton_df,
        lgd=lgd, ead_col=ead_col,
        pd_samples_matrix=pd_matrix,
        sample_defaults=sample_defaults,
        seed=seed_for_bernoulli,
    )
    agg["merton_df"] = merton_df
    agg["mc_df"] = mc_df
    return agg


# ============================================================================
#  VALIDATION  ·  python backend/portfolio.py
# ============================================================================

def _atomic_metrics_test() -> bool:
    """EL-, HHI-, VaR/CVaR-Atomarformeln gegen handgerechnete Werte."""
    print("  [1] Atomic Metrics:")
    pds  = np.array([0.01, 0.02, 0.05, 0.10])
    eads = np.array([100., 200., 300., 400.])
    lgd  = 0.45

    # EL = (0.01·100 + 0.02·200 + 0.05·300 + 0.10·400) · 0.45
    #    = (1 + 4 + 15 + 40) · 0.45 = 60 · 0.45 = 27.0
    el = portfolio_expected_loss(pds, eads, lgd)
    err_el = abs(el - 27.0)
    ok_el = err_el < 1e-10
    print(f"      EL: {el:.4f}  (expected 27.0)  {'OK' if ok_el else 'FAIL'}")

    # HHI: weights = [100, 200, 300, 400] / 1000 = [0.1, 0.2, 0.3, 0.4]
    # HHI = 0.01 + 0.04 + 0.09 + 0.16 = 0.30
    hhi = portfolio_hhi(eads)
    err_hhi = abs(hhi - 0.30)
    ok_hhi = err_hhi < 1e-10
    print(f"      HHI: {hhi:.4f}  (expected 0.30)  {'OK' if ok_hhi else 'FAIL'}")

    # Effective N = 1 / 0.30 = 3.333...
    eff_n = effective_n(eads)
    ok_n = abs(eff_n - 1/0.30) < 1e-10
    print(f"      Eff-N: {eff_n:.4f}  (expected 3.333)  {'OK' if ok_n else 'FAIL'}")

    # VaR/CVaR: deterministische Loss-Reihe [10, 20, 30, ..., 100]
    losses = np.arange(10, 101, 10, dtype=float)  # 10 Werte
    var_95, cvar_95 = portfolio_var_cvar(losses, alpha=0.95)
    # 95%-Quantil bei 10 Werten: zwischen Index 8 und 9 → ~95
    print(f"      VaR_95={var_95:.2f}  CVaR_95={cvar_95:.2f}")
    ok_var = 90 <= var_95 <= 100 and cvar_95 >= var_95

    return ok_el and ok_hhi and ok_n and ok_var


def _conditional_vs_bernoulli_test() -> bool:
    """Conditional EL ≈ Mean(Bernoulli Losses) für grosses n_sims."""
    print("  [2] Conditional EL vs. Bernoulli-Sampling:")
    rng = np.random.default_rng(0)
    n_sims, n_firms = 50_000, 5
    pds = np.full((n_sims, n_firms), 0.05)
    eads = np.array([100., 200., 150., 250., 100.])
    lgd = 0.40

    losses_cond = portfolio_loss_distribution(pds, eads, lgd,
                                              sample_defaults=False)
    losses_bern = portfolio_loss_distribution(pds, eads, lgd,
                                              sample_defaults=True, rng=rng)
    mean_cond = float(losses_cond.mean())
    mean_bern = float(losses_bern.mean())
    rel_err = abs(mean_bern - mean_cond) / mean_cond
    ok = rel_err < 0.01
    print(f"      Mean Conditional = {mean_cond:.4f}")
    print(f"      Mean Bernoulli   = {mean_bern:.4f}  (rel err {rel_err:.2%})  "
          f"{'OK' if ok else 'FAIL'}")
    # Bernoulli hat höhere Varianz (Default-Jumps)
    std_cond = float(losses_cond.std())
    std_bern = float(losses_bern.std())
    print(f"      Std Conditional  = {std_cond:.4f}  (deterministisch — keine Streuung)")
    print(f"      Std Bernoulli    = {std_bern:.4f}  (>0, Default-Jumps)")
    return ok


def _hhi_edge_test() -> bool:
    """HHI-Grenzen: vollständig konzentriert → 1; gleich-gewichtet → 1/N."""
    print("  [3] HHI-Grenzfälle:")
    one_only = np.array([1000.0, 0.0, 0.0, 0.0])
    equal = np.array([100.0, 100.0, 100.0, 100.0])
    h1 = portfolio_hhi(one_only)
    h2 = portfolio_hhi(equal)
    print(f"      [1000,0,0,0] → HHI = {h1:.4f}  (expected 1.0)  "
          f"{'OK' if abs(h1-1.0) < 1e-10 else 'FAIL'}")
    print(f"      [100×4]      → HHI = {h2:.4f}  (expected 0.25)  "
          f"{'OK' if abs(h2-0.25) < 1e-10 else 'FAIL'}")
    return abs(h1-1.0) < 1e-10 and abs(h2-0.25) < 1e-10


def _real_dax40_run() -> bool:
    """End-to-End-DAX-40-Run."""
    print("  [4] DAX-40 End-to-End:")
    try:
        from config import CACHE_DIR  # type: ignore
    except ImportError:
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
    res = run_dax40(prices, brent, sven, fund)
    elapsed = time.perf_counter() - t0

    print(f"      Pipeline-Laufzeit: {elapsed:.1f}s")
    print(f"      Firmen total/converged: {res['n_firms']}/{res['n_converged']}")
    print(f"      Total EAD (Σ DPT):     {res['total_ead']/1e9:.2f} bn EUR")
    print(f"      LGD:                   {res['lgd']*100:.0f}%")
    print()
    print(f"      Expected Loss Baseline:    {res['el_baseline']/1e6:>8.2f} m EUR  "
          f"({res['el_baseline']/res['total_ead']*100:.4f}% des EAD)")
    print(f"      Expected Loss Stress μ:    {res['el_stress_mean']/1e6:>8.2f} m EUR  "
          f"({res['el_stress_mean']/res['total_ead']*100:.4f}%)")
    print(f"      Expected Loss Stress p99:  {res['el_stress_p99']/1e6:>8.2f} m EUR  "
          f"({res['el_stress_p99']/res['total_ead']*100:.4f}%)")
    print()
    if "var_99" in res:
        print(f"      VaR_95:    {res['var_95']/1e6:>8.2f} m EUR")
        print(f"      CVaR_95:   {res['cvar_95']/1e6:>8.2f} m EUR")
        print(f"      VaR_99:    {res['var_99']/1e6:>8.2f} m EUR")
        print(f"      CVaR_99:   {res['cvar_99']/1e6:>8.2f} m EUR")
    print()
    print(f"      HHI:                {res['hhi']:.4f}")
    print(f"      Effective N (1/HHI): {res['effective_n']:.1f}  "
          f"(N nominal = {res['n_converged']})")
    print()
    print(f"      Sektor-Breakdown (top-5 nach EAD):")
    sb = res["sector_breakdown"].head(5)
    for _, row in sb.iterrows():
        print(f"        {row['Sector']:<22} N={row['NFirms']:>2}  "
              f"EAD={row['EAD']/1e9:>6.1f}bn ({row['EADShare']*100:5.1f}%)  "
              f"EL_base={row['ELBaseline']/1e6:>6.1f}m  "
              f"EL_p99={row['ELStressP99']/1e6:>6.1f}m")

    # Sanity: EL_p99 ≥ EL_baseline (Stress sollte EL erhöhen)
    ok_monoton = res["el_stress_p99"] >= res["el_baseline"]
    # Sanity: Loss-Distribution > 0
    ok_dist = "loss_distribution" in res and res["loss_distribution"].mean() > 0
    print(f"      EL_p99 ≥ EL_base: {'OK' if ok_monoton else 'FAIL'}    "
          f"Loss-Dist > 0: {'OK' if ok_dist else 'FAIL'}")
    return ok_monoton and ok_dist


if __name__ == "__main__":
    print("=" * 70)
    print(" portfolio.py · Validierung")
    print("=" * 70)

    results = [
        _atomic_metrics_test(),
        _conditional_vs_bernoulli_test(),
        _hhi_edge_test(),
        _real_dax40_run(),
    ]

    print("=" * 70)
    if all(results):
        print(f" [PASS]  Alle {len(results)} Test-Blöcke bestanden.")
    else:
        n_fail = sum(1 for r in results if not r)
        print(f" [FAIL]  {n_fail}/{len(results)} Test-Blöcke fehlgeschlagen.")
    print("=" * 70)
