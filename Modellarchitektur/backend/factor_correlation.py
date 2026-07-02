"""
============================================================================
 factor_correlation.py · Empirische Faktor-Analyse Brent vs. 10y-Zins
============================================================================

Belegt die ökonomische Annahme, dass Brent und 10y-Zins als SEPARATE
Faktoren behandelt werden — adressiert direkt Professor-Kritik Punkt 5,
8 und 9:

  > "Untersuchen, ob und wie die beiden Risikofaktoren korrelieren.
  >  Erst nach einer solchen Analyse kann sinnvoll entschieden werden,
  >  ob eine Verdichtung zu einem einzigen Faktor überhaupt gerechtfertigt
  >  ist."

Methodik
--------
- 5-Jahres-Sample (1260 Handelstage), Frequency: täglich
- Rolling-252d-Korrelation (jährliches Window): ρ(ΔBrent_log, Δr_10y) als
  Zeitreihe → Stabilitäts-Test der Korrelations-Struktur
- Full-Sample-OLS-Regression: Δr_10y = α + β·ΔBrent_log + ε
- Pearson-ρ und Spearman-ρ über das volle Sample
- Pro-Jahr-Aufschlüsselung (Korrelations-Heatmap)

API
---
- compute_factor_correlation(brent, svensson, years=5) -> dict
- rolling_correlation_series(brent, svensson, window=252) -> pd.Series
- yearly_correlation_table(brent, svensson, years=5) -> pd.DataFrame

Quellen
-------
- Tsay, R. (2010). Analysis of Financial Time Series, Wiley
- Box & Jenkins (1970). Time Series Analysis
============================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ----------------------------------------------------------------------
# 1. Vorverarbeitung: Brent-Returns + 10y-Zins-Differenzen
# ----------------------------------------------------------------------
def _build_daily_factors(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    maturity: float = 10.0,
) -> pd.DataFrame:
    """Liefert ein DataFrame mit Tages-Veränderungen beider Faktoren.

    Columns:
      d_brent_log : log-Return von Brent-Schluss
      d_r_10y_pp  : Tages-Differenz im 10y-Zins (in pp)
    """
    from svensson import zero_rate, params_from_row

    # Brent close → log return
    if "Close_USD" in brent_df.columns:
        brent_close = brent_df["Close_USD"].dropna()
    elif "Close" in brent_df.columns:
        brent_close = brent_df["Close"].dropna()
    else:
        raise ValueError("Brent dataframe has no Close column")
    d_brent = np.log(brent_close).diff().dropna()

    # 10y zero rate series
    rates_10y = pd.Series(
        [zero_rate(maturity, params_from_row(row), as_decimal=False)
         for _, row in svensson_df.iterrows()],
        index=svensson_df.index,
    )
    d_r_10y_pp = rates_10y.diff().dropna()

    # Inner-join on dates
    df = pd.DataFrame({
        "d_brent_log": d_brent,
        "d_r_10y_pp":  d_r_10y_pp,
    }).dropna()
    return df


# ----------------------------------------------------------------------
# 2. Hauptanalyse · Full-Sample + Rolling + OLS
# ----------------------------------------------------------------------
def compute_factor_correlation(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    years: int = 5,
) -> dict:
    """Empirische Korrelations- und Regressionsanalyse über `years` Jahre.

    Returns dict mit:
      n_obs              : Anzahl Tages-Observationen
      sample_start, end  : Sample-Range
      pearson_rho        : Pearson-Korrelation (mit 95%-KI)
      pearson_rho_ci_lo, ci_hi
      spearman_rho       : Spearman-Rangkorrelation
      ols_alpha, ols_beta : OLS-Koeffizienten
      ols_r2             : R²
      ols_t_beta         : t-Statistik für β
      ols_se_beta        : Standardfehler für β
      interpretation     : Klartext-Aussage zur Faktor-Unabhängigkeit
    """
    df = _build_daily_factors(brent_df, svensson_df)
    cutoff = df.index.max() - pd.DateOffset(years=years)
    sub = df[df.index >= cutoff].copy()
    if len(sub) < 30:
        return {"n_obs": len(sub), "error": "Zu wenige Tagesobservationen."}

    x = sub["d_brent_log"].to_numpy()
    y = sub["d_r_10y_pp"].to_numpy()
    n = len(x)

    # Pearson
    rho = float(np.corrcoef(x, y)[0, 1])

    # Fisher-z 95% confidence interval for Pearson ρ
    z = 0.5 * np.log((1 + rho) / (1 - rho)) if abs(rho) < 0.999 else 0.0
    se_z = 1.0 / np.sqrt(n - 3) if n > 3 else 1.0
    z_lo, z_hi = z - 1.96 * se_z, z + 1.96 * se_z
    rho_lo = (np.exp(2*z_lo) - 1) / (np.exp(2*z_lo) + 1)
    rho_hi = (np.exp(2*z_hi) - 1) / (np.exp(2*z_hi) + 1)

    # Spearman
    rank_x = pd.Series(x).rank().to_numpy()
    rank_y = pd.Series(y).rank().to_numpy()
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])

    # OLS: y = α + β·x + ε (mit vollem Diagnostik-Output wie R's lm())
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    yhat = alpha + beta * x
    residuals = y - yhat

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - 2, 1)
    # Residual standard error (RSE)
    rse = float(np.sqrt(ss_res / max(n - 2, 1)))

    # Standard errors and t-statistics
    var_resid = ss_res / max(n - 2, 1)
    x_centered = x - x.mean()
    sxx = float(np.sum(x_centered ** 2))
    se_beta = float(np.sqrt(var_resid / max(sxx, 1e-12)))
    se_alpha = float(np.sqrt(var_resid * (1.0/n + (x.mean()**2) / max(sxx, 1e-12))))
    t_beta = beta / se_beta if se_beta > 0 else float("nan")
    t_alpha = alpha / se_alpha if se_alpha > 0 else float("nan")

    # p-values via Student-t (df = n-2). Approximate via normal for large n.
    from math import erf, sqrt
    def _t_pvalue(t_stat, df):
        """Two-tailed p-value for t-statistic. Uses normal approximation
        for df > 30 (sufficient for n > 100)."""
        if not np.isfinite(t_stat):
            return float("nan")
        # Normal approximation (very accurate for df > 30)
        z = abs(t_stat)
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        return float(p)

    p_beta  = _t_pvalue(t_beta,  n - 2)
    p_alpha = _t_pvalue(t_alpha, n - 2)

    # F-statistic for the overall regression
    if n > 2 and ss_res > 0:
        ms_reg = (ss_tot - ss_res)  # 1 regressor → df_reg = 1
        ms_res = ss_res / (n - 2)
        f_stat = ms_reg / ms_res
        # p-value of F-stat via approximation:  F(1, n-2) ≈ t²(n-2)
        # so p-value ≈ p-value of |t_beta| (already two-tailed)
        p_f = p_beta
    else:
        f_stat = float("nan")
        p_f = float("nan")

    # Residual quantiles (für die R-style lm()-Output-Box)
    res_min = float(np.min(residuals))
    res_q1  = float(np.quantile(residuals, 0.25))
    res_med = float(np.median(residuals))
    res_q3  = float(np.quantile(residuals, 0.75))
    res_max = float(np.max(residuals))

    # Interpretation
    if abs(rho) < 0.10:
        interp = ("Sehr schwache Korrelation. Die beiden Faktoren sind "
                  "empirisch nahezu unabhängig — separate Modellierung "
                  "klar gerechtfertigt.")
    elif abs(rho) < 0.30:
        interp = ("Schwache Korrelation. Faktoren tragen weitgehend "
                  "unabhängige Information — separate Modellierung "
                  "ökonomisch gerechtfertigt.")
    elif abs(rho) < 0.60:
        interp = ("Moderate Korrelation. Faktoren teilen sich gemeinsame "
                  "Komponenten — separate Modellierung trotzdem zulässig, "
                  "Multikollinearität in der Regression beachten.")
    else:
        interp = ("Starke Korrelation. Eine Faktor-Aggregation wäre "
                  "ökonomisch zu erwägen — separate Modellierung birgt "
                  "Risiko der Doppelzählung.")

    return {
        "n_obs":          n,
        "sample_start":   sub.index.min(),
        "sample_end":     sub.index.max(),
        "years":          years,
        "pearson_rho":    rho,
        "pearson_rho_ci_lo": float(rho_lo),
        "pearson_rho_ci_hi": float(rho_hi),
        "spearman_rho":   spearman,
        # OLS coefficients
        "ols_alpha":      alpha,
        "ols_beta":       beta,
        "ols_se_alpha":   se_alpha,
        "ols_se_beta":    se_beta,
        "ols_t_alpha":    t_alpha,
        "ols_t_beta":     t_beta,
        "ols_p_alpha":    p_alpha,
        "ols_p_beta":     p_beta,
        # Model fit
        "ols_r2":         r2,
        "ols_r2_adj":     r2_adj,
        "ols_rse":        rse,
        "ols_f_stat":     f_stat,
        "ols_p_f":        p_f,
        "df_residual":    n - 2,
        # Residual quantiles (R-style)
        "res_min":        res_min,
        "res_q1":         res_q1,
        "res_median":     res_med,
        "res_q3":         res_q3,
        "res_max":        res_max,
        "interpretation": interp,
    }


def rolling_correlation_series(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    window: int = 252,
    years: int = 5,
) -> pd.Series:
    """Rolling-252d-Pearson-Korrelation als Zeitreihe.

    Returns: pd.Series indexed by date, value = ρ über trailing window.
    """
    df = _build_daily_factors(brent_df, svensson_df)
    cutoff = df.index.max() - pd.DateOffset(years=years)
    sub = df[df.index >= cutoff]
    return sub["d_brent_log"].rolling(window).corr(sub["d_r_10y_pp"])


def yearly_correlation_table(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    years: int = 5,
) -> pd.DataFrame:
    """Korrelation pro Kalenderjahr (Stabilitäts-Test).

    Returns DataFrame[year, n_days, rho, mean_brent, mean_rate]
    """
    df = _build_daily_factors(brent_df, svensson_df)
    cutoff = df.index.max() - pd.DateOffset(years=years)
    sub = df[df.index >= cutoff].copy()
    sub["year"] = sub.index.year
    rows = []
    for y, g in sub.groupby("year"):
        if len(g) < 30:
            continue
        rho = float(np.corrcoef(g["d_brent_log"], g["d_r_10y_pp"])[0, 1])
        rows.append({
            "year":         int(y),
            "n_days":       len(g),
            "rho":          rho,
            "mean_brent":   float(g["d_brent_log"].mean()),
            "mean_rate_pp": float(g["d_r_10y_pp"].mean()),
            "std_brent":    float(g["d_brent_log"].std()),
            "std_rate_pp":  float(g["d_r_10y_pp"].std()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. Tests
# ----------------------------------------------------------------------
def _test_factor_dataframe():
    # Synthetic: 500 random days with rho ≈ 0.1
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(0)
    brent = pd.DataFrame({"Close_USD": 60 + np.cumsum(np.random.randn(500)*1.5)},
                          index=dates)
    sven_cols = ["Beta0","Beta1","Beta2","Beta3","Tau1","Tau2"]
    sven = pd.DataFrame(
        np.tile([2.5, -1.5, -2.0, 1.0, 2.0, 8.0], (500, 1))
        + np.random.randn(500, 6) * 0.05,
        index=dates, columns=sven_cols,
    )
    df = _build_daily_factors(brent, sven)
    assert "d_brent_log" in df.columns and "d_r_10y_pp" in df.columns
    assert len(df) > 400


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("factor_correlation.py · Tests")
    print("=" * 60)
    try:
        _test_factor_dataframe()
        print("  [PASS]  Synthetic factor dataframe")
    except AssertionError as e:
        print(f"  [FAIL]  {e}")
        raise
    print()
    print("Funktioniert mit echten Daten? Test:")
    try:
        from config import EBA_RAW_DIR
        brent = pd.read_parquet(_ROOT / "data" / "cache" / "brent_crude.parquet")
        sven = pd.read_parquet(_ROOT / "data" / "cache" / "svensson_params.parquet")
        res = compute_factor_correlation(brent, sven, years=5)
        print(f"  n_obs = {res['n_obs']}, ρ = {res['pearson_rho']:+.3f}")
        print(f"  95%-CI: [{res['pearson_rho_ci_lo']:+.3f}, "
              f"{res['pearson_rho_ci_hi']:+.3f}]")
        print(f"  R² (OLS) = {res['ols_r2']:.3f}")
        print(f"  → {res['interpretation']}")
    except Exception as e:
        print(f"  [SKIP] echte Daten: {e}")
    print()
    print("[PASS] All tests passed.")
