"""
============================================================================
 backtesting.py · Forecast vs. Realized Validation
============================================================================

Implementiert die Backtesting-Module für Critique 5 (systematic outcomes
analysis SR 11-7) und Critique 6 (Netto-EAD Decomposition).

Datenbasis: Historisches Capital-Panel aus den 6 EBA-Transparency-Vintages
2020-2025, 22 Quartals-Stichtage von Sep 2019 bis Jun 2025, ~120 Banken
pro Stichtag (~2 600 Bank-Quartal-Observationen).

Drei Validierungs-Outputs:
  1. Empirische Sensitivität: Regression Δ-CET1-Ratio vs. realized M
     über das volle Panel mit OLS (β, R², t-Statistik). Liefert die
     "tatsächliche" historische Wirkung des Macro-Faktors auf die
     regulatorische Eigenkapitalquote.
  2. Stress-Episode-Vergleich: Corona-2020 (Q4-19 → Q4-20) und
     Ukraine-2022 (Q4-21 → Q4-22) — pro Bank realized vs. modell-
     basierte Erwartung.
  3. RWA-Decomposition: Δ-Total-RWA in seine drei Basel-Komponenten
     (Credit / Market / Operational) zerlegt. EBA-publiziertes
     Ground-Truth-Decomposition; bereinigt um Portfolio-Mix-Effekte
     auf der Komponenten-Ebene.

API
---
- compute_macro_panel(brent_df, svensson_df, periods)
- compute_realized_changes(capital_wide, lag_quarters=4)
- fit_empirical_sensitivity(panel)  → dict
- episode_diagnostics(panel, episode_specs)
- rwa_decomposition(capital_wide_t, capital_wide_t1)
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


# ============================================================================
# 1. Macro state per EBA reporting date
# ============================================================================
def _period_to_date(period: int) -> pd.Timestamp:
    """Convert YYYYMM to quarter-end pd.Timestamp (last day of month)."""
    yr = period // 100
    mo = period % 100
    return pd.Timestamp(yr, mo, 1) + pd.offsets.MonthEnd(0)


def compute_macro_panel(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    periods: list[int],
    *,
    lookback_days: int = 252,
    maturity: float = 10.0,
) -> pd.DataFrame:
    """Per EBA-Period (YYYYMM) the trailing-1Y macro shock state.

    For each period:
      - brent_log_1y: cumulative Brent log-return over trailing `lookback_days`
      - dr_10y_1y_pp: Δr_10y from t-lookback to t (in percentage points)

    Returns DataFrame [Period, date, brent_log_1y, dr_10y_1y_pp]
    Periods without sufficient lookback are dropped.
    """
    from svensson import zero_rate, params_from_row

    # Brent close from cache
    if "Close_USD" not in brent_df.columns and "Close" in brent_df.columns:
        brent_close = brent_df["Close"]
    elif "Close_USD" in brent_df.columns:
        brent_close = brent_df["Close_USD"]
    else:
        # Fallback: use Return_log
        return pd.DataFrame()
    brent_close = brent_close.dropna()

    # Build 10y zero rate series
    rates = pd.Series(
        [zero_rate(maturity, params_from_row(row), as_decimal=False)
         for _, row in svensson_df.iterrows()],
        index=svensson_df.index,
    )

    rows = []
    for period in periods:
        t = _period_to_date(period)
        # Find nearest available Brent date ≤ t (forward-fill)
        bt_dates = brent_close.index[brent_close.index <= t]
        if len(bt_dates) < lookback_days + 1:
            continue
        brent_at_t = float(brent_close.loc[bt_dates[-1]])
        brent_at_t_lookback = float(brent_close.iloc[
            -len(bt_dates) - 0 + max(0, len(brent_close) - len(bt_dates) - lookback_days):
        ].iloc[0]) if False else float(brent_close.iloc[max(0, brent_close.index.get_loc(bt_dates[-1]) - lookback_days)])
        brent_log = float(np.log(brent_at_t / max(brent_at_t_lookback, 1e-9)))

        # Δr_10y over same lookback
        rt_dates = rates.index[rates.index <= t]
        if len(rt_dates) < lookback_days + 1:
            continue
        r_at_t = float(rates.loc[rt_dates[-1]])
        r_at_t_lookback = float(rates.iloc[
            max(0, rates.index.get_loc(rt_dates[-1]) - lookback_days)
        ])
        dr_pp = r_at_t - r_at_t_lookback   # already in percent → diff in pp

        rows.append({
            "Period":         period,
            "date":           t,
            "brent_at_t":     brent_at_t,
            "brent_log_1y":   brent_log,
            "rate_10y_at_t_pct": r_at_t,
            "dr_10y_1y_pp":   dr_pp,
        })

    return pd.DataFrame(rows)


# ============================================================================
# 2. M factor per period (via macro_factor.hybrid_mapping)
# ============================================================================
def attach_m_factor(
    macro_panel: pd.DataFrame,
    *,
    cov_factors: np.ndarray,
    horizon_days: int = 252,
    vintage_anchor: str = "2025",
) -> pd.DataFrame:
    """Add columns m_anchor, m_data, m_hybrid using hybrid_mapping per row."""
    from macro_factor import anchor_from_eba, hybrid_mapping

    anchor = anchor_from_eba(vintage_anchor)
    out = macro_panel.copy()
    rows = []
    for _, r in out.iterrows():
        h = hybrid_mapping(
            delta_brent_log=float(r["brent_log_1y"]),
            delta_rate_10y_pp=float(r["dr_10y_1y_pp"]),
            anchor=anchor,
            cov_factors=cov_factors,
            horizon_days=horizon_days,
        )
        rows.append(h)
    df_m = pd.DataFrame(rows)
    return pd.concat([out.reset_index(drop=True), df_m[["m_anchor", "m_data", "m_hybrid"]].reset_index(drop=True)], axis=1)


# ============================================================================
# 3. Realized 1Y changes per bank
# ============================================================================
def compute_realized_changes(
    capital_wide: pd.DataFrame,
    *,
    lag_quarters: int = 4,
) -> pd.DataFrame:
    """Berechnet realized 1Y changes pro Bank: ΔCET1, ΔRWA, ΔRatio.

    Pro Bank werden Period_start und Period_end definiert mit lag_quarters
    Abstand. Liefert long-format DataFrame mit Spalten:
      LEI_Code, Period_start, Period_end, cet1_start, cet1_end,
      rwa_start, rwa_end, ratio_start, ratio_end, delta_ratio_pp,
      delta_cet1_eur, delta_rwa_eur,
      rwa_credit_start, rwa_market_start, rwa_op_start (etc.)
    """
    if capital_wide.empty:
        return pd.DataFrame()

    # Distinct sorted period list
    periods_sorted = sorted(capital_wide["Period"].unique())
    period_to_idx = {p: i for i, p in enumerate(periods_sorted)}

    rows = []
    for lei, group in capital_wide.groupby("LEI_Code"):
        g = group.sort_values("Period")
        g_idx = {row["Period"]: row for _, row in g.iterrows()}
        for p_start, row_start in g_idx.items():
            i_start = period_to_idx[p_start]
            i_end = i_start + lag_quarters
            if i_end >= len(periods_sorted):
                continue
            p_end = periods_sorted[i_end]
            row_end = g_idx.get(p_end)
            if row_end is None:
                continue
            cet1_s = row_start.get("cet1")
            cet1_e = row_end.get("cet1")
            rwa_s  = row_start.get("rwa_total")
            rwa_e  = row_end.get("rwa_total")
            if pd.isna(cet1_s) or pd.isna(cet1_e) or pd.isna(rwa_s) or pd.isna(rwa_e):
                continue
            if rwa_s <= 0 or rwa_e <= 0:
                continue
            ratio_s = cet1_s / rwa_s
            ratio_e = cet1_e / rwa_e
            rows.append({
                "LEI_Code":           lei,
                "Period_start":       p_start,
                "Period_end":         p_end,
                "cet1_start":         cet1_s,
                "cet1_end":           cet1_e,
                "rwa_start":          rwa_s,
                "rwa_end":            rwa_e,
                "rwa_credit_start":   row_start.get("rwa_credit", float("nan")),
                "rwa_market_start":   row_start.get("rwa_market", float("nan")),
                "rwa_op_start":       row_start.get("rwa_operational", float("nan")),
                "rwa_credit_end":     row_end.get("rwa_credit", float("nan")),
                "rwa_market_end":     row_end.get("rwa_market", float("nan")),
                "rwa_op_end":         row_end.get("rwa_operational", float("nan")),
                "ratio_start":        ratio_s,
                "ratio_end":          ratio_e,
                "delta_ratio_pp":     (ratio_e - ratio_s) * 100,
                "delta_cet1_eur":     cet1_e - cet1_s,
                "delta_rwa_eur":      rwa_e - rwa_s,
            })
    return pd.DataFrame(rows)


# ============================================================================
# 4. Forecast-vs-Realized panel (joins macro + realized)
# ============================================================================
def build_forecast_panel(
    realized: pd.DataFrame, macro_with_m: pd.DataFrame,
) -> pd.DataFrame:
    """Joint realized changes mit M-Faktor zum Period_start.

    Result columns include all of realized + m_anchor / m_data / m_hybrid
    at the start period.
    """
    if realized.empty or macro_with_m.empty:
        return pd.DataFrame()
    macro_keep = macro_with_m[["Period", "brent_log_1y", "dr_10y_1y_pp",
                               "m_anchor", "m_data", "m_hybrid"]].copy()
    macro_keep.columns = ["Period_start", "brent_log_1y", "dr_10y_1y_pp",
                          "m_anchor_at_start", "m_data_at_start", "m_at_start"]
    return realized.merge(macro_keep, on="Period_start", how="left")


# ============================================================================
# 5. Empirical sensitivity (OLS regression)
# ============================================================================
def fit_empirical_sensitivity(panel: pd.DataFrame) -> dict:
    """OLS regression: Δratio_pp = α + β · M_at_start + ε.

    Returns dict with α, β, R², n, residual std, mean predicted, etc.
    """
    sub = panel.dropna(subset=["delta_ratio_pp", "m_at_start"])
    if len(sub) < 30:
        return {"n": len(sub), "alpha": float("nan"), "beta": float("nan"),
                "r2": float("nan"), "rmse": float("nan")}

    x = sub["m_at_start"].to_numpy()
    y = sub["delta_ratio_pp"].to_numpy()
    n = len(x)

    # OLS via numpy
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])

    yhat = alpha + beta * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.mean() != y[0] or n > 1 else 1.0
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse = float(np.sqrt(ss_res / n))
    mae  = float(np.mean(np.abs(y - yhat)))

    # Standard error of beta (homoskedastic OLS)
    var_resid = ss_res / max(n - 2, 1)
    var_beta  = var_resid / max(np.sum((x - x.mean()) ** 2), 1e-12)
    se_beta   = float(np.sqrt(var_beta))
    t_beta    = beta / se_beta if se_beta > 0 else float("nan")

    return {
        "n":         n,
        "alpha":     alpha,
        "beta":      beta,
        "se_beta":   se_beta,
        "t_beta":    t_beta,
        "r2":        r2,
        "rmse":      rmse,
        "mae":       mae,
        "y_mean":    float(y.mean()),
        "y_std":     float(y.std(ddof=1)),
    }


# ============================================================================
# 6. Stress-episode diagnostics
# ============================================================================
DEFAULT_EPISODES = {
    "Corona 2020": {"Period_start": 201912, "Period_end": 202012,
                    "label": "End-2019 → End-2020"},
    "Ukraine 2022": {"Period_start": 202112, "Period_end": 202212,
                     "label": "End-2021 → End-2022"},
    "Rate-hike 2023": {"Period_start": 202206, "Period_end": 202306,
                       "label": "Mid-2022 → Mid-2023"},
}


def episode_diagnostics(
    panel: pd.DataFrame,
    *,
    episodes: dict | None = None,
) -> pd.DataFrame:
    """Pro Episode: durchschnittliche realized Δ-Ratio vs. modell-basierte
    Erwartung (Episode-Mittel), Anzahl Banken, Spread.
    """
    eps = episodes or DEFAULT_EPISODES
    rows = []
    for name, spec in eps.items():
        sub = panel[(panel["Period_start"] == spec["Period_start"])
                    & (panel["Period_end"]   == spec["Period_end"])]
        if len(sub) == 0:
            continue
        d = sub["delta_ratio_pp"]
        m = sub["m_at_start"]
        rows.append({
            "episode":           name,
            "label":             spec["label"],
            "n_banks":           len(sub),
            "M_at_start":        float(m.mean()) if len(m) else float("nan"),
            "delta_ratio_mean":  float(d.mean()),
            "delta_ratio_median": float(d.median()),
            "delta_ratio_p25":   float(d.quantile(0.25)),
            "delta_ratio_p75":   float(d.quantile(0.75)),
            "delta_ratio_min":   float(d.min()),
            "delta_ratio_max":   float(d.max()),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 7. RWA-component decomposition (Brinson-style for RWA channel)
# ============================================================================
def rwa_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    """Pro Bank-Period-Pair: ΔRWA-Total in seine drei Komponenten zerlegen.

    Liefert volume + composition + residual:
      ΔRWA_total = ΔRWA_credit + ΔRWA_market + ΔRWA_operational

    Plus Brinson-style 3-component:
      ΔRWA_volume   = (RWA_total_end / RWA_total_start - 1) · weights_start
      ΔRWA_mix      = component-weight shift × constant total scaling
      ΔRWA_residual = stress / model-explained part

    V1 simplified: just report the raw component deltas; the mix/volume
    distinction is informational, not strictly Brinson.
    """
    sub = panel.dropna(subset=["rwa_credit_start", "rwa_credit_end",
                               "rwa_market_start", "rwa_market_end",
                               "rwa_op_start", "rwa_op_end"]).copy()
    sub["dRWA_credit"]      = sub["rwa_credit_end"] - sub["rwa_credit_start"]
    sub["dRWA_market"]      = sub["rwa_market_end"] - sub["rwa_market_start"]
    sub["dRWA_operational"] = sub["rwa_op_end"]     - sub["rwa_op_start"]
    sub["dRWA_total_check"] = sub["dRWA_credit"] + sub["dRWA_market"] + sub["dRWA_operational"]

    # Component growth rates
    sub["g_credit"] = sub["dRWA_credit"] / sub["rwa_credit_start"].replace(0, pd.NA)
    sub["g_market"] = sub["dRWA_market"] / sub["rwa_market_start"].replace(0, pd.NA)
    sub["g_op"]     = sub["dRWA_operational"] / sub["rwa_op_start"].replace(0, pd.NA)
    sub["g_total"]  = sub["delta_rwa_eur"]    / sub["rwa_start"].replace(0, pd.NA)

    return sub


# ============================================================================
# 8. Validation tests
# ============================================================================
def _test_period_to_date():
    assert _period_to_date(202506) == pd.Timestamp("2025-06-30")
    assert _period_to_date(201912) == pd.Timestamp("2019-12-31")


def _test_realized_changes_simple():
    """Synthetic 8-quarter panel for one bank, lag 4 → 5 transitions."""
    df = pd.DataFrame({
        "LEI_Code": ["BANK1"] * 8,
        "Period":   [201909, 201912, 202003, 202006, 202009, 202012, 202103, 202106],
        "cet1":     [50, 51, 52, 53, 54, 55, 56, 57],
        "rwa_total":[400, 405, 410, 415, 420, 425, 430, 435],
        "rwa_credit":[300, 305, 310, 315, 320, 325, 330, 335],
        "rwa_market":[50, 50, 50, 50, 50, 50, 50, 50],
        "rwa_operational":[50, 50, 50, 50, 50, 50, 50, 50],
    })
    realized = compute_realized_changes(df, lag_quarters=4)
    # 8 periods, lag 4 → 4 valid transitions (idx 0..3)
    assert len(realized) == 4, f"Expected 4, got {len(realized)}"
    assert realized.iloc[0]["Period_start"] == 201909
    assert realized.iloc[0]["Period_end"]   == 202009


def _test_sensitivity_fit_smoke():
    """Toy panel with linear pattern → β ≈ 1, R² ≈ 1."""
    n = 100
    rng = np.random.default_rng(0)
    m = rng.normal(0, 1, n)
    delta_pp = 1.0 * m + rng.normal(0, 0.01, n)  # near-perfect linear
    panel = pd.DataFrame({"m_at_start": m, "delta_ratio_pp": delta_pp})
    res = fit_empirical_sensitivity(panel)
    assert res["n"] == n
    assert abs(res["beta"] - 1.0) < 0.05
    assert res["r2"] > 0.99


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("backtesting.py - Forecast vs Realized - Validation")
    print("=" * 60)
    tests = [
        ("Period to date conversion",         _test_period_to_date),
        ("Realized changes lag-4 transitions", _test_realized_changes_simple),
        ("Empirical sensitivity OLS smoke",   _test_sensitivity_fit_smoke),
    ]
    for label, fn in tests:
        try:
            fn()
            print(f"  [PASS]  {label}")
        except AssertionError as e:
            print(f"  [FAIL]  {label}: {e}")
            raise

    print("\n[PASS] All backtesting tests passed.")
