"""
============================================================================
 backtesting_walkforward.py · Quarterly Walk-Forward Backtest
============================================================================

Adaption des klassischen Walk-Forward-Konzepts auf unser Use-Case
(Vasicek/ASRF + EBA Transparency 2025 + 3-Channel-CET1).

Für jedes Quartal-Paar (t → t+1) in den historischen EBA-Vintages 2019-2025
und für jede der 67 IRB-Banken:

  Schritt A · Bestand bei t aus EBA holen (RWA_credit, RWA_market, RWA_op,
              CET1, RWA_total)
  Schritt B · Realisierten Macro-Schock zwischen t und t+1 messen
              (Δ Brent log, Δ r_10y in pp)
  Schritt C · Implizites M via hybrid_mapping berechnen
  Schritt D · Modell-Vorhersage: ΔRWA_credit = RWA_credit_t × scale(M)
              wobei scale(M) ein EU-aggregat-kalibrierter Faktor ist
  Schritt E · Realität bei t+1 holen und Volumeneffekt herausrechnen:
                Volume_ΔRWA_credit = RWA_credit_t × (NonCredit_RWA_t+1 /
                                                    NonCredit_RWA_t − 1)
                Risk_driven_ΔRWA_credit = Actual_ΔRWA_credit − Volume_ΔRWA

Output: pro Bank × Quartal-Paar eine Zeile mit Vorhersage vs. bereinigter
Realität — zum Plot, zur Fehlerstatistik, zum Hit-Rate-Test.

Datenbasis
----------
- capital_wide aus load_historical_capital_panel(...) → panel_to_wide(...)
- Brent + Svensson aus dem täglichen Cache (load_data_layer())
- IRB-Universum (LEI-Filter aus tr_cre.csv)

API
---
- compute_quarterly_macro(brent_df, svensson_df, periods)
- attach_m_factor_quarterly(macro_q, cov_factors)
- m_to_credit_rwa_scale(m, ref_universe=None) → float
- build_walkforward_panel(wide, macro_q_with_m, irb_leis=None)
- walkforward_error_stats(panel)
- per_bank_summary(panel)
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
# 1. Period helpers
# ============================================================================
def _period_to_date(period: int) -> pd.Timestamp:
    """YYYYMM → quarter-end Timestamp."""
    yr = period // 100
    mo = period % 100
    return pd.Timestamp(yr, mo, 1) + pd.offsets.MonthEnd(0)


def _period_label(period: int) -> str:
    """201912 → 'Q4 2019', 202506 → 'Q2 2025'."""
    yr = period // 100
    mo = period % 100
    q = (mo - 1) // 3 + 1
    return f"Q{q} {yr}"


# ============================================================================
# 2. Quarterly macro shocks (NOT trailing-1Y — quarter-on-quarter)
# ============================================================================
def compute_quarterly_macro(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    periods: list[int],
    *,
    maturity: float = 10.0,
) -> pd.DataFrame:
    """Pro Quartal-Paar (Period_t → Period_t+1) der realisierte Macro-Schock.

    Anders als compute_macro_panel() in backtesting.py (das verwendet eine
    rollende 252-Tage-Window) misst dieser Walk-Forward den Schock
    *exakt zwischen zwei EBA-Stichtagen*.

    Output-Schema:
      Period_start, Period_end, date_start, date_end,
      brent_at_start, brent_at_end, brent_log_q,
      rate_10y_at_start, rate_10y_at_end, dr_10y_pp_q
    """
    from svensson import zero_rate, params_from_row

    # Brent close
    if "Close_USD" in brent_df.columns:
        brent_close = brent_df["Close_USD"].dropna()
    elif "Close" in brent_df.columns:
        brent_close = brent_df["Close"].dropna()
    else:
        return pd.DataFrame()

    # 10y rate series
    rates = pd.Series(
        [zero_rate(maturity, params_from_row(row), as_decimal=False)
         for _, row in svensson_df.iterrows()],
        index=svensson_df.index,
    )

    periods_sorted = sorted(periods)
    rows = []
    for i in range(len(periods_sorted) - 1):
        p_start = periods_sorted[i]
        p_end = periods_sorted[i + 1]
        t_start = _period_to_date(p_start)
        t_end = _period_to_date(p_end)

        # Brent
        b_start_dates = brent_close.index[brent_close.index <= t_start]
        b_end_dates = brent_close.index[brent_close.index <= t_end]
        if len(b_start_dates) == 0 or len(b_end_dates) == 0:
            continue
        brent_at_start = float(brent_close.loc[b_start_dates[-1]])
        brent_at_end = float(brent_close.loc[b_end_dates[-1]])
        brent_log = float(np.log(brent_at_end / max(brent_at_start, 1e-9)))

        # Rate
        r_start_dates = rates.index[rates.index <= t_start]
        r_end_dates = rates.index[rates.index <= t_end]
        if len(r_start_dates) == 0 or len(r_end_dates) == 0:
            continue
        r_at_start = float(rates.loc[r_start_dates[-1]])
        r_at_end = float(rates.loc[r_end_dates[-1]])
        dr_pp = r_at_end - r_at_start

        rows.append({
            "Period_start":       p_start,
            "Period_end":         p_end,
            "date_start":         t_start,
            "date_end":           t_end,
            "brent_at_start":     brent_at_start,
            "brent_at_end":       brent_at_end,
            "brent_log_q":        brent_log,
            "rate_10y_at_start":  r_at_start,
            "rate_10y_at_end":    r_at_end,
            "dr_10y_pp_q":        dr_pp,
        })
    return pd.DataFrame(rows)


def attach_m_factor_quarterly(
    macro_q: pd.DataFrame,
    *,
    cov_factors: np.ndarray,
    horizon_days: int = 63,   # ~1 Quartal
    vintage_anchor: str = "2025",
) -> pd.DataFrame:
    """Implizites M pro Quartal-Paar via hybrid_mapping.

    Kürzerer Horizon (63d ≈ 1 Quartal) als der Default-1Y im
    Standard-Backtest — entscheidend, weil wir hier Quartal-zu-Quartal-
    Bewegung erklären wollen, nicht jährliche.
    """
    from macro_factor import anchor_from_eba, hybrid_mapping

    if macro_q.empty:
        return macro_q
    anchor = anchor_from_eba(vintage_anchor)
    rows = []
    for _, r in macro_q.iterrows():
        h = hybrid_mapping(
            delta_brent_log=float(r["brent_log_q"]),
            delta_rate_10y_pp=float(r["dr_10y_pp_q"]),
            anchor=anchor,
            cov_factors=cov_factors,
            horizon_days=horizon_days,
        )
        rows.append(h)
    df_m = pd.DataFrame(rows)
    out = pd.concat([macro_q.reset_index(drop=True),
                     df_m[["m_anchor", "m_data", "m_hybrid"]]
                          .reset_index(drop=True)], axis=1)
    return out


# ============================================================================
# 3. M → ΔRWA_credit scale factor (EU-aggregate calibration)
# ============================================================================
# Wir berechnen einmalig: wie skaliert das aggregierte Loan-Book-RWA als
# Funktion von M? Dieser Skalierungs-Faktor ist dann universell für alle
# Banken (uniform-scaling-assumption — Vereinfachung, dokumentiert im
# Doku-Tab).

_CACHE_SCALE = None


def _build_scale_curve(m_grid: np.ndarray | None = None) -> pd.DataFrame:
    """Berechnet scale(M) = RWA_credit_stressed(M) / RWA_credit_baseline
    auf einem M-Gitter ∈ [-3, +3] mit dem EU-aggregate-Portfolio.

    Wird beim ersten Aufruf einmal gerechnet, danach gecached.
    """
    global _CACHE_SCALE
    if _CACHE_SCALE is not None:
        return _CACHE_SCALE

    from config import EBA_RAW_DIR
    from eba_loader import load_eba_universe

    universe = load_eba_universe(vintage="2025", top_n=None, prefer_real=True)
    agg = universe.aggregated_portfolio("EU-Aggregate")

    if m_grid is None:
        m_grid = np.linspace(-3.0, 3.0, 31)

    # Normalise relative to stressed_kpis(M=0) to guarantee scale(0)=1.
    # The Basel-MA term in stressed_kpis vs. baseline_kpis can differ
    # slightly even at the no-shock point — using M=0 as the
    # reference removes that offset cleanly.
    ref_stressed = agg.stressed_kpis(
        z_factor=0.0, confidence=0.999, kappa_lgd=0.30,
    )
    rwa_ref = float(ref_stressed["rwa"])

    rows = []
    for m in m_grid:
        stressed = agg.stressed_kpis(
            z_factor=float(m), confidence=0.999, kappa_lgd=0.30,
        )
        scale = float(stressed["rwa"]) / max(rwa_ref, 1e-9)
        rows.append({"M": float(m), "scale": scale})

    _CACHE_SCALE = pd.DataFrame(rows)
    return _CACHE_SCALE


def m_to_credit_rwa_scale(m: float) -> float:
    """Interpoliert scale(M) = stressed_RWA / baseline_RWA aus der Kurve."""
    df = _build_scale_curve()
    # Linear interp
    return float(np.interp(m, df["M"].to_numpy(), df["scale"].to_numpy()))


# ============================================================================
# 4. Walk-Forward Panel
# ============================================================================
def build_walkforward_panel(
    capital_wide: pd.DataFrame,
    macro_q_with_m: pd.DataFrame,
    *,
    irb_leis: set[str] | None = None,
    lag_quarters: int = 1,
) -> pd.DataFrame:
    """Pro Bank × Quartal-Paar: Modell-Vorhersage vs. bereinigte Realität.

    Pflicht-Spalten in capital_wide:
      LEI_Code, Period, cet1, rwa_total, rwa_credit, rwa_market, rwa_op

    Output-Schema:
      LEI_Code, Period_start, Period_end, period_label_end,
      m_quarter,
      rwa_credit_start, rwa_credit_end,
      rwa_noncredit_start, rwa_noncredit_end,
      cet1_start, cet1_end,
      pred_dRWA_credit_eur,
      actual_dRWA_credit_eur,
      volume_dRWA_credit_eur,
      risk_driven_dRWA_credit_eur,
      pred_dK_eur,            (= pred × 8%)
      actual_dK_eur,          (= actual × 8%)
      risk_driven_dK_eur,
      error_eur               (= pred - risk_driven)
      error_pct_of_start      (= error / RWA_credit_start)
      sign_match              (bool)
    """
    if capital_wide.empty or macro_q_with_m.empty:
        return pd.DataFrame()

    # Filter to IRB universe if provided
    if irb_leis is not None:
        capital_wide = capital_wide[capital_wide["LEI_Code"].isin(irb_leis)].copy()

    # Required columns sanity check
    for col in ("cet1", "rwa_total", "rwa_credit", "rwa_market"):
        if col not in capital_wide.columns:
            return pd.DataFrame()

    # Pre-fill non-credit RWA (market + operational if available)
    cw = capital_wide.copy()
    cw["rwa_op"] = cw.get("rwa_operational", pd.Series(0, index=cw.index))
    cw["rwa_noncredit"] = (cw["rwa_market"].fillna(0)
                            + cw["rwa_op"].fillna(0))

    # Build dict period→row per bank
    rows = []
    macro_lookup = {(int(r["Period_start"]), int(r["Period_end"])): r
                    for _, r in macro_q_with_m.iterrows()}
    periods_sorted = sorted(cw["Period"].unique())
    period_idx = {p: i for i, p in enumerate(periods_sorted)}

    for lei, group in cw.groupby("LEI_Code"):
        g = group.sort_values("Period")
        g_idx = {row["Period"]: row for _, row in g.iterrows()}
        for p_start, row_start in g_idx.items():
            i_start = period_idx[p_start]
            i_end = i_start + lag_quarters
            if i_end >= len(periods_sorted):
                continue
            p_end = periods_sorted[i_end]
            row_end = g_idx.get(p_end)
            if row_end is None:
                continue

            # Pull macro for this quarter pair
            macro_row = macro_lookup.get((p_start, p_end))
            if macro_row is None:
                continue
            m_q = float(macro_row["m_hybrid"])

            # Bank inputs
            rwa_c_s = row_start.get("rwa_credit")
            rwa_c_e = row_end.get("rwa_credit")
            rwa_nc_s = row_start.get("rwa_noncredit")
            rwa_nc_e = row_end.get("rwa_noncredit")
            cet1_s = row_start.get("cet1")
            cet1_e = row_end.get("cet1")

            if any(pd.isna(x) for x in (rwa_c_s, rwa_c_e, rwa_nc_s,
                                         rwa_nc_e, cet1_s, cet1_e)):
                continue
            if rwa_c_s <= 0:
                continue

            # Model prediction
            scale = m_to_credit_rwa_scale(m_q)
            pred_d = float(rwa_c_s) * (scale - 1.0)

            # Actual
            actual_d = float(rwa_c_e) - float(rwa_c_s)

            # Volume adjustment via non-credit RWA growth
            if rwa_nc_s > 0:
                vol_factor = (float(rwa_nc_e) / float(rwa_nc_s)) - 1.0
            else:
                vol_factor = 0.0
            volume_d = float(rwa_c_s) * vol_factor
            risk_driven_d = actual_d - volume_d

            # Sign match (both predicted and risk-driven same sign,
            # or both near zero)
            sign_match = ((pred_d >= 0 and risk_driven_d >= 0)
                          or (pred_d < 0 and risk_driven_d < 0))

            rows.append({
                "LEI_Code":                 lei,
                "Period_start":             int(p_start),
                "Period_end":               int(p_end),
                "period_label_end":         _period_label(int(p_end)),
                "m_quarter":                m_q,
                "brent_log_q":              float(macro_row["brent_log_q"]),
                "dr_10y_pp_q":              float(macro_row["dr_10y_pp_q"]),
                "rwa_credit_start":         float(rwa_c_s),
                "rwa_credit_end":           float(rwa_c_e),
                "rwa_noncredit_start":      float(rwa_nc_s),
                "rwa_noncredit_end":        float(rwa_nc_e),
                "cet1_start":               float(cet1_s),
                "cet1_end":                 float(cet1_e),
                "pred_dRWA_credit_eur":     pred_d,
                "actual_dRWA_credit_eur":   actual_d,
                "volume_dRWA_credit_eur":   volume_d,
                "risk_driven_dRWA_credit_eur": risk_driven_d,
                "pred_dK_eur":              pred_d * 0.08,
                "actual_dK_eur":            actual_d * 0.08,
                "risk_driven_dK_eur":       risk_driven_d * 0.08,
                "error_eur":                pred_d - risk_driven_d,
                "error_pct_of_start":       (pred_d - risk_driven_d)
                                            / float(rwa_c_s),
                "sign_match":               bool(sign_match),
            })
    return pd.DataFrame(rows)


# ============================================================================
# 5. Aggregated diagnostics
# ============================================================================
def walkforward_error_stats(panel: pd.DataFrame) -> dict:
    """System-level Fehlerstatistiken: MAE, RMSE, Hit-Rate, R²."""
    if panel.empty:
        return {"n": 0}

    sub = panel.dropna(subset=["pred_dRWA_credit_eur",
                                "risk_driven_dRWA_credit_eur"])
    if len(sub) == 0:
        return {"n": 0}

    pred = sub["pred_dRWA_credit_eur"].to_numpy()
    act = sub["risk_driven_dRWA_credit_eur"].to_numpy()
    err = pred - act

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    hit_rate = float(sub["sign_match"].mean())

    # Linear R²: pred vs act
    if np.var(act) > 0:
        corr = float(np.corrcoef(pred, act)[0, 1])
        r2 = corr ** 2 * np.sign(corr)
    else:
        corr, r2 = float("nan"), float("nan")

    return {
        "n":         int(len(sub)),
        "mae_eur":   mae,
        "rmse_eur":  rmse,
        "bias_eur":  bias,
        "hit_rate":  hit_rate,
        "corr":      corr,
        "r2_signed": r2,
        "mae_pct":   float(np.mean(np.abs(sub["error_pct_of_start"]))),
    }


def per_bank_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Pro Bank: n_obs, mean error, MAE, hit-rate."""
    if panel.empty:
        return pd.DataFrame()
    grouped = panel.groupby("LEI_Code").agg(
        n_obs=("error_eur", "size"),
        mae_eur=("error_eur", lambda x: float(np.mean(np.abs(x)))),
        bias_eur=("error_eur", "mean"),
        hit_rate=("sign_match", "mean"),
        avg_rwa_credit_start=("rwa_credit_start", "mean"),
    ).reset_index()
    grouped["mae_pct_of_avg_rwa"] = (grouped["mae_eur"]
                                     / grouped["avg_rwa_credit_start"]
                                              .replace(0, np.nan))
    return grouped


def system_aggregate_timeseries(panel: pd.DataFrame) -> pd.DataFrame:
    """Σ über alle Banken pro Quartal-Paar: Modell vs bereinigte Realität."""
    if panel.empty:
        return pd.DataFrame()
    agg = panel.groupby(
        ["Period_end", "period_label_end", "m_quarter"], as_index=False
    ).agg(
        pred_sum=("pred_dRWA_credit_eur", "sum"),
        actual_sum=("actual_dRWA_credit_eur", "sum"),
        volume_sum=("volume_dRWA_credit_eur", "sum"),
        risk_driven_sum=("risk_driven_dRWA_credit_eur", "sum"),
        n_banks=("LEI_Code", "nunique"),
    ).sort_values("Period_end")
    return agg


# ============================================================================
# 6. Crisis-episode annotations
# ============================================================================
EPISODES_QUARTERLY = [
    {"period_end": 202003, "label": "COVID Q1 2020",
     "color": "#A52F4D"},
    {"period_end": 202203, "label": "Ukraine Q1 2022",
     "color": "#A52F4D"},
    {"period_end": 202209, "label": "Rate-hike Q3 2022",
     "color": "#C9A227"},
    {"period_end": 202303, "label": "SVB/CS Q1 2023",
     "color": "#A52F4D"},
]


# ============================================================================
# 7. Tests
# ============================================================================
def _test_period_label():
    assert _period_label(201912) == "Q4 2019"
    assert _period_label(202506) == "Q2 2025"
    assert _period_label(202003) == "Q1 2020"


def _test_scale_curve_monotone():
    """scale(M) should be a decreasing function of M (negative M = stress
    → larger RWA → scale > 1; positive M = benign → scale < 1)."""
    df = _build_scale_curve(np.linspace(-3, 3, 13))
    assert df["scale"].iloc[0] > df["scale"].iloc[-1], \
        "scale(-3) must exceed scale(+3) — adverse shock raises RWA"
    assert abs(df["scale"].iloc[len(df)//2] - 1.0) < 0.05, \
        f"scale(0) ≈ 1, got {df['scale'].iloc[len(df)//2]:.3f}"


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("backtesting_walkforward.py · Tests")
    print("=" * 60)
    for label, fn in [
        ("period label",             _test_period_label),
        ("scale-curve monotonicity", _test_scale_curve_monotone),
    ]:
        try:
            fn()
            print(f"  [PASS]  {label}")
        except AssertionError as e:
            print(f"  [FAIL]  {label}: {e}")
            raise
    print("\n[PASS] All walk-forward tests passed.")
