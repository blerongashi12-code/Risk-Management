"""
============================================================================
 backtesting_walkforward.py · Quarterly Walk-Forward Backtest
============================================================================

Adaption des klassischen Walk-Forward-Konzepts auf unser Use-Case
(Vasicek/ASRF + EBA Transparency + 2-Kanal-CET1: Loan-Book ΔRWA+ΔEL und
Sovereign ΔMtM; der frühere Trading-Book-Kanal wurde entfernt).

Methodische Erweiterung (PD/LGD-Zeitreihe): Das Portfolio wird an jedem
historischen T0 mit den *damals gültigen* Pillar-3-PD/LGD eingefroren
(`vintage_for_period` → Jahresend-Vintage, hold-flat über die 4 Quartale
des Jahres). Quelle der PD/LGD ist ausschließlich Pillar-3 (siehe
MODEL_ASSUMPTIONS A-02c); kein Look-ahead aus dem 2024-Snapshot.

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


def vintage_for_period(period: int,
                       available: list[str] | None = None,
                       *, lag_years: int = 1) -> str:
    """Mappt ein EBA-Quartal (YYYYMM) auf den Pillar-3-Jahresend-Stichtag,
    der zum Forward-Stress an diesem T0 verwendet wird (MODEL_ASSUMPTIONS A-02c).

    **No-look-ahead (Default ``lag_years=1``):** Zum Stichtag eines Quartals im
    Jahr Y ist das jüngste *veröffentlichte* Pillar-3-Jahresende 31.12.(Y−1)
    (FY_Y wird erst ~Q1 von Y+1 publiziert). Das eingefrorene Portfolio nutzt
    also nur Information, die zu T0 real verfügbar war — kein Look-ahead.
    Der gewählte Jahresend-Wert gilt flach über alle vier Quartale des Jahres.
    ``lag_years=0`` liefert die (looking-ahead) Same-Year-Variante zum Vergleich.

    Liegt ``available`` vor, wird auf den jüngsten *vorhandenen* Stichtag
    ≤ Ziel geclippt (Banken ohne den exakten Jahrgang fallen auf den ältesten
    verfügbaren zurück; dokumentierte Limitation bis zum Backfill).
    """
    yr = period // 100
    target = f"{yr - int(lag_years)}-12-31"
    if not available:
        return target
    le = [v for v in sorted(available) if v <= target]
    return le[-1] if le else min(available)


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
# 3b. PD/LGD-Zeitreihe → frozen-portfolio prediction (faithful core)
# ============================================================================
# Statt des EU-aggregierten Einheits-Skalierungsfaktors (m_to_credit_rwa_scale)
# friert dieser Pfad das Portfolio je Bank an jedem historischen T0 mit den
# *damals gültigen* Pillar-3-PD/LGD UND -EAD ein (alles aus derselben EU-CR6-
# Vintage → input-seitig 100 % Pillar-3, MODEL_ASSUMPTIONS A-02c) und rechnet
# die Stress-Wirkung durch das echte 2-Faktor-/IRB-K-Modell (vasicek.py).
# Die bank- und vintage-spezifische RWA-Skalierung wird dann auf die *gemeldete*
# RWA_credit der Bank angewandt — so bleibt die Prognose auf der realisierten
# Skala vergleichbar (die rekonstruierte IRB-K trifft nicht 1:1 die bank-
# gemeldete RWA, wohl aber die relative Stress-Reaktion).

_PORTFOLIO_CSV = _ROOT / "data" / "pillar3_portfolio_timeseries.csv"


def load_portfolio_timeseries(path=None) -> pd.DataFrame:
    """EAD/Restlaufzeit je (LEI, vasicek_class, vintage_date) aus Pillar-3 EU-CR6
    — die Frozen-Portfolio-Gewichte für den Walk-Forward (input-seitig
    ausschließlich Pillar-3)."""
    p = Path(path) if path else _PORTFOLIO_CSV
    df = pd.read_csv(p)
    df["ead_eur"] = pd.to_numeric(df["ead_eur_m"], errors="coerce") * 1e6
    df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="coerce")
    df["vintage_date"] = df["vintage_date"].astype(str)
    return df


def build_frozen_portfolio(lei: str, pd_vintage: str, *,
                           port_df: pd.DataFrame | None = None,
                           pd_path=None):
    """Eingefrorenes BankPortfolio für (Bank, Stichtag): EAD aus Pillar-3 EU-CR6,
    PD/LGD aus derselben Pillar-3-Vintage. None, wenn Daten fehlen."""
    from eba_pd_loader import load_pd_table
    from vasicek import PortfolioSegment, BankPortfolio
    if port_df is None:
        try:
            port_df = load_portfolio_timeseries()
        except FileNotFoundError:
            return None
    pdf = load_pd_table(path=pd_path, vintage=pd_vintage)
    pdf = pdf[pdf["LEI"] == lei]
    sub = port_df[(port_df["LEI"] == lei)
                  & (port_df["vintage_date"] == pd_vintage)]
    if sub.empty or pdf.empty:
        return None
    lk = pdf.set_index("vasicek_class")[["pd_pct", "lgd_pct", "status"]]
    segs = []
    for _, r in sub.iterrows():
        vclass = str(r["vasicek_class"])
        if vclass not in lk.index:
            continue
        if str(lk.loc[vclass, "status"]) == "standardised_not_applicable":
            continue
        pp = float(lk.loc[vclass, "pd_pct"]); ll = float(lk.loc[vclass, "lgd_pct"])
        ead = float(r["ead_eur"])
        mat = float(r["maturity_years"]) if r["maturity_years"] == r["maturity_years"] else 2.5
        if not (ead > 0) or pp != pp or ll != ll:
            continue
        segs.append(PortfolioSegment(
            name=f"{lei}:{vclass}", exposure_class=vclass,
            ead=ead, pd=pp / 100.0, lgd=ll / 100.0,
            maturity_years=(mat if mat > 0 else 2.5),
            # Basel-SME-Größenanpassung: gleiche Konvention wie eba_loader
            # (repräsentativer Jahresumsatz €20 Mio. für das aggregierte SME-Buch)
            sales_m_eur=(20.0 if vclass == "sme_corporate" else None),
        ))
    if not segs:
        return None
    return BankPortfolio(name=f"{lei}@{pd_vintage}", segments=segs, lei=lei)


def frozen_rwa_scale(frozen, m: float, *, kappa_lgd: float = 0.30):
    """RWA-Skalierung = stressed_RWA / baseline_RWA des eingefrorenen
    Portfolios unter Systemfaktor M. None bei leerem Portfolio."""
    if frozen is None:
        return None
    base = frozen.portfolio_kpis()
    if not base or base.get("rwa", 0) <= 0:
        return None
    st = frozen.stressed_kpis(z_factor=float(m), kappa_lgd=kappa_lgd)
    return (base["rwa"] + st["delta_rwa"]) / base["rwa"]


def build_pdlgd_walkforward(
    lei: str,
    macro_q_with_m: pd.DataFrame,
    capital_wide: pd.DataFrame,
    *,
    port_df: pd.DataFrame | None = None,
    kappa_lgd: float = 0.30,
    lag_years: int = 1,
    lag_quarters: int = 1,
) -> pd.DataFrame:
    """Walk-Forward für EINE Bank mit Pillar-3-PD/LGD-Zeitreihe.

    Pro Quartal-Paar (t→t+1): Portfolio mit der no-look-ahead-Vintage einfrieren
    → RWA-Skalierung unter realisiertem M(t→t+1) → Prognose
    ΔRWA_credit = gemeldete RWA_credit(t) · (scale−1) → gegen die um Volumen
    (Non-Credit-RWA-Wachstum) bereinigte Realität vergleichen.

    Output-Schema kompatibel mit walkforward_error_stats / per_bank_summary.
    """
    from eba_pd_loader import available_vintages
    if macro_q_with_m.empty or capital_wide.empty:
        return pd.DataFrame()
    if port_df is None:
        try:
            port_df = load_portfolio_timeseries()
        except FileNotFoundError:
            return pd.DataFrame()
    vintages = available_vintages()

    cw = capital_wide[capital_wide["LEI_Code"] == lei].copy()
    if cw.empty:
        return pd.DataFrame()
    cw["rwa_op"] = cw.get("rwa_operational", pd.Series(0, index=cw.index))
    cw["rwa_noncredit"] = cw["rwa_market"].fillna(0) + cw["rwa_op"].fillna(0)
    g_idx = {int(r["Period"]): r for _, r in cw.iterrows()}
    periods_sorted = sorted(g_idx.keys())
    pidx = {p: i for i, p in enumerate(periods_sorted)}
    macro_lookup = {(int(r["Period_start"]), int(r["Period_end"])): r
                    for _, r in macro_q_with_m.iterrows()}

    _frozen: dict[str, object] = {}
    rows = []
    for p_start in periods_sorted:
        i_end = pidx[p_start] + lag_quarters
        if i_end >= len(periods_sorted):
            continue
        p_end = periods_sorted[i_end]
        macro = macro_lookup.get((p_start, p_end))
        if macro is None:
            continue
        m_q = float(macro["m_hybrid"])
        pd_vintage = vintage_for_period(p_start, vintages, lag_years=lag_years)
        if pd_vintage not in _frozen:
            _frozen[pd_vintage] = build_frozen_portfolio(lei, pd_vintage, port_df=port_df)
        scale = frozen_rwa_scale(_frozen[pd_vintage], m_q, kappa_lgd=kappa_lgd)
        if scale is None:
            continue
        rs, re_ = g_idx[p_start], g_idx[p_end]
        rwa_c_s = rs.get("rwa_credit"); rwa_c_e = re_.get("rwa_credit")
        rwa_nc_s = rs.get("rwa_noncredit"); rwa_nc_e = re_.get("rwa_noncredit")
        if any(pd.isna(x) for x in (rwa_c_s, rwa_c_e, rwa_nc_s, rwa_nc_e)) or rwa_c_s <= 0:
            continue
        pred_d = float(rwa_c_s) * (scale - 1.0)
        actual_d = float(rwa_c_e) - float(rwa_c_s)
        vol_factor = (float(rwa_nc_e) / float(rwa_nc_s) - 1.0) if rwa_nc_s > 0 else 0.0
        volume_d = float(rwa_c_s) * vol_factor
        risk_driven_d = actual_d - volume_d
        sign_match = ((pred_d >= 0 and risk_driven_d >= 0)
                      or (pred_d < 0 and risk_driven_d < 0))
        rows.append({
            "LEI_Code":                   lei,
            "Period_start":               int(p_start),
            "Period_end":                 int(p_end),
            "period_label_end":           _period_label(int(p_end)),
            "pd_vintage":                 pd_vintage,
            "m_quarter":                  m_q,
            "brent_log_q":                float(macro["brent_log_q"]),
            "dr_10y_pp_q":                float(macro["dr_10y_pp_q"]),
            "rwa_scale":                  scale,
            "rwa_credit_start":           float(rwa_c_s),
            "rwa_credit_end":             float(rwa_c_e),
            "rwa_noncredit_start":        float(rwa_nc_s),
            "rwa_noncredit_end":          float(rwa_nc_e),
            "pred_dRWA_credit_eur":       pred_d,
            "actual_dRWA_credit_eur":     actual_d,
            "volume_dRWA_credit_eur":     volume_d,
            "risk_driven_dRWA_credit_eur": risk_driven_d,
            "error_eur":                  pred_d - risk_driven_d,
            "error_pct_of_start":         (pred_d - risk_driven_d) / float(rwa_c_s),
            "sign_match":                 bool(sign_match),
        })
    return pd.DataFrame(rows)


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

    # Random-Walk-Benchmark: naive "keine-Änderung"-Prognose (pred=0).
    # Ein MAE ist nur RELATIV zu einer Baseline interpretierbar — schlägt
    # das Modell den Random-Walk nicht, hat es keinen prognostischen Wert.
    mae_rw = float(np.mean(np.abs(act)))                 # |0 − act|
    skill_vs_rw = (1.0 - mae / mae_rw) if mae_rw > 0 else float("nan")

    # Konservativ-Check (gerichtet): Anteil der Fälle, in denen das Modell
    # den risk-driven Effekt betrags­mäßig NICHT unterschätzt (|pred| ≥ |act|
    # bei gleichem Vorzeichen) — stützt die "obere-Schranke"-Erzählung.
    same_sign = np.sign(pred) == np.sign(act)
    conservative = float(np.mean(same_sign & (np.abs(pred) >= np.abs(act))))

    # Linear R²: pred vs act
    if np.var(act) > 0:
        corr = float(np.corrcoef(pred, act)[0, 1])
        r2 = corr ** 2 * np.sign(corr)
    else:
        corr, r2 = float("nan"), float("nan")

    return {
        "n":            int(len(sub)),
        "mae_eur":      mae,
        "rmse_eur":     rmse,
        "bias_eur":     bias,
        "hit_rate":     hit_rate,
        "mae_randomwalk_eur": mae_rw,
        "skill_vs_rw":  skill_vs_rw,
        "conservative_share": conservative,
        "corr":         corr,
        "r2_signed":    r2,
        "mae_pct":      float(np.mean(np.abs(sub["error_pct_of_start"]))),
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


def _test_vintage_for_period():
    # no-look-ahead (Default lag_years=1): Quartal in Jahr Y → 31.12.(Y−1)
    assert vintage_for_period(202203) == "2021-12-31"
    assert vintage_for_period(202209) == "2021-12-31"
    assert vintage_for_period(202412) == "2023-12-31"
    # look-ahead-Variante (lag_years=0) = Same-Year
    assert vintage_for_period(202203, lag_years=0) == "2022-12-31"
    # mit available-Liste: clip auf jüngsten vorhandenen Stichtag ≤ Ziel
    av = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]
    assert vintage_for_period(202306, av) == "2022-12-31"   # 2023er Quartal → FY2022
    assert vintage_for_period(202506, av) == "2024-12-31"   # 2025er Quartal → FY2024
    assert vintage_for_period(202106, av) == "2021-12-31"   # FY2020 fehlt → ältester


def _test_frozen_portfolio_pdlgd():
    """Frozen-Portfolio aus der Pillar-3-Zeitreihe (DB): RWA-Skalierung unter
    Adverse-M ist > 1 UND vintage-abhängig → die PD/LGD-Zeitreihe fließt
    tatsächlich durch das IRB-K-Modell (Kern des Umbaus)."""
    try:
        port = load_portfolio_timeseries()
    except FileNotFoundError:
        print("  [SKIP] frozen-portfolio (kein pillar3_portfolio_timeseries.csv)")
        return
    DB = "7LTWFZYICNSX8D621K86"
    scales = {}
    for v in ("2022-12-31", "2023-12-31"):
        fp = build_frozen_portfolio(DB, v, port_df=port)
        assert fp is not None and len(fp.segments) == 7, f"frozen {v} unvollständig"
        s = frozen_rwa_scale(fp, -2.0)
        assert s is not None and s > 1.2, f"Adverse-Skalierung {v} zu klein: {s}"
        scales[v] = s
    assert abs(scales["2022-12-31"] - scales["2023-12-31"]) > 1e-3, \
        "RWA-Skalierung muss vintage-abhängig sein (Zeitreihe wirkt)"


def _test_randomwalk_benchmark():
    panel = pd.DataFrame({
        "pred_dRWA_credit_eur":        [10.0, -5.0, 8.0, 2.0],
        "risk_driven_dRWA_credit_eur": [12.0, -4.0, 6.0, 3.0],
        "sign_match":                  [True, True, True, True],
        "error_pct_of_start":          [0.01, -0.01, 0.02, -0.005],
    })
    stats = walkforward_error_stats(panel)
    assert "mae_randomwalk_eur" in stats and "skill_vs_rw" in stats
    # RW-MAE = mean(|act|) = mean(12,4,6,3) = 6.25 ; Modell-MAE = mean(2,1,2,1)=1.5
    assert abs(stats["mae_randomwalk_eur"] - 6.25) < 1e-9
    assert abs(stats["mae_eur"] - 1.5) < 1e-9
    assert stats["skill_vs_rw"] > 0   # Modell schlägt Random-Walk


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
        ("vintage_for_period map",   _test_vintage_for_period),
        ("frozen-portfolio PD/LGD",  _test_frozen_portfolio_pdlgd),
        ("random-walk benchmark",    _test_randomwalk_benchmark),
        ("scale-curve monotonicity", _test_scale_curve_monotone),
    ]:
        try:
            fn()
            print(f"  [PASS]  {label}")
        except AssertionError as e:
            print(f"  [FAIL]  {label}: {e}")
            raise
    print("\n[PASS] All walk-forward tests passed.")
