"""
============================================================================
 macro_factor.py · Macro Shocks → Vasicek Systemic Factor M
============================================================================

Brücke zwischen dem 2-Faktor-Schock-Vektor (Brent log-Return + Δr_10y)
und dem Vasicek-Systemfaktor M ~ N(0,1), der die ASRF-Conditional-PDs
treibt.

Methodische Doppelstruktur (siehe MODEL_ASSUMPTIONS.md §11.2):

  Route 1 — Anchor-basiert (Primary):
      EBA-2025-Adverse-Szenario liefert ein Tupel
        (ΔBrent_log_anchor, Δr_10y_pp_anchor, M_anchor = z_factor_implied)
      Lineare Skalierung in der Schock-Norm:
        scale = ||shock|| / ||anchor||
        M_implied = M_anchor · scale · sign(shock_alignment)

  Route 2 — Datenbasiert (Validation):
      Faktor-Sensitivität gegenüber dem Schock-Vektor wird aus der
      historischen Brent + Δr_10y Kovarianz abgeleitet, und in der
      Schock-Mahalanobis-Norm ausgedrückt.

Beide Routen werden parallel berechnet und im Output transparent
nebeneinander gestellt — der User wählt im Cockpit, welche er als
"primäre" Mapping-Route nutzt.
============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Project root (directory above `backend/`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eba_loader import EBA_2025_ADVERSE_ANCHOR, EBA_2023_ADVERSE_ANCHOR


# ============================================================================
# 0. Factor returns and statistics (Brent + Δr_10y)
# ============================================================================
def factor_returns(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    maturity: float = 10.0,
) -> pd.DataFrame:
    """Tägliche Faktor-Returns r_brent (log) und Δr_10y (pp).

    Parameters
    ----------
    brent_df : DataFrame mit DatetimeIndex und Spalte 'Return_log'.
    svensson_df : Bundesbank-Parameter-Zeitreihe.
    maturity : Δr-Maturity in Jahren (default 10).
    """
    from svensson import zero_rate, params_from_row  # type: ignore

    if "Return_log" not in brent_df.columns:
        raise KeyError("brent_df must contain a 'Return_log' column "
                       "(from the Brent fetcher).")
    r_brent = brent_df["Return_log"].rename("r_brent")

    rates = pd.Series(
        [zero_rate(maturity, params_from_row(row), as_decimal=False)
         for _, row in svensson_df.iterrows()],
        index=svensson_df.index,
        name=f"r_{maturity:.0f}y",
    )
    d_r = rates.diff().rename(f"d_r_{maturity:.0f}y")

    return pd.concat([r_brent, d_r], axis=1, sort=False).sort_index()


def factor_stats(
    brent_df: pd.DataFrame,
    svensson_df: pd.DataFrame,
    *,
    lookback: int = 252,
    maturity: float = 10.0,
) -> dict:
    """Empirische μ, Σ, ρ der Faktor-Returns über die letzten `lookback` Tage."""
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


# ============================================================================
# 1. Anchor-Mapping (Route 1)
# ============================================================================
@dataclass
class MacroAnchor:
    """Anker-Punkt im (ΔBrent, Δr_10y) → M Raum."""
    brent_log: float
    rate_10y_pp: float
    z_factor: float
    label: str = ""

    @property
    def shock_norm(self) -> float:
        """L2-Norm des Schock-Vektors (Brent in log, Rate in pp)."""
        return float(np.hypot(self.brent_log, self.rate_10y_pp))


def anchor_from_eba(vintage: str = "2025") -> MacroAnchor:
    """Konstruiert MacroAnchor aus EBA-Stress-Test-Anker."""
    spec = EBA_2025_ADVERSE_ANCHOR if vintage == "2025" else EBA_2023_ADVERSE_ANCHOR
    return MacroAnchor(
        brent_log=spec["brent_log_shock"],
        rate_10y_pp=spec["rate_10y_pp_shock"],
        z_factor=spec["z_factor_implied"],
        label=f"EBA {spec['vintage']} adverse",
    )


def m_from_anchor(
    delta_brent_log: float,
    delta_rate_10y_pp: float,
    anchor: MacroAnchor,
) -> float:
    """Mappt einen Schock-Vektor auf M via Anchor-Skalierung.

    Methodik:
      1. Projiziere Schock auf den Anchor-Vektor (Cosine-Direction).
      2. Skaliere die Schock-Norm relativ zur Anchor-Norm.
      3. M = z_factor_anchor · cosine · (norm_ratio).

    Wenn der Schock orthogonal zum Anchor steht, ist die Projektion 0
    und das Modell liefert M = 0 (kein Stress). Wenn er anti-aligned
    ist, kehrt sich das Vorzeichen — ein Brent-Crash + Rate-Cut ist
    benigner, nicht adverser, was korrekt abgebildet wird.

    Parameters
    ----------
    delta_brent_log : kumulativer Brent log-Return über den Horizon
    delta_rate_10y_pp : Δr_10y in Prozentpunkten (NICHT Basispunkten)
    anchor : MacroAnchor

    Returns
    -------
    float — implizierter M-Faktor (Vasicek-Systemfaktor)
    """
    shock = np.array([delta_brent_log, delta_rate_10y_pp])
    anchor_vec = np.array([anchor.brent_log, anchor.rate_10y_pp])

    anchor_norm = float(np.linalg.norm(anchor_vec))
    if anchor_norm < 1e-12:
        return 0.0

    # Projizierte Komponente entlang Anchor (positiv = aligned mit Adverse)
    cos_align = float(np.dot(shock, anchor_vec) / (anchor_norm * max(np.linalg.norm(shock), 1e-12)))
    shock_norm = float(np.linalg.norm(shock))
    norm_ratio = shock_norm / anchor_norm

    # M_anchor ist negativ (adverse). Aligned shock skaliert ihn proportional.
    return anchor.z_factor * cos_align * norm_ratio


# ============================================================================
# 2. Datenbasiertes Mapping (Route 2)
# ============================================================================
def m_from_data(
    delta_brent_log: float,
    delta_rate_10y_pp: float,
    cov_factors: np.ndarray,
    horizon_days: int = 252,
) -> dict:
    """Mahalanobis-Distanz des Schock-Vektors als M-Schätzung.

    Idee: M ist ein N(0,1)-Faktor, also entspricht das negative der
    Mahalanobis-Norm des Schocks (in Standard-Abweichungen) einem
    plausiblen Ausdruck dafür, "wie viele σ vom Mean wir entfernt sind".

    Sign-Konvention: positives Adverse-Alignment (Brent rauf + Rate rauf)
    → negatives M.

    Parameters
    ----------
    cov_factors : 2x2 Kovarianz-Matrix [Brent_log, Δr_10y_pp] tagesweise
    horizon_days : skaliert die Kovarianz auf den Horizon (iid-Annahme)

    Returns
    -------
    dict mit
        m_data : signiertes M
        mahalanobis_distance : positive Distanz in σ-Einheiten
        adverse_direction : Cosine-Alignment mit "rauf+rauf" als Adverse
    """
    shock = np.array([delta_brent_log, delta_rate_10y_pp])
    cov_h = np.asarray(cov_factors) * horizon_days   # iid scaling
    inv_cov = np.linalg.pinv(cov_h)

    # Mahalanobis-Distanz |shock|_Σ (immer ≥ 0)
    md = float(np.sqrt(shock @ inv_cov @ shock))

    # Adverse-Direction: aligned mit (positiv, positiv) → negatives M
    # (Brent-Spike + Rate-Hike = klassisches Inflation-Schock-Muster)
    adv_dir = np.array([1.0, 1.0])
    adv_dir = adv_dir / np.linalg.norm(adv_dir)
    cos_adv = float(np.dot(shock / max(np.linalg.norm(shock), 1e-12), adv_dir))

    return {
        "m_data":               -md * cos_adv,
        "mahalanobis_distance": md,
        "adverse_direction":    cos_adv,
    }


def factor_covariance(
    brent_returns: pd.Series, rate_changes_pp: pd.Series,
    *, lookback_days: int = 252,
) -> np.ndarray:
    """Empirische 2x2 Kovarianz aus den letzten `lookback_days`."""
    df = pd.concat({"brent": brent_returns, "rate_pp": rate_changes_pp}, axis=1)
    df = df.dropna().tail(lookback_days)
    return df.cov().to_numpy()


# ============================================================================
# 3. Hybrid-Mapping (Anchor + Data nebeneinander)
# ============================================================================
def hybrid_mapping(
    delta_brent_log: float,
    delta_rate_10y_pp: float,
    *,
    anchor: MacroAnchor | None = None,
    cov_factors: np.ndarray | None = None,
    horizon_days: int = 252,
) -> dict:
    """Liefert beide Routen + abgeleitete Diagnostik.

    Returns
    -------
    dict mit
        m_anchor      : aus Anchor-Skalierung (Primary)
        m_data        : aus Mahalanobis-Distanz (Validation)
        m_hybrid      : Mittel der beiden (falls beide vorhanden)
        anchor_label  : Anker-Beschreibung
        consistency   : |m_anchor − m_data| (Δ in σ-Einheiten)
    """
    out = {
        "delta_brent_log":   delta_brent_log,
        "delta_rate_10y_pp": delta_rate_10y_pp,
        "m_anchor":          None,
        "m_data":            None,
        "m_hybrid":          None,
        "anchor_label":      None,
        "mahalanobis":       None,
        "consistency":       None,
    }
    if anchor is not None:
        out["m_anchor"] = m_from_anchor(delta_brent_log, delta_rate_10y_pp, anchor)
        out["anchor_label"] = anchor.label

    if cov_factors is not None:
        d = m_from_data(delta_brent_log, delta_rate_10y_pp, cov_factors, horizon_days)
        out["m_data"] = d["m_data"]
        out["mahalanobis"] = d["mahalanobis_distance"]

    if out["m_anchor"] is not None and out["m_data"] is not None:
        out["m_hybrid"] = 0.5 * (out["m_anchor"] + out["m_data"])
        out["consistency"] = abs(out["m_anchor"] - out["m_data"])
    elif out["m_anchor"] is not None:
        out["m_hybrid"] = out["m_anchor"]
    elif out["m_data"] is not None:
        out["m_hybrid"] = out["m_data"]

    return out


# ============================================================================
# 4. Validation
# ============================================================================
def _test_anchor_baseline_zero():
    """Schock = 0 → M = 0."""
    a = anchor_from_eba("2025")
    assert abs(m_from_anchor(0.0, 0.0, a)) < 1e-9


def _test_anchor_replicates_anchor_point():
    """Schock = Anchor → M = M_anchor (Selbst-Konsistenz)."""
    a = anchor_from_eba("2025")
    m = m_from_anchor(a.brent_log, a.rate_10y_pp, a)
    assert abs(m - a.z_factor) < 1e-6, f"Got M={m:.4f}, expected {a.z_factor:.4f}"


def _test_anchor_sign_flips_on_anti_aligned():
    """Anti-aligned Schock (Crash + Rate-Cut) → M positiv (benign)."""
    a = anchor_from_eba("2025")
    m = m_from_anchor(-a.brent_log, -a.rate_10y_pp, a)
    assert m > 0, f"Anti-aligned shock should yield positive M, got {m:.4f}"


def _test_anchor_scales_linearly():
    """2x Schock → 2x M."""
    a = anchor_from_eba("2025")
    m1 = m_from_anchor(a.brent_log, a.rate_10y_pp, a)
    m2 = m_from_anchor(2 * a.brent_log, 2 * a.rate_10y_pp, a)
    assert abs(m2 - 2 * m1) < 1e-6, f"Linear scaling broken: m1={m1:.4f}, m2={m2:.4f}"


def _test_data_mapping_signs():
    """Adverse Schock → negatives m_data; benign Schock → positives."""
    cov = np.array([[0.0004, 0.00002], [0.00002, 0.0001]])
    d_adv = m_from_data(0.4, 1.5, cov, horizon_days=252)
    d_ben = m_from_data(-0.4, -1.5, cov, horizon_days=252)
    assert d_adv["m_data"] < 0, f"Adverse shock should yield m_data<0, got {d_adv['m_data']}"
    assert d_ben["m_data"] > 0, f"Benign shock should yield m_data>0, got {d_ben['m_data']}"


def _test_hybrid_consistency():
    """Hybrid liefert m_anchor und m_data nebeneinander."""
    a = anchor_from_eba("2025")
    cov = np.array([[0.0004, 0.00002], [0.00002, 0.0001]])
    h = hybrid_mapping(a.brent_log, a.rate_10y_pp, anchor=a, cov_factors=cov)
    assert h["m_anchor"] is not None
    assert h["m_data"] is not None
    assert h["consistency"] is not None
    assert h["m_hybrid"] is not None


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("macro_factor.py - Macro Shock to M Mapping - Validation")
    print("=" * 60)
    tests = [
        ("Baseline shock yields M=0",              _test_anchor_baseline_zero),
        ("Anchor self-consistency (M = z_anchor)", _test_anchor_replicates_anchor_point),
        ("Anti-aligned shock yields positive M",   _test_anchor_sign_flips_on_anti_aligned),
        ("Linear scaling in shock magnitude",      _test_anchor_scales_linearly),
        ("Data route signs (adverse vs benign)",   _test_data_mapping_signs),
        ("Hybrid mapping returns both routes",     _test_hybrid_consistency),
    ]
    for label, fn in tests:
        try:
            fn()
            print(f"  [PASS]  {label}")
        except AssertionError as e:
            print(f"  [FAIL]  {label}: {e}")
            raise

    print("\n--- Sample shock vectors mapped to M ---\n")
    a = anchor_from_eba("2025")
    cov = np.array([[0.0004, 0.00002], [0.00002, 0.0001]])

    samples = [
        ("EBA 2025 adverse (anchor)",  a.brent_log, a.rate_10y_pp),
        ("Half adverse",               a.brent_log/2, a.rate_10y_pp/2),
        ("Iran 2026 hypothetical",     0.55, 0.50),
        ("Ukraine 2022 historical",    0.12, 0.67),
        ("Corona 2020 literature",    -1.20, -0.65),
        ("Mild stagflation",           0.20, 0.50),
        ("Disinflation rally",        -0.15, -0.50),
        ("Pure rate spike",            0.00, 1.00),
        ("Pure oil shock",             0.30, 0.00),
    ]
    rows = []
    for label, db, dr in samples:
        h = hybrid_mapping(db, dr, anchor=a, cov_factors=cov)
        rows.append({
            "Shock":             label,
            "ΔBrent_log":        f"{db:+.2f}",
            "Δr_10y_pp":         f"{dr:+.2f}",
            "M (anchor)":        f"{h['m_anchor']:+.2f}",
            "M (data)":          f"{h['m_data']:+.2f}",
            "M (hybrid)":        f"{h['m_hybrid']:+.2f}",
            "|Δ| consistency":   f"{h['consistency']:.2f}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n[PASS] All macro-factor tests passed.")
