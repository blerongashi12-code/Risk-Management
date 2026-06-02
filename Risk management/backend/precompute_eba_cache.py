"""
============================================================================
 precompute_eba_cache.py · Build small committable EBA artefacts
============================================================================

Die EBA-Rohdateien (tr_cre.csv ~123 MB, tr_sov.csv ~91 MB, tr_oth.csv,
TR_Metadata.xlsx, SDD.xlsx, transparency_YYYY/) sind zu groß für Git und
werden per .gitignore ausgeschlossen. In der Cloud (Streamlit Community
Cloud) liegen sie deshalb nicht vor — alle Seiten, die sie direkt parsen
(Marktbuch, Eigenkapital, Validierung), liefen in FileNotFoundError.

Dieses Skript wird **einmal lokal** ausgeführt (wo die Rohdateien
vorliegen) und schreibt die *bereits gefilterten/geparsten* Ergebnis-
DataFrames als kleine Parquet-Dateien nach `data/derived/`. Diese werden
committet. `eba_loader` greift in der Cloud automatisch auf diese Parquets
zurück, wenn die Rohdatei fehlt (siehe `_load_or_fallback` dort).

Ausführen:
    python backend/precompute_eba_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_ROOT / "backend"))

from config import EBA_RAW_DIR, DERIVED_DIR  # type: ignore
import eba_loader as L  # type: ignore

PERIOD = 202506


def _save(df: pd.DataFrame, name: str) -> None:
    out = DERIVED_DIR / name
    df.to_parquet(out, index=False)
    size_kb = out.stat().st_size / 1024
    print(f"  [OK] {name:32s} {len(df):>8,d} rows  {size_kb:>9,.1f} KB")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 64)
    print("Precompute EBA derived cache")
    print("=" * 64)
    print(f"  Raw dir:     {EBA_RAW_DIR}")
    print(f"  Derived dir: {DERIVED_DIR}")

    cre_path  = EBA_RAW_DIR / "tr_cre.csv"
    sov_path  = EBA_RAW_DIR / "tr_sov.csv"
    oth_path  = EBA_RAW_DIR / "tr_oth.csv"
    meta_path = EBA_RAW_DIR / "TR_Metadata.xlsx"

    missing = [p.name for p in (cre_path, sov_path, oth_path, meta_path)
               if not p.exists()]
    if missing:
        raise SystemExit(
            f"Rohdateien fehlen in {EBA_RAW_DIR}: {missing}. "
            f"Dieses Skript nur dort ausführen, wo die EBA-Rohdaten liegen."
        )

    print("\n-- Bank directory + country dim (TR_Metadata.xlsx) --")
    _save(L.load_bank_directory(meta_path), "bank_dir.parquet")
    _save(L.load_country_dim(meta_path), "country_dim.parquet")

    print("\n-- Credit-risk long table (tr_cre.csv, all periods) --")
    cre = L.parse_credit_risk_csv(cre_path, period=None)
    _save(cre, "cre_raw.parquet")

    print("\n-- Sovereign long table (tr_sov.csv, all periods) --")
    sov = L.parse_sovereign_csv(sov_path, period=None)
    _save(sov, "sov_raw.parquet")

    print(f"\n-- Capital overview (tr_oth.csv, period {PERIOD}) --")
    cap = L.parse_capital_overview(oth_path, period=PERIOD)
    _save(cap, f"cap_df_{PERIOD}.parquet")

    print("\n-- Historical capital panel (all vintages) --")
    panel = L.load_historical_capital_panel(EBA_RAW_DIR)
    _save(panel, "hist_capital_panel.parquet")

    print("\n[DONE] Derived cache written. Commit data/derived/*.parquet.")


if __name__ == "__main__":
    main()
