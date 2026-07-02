"""
============================================================================
 04_fetch_bundesbank_svensson.py
 ---------------------------------------------------------------------------
 Parst die Svensson-Parameter aus dem offiziellen Bundesbank-CSV-Export.

 Quelle:
   https://www.bundesbank.de/dynamic/action/de/statistiken/zeitreihen-datenbanken
     /zeitreihen-datenbank/759778/759778

 Du lädst die CSV manuell herunter und legst sie ab unter:
   data/bundesbank_svensson.csv

 Output:
   - DataFrame `svensson_params`:
       Index   = Date
       Columns = [Beta0, Beta1, Beta2, Beta3, Tau1, Tau2]
   - Cache: data/cache/svensson_params.parquet

 Charakteristika:
   - Wochenenden/Feiertage werden automatisch gefiltert
   - Deutsches Dezimalformat (Komma) wird konvertiert
============================================================================
"""

import time
import pandas as pd
from pathlib import Path

# Pfad-Bootstrap: Skript liegt in tools/data_fetch/ — config.py am App-Root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATA_DIR, CACHE_DIR, BBK_COLUMNS, BBK_RAW_INDICES


def parse_bundesbank_csv(csv_path: Path) -> pd.DataFrame:
    """
    Parst das offizielle BBK-Format:
        Zeile 1-5: Metadaten
        Zeile 6+:  Date;Beta0;Flag;Beta1;Flag;...;Tau2;Flag
    """
    raw = pd.read_csv(
        csv_path,
        sep=";",
        skiprows=5,
        header=None,
        encoding="utf-8-sig",
        dtype=str,
        na_values=[".", ""],
        keep_default_na=False,
    )

    # Nur Zeilen mit echtem Datum behalten (letzte Zeilen sind Bemerkungen)
    raw = raw[raw[0].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]

    # Relevante Spalten herausfiltern (Flag-Spalten überspringen)
    df = raw[BBK_RAW_INDICES].copy()
    df.columns = BBK_COLUMNS

    # Deutsches Dezimalformat → Punkt
    for c in BBK_COLUMNS[1:]:
        df[c] = df[c].str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Handelstage: alle 6 Parameter müssen vorhanden sein
    df_clean = df.dropna(how="any").sort_index()

    return df_clean


def fetch_svensson_params(
    csv_filename: str = "bundesbank_svensson.csv",
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
) -> pd.DataFrame:
    """
    Zentrale Funktion: lädt & parst Svensson-Parameter.

    Sucht die CSV-Datei unter data/<csv_filename>.
    Falls nicht vorhanden → informative Error-Message mit Download-Link.
    """
    csv_path = DATA_DIR / csv_filename
    cache_file = CACHE_DIR / "svensson_params.parquet"

    # --- Cache Check ---
    if use_cache and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < cache_max_age_hours:
            # Nur Cache nutzen wenn CSV nicht neuer ist
            if csv_path.exists():
                if cache_file.stat().st_mtime >= csv_path.stat().st_mtime:
                    print(f"[CACHE] Lade Svensson aus Cache ({age_h:.1f}h alt)")
                    return pd.read_parquet(cache_file)
            else:
                print(f"[CACHE] Lade Svensson aus Cache (CSV nicht gefunden, "
                      f"{age_h:.1f}h alt)")
                return pd.read_parquet(cache_file)

    # --- CSV-Datei vorhanden? ---
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n❌ Bundesbank-CSV nicht gefunden: {csv_path}\n\n"
            f"   Bitte manuell herunterladen:\n"
            f"   https://www.bundesbank.de/dynamic/action/de/statistiken/\n"
            f"   zeitreihen-datenbanken/zeitreihen-datenbank/759778/759778\n\n"
            f"   Und speichern unter: {csv_path}\n"
        )

    print(f"[FETCH] Parse Bundesbank-CSV: {csv_path.name}")
    df = parse_bundesbank_csv(csv_path)

    print(f"[DONE]  {len(df):,} Handelstage")
    print(f"        Von: {df.index.min().date()}")
    print(f"        Bis: {df.index.max().date()}")

    df.to_parquet(cache_file)
    print(f"[CACHE] Gespeichert nach {cache_file}")

    return df


def compute_historical_stats(svensson_df: pd.DataFrame) -> dict:
    """
    Berechnet Statistiken, die das Monte-Carlo später braucht:
       - Annualisierte Vola der täglichen Δβ
       - Korrelationsmatrix der täglichen Δβ
       - Kovarianzmatrix (annualisiert)
    """
    deltas = svensson_df.diff().dropna()

    stats = {
        "levels_latest":  svensson_df.iloc[-1].to_dict(),
        "levels_mean":    svensson_df.mean().to_dict(),
        "deltas_mean":    deltas.mean().to_dict(),
        "deltas_std":     deltas.std().to_dict(),
        "annualized_vol": (deltas.std() * (252 ** 0.5)).to_dict(),
        "correlation":    deltas.corr(),
        "covariance_ann": deltas.cov() * 252,
    }
    return stats


# ======================================================================
if __name__ == "__main__":
    svensson_params = fetch_svensson_params()

    print("\n[PREVIEW] Letzte 5 Handelstage:")
    print(svensson_params.tail(5).round(5))

    print("\n[STATS] Historische Statistiken (annualisiert):")
    stats = compute_historical_stats(svensson_params)

    print("\n   Levels (aktuell):")
    for k, v in stats["levels_latest"].items():
        print(f"     {k:<8} {v:>10.4f}")

    print("\n   Annualisierte Volatilität täglicher Δ:")
    for k, v in stats["annualized_vol"].items():
        print(f"     Δ{k:<8} {v:>10.4f}  ({v*100:.0f}bp)")

    print("\n   Korrelationsmatrix Δβ₀/Δβ₁/Δβ₂ (Level/Slope/Curvature):")
    print(stats["correlation"].iloc[:3, :3].round(3))
