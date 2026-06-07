"""
============================================================================
 eba_pd_loader.py · Loader für bank-spezifische A-IRB-PDs und -LGDs
============================================================================

Lädt die exposure-gewichteten 1-Jahres-PDs und -LGDs aus den Pillar-3-
Disclosures der zehn größten EU-IRB-Banken. Primary source:
`data/pillar3_bank_pd_lgd.csv`.

Methodische Eckpunkte
---------------------
- **Primary source:** bank-publizierte EU CR6 Sub-totals (EBA ITS on
  Disclosure ITS/2020/04, CRR Art. 431–455) — pro Bank × IRB-Klasse die
  EAD-gewichtete Average-PD und Average-LGD direkt aus dem Pillar-3-Report
  der jeweiligen Bank. PD-Definition: regulatorisch publizierte 1-Jahres-
  PD aus internen Rating-Modellen (CRR Art. 180).
- **Fallback bei nicht-extrahierbarer Pillar-3-Disclosure:** Country-
  Aggregate aus dem EBA Risk Dashboard Credit Risk Parameters Annex
  Q4 2025 (Source: COREP C 9.02 / Heimatland-Proxy). Diese Zeilen sind in
  der Spalte `status` mit ``country_proxy_pending_pillar3`` markiert; der
  Loader emittiert beim Laden eine sichtbare Warnung.
- **Bank- und Sovereign-Klasse:** falls bank-spezifisch nicht verfügbar →
  Basel III F-IRB-Defaults (0,20 % PD / 45 % LGD für Banken, 0,05 % PD /
  45 % LGD für Sovereign) markiert als ``basel_default_pending_pillar3``.

Status-Werte in der CSV
-----------------------
- ``pillar3_verified``                 — direkt aus dem Pillar-3 EU CR6 der
                                         Bank, mit `source_url`, Seite,
                                         Periode dokumentiert
- ``country_proxy_pending_pillar3``    — EBA Risk Dashboard Country-
                                         Aggregat als Übergangs-Wert, bis
                                         Pillar-3 extrahiert wurde
- ``basel_default_pending_pillar3``    — F-IRB-Default; nur für Bank- und
                                         Sovereign-Klasse, wo die meisten
                                         Banken keine separate IRB-PD
                                         publizieren

API
---
- load_pd_table(path=None) -> pd.DataFrame
- get_pd_for_bank(lei, vasicek_class) -> dict
- filter_universe_to_top10(universe) -> EbaUniverse
- enrich_segments(seg_df) -> pd.DataFrame
- coverage_report() -> dict  — Anteile verified / country_proxy / default
============================================================================
"""
from __future__ import annotations

import sys
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ----------------------------------------------------------------------
# 1. CSV-Pfad
# ----------------------------------------------------------------------
DEFAULT_CSV_PATH = _ROOT / "data" / "pillar3_bank_pd_lgd.csv"

_CACHED: pd.DataFrame | None = None


def load_pd_table(path: Path | str | None = None) -> pd.DataFrame:
    """Lädt die bank-spezifische PD/LGD-Tabelle.

    Returns DataFrame mit Spalten
      bank_name, LEI, bank_country, vasicek_class, pd_pct, lgd_pct,
      status, source, source_period, source_table, source_page, source_url
    """
    global _CACHED
    if _CACHED is not None and path is None:
        return _CACHED.copy()
    p = Path(path) if path else DEFAULT_CSV_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"PD/LGD-Tabelle nicht gefunden: {p}. Bitte sicherstellen, "
            f"dass `data/pillar3_bank_pd_lgd.csv` existiert (extrahiert "
            f"aus den Pillar-3-Reports der 10 Banken)."
        )
    df = pd.read_csv(p)
    df["pd_pct"]  = pd.to_numeric(df["pd_pct"],  errors="coerce")
    df["lgd_pct"] = pd.to_numeric(df["lgd_pct"], errors="coerce")

    # Defensive: alte CSV ohne status-Spalte → markiere alle Zeilen als
    # country_proxy (legacy view)
    if "status" not in df.columns:
        df["status"] = "country_proxy_pending_pillar3"
        df["source_url"] = ""

    if path is None:
        _CACHED = df.copy()
        _emit_coverage_warning(df)
    return df


# Backward-Kompat-Alias
load_eba_pd_table = load_pd_table


# ----------------------------------------------------------------------
# 2. Coverage-Warning (sichtbarer Status-Banner beim Laden)
# ----------------------------------------------------------------------
def _emit_coverage_warning(df: pd.DataFrame) -> None:
    """Druckt einen Coverage-Report nach stderr, sobald die Tabelle das
    erste Mal geladen wird. Macht den Datenzustand für jeden Cockpit-Run
    sichtbar.
    """
    n_total = len(df)
    by_status = df["status"].value_counts().to_dict()
    n_verified = by_status.get("pillar3_verified", 0)
    n_proxy    = by_status.get("country_proxy_pending_pillar3", 0)
    n_default  = by_status.get("basel_default_pending_pillar3", 0)
    pct_verified = 100.0 * n_verified / n_total if n_total else 0.0

    banks_verified = df.loc[df["status"] == "pillar3_verified",
                              "bank_name"].nunique()
    banks_total    = df["bank_name"].nunique()

    msg = (
        f"PD/LGD-Tabelle geladen · {n_total} Zeilen "
        f"({banks_verified}/{banks_total} Banken Pillar-3-verifiziert, "
        f"{pct_verified:.0f} % der Zeilen verified) · "
        f"verified={n_verified}, country_proxy={n_proxy}, "
        f"basel_default={n_default}"
    )
    print(f"[eba_pd_loader] {msg}", file=sys.stderr)


def coverage_report() -> dict:
    """Strukturierter Coverage-Report — für UI-Banner und Diagnostik."""
    df = load_pd_table()
    n_total = len(df)
    by_status = df["status"].value_counts().to_dict()
    banks_by_status = (df.groupby("status")["bank_name"]
                         .apply(lambda s: sorted(s.unique().tolist())))
    return {
        "n_rows_total":    n_total,
        "n_banks_total":   df["bank_name"].nunique(),
        "by_status":       by_status,
        "banks_by_status": banks_by_status.to_dict(),
        "pct_verified":    100.0 * by_status.get("pillar3_verified", 0)
                            / n_total if n_total else 0.0,
        "verified_banks":  sorted(df.loc[df["status"] == "pillar3_verified",
                                          "bank_name"].unique().tolist()),
    }


# ----------------------------------------------------------------------
# 3. Bank- und Klassen-spezifische Lookup-Funktionen
# ----------------------------------------------------------------------
def get_top10_leis() -> set[str]:
    df = load_pd_table()
    return set(df["LEI"].unique())


def get_top10_bank_names() -> dict:
    df = load_pd_table()
    return dict(zip(df["LEI"], df["bank_name"]))


def _strip_accents(s: str) -> str:
    """Entfernt diakritische Zeichen (é→e, ö→o, …) via NFKD-Normalisierung.

    Nötig, weil die kuratierte CSV ASCII-Namen führt ("Societe Generale"),
    die EBA-Metadaten aber Akzente ("Société générale S.A.") — ohne diesen
    Schritt scheiterte das Matching und SocGen fiel aus der 10er-Liste.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _name_match(eba_name: str, universe_name: str) -> bool:
    """Robustes, akzent-insensitives Substring-Matching."""
    a = _strip_accents(eba_name).lower()
    b = _strip_accents(universe_name).lower()
    for tok in ("groupe ", "banco ", "cooperatieve ",
                "confederation nationale du ", " n.v.", " s.a.", "groep"):
        a = a.replace(tok, "")
        b = b.replace(tok, "")
    a, b = a.strip(), b.strip()
    return bool(a) and bool(b) and (a in b or b in a)


def filter_universe_to_top10(universe):
    """Filtert ein EbaUniverse auf die Top-10-Banken UND ersetzt die
    segment-PDs/-LGDs durch die bank-spezifischen Werte aus der
    pillar3_bank_pd_lgd.csv.

    Effekt:
      - universe.banks enthält nur noch die 10 Banken aus der CSV
      - Für jede Bank × Klasse werden segment.pd und segment.lgd auf die
        publizierten Werte gesetzt (Pillar-3-verified, country-proxy oder
        F-IRB-default — je nach `status`)
    """
    df = load_pd_table()
    target_names = df["bank_name"].unique().tolist()
    name_to_lei  = dict(zip(df["bank_name"], df["LEI"]))
    curated_leis = set(df["LEI"].unique())

    filtered = {}
    universe_to_lei = {}
    for bank_name, portfolio in universe.banks.items():
        # 1) Primär: exakter LEI-Match (robust, akzent-/schreibweise-immun)
        port_lei = getattr(portfolio, "lei", "") or ""
        if port_lei and port_lei in curated_leis:
            filtered[bank_name] = portfolio
            universe_to_lei[bank_name] = port_lei
            continue
        # 2) Fallback: akzent-insensitiver Namensvergleich
        for eba_name in target_names:
            if _name_match(eba_name, bank_name):
                filtered[bank_name] = portfolio
                universe_to_lei[bank_name] = name_to_lei[eba_name]
                break

    # Guard: stilles Verschlucken einer kuratierten Bank verhindern.
    matched_leis = set(universe_to_lei.values())
    missing_leis = curated_leis - matched_leis
    if missing_leis:
        lei_to_name = dict(zip(df["LEI"], df["bank_name"]))
        missing_names = sorted(lei_to_name.get(l, l) for l in missing_leis)
        warnings.warn(
            f"filter_universe_to_top10: {len(missing_leis)} kuratierte "
            f"Bank(en) konnten NICHT im EBA-Universe gematcht werden und "
            f"fehlen daher im Modell: {missing_names}. Erwartet wurden alle "
            f"{len(curated_leis)} Banken aus pillar3_bank_pd_lgd.csv.",
            stacklevel=2,
        )

    pd_lookup = df.set_index(["LEI", "vasicek_class"])[["pd_pct", "lgd_pct",
                                                          "status"]]
    n_overrides, n_verified = 0, 0
    for bank_name, portfolio in filtered.items():
        lei = universe_to_lei[bank_name]
        for seg in portfolio.segments:
            v_class = seg.exposure_class
            key = (lei, v_class)
            if key in pd_lookup.index:
                seg.pd  = float(pd_lookup.loc[key, "pd_pct"])  / 100.0
                seg.lgd = float(pd_lookup.loc[key, "lgd_pct"]) / 100.0
                n_overrides += 1
                if pd_lookup.loc[key, "status"] == "pillar3_verified":
                    n_verified += 1

    universe.banks = filtered
    universe.source = (f"{universe.source} | PDs/LGDs aus "
                       f"pillar3_bank_pd_lgd.csv ({len(filtered)} Banken, "
                       f"{n_overrides} Segmente, davon {n_verified} "
                       f"Pillar-3-verifiziert)")
    return universe


def get_pd_for_bank(lei: str, vasicek_class: str) -> dict:
    """Liefert PD% und LGD% (plus Source) für (LEI, Klasse).

    Returns dict mit Keys: pd_pct, lgd_pct, pd_decimal, lgd_decimal,
                          status, source, source_period, source_url
    Raises KeyError wenn Kombination nicht gefunden.
    """
    df = load_pd_table()
    row = df[(df["LEI"] == lei) & (df["vasicek_class"] == vasicek_class)]
    if len(row) == 0:
        raise KeyError(
            f"Keine PD/LGD-Daten für LEI={lei}, Klasse={vasicek_class}. "
            f"Bank evtl. nicht in der Top-10-Universe."
        )
    r = row.iloc[0]
    return {
        "pd_pct":        float(r["pd_pct"]),
        "lgd_pct":       float(r["lgd_pct"]),
        "pd_decimal":    float(r["pd_pct"]) / 100.0,
        "lgd_decimal":   float(r["lgd_pct"]) / 100.0,
        "status":        str(r.get("status", "unknown")),
        "source":        str(r["source"]),
        "source_period": str(r["source_period"]),
        "source_table":  str(r.get("source_table", "")),
        "source_page":   str(r.get("source_page", "")),
        "source_url":    str(r.get("source_url", "")),
    }


# ----------------------------------------------------------------------
# 4. Segment-Anreicherung (für Pipeline-Kompatibilität)
# ----------------------------------------------------------------------
def enrich_segments(seg_df: pd.DataFrame) -> pd.DataFrame:
    """Reicht seg_df (Bank × Klasse) um PD und LGD an."""
    pd_table = load_pd_table()
    lookup = pd_table.set_index(["LEI", "vasicek_class"])[["pd_pct", "lgd_pct",
                                                             "status"]]
    out = seg_df.copy()

    pd_vals, lgd_vals, status_vals, has_data = [], [], [], []
    for _, row in out.iterrows():
        key = (row["LEI_Code"], row["v_class"])
        if key in lookup.index:
            pd_vals.append(float(lookup.loc[key, "pd_pct"]))
            lgd_vals.append(float(lookup.loc[key, "lgd_pct"]))
            status_vals.append(str(lookup.loc[key, "status"]))
            has_data.append(True)
        else:
            pd_vals.append(float("nan"))
            lgd_vals.append(float("nan"))
            status_vals.append("missing")
            has_data.append(False)
    out["pd_eba_pct"]  = pd_vals
    out["lgd_eba_pct"] = lgd_vals
    out["pd_eba_dec"]  = [v/100 if pd.notna(v) else float("nan") for v in pd_vals]
    out["lgd_eba_dec"] = [v/100 if pd.notna(v) else float("nan") for v in lgd_vals]
    out["pd_status"]   = status_vals
    out["has_eba_data"] = has_data
    return out


# ----------------------------------------------------------------------
# 5. Self-Tests
# ----------------------------------------------------------------------
def _test_load():
    df = load_pd_table()
    assert len(df) == 70, f"Erwartet 70 Zeilen, gefunden {len(df)}"
    assert df["bank_name"].nunique() == 10, "Erwartet 10 Banken"
    assert set(df["vasicek_class"].unique()) >= {
        "corporate", "sme_corporate", "mortgage", "qrre",
        "other_retail", "bank", "sovereign",
    }
    assert "status" in df.columns, "Spalte 'status' fehlt"
    assert "vintage_date" in df.columns, "Spalte 'vintage_date' fehlt"


def _test_vintage_consistency():
    """Stichtag-Einheitlichkeit: ALLE Zeilen müssen denselben vintage_date haben.

    Methodische Anforderung: Stress-Modelle starten von einem definierten
    "Heute"-Snapshot. Das ist EBA-ST-2025-Standard und unsere Konvention.
    Bei Mix-Stichtag würde der Macro-Schock auf inkonsistente Baselines
    angewendet (manche Banken hätten schon 6+ Monate Macro-Evolution
    absorbiert) — methodisch unzulässig.
    """
    df = load_pd_table()
    vintages = df["vintage_date"].unique()
    assert len(vintages) == 1, (
        f"Stichtag-Inkonsistenz! Gefundene Vintage-Daten: {sorted(vintages)} "
        f"— alle Zeilen müssen denselben vintage_date haben "
        f"(EBA-ST-2025-Standard, einheitlicher Snapshot für Forward-Stress)."
    )
    assert vintages[0] == "2024-12-31", (
        f"Erwartete Vintage 2024-12-31, gefunden: {vintages[0]}. "
        f"Bei Vintage-Umstellung: alle Banken konsistent re-extrahieren."
    )


def _test_lookup_pillar3_verified():
    # Deutsche Bank Corporate sollte 4.12 % sein (Pillar 3 Q4 2024)
    r = get_pd_for_bank("7LTWFZYICNSX8D621K86", "corporate")
    assert abs(r["pd_pct"] - 4.12) < 0.01, f"DB Corp PD: {r['pd_pct']}"
    assert r["status"] == "pillar3_verified", f"DB Corp Status: {r['status']}"
    assert "Deutsche Bank Pillar 3" in r["source"]


def _test_coverage():
    cov = coverage_report()
    assert cov["n_banks_total"] == 10
    assert cov["n_rows_total"] == 70
    # Mindestens 5 Banken sollten Pillar-3-verifiziert sein
    # (aktuelle Coverage: DB, ING, SocGen, Rabobank, UniCredit, CA, CM=partial)
    assert len(cov["verified_banks"]) >= 5, \
        f"Erwartet ≥ 5 verified Banken, gefunden {cov['verified_banks']}"
    assert "Deutsche Bank" in cov["verified_banks"]


def _test_enrich():
    seg = pd.DataFrame({
        "LEI_Code": ["7LTWFZYICNSX8D621K86", "UNKNOWN_LEI"],
        "v_class":  ["corporate", "corporate"],
        "ead_eur":  [1e9, 5e8],
    })
    out = enrich_segments(seg)
    assert bool(out.iloc[0]["has_eba_data"]) is True
    assert out.iloc[0]["pd_status"] == "pillar3_verified"
    assert abs(out.iloc[0]["pd_eba_pct"] - 4.12) < 0.01
    assert bool(out.iloc[1]["has_eba_data"]) is False


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("eba_pd_loader.py · Tests")
    print("=" * 60)
    for label, fn in [
        ("CSV-Load + Schema",          _test_load),
        ("Vintage-Date-Konsistenz",    _test_vintage_consistency),
        ("Pillar-3-verified-Lookup",   _test_lookup_pillar3_verified),
        ("Coverage-Report",            _test_coverage),
        ("Segment-Anreicherung",       _test_enrich),
    ]:
        try:
            fn()
            print(f"  [PASS]  {label}")
        except AssertionError as e:
            print(f"  [FAIL]  {label}: {e}")
            raise
    print()
    cov = coverage_report()
    print("Coverage-Status:")
    for k, v in cov["by_status"].items():
        print(f"  {k:40s} {v:>3d} rows")
    print(f"\nVerified banks ({len(cov['verified_banks'])}/10):")
    for b in cov["verified_banks"]:
        print(f"  · {b}")
    print()
    print("[PASS] All tests passed.")
