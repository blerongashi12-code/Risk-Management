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
- **Aktuelle Abdeckung:** Alle 69 IRB-fähigen Bank-Klassen-Kombinationen
  stammen direkt aus bankpublizierten EU-CR6-Sub-totals. Country-Proxies
  sind nach erneuter Originalbericht-Extraktion nicht mehr erforderlich.
- **Strukturelle Ausnahme:** Santander Sovereign wird vollständig im
  Standardansatz geführt. Die Position wird daher aus dem IRB-
  Kreditbuchkanal entfernt, bleibt aber im separaten Sovereign-
  Marktwertkanal auf Basis der EBA-Bestände enthalten.

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
- ``standardised_not_applicable``      — keine IRB-PD/LGD, da die Position
                                         regulatorisch im Standardansatz
                                         geführt wird; aus IRB-Kreditbuch
                                         ausgeschlossen

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

_CACHED_FULL: pd.DataFrame | None = None


def _load_full(path: Path | str | None = None) -> pd.DataFrame:
    """Lädt das vollständige (potenziell multi-vintage) PD/LGD-Long-Panel.

    Spalten: bank_name, LEI, bank_country, vasicek_class, pd_pct, lgd_pct,
    vintage_date, status, source, source_period, source_table, source_page,
    source_url. Eine Zeile pro (LEI, vasicek_class, vintage_date).
    """
    global _CACHED_FULL
    if _CACHED_FULL is not None and path is None:
        return _CACHED_FULL
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
    # Defensive: alte CSV ohne vintage_date → einheitlich 31.12.2024
    if "vintage_date" not in df.columns:
        df["vintage_date"] = "2024-12-31"
    df["vintage_date"] = df["vintage_date"].astype(str)

    if path is None:
        _CACHED_FULL = df
        # Coverage-Banner bezieht sich auf den Live-Snapshot (jüngster Stichtag)
        latest = df["vintage_date"].max()
        _emit_coverage_warning(df[df["vintage_date"] == latest])
    return df


def available_vintages(path: Path | str | None = None) -> list[str]:
    """Sortierte Liste der vorhandenen Stichtage (vintage_date)."""
    return sorted(_load_full(path)["vintage_date"].dropna().unique().tolist())


def load_pd_table(path: Path | str | None = None,
                  vintage: str = "latest") -> pd.DataFrame:
    """Bank-spezifische PD/LGD-Tabelle, gefiltert auf EINEN Stichtag.

    ``vintage``:
      - ``"latest"`` (Default) → jüngster vintage_date. Das ist der
        Live-Modell-Snapshot (31.12.2024); Verhalten identisch zum
        früheren Single-Vintage-Loader (eine Zeile je LEI×Klasse).
      - ``"all"`` → vollständiges Long-Panel über alle Stichtage.
      - ``"YYYY-MM-DD"`` → genau dieser Stichtag (für den Walk-Forward-
        Backtest, der das Portfolio mit den *damals gültigen* PD/LGD
        einfriert — kein Look-ahead aus 2024).

    Banken ohne den angefragten Stichtag liefern entsprechend keine Zeilen
    (der Backtest überspringt sie für dieses Vintage).
    """
    full = _load_full(path)
    if vintage == "all":
        return full.copy()
    target = full["vintage_date"].max() if vintage == "latest" else str(vintage)
    return full[full["vintage_date"] == target].copy()


def load_pd_panel(path: Path | str | None = None) -> pd.DataFrame:
    """Vollständiges Multi-Vintage-Panel (= load_pd_table(vintage='all'))."""
    return load_pd_table(path, vintage="all")


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
    n_na       = by_status.get("standardised_not_applicable", 0)
    n_eligible = n_total - n_na
    pct_verified = 100.0 * n_verified / n_eligible if n_eligible else 0.0

    banks_verified = df.loc[df["status"] == "pillar3_verified",
                              "bank_name"].nunique()
    banks_total    = df["bank_name"].nunique()

    msg = (
        f"PD/LGD-Tabelle geladen · {n_total} Zeilen "
        f"({banks_verified}/{banks_total} Banken Pillar-3-verifiziert, "
        f"{pct_verified:.0f} % der IRB-fähigen Kombinationen direkt) · "
        f"verified={n_verified}, country_proxy={n_proxy}, "
        f"basel_default={n_default}, standardised_na={n_na}"
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
        "n_irb_eligible":  n_total
                            - by_status.get("standardised_not_applicable", 0),
        "pct_verified":    100.0 * by_status.get("pillar3_verified", 0)
                            / (n_total
                               - by_status.get("standardised_not_applicable", 0))
                            if (n_total
                                - by_status.get("standardised_not_applicable", 0))
                            else 0.0,
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


def _norm_name(s: str) -> str:
    """Normalisiert einen Banknamen: Akzente weg, lowercase, Interpunktion
    zu Leerzeichen, Mehrfach-Spaces kollabiert. KEINE aggressive Wort-
    Entfernung (ein früherer Versuch reduzierte "ING Groep" auf "ing" und
    matchte dann InvesterING/HoldING/… → 16 statt 10 Banken)."""
    a = _strip_accents(s).lower()
    for ch in (",", ".", "-", "/", "'", '"'):
        a = a.replace(ch, " ")
    return " ".join(a.split())


def _name_match(curated_name: str, universe_name: str) -> bool:
    """Akzent-/interpunktions-insensitiver Substring-Match auf vollständigen
    Namen (keine Mini-Cores). Töchter, die denselben Substring teilen
    (z. B. "BNP Paribas Fortis"), werden NICHT hier ausgeschlossen — das
    erledigt die Größten-EAD-Auswahl in filter_universe_to_top10."""
    a = _norm_name(curated_name)
    b = _norm_name(universe_name)
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
    # Genau die kuratierten (Name, LEI) — Reihenfolge der CSV, dedupliziert.
    curated = list(dict.fromkeys(zip(df["bank_name"], df["LEI"])))
    items = list(universe.banks.items())   # (universe_name, portfolio)

    # INVERTIERT: pro kuratierter Bank GENAU EINE Universe-Bank wählen.
    # So kann das Universe niemals mehr als 10 Banken zurückgeben, auch
    # wenn es (top_n=None) alle ~67 IRB-Banken inkl. Töchtern enthält.
    filtered = {}
    universe_to_lei = {}
    used = set()
    unmatched = []
    for cname, clei in curated:
        chosen = None
        # 1) Exakter LEI-Match (robust, schreibweise-/akzent-immun)
        for bn, pf in items:
            if bn in used:
                continue
            if (getattr(pf, "lei", "") or "") == clei:
                chosen = (bn, pf)
                break
        # 2) Sonst: Namens-Match; bei mehreren Treffern die größte EAD
        #    (= Konzern-Mutter, nicht Tochter wie "Santander Consumer").
        if chosen is None:
            cands = [(bn, pf) for bn, pf in items
                     if bn not in used and _name_match(cname, bn)]
            if cands:
                chosen = max(cands, key=lambda t: t[1].total_ead)
        if chosen is None:
            unmatched.append(cname)
            continue
        bn, pf = chosen
        used.add(bn)
        filtered[bn] = pf
        universe_to_lei[bn] = clei

    # Guard: stilles Verschlucken einer kuratierten Bank verhindern.
    if unmatched:
        warnings.warn(
            f"filter_universe_to_top10: {len(unmatched)} kuratierte "
            f"Bank(en) konnten NICHT im EBA-Universe gematcht werden und "
            f"fehlen daher im Modell: {sorted(unmatched)}. Erwartet wurden "
            f"alle {len(curated)} Banken aus pillar3_bank_pd_lgd.csv.",
            stacklevel=2,
        )

    pd_lookup = df.set_index(["LEI", "vasicek_class"])[["pd_pct", "lgd_pct",
                                                          "status"]]
    n_overrides, n_verified, n_excluded = 0, 0, 0
    for bank_name, portfolio in filtered.items():
        lei = universe_to_lei[bank_name]
        retained_segments = []
        for seg in portfolio.segments:
            v_class = seg.exposure_class
            key = (lei, v_class)
            if key in pd_lookup.index:
                status = str(pd_lookup.loc[key, "status"])
                if status == "standardised_not_applicable":
                    n_excluded += 1
                    continue
                seg.pd  = float(pd_lookup.loc[key, "pd_pct"])  / 100.0
                seg.lgd = float(pd_lookup.loc[key, "lgd_pct"]) / 100.0
                n_overrides += 1
                if status == "pillar3_verified":
                    n_verified += 1
            retained_segments.append(seg)
        portfolio.segments = retained_segments

    universe.banks = filtered
    universe.source = (f"{universe.source} | PDs/LGDs aus "
                       f"pillar3_bank_pd_lgd.csv ({len(filtered)} Banken, "
                       f"{n_overrides} Segmente, davon {n_verified} "
                       f"Pillar-3-verifiziert; {n_excluded} SA-Segment "
                       f"aus IRB-Kanal ausgeschlossen)")
    return universe


def get_pd_for_bank(lei: str, vasicek_class: str,
                    vintage: str = "latest") -> dict:
    """Liefert PD% und LGD% (plus Source) für (LEI, Klasse[, Stichtag]).

    Returns dict mit Keys: pd_pct, lgd_pct, pd_decimal, lgd_decimal,
                          status, source, source_period, source_url
    Raises KeyError wenn Kombination (für den Stichtag) nicht gefunden.
    """
    df = load_pd_table(vintage=vintage)
    row = df[(df["LEI"] == lei) & (df["vasicek_class"] == vasicek_class)]
    if len(row) == 0:
        raise KeyError(
            f"Keine PD/LGD-Daten für LEI={lei}, Klasse={vasicek_class}, "
            f"vintage={vintage}. Bank evtl. nicht in der Top-10-Universe "
            f"oder dieser Stichtag noch nicht extrahiert."
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
    df = load_pd_table()   # Default "latest" = Live-Snapshot 31.12.2024
    assert len(df) == 70, f"Erwartet 70 Zeilen (latest), gefunden {len(df)}"
    assert df["bank_name"].nunique() == 10, "Erwartet 10 Banken (latest)"
    assert set(df["vasicek_class"].unique()) >= {
        "corporate", "sme_corporate", "mortgage", "qrre",
        "other_retail", "bank", "sovereign",
    }
    assert "status" in df.columns, "Spalte 'status' fehlt"
    assert "vintage_date" in df.columns, "Spalte 'vintage_date' fehlt"


def _test_vintage_panel_integrity():
    """Multi-Vintage-Panel-Integrität (löst die alte Single-Vintage-Annahme ab).

    Das Live-Modell startet weiterhin von einem definierten Snapshot
    (jüngster Stichtag = 31.12.2024); load_pd_table() liefert per Default
    GENAU diesen Snapshot. Zusätzlich darf die CSV nun ältere Pillar-3-
    Stichtage für die Walk-Forward-Validierung enthalten (eine Zeile je
    LEI×Klasse×Stichtag). Geprüft wird:
      1. jüngster Stichtag == 2024-12-31 und in sich konsistent,
      2. jede (LEI, vasicek_class, vintage_date)-Kombination eindeutig,
      3. der Live-Snapshot deckt 10 Banken ab.
    """
    full = load_pd_panel()
    vintages = sorted(full["vintage_date"].unique())
    latest = max(vintages)
    assert latest == "2024-12-31", (
        f"Jüngster Stichtag muss 2024-12-31 sein (Live-Snapshot), "
        f"gefunden: {latest}. Vorhandene Stichtage: {vintages}."
    )
    dup = full.duplicated(subset=["LEI", "vasicek_class", "vintage_date"])
    assert not dup.any(), (
        f"Doppelte (LEI, Klasse, Stichtag)-Kombinationen: "
        f"{full[dup][['LEI','vasicek_class','vintage_date']].values.tolist()}"
    )
    snap = load_pd_table(vintage="latest")
    assert snap["bank_name"].nunique() == 10, "Live-Snapshot ≠ 10 Banken"
    assert (snap["vintage_date"] == "2024-12-31").all(), \
        "Live-Snapshot enthält fremde Stichtage"


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
    assert len(cov["verified_banks"]) == 10
    assert cov["by_status"].get("pillar3_verified", 0) == 69
    assert cov["by_status"].get("country_proxy_pending_pillar3", 0) == 0
    assert cov["by_status"].get("basel_default_pending_pillar3", 0) == 0
    assert cov["by_status"].get("standardised_not_applicable", 0) == 1
    assert cov["n_irb_eligible"] == 69
    assert abs(cov["pct_verified"] - 100.0) < 1e-9


def _test_recovered_pillar3_values():
    bpce_mortgage = get_pd_for_bank("FR9695005MSX1OYEMGDF", "mortgage")
    assert bpce_mortgage["status"] == "pillar3_verified"
    assert abs(bpce_mortgage["pd_pct"] - 14.68) < 0.01
    assert abs(bpce_mortgage["lgd_pct"] - 10.70) < 0.01

    bpce_qrre = get_pd_for_bank("FR9695005MSX1OYEMGDF", "qrre")
    assert abs(bpce_qrre["pd_pct"] - 9.63) < 0.01
    assert abs(bpce_qrre["lgd_pct"] - 33.85) < 0.01

    cm_qrre = get_pd_for_bank("9695000CG7B84NLR5984", "qrre")
    assert abs(cm_qrre["pd_pct"] - 3.13) < 0.01
    assert abs(cm_qrre["lgd_pct"] - 33.00) < 0.01

    cm_bank = get_pd_for_bank("9695000CG7B84NLR5984", "bank")
    assert abs(cm_bank["pd_pct"] - 0.12) < 0.01
    assert abs(cm_bank["lgd_pct"] - 34.00) < 0.01


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
        ("Vintage-Panel-Integrität",   _test_vintage_panel_integrity),
        ("Pillar-3-verified-Lookup",   _test_lookup_pillar3_verified),
        ("Coverage-Report",            _test_coverage),
        ("Recovered Pillar-3 values",  _test_recovered_pillar3_values),
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
