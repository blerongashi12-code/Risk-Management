# Data Layer · DAX Credit Stress Model

Zieht alle Rohdaten für das Credit-Stress-Modell. Modular, mit Cache, einzeln oder gemeinsam ausführbar.

## Ordnerstruktur

```
C:\Users\blero\Downloads\Risk management\
│
├── config.py                         ← zentrale Konfiguration (Pfade, Ticker, Konstanten)
├── 00_build_data_layer.py            ← Master-Orchestrator (ruft alle Module auf)
├── 01_fetch_dax40_prices.py          ← DAX-40 Aktienkurse (yfinance)
├── 02_fetch_dax40_fundamentals.py    ← DAX-40 Bilanzdaten (yfinance)
├── 03_fetch_brent_crude.py           ← Brent Crude (Energy-Faktor)
├── 04_fetch_bundesbank_svensson.py   ← Svensson-Parameter (Bundesbank-CSV)
├── 05_fetch_market_proxy.py          ← DAX-Index (Markt-Faktor)
├── DataLayer_Runner.ipynb            ← Jupyter Notebook zum Ausführen
│
├── data/                             ← Inputs (vom User)
│   └── bundesbank_svensson.csv       ← Bundesbank-Download (manuell)
│
└── data/cache/                       ← Parquet-Cache (automatisch)
    ├── dax40_prices.parquet
    ├── dax40_fundamentals.parquet
    ├── brent_crude.parquet
    ├── svensson_params.parquet
    ├── market_proxy.parquet
    └── data_layer_state.pkl
```

## Einmaliges Setup

```bash
pip install yfinance pandas pyarrow scipy openpyxl matplotlib jupyter
```

## Svensson-Daten besorgen

Die Bundesbank-Parameter musst du einmalig manuell herunterladen:

1. Gehe zu https://www.bundesbank.de/dynamic/action/de/statistiken/zeitreihen-datenbanken/zeitreihen-datenbank/759778/759778
2. Wähle "Parameter der Zinsstrukturkurve (Svensson-Methode) — Börsennotierte Bundeswertpapiere, Tageswerte"
3. Setze Zeitraum auf z.B. 2020-01-01 bis heute
4. Exportiere als CSV
5. Speichere unter `data/bundesbank_svensson.csv`

(Du hast bereits `bbk_paket1__8_.csv` — benenne es einfach um.)

## Nutzung

### Variante A — Jupyter Notebook (empfohlen)

Öffne `DataLayer_Runner.ipynb` und führe die Zellen aus. Du siehst:
- Plots der Kurse, Volatilitäten, Zinskurven-Parameter
- Quality-Checks für jede Datenquelle
- Finales `data_layer` Dictionary

### Variante B — Einzeln per Kommandozeile

Jedes Modul kann standalone laufen:

```bash
python config.py                        # zeigt Konfiguration
python 01_fetch_dax40_prices.py         # nur Kurse
python 02_fetch_dax40_fundamentals.py   # nur Bilanzen
python 03_fetch_brent_crude.py          # nur Brent
python 04_fetch_bundesbank_svensson.py  # nur Svensson
python 05_fetch_market_proxy.py         # nur Markt-Index
```

### Variante C — Master ausführen

```bash
python 00_build_data_layer.py
```

→ Baut alles auf einmal, Ergebnis im Dictionary `data_layer`.

### Variante D — Als Import

```python
from build_data_layer import build_all
data_layer = build_all()

dax_prices       = data_layer['dax_prices']
dax_fundamentals = data_layer['dax_fundamentals']
brent            = data_layer['brent']
svensson         = data_layer['svensson']
svensson_stats   = data_layer['svensson_stats']
market           = data_layer['market']
```

## Cache-Verhalten

- **Erster Lauf:** ca. 30–60 Sek (yfinance Downloads)
- **Folgende Läufe:** < 1 Sek (aus Parquet)
- **Cache-Ablauf:** 12h für Marktdaten, 24h für Bilanzen
- **Force-Refresh:** `use_cache=False` als Parameter übergeben

## Ausgabe-Preview

```
======================================================================
BUILD COMPLETE
======================================================================
  DAX-40 Prices:       (1524, 38)
  DAX-40 Fundamentals: (38, 11)
  Brent:               (1568, 3)
  Svensson:            (1606, 6)
  Market:              (1524, 3)
  Timestamp:           2026-04-23 14:32:17
======================================================================
```

## Nächste Schritte

Nach erfolgreichem Data-Layer-Build baut darauf auf:

- **`svensson.py`** — Zero-Rate & Diskontierungs-Engine
- **`merton.py`** — KMV-Solver für Asset Value & Vola
- **`factor_model.py`** — 2-Faktor Betas (Energie + Zins)
- **`monte_carlo.py`** — Stress-Engine
- **`excel_writer.py`** — schreibt Ergebnisse ins Dashboard

## Bekannte Einschränkungen

- **yfinance Bilanzdaten** sind lückenhaft für manche DAX-Mitglieder — Total Debt fehlt gelegentlich. Fallback: Komponenten-Rekonstruktion aus kurz- + langfristiger Verschuldung.
- **Neulistings:** P911.DE (Porsche AG) und ENR.DE (Siemens Energy) haben kürzere Historien als 6 Jahre — wird automatisch behandelt.
- **Währungen:** Qiagen bilanziert in USD. Wird in `dax_fundamentals['Currency']` dokumentiert.
