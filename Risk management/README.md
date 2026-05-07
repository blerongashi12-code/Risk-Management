# EU Banking Credit Stress Cockpit

Regulatorische Credit-Risk-Sicht auf das europäische Bankensystem, gebaut auf der EBA EU-wide Transparency Exercise 2025 + Vasicek/ASRF (Basel-III IRB). Drei Stress-Channels, einheitliche CET1-Wirkungskette:

1. **Loan-Book-Channel** — Vasicek-conditional-PD + Downturn-LGD → Δ-EL / Δ-RWA
2. **Sovereign-/Bonds-Channel** — IFRS-9-Accounting-Class-Split (HfT/FVTPL/FVOCI/AC), Modified-Duration-MtM → ΔFV → CET1
3. **Trading-Book-Channel** — FRTB-style Market-RWA-Multiplier + TB-P&L-Haircut

Bundesbank-Svensson-Zero-Curve treibt den Δr-Schock; ICE Brent treibt den Energy-Faktor.

## Repository-Struktur

```
Risk management/
├── config.py                    ← Pfade + Vasicek/EBA-Konfig
├── 03_fetch_brent_crude.py      ← Brent (yfinance)
├── 04_fetch_bundesbank_svensson.py  ← Svensson-Params (Bundesbank)
│
├── backend/
│   ├── svensson.py              ← Zero-Curve-Engine
│   ├── vasicek.py               ← Vasicek/ASRF + Basel-III IRB
│   ├── eba_loader.py            ← EBA-CSV-Parsing + Sovereign + Capital
│   ├── macro_factor.py          ← (Brent, Δr) → Vasicek-M
│   └── backtesting.py           ← historisches Panel + Forecast vs Realized
│
├── streamlit_app/
│   ├── app.py                   ← Landing-Page
│   ├── components/              ← theme · sidebar · methodology · data_loader
│   └── pages/
│       ├── 1_Credit_Risk.py     ← Vasicek IRB + NPL + CET1-Strip
│       ├── 2_Bonds.py           ← 3 Sub-Tabs: Sovereigns · Banking-Book · TB+ABS
│       ├── 3_Capital_Adequacy.py ← 3-Channel CET1-Ratio
│       ├── 4_Yield_Curve.py     ← Bundesbank-Svensson + β-Shifts
│       ├── 5_Backtesting.py     ← 22 Quartals-Stichtage 2019-2025
│       ├── 6_Annahmen.py        ← Governance-Doku (3 Layer)
│       └── 7_Methodology.py     ← Vollständige MODEL_ASSUMPTIONS.md
│
├── data/
│   ├── bundesbank_svensson.csv  ← Bundesbank-Download (manuell)
│   ├── tr_cre.csv  (~123 MB, EBA, gitignored)
│   ├── tr_sov.csv  (~91 MB, EBA, gitignored)
│   ├── tr_oth.csv  (~14 MB, EBA, gitignored)
│   ├── tr_mrk.csv  (~3.6 MB, EBA, gitignored)
│   ├── TR_Metadata.xlsx, SDD.xlsx  ← EBA-Dictionaries
│   ├── cache/
│   │   ├── brent_crude.parquet
│   │   └── svensson_params.parquet
│   └── transparency_2020/ … 2024/  ← historische Vintages für Backtesting
│
├── MODEL_ASSUMPTIONS.md         ← Single Source of Truth
└── README.md                    ← (dieses File)
```

## Setup

```bash
pip install streamlit pandas pyarrow numpy scipy openpyxl yfinance plotly
```

## Daten beschaffen

### Bundesbank Svensson (einmalig)

```text
URL:  https://www.bundesbank.de/dynamic/action/de/statistiken/zeitreihen-datenbanken/zeitreihen-datenbank/759778/759778
```

Als CSV exportieren → `data/bundesbank_svensson.csv` → `python 04_fetch_bundesbank_svensson.py`

### EBA Transparency 2025

Download von [eba.europa.eu/risk-and-data-analysis/eu-wide-transparency-exercise](https://www.eba.europa.eu/risk-and-data-analysis/eu-wide-transparency-exercise):
- `tr_cre.csv`, `tr_sov.csv`, `tr_oth.csv`, `tr_mrk.csv`
- `TR_Metadata.xlsx`, `SDD.xlsx`

Nach `data/`. Der Loader (`backend/eba_loader.py`) findet sie automatisch via `config.EBA_RAW_DIR`.

### Historische Vintages (für Backtesting-Page)

Optional. Die Vintages 2020–2024 nach `data/transparency_2020/` … `data/transparency_2024/` ablegen — gleicher 6-File-Satz. URLs in der Backtesting-Page-Doku.

## Cockpit starten

```bash
cd "Risk management"
streamlit run streamlit_app/app.py
```

→ Browser öffnet `http://localhost:8501`. Linke Sidebar: Live-Macro-Slider (ΔBrent + Svensson-β → Δr_10y). Alle Pages reagieren in Echtzeit.

### Falls Pages nicht öffnen oder fehlerhaft sind

99% der Fälle: **stale `__pycache__/`-Files** aus einer früheren Version (insb. nach Page-Renamings). Einmalig aufräumen:

```bash
# vom Risk-management-Ordner aus
find . -type d -name "__pycache__" -exec rm -rf {} +
```

Dann `streamlit run streamlit_app/app.py` neu starten. Python regeneriert die Caches korrekt für die aktuellen Pages.

Alternativ einmal `python run_clean.py` (bei Fix beigelegt — ruft den obigen Befehl + Streamlit-Start zusammen auf).

## Backend Self-Tests

```bash
python backend/svensson.py        # Svensson Excel-Cross-Check + 6 Tests
python backend/vasicek.py         # 9 ASRF/IRB Tests (BCBS-Referenz, Bridge, Downturn-LGD)
python backend/eba_loader.py      # Synthetic + Real-Loader-Smoke-Tests
python backend/macro_factor.py    # 6 Anchor + Data-Route Tests
python backend/backtesting.py     # 3 Forecast-vs-Realized Tests
```

Expected output: `[PASS] All tests passed.`

## Bekannte Limitationen

Vollständige Liste in `MODEL_ASSUMPTIONS.md`. Highlights:

- **Implied PD** = beobachteter Default-Ratio (backward-looking) — Stress-Sensitivität korrekt, absolutes Niveau konservativ.
- **F-IRB-LGD** statt A-IRB-Modell (CRR Art. 181, EBA Stress Test 2023).
- **Sovereign-Book**: Parallel-Shift only, kein Credit-Spread, kein Hedging.
- **Banking-Book Bonds (Financials/Corporates/Covered)**: in EBA-Disclosure mit Loans aggregiert — keine isolierte Bond-Position rekonstruierbar.
- **Trading-Book**: nur Market-RWA-Aggregat, keine Issuer-Granularität.
- **EBA Stress Test 2025 File**: passwortgeschützt — Adverse-Anker hardcoded aus Methodology Note.

## Wichtige Quellen

- Vasicek, O. (2002). *Loan Portfolio Value*. Risk Magazine.
- BCBS (2017). *Basel III: Finalising post-crisis reforms*.
- EBA (2025). *EU-wide Transparency Exercise 2025 — Public Disclosure*.
- EBA GL 14 (ICAAP / Stress-Testing).
- SR 11-7 (Federal Reserve / OCC Supervisory Guidance on Model Risk Management).
