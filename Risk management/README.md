# EU Banking Credit Stress Cockpit

Regulatory credit-risk view of the European banking system, built on the EBA EU-wide Transparency Exercise 2025 and a Vasicek/ASRF — Basel-III IRB engine. Live macroeconomic stress in two channels:

1. **Loan-book channel** — bank-portfolio expected loss, RWA and capital under macro shocks via Vasicek/ASRF.
2. **Sovereign-book channel** — modified-duration mark-to-market on the sovereign portfolio, driven by the same Δr_10y.

The Bundesbank Svensson zero-curve drives both the rate shock vector and the discounting framework.

## Repository layout

```
Risk management/
│
├── config.py                              ← paths + Vasicek/EBA config
├── 03_fetch_brent_crude.py                ← Brent crude (yfinance)
├── 04_fetch_bundesbank_svensson.py        ← Svensson params (Bundesbank CSV)
│
├── backend/
│   ├── svensson.py                        ← Svensson zero-curve engine
│   ├── vasicek.py                         ← Vasicek/ASRF + Basel-III IRB
│   ├── eba_loader.py                      ← EBA Transparency CSV parsing + sovereign
│   └── macro_factor.py                    ← (Brent, Δr_10y) → Vasicek M
│
├── streamlit_app/
│   ├── app.py                             ← landing page
│   ├── components/                        ← theme, sidebar, data loader
│   └── pages/
│       ├── 1_Bank_Portfolio.py            ← Vasicek IRB capital under stress
│       ├── 2_Sovereign_Risk.py            ← doom-loop heatmap + duration P&L
│       ├── 3_Yield_Curve.py               ← Bundesbank Svensson + factor history
│       └── 4_Methodology.py               ← rendered MODEL_ASSUMPTIONS.md
│
├── data/
│   ├── bundesbank_svensson.csv            ← Bundesbank download (manual)
│   ├── tr_cre.csv  (~123 MB, EBA, gitignored)
│   ├── tr_sov.csv  (~91 MB, EBA, gitignored)
│   ├── TR_Metadata.xlsx, SDD.xlsx         ← EBA dictionaries
│   └── cache/
│       ├── brent_crude.parquet
│       └── svensson_params.parquet
│
├── MODEL_ASSUMPTIONS.md                   ← single source of truth for assumptions
└── README.md
```

## Setup

```bash
pip install streamlit pandas pyarrow numpy scipy openpyxl yfinance plotly
```

## Bringing in Bundesbank Svensson parameters

Download once from the Bundesbank time-series database (Svensson method, daily values for listed federal securities):
- https://www.bundesbank.de/dynamic/action/de/statistiken/zeitreihen-datenbanken/zeitreihen-datenbank/759778/759778
- Export as CSV, save to `data/bundesbank_svensson.csv`
- Run `python 04_fetch_bundesbank_svensson.py` to build the parquet cache

## Bringing in EBA Transparency 2025 data

Download the public EBA EU-wide Transparency Exercise 2025 release (~210 MB total):
- https://www.eba.europa.eu/risk-and-data-analysis/eu-wide-transparency-exercise
- Save the four CSVs (`tr_cre.csv`, `tr_sov.csv`, `tr_oth.csv`, `tr_mrk.csv`) and the two Excel dictionaries (`TR_Metadata.xlsx`, `SDD.xlsx`) into `data/`.

The cockpit's loader (`backend/eba_loader.py`) auto-discovers them via `config.EBA_RAW_DIR`.

## Running

```bash
cd "Risk management"
streamlit run streamlit_app/app.py
```

Sidebar drives the live macro stress (ΔBrent, Svensson β-shifts → Δr_10y). All charts on every page recompute on change.

## Backend self-tests

Each backend module has a `__main__` validation block:

```bash
python backend/svensson.py        # Svensson Excel cross-check + 6 tests
python backend/vasicek.py         # 6 ASRF / IRB tests (incl BCBS reference)
python backend/eba_loader.py      # synthetic + real-loader smoke tests
python backend/macro_factor.py    # 6 anchor + data-route tests
```

Expected output: `[PASS] All tests passed.`

## Known limitations

See `MODEL_ASSUMPTIONS.md` §8 for the full list. Highlights:

- **Implied PD** is the observed default ratio (backward-looking) — stress sensitivity is correct, absolute level conservative.
- **F-IRB LGD assumptions** stand in for unpublished A-IRB internal models.
- **Sovereign book**: parallel-shift only, no credit-spread or hedging adjustments.
- **EBA stress-test 2025 file**: password-protected — anchor values hardcoded from the published Methodology Note.
