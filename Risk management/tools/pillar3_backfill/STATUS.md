# Pillar-3 PD/LGD Backfill — Tooling & Status

Zweck: eine **konsistente, rohe EU-CR6-AIRB-PD/LGD/EAD-Zeitreihe** (alle Banken,
alle Jahre dieselbe Definition) für den Walk-Forward-Backtest aufbauen — extrahiert
aus den offiziellen Pillar-3-Reports der Banken. Quelle ausschließlich Pillar-3
(MODEL_ASSUMPTIONS A-02c). Das **Live-Modell** behält seine kuratierte
31.12.2024-Baseline (`data/pillar3_bank_pd_lgd.csv`) unverändert.

## Pipeline
1. **PDFs** liegen in `data/pillar3_reports/<bank>_<jahr>.pdf` (git-ignoriert, ~210 MB).
   Download via `dl_batch.py` (URLs) bzw. WebFetch.
2. **Per-Bank-Extraktoren** lesen die EU-CR6-AIRB-Subtotals (Spalten i. d. R.
   idx3=EAD, idx4=PD, idx6=LGD) und schreiben `data/backtest_raw/<bank>_*_rows.csv`.
   Verifikation: Spalten gegen FY2024 kalibriert/value-gematcht + Dichte-Check
   (RWA/EAD, Bruch ODER Prozent) + bankinterne EAD-Kontinuität über die Jahre.
   **Keine ungeprüften Werte** — was nicht verifiziert, wird nicht geschrieben.
3. **`consolidate_backtest.py`** fasst die verlässlichen Zeilen zu
   `data/pillar3_backtest_pdlgd.csv` zusammen (die Backtest-Roh-Reihe).

## Per-Bank-Layout-Eigenheiten (gelöst)
- **Deutsche Bank**: Multi-Klassen-Tabelle, Dec-31- vs. Jun-30-Block, Dichte %.
- **ING**: eine Klasse pro Seite, Label-Drift „sub-total" (2021) vs.
  „Subtotal (exposure class)" (2024), Dichte als **Bruch**; Klassen-Mapping
  qrre=SRE-SME-Proxy, other_retail=Retail-Other-SME.
- **UniCredit**: fließende Multi-Klassen-Tabelle, „%"-Spalten, Header-Bleed
  (Sovereign-/Mortgage-Zeile); sovereign = relabelte Central-gov-Zeile.
- **SocGen**: value-matchbar (CSV=roh) außer corporate; Dichte als Bruch.
- **BPCE**: Label **in** der Subtotal-Zeile, teils zeilen-umbrochen
  („… sub-" / „total …"), „-"-Platzhalter → positionserhaltend parsen.

## Stand (Stand dieser Sitzung)
Authoritative Reihe: `data/pillar3_backtest_pdlgd.csv`.

| Bank | Klassen | Jahre | Status |
|---|---|---|---|
| Deutsche Bank | 7/7 | 2021–2024 | ✅ komplett |
| ING Groep | 6/7 | 2021, 2024 | 🟡 2022 (Doku-Layout), 2023 (Report fehlte) offen |
| Société Générale | 6/7 (ohne corporate) | 2022–2024 | 🟡 corporate (CSV≠roh), bank-Vorjahre |
| UniCredit | 6/7 (ohne mortgage) | 2021–2024 | 🟡 echte Retail-Mortgage (Header-Bleed) |
| Groupe BPCE | 7/7 | 2024 | 🟡 2022/2023 (Multi-Block) offen |
| Crédit Agricole | – | – | ⏳ Download + Extraktion offen |
| BNP Paribas | – | – | ⏳ („Table 39 IRBA by PD scale"-Layout) |
| Crédit Mutuel | – | – | ⏳ (FR-Format) |
| Rabobank | – | – | ⏳ (PDF text-extraction problematisch) |
| Banco Santander | – | – | ⏳ (Multi-Geografie, forensisch) |

## Wichtiger Befund
Für etliche Banken (UniCredit, SocGen-corporate, …) sind die **kuratierten
FY2024-CSV-Werte nicht identisch mit den rohen EU-CR6-Subtotals** (offenbar beim
CSV-Aufbau abgeleitet/proxied). Deshalb diese separate **rohe** Reihe für den
Backtest (konsistent über die Zeit), während das Live-Modell die kuratierten
Werte behält.
