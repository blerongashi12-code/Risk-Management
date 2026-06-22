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
- **BNP Paribas**: „IRBA EXPOSURE BY PD SCALE" (Tab. 38/39/41/42), Spalten
  idx3=EAD, 4=PD, 5=LGD, 7=RWA, 8=Dichte. Jedes Doc liefert Stichjahr **plus
  Vorjahres-Komparativ** (2024-Doc → 2024+2023, 2023-Doc → 2023+2022). Gemischte
  Zahlformate je Seite: Anglo (`375,110` / `2.87%`) vs. EU-Retail-Vorjahr
  (`185 085` / `1,76 %`) — Lookahead `\d{3}(?!\s*%)` verhindert, dass die
  CCF-100%-Spalte als Tausender-Gruppe verschluckt wird. **2021**: Primärseiten
  runden PD auf ganze % (`0%`/`5%`) → ausgeschlossen (Präzisions-Inkonsistenz);
  **2020**-Komparativ (p401/p407 im 2021-Doc) hat zwar Dezimalen, aber abweichendes
  12-Spalten-Layout + fehlende SME → bei Bedarf nachziehbar, derzeit nicht
  Teil der Reihe.
- **Banco Santander**: „Table NN.CR6 - AIRB approach" (+ „… Retail"), eine Klasse
  pro Seite, jede Seite mit **Jahres-Marker** (`2024`/`2023`) + Klassen-Label;
  Abschluss „Subtotal (exposure class)". Spalten idx3=EAD, 4=PD, **6=LGD** (eine
  Schuldnerzahl-Spalte zwischen PD und LGD!), 8=RWA, 9=Dichte; `—` als
  Maturity-Platzhalter positionserhaltend. Jedes Doc liefert Stichjahr +
  Vorjahr; Primär-Disclosure bevorzugt, Überlappungen cross-validiert. Kein
  sovereign (Standardisierter Ansatz). FIRB-Tabellen (Table 28.CR6 - FIRB)
  ausgeschlossen. **Kuratierte Retail-Werte ≠ Roh-Subtotals** (abweichende
  Schuldnerzahlen → andere Quelle); Roh-Reihe nutzt die Report-Subtotals.

## Stand (Stand dieser Sitzung)
Authoritative Reihe: `data/pillar3_backtest_pdlgd.csv`.

| Bank | Klassen | Jahre | Status |
|---|---|---|---|
| Deutsche Bank | 7/7 | 2021–2024 | ✅ komplett |
| ING Groep | 6/7 | 2021–2024 | ✅ alle 4 Jahre. **„qrre" → `mortgage_sme` umbenannt** (Verifikation: ING hat KEINEN QRRE-A-IRB-Block; die Zeilen sind „Retail – Secured by immovable property SME"). Retail-Other-non-SME nicht separat erfasst (Lücke) |
| Société Générale | 6/7 (ohne corporate) | 2021–2024 | 🟢 2021 ergänzt; corporate (CSV≠roh) fehlt durchgängig, bank nur 2024, sovereign-2021-PD=1,00 (verifiziert, Level-Shift) |
| UniCredit | 6/7 (ohne mortgage) | 2021–2024 | ✅ alle 4 Jahre komplett. 2021 sovereign + other_retail via Verifikation aus Quelle ergänzt (p134/p138, dichte-verifiziert). „sovereign" = Central-gov-Zeile (relabel) |
| Groupe BPCE | 7/7 | 2024 | 🔴 2022/2023 Multi-Block (3 Sub-Entity-Blöcke: Gruppe/BP/CE bzw. A-/F-IRB; Klassen-Labels zeilenumbrochen) → Vorjahres-Disambiguierung unzuverlässig (sme/qrre/sovereign mehrdeutig), nicht geschrieben |
| Crédit Mutuel | 5/7 (2024) · 4/7 (2023) | 2023, 2024 | 🔴 sovereign ENTFERNT (Verifikation: CM hat keinen IRB-sovereign; „sovereign 2024" war byte-identisches Duplikat von bank → Phantom). 2021/2022-PDFs geladen, aber degenerierte Anker + Layout-Drift → Vorjahres-Mapping unzuverlässig, nicht geschrieben |
| BNP Paribas | 7/7 | 2022–2024 | ✅ 3 Jahre, anchor-/dichte-/EAD-verifiziert; 2023 doppelt validiert. **`sme_corporate` korrigiert** (war fälschlich „Corporates – Specialised financing", jetzt echtes „SME corporates", Verifikation an Quell-PDF-Labels p441 bestätigt) |
| Crédit Agricole | – | – | 🔴 doku-blockiert: nur Halbjahres-`pdfPreview`-Viewer; Jahresend-IDs nicht auffindbar |
| Banco Santander | 6/6 IRB | 2021–2024 | ✅ 4 Jahre komplett (keine sovereign = Standardised); label+jahr-getrieben, Dichte-verifiziert. bank/corporate-AIRB-EAD schrumpft 2022→2023 (realer IRB-Rollback zu FIRB, verifiziert) |
| Rabobank | 6/6 (2021·2022) · 4 (2024) | 2021, 2022, 2024 | ✅ entsperrt: dedizierter text-basierter „Pillar 3 Year Report" (nicht der bild-basierte Jahresbericht). Layout-agnostischer Parser (Dichte-validiert: Row#-Praefix ja/nein × Maturity ja/nein). 2021+2022 6/6 dichte-verifiziert; 2024 mortgage+sme stabil, sovereign/corporate = A-IRB-Residuum nach IRB-Rollback, bank/other_retail-2024 verworfen. 2020 (Frakt.-Format) + 2023 (Hash nicht auffindbar) offen. In Transparency-Panel ✓ |

## Adversariale Verifikation (8-Banken-Workflow, je 1 Prüf-Agent gegen Quell-PDFs)
Ergebnis: **die Zahl-Extraktion ist durchgängig korrekt** (jeder stichprobenartig
gegen die Quell-„Sub-total"-Zeile geprüfte PD/LGD/EAD-Wert stimmt). Gefundene und
**behobene** Defekte waren ausschließlich Klassen-Zuordnungen:
- **BNP `sme_corporate`**: war „Corporates – Specialised financing" → auf echtes
  „SME corporates" korrigiert (occ #1 statt #0 auf der Corporate-Seite).
- **ING `qrre`**: ING hat keinen QRRE-A-IRB-Block → in `mortgage_sme` umbenannt
  (echtes „Secured by immovable property SME"), damit die echte QRRE-Klasse
  (DB/Santander/BNP/UniCredit) nicht kontaminiert wird.
- **CM `sovereign` 2024**: Phantom (Duplikat der bank-Zeile) → entfernt.
- **Lücken-Fill aus Verifikation**: UniCredit 2021 sovereign + other_retail
  (aus Quelle gelesen, dichte-verifiziert) ergänzt.
Als **real disclosed** (kein Fehler) bestätigt: DB-Institutions-PD-Spitzen
(100%-Default-Band), DB-sovereign-LGD-Rekalibrierung, Santander-IRB-Rollback,
BPCE-Retail-PDs (Default-Band). Voller Report: `VERIFICATION_REPORT.json`.
Offene Lücken (dokumentiert, nicht fabriziert): SocGen large-corporate (alle Jahre)
+ bank-Vorjahre, ING-Retail-Other-non-SME, CM-bank-2023-F-IRB.

## Wichtiger Befund
Für etliche Banken (UniCredit, SocGen-corporate, …) sind die **kuratierten
FY2024-CSV-Werte nicht identisch mit den rohen EU-CR6-Subtotals** (offenbar beim
CSV-Aufbau abgeleitet/proxied). Deshalb diese separate **rohe** Reihe für den
Backtest (konsistent über die Zeit), während das Live-Modell die kuratierten
Werte behält.
