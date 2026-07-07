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

## Stand (nach Lückenschluss-Sweep Juli 2026)
Authoritative Reihe: `data/pillar3_backtest_pdlgd.csv` — **255
quellenbelegte Zeilen, 99,6 % Abdeckung** (255 von 256 gemeldeten
Bank×Jahr×Klasse-Zellen). Davon sind **253 Zeilen modellfähig**
(`include_in_backtest = 1`); zwei quellenbelegte Sonderfälle bleiben für
Audit/Coverage sichtbar, laufen aber wegen Quellen-/Perimeterproblemen nicht
in die Backtest-Rechnung (`include_in_backtest = 0`).
Der Sweep hat 27 zuvor offene Zellen geschlossen; Methoden: (a)
**Vorjahres-Vergleichsspalten** der Folgejahres-Berichte (BNP-URD 2022 →
31.12.2021 mit Dezimal-PDs; bpce_2023 → 2022-Doppelbestätigung; socgen_2023
→ 2022-Komparativ; ca_sa_2023h1 → 31.12.2022), (b) **Block-Anker-Matching**
(Tabellenblock über bereits verifizierte Nachbarwerte identifizieren, fehlende
Klasse aus demselben Block lesen — löst umbrochene BPCE-Subtotal-Labels).
Jede Zelle wort-wörtlich + Seitenbeleg + Dichte-Check + EAD-Kontinuität.

| Bank | Klassen | Jahre | Status |
|---|---|---|---|
| Deutsche Bank | 7/7 | 2021–2024 | ✅ komplett |
| ING Groep | 6/7 | 2021–2024 | ✅ alle 4 Jahre. **„qrre" → `mortgage_sme` umbenannt** (Verifikation: ING hat KEINEN QRRE-A-IRB-Block; die Zeilen sind „Retail – Secured by immovable property SME"). Retail-Other-non-SME nicht separat erfasst (Lücke) |
| Société Générale | 7/7 | 2021–2024 | ✅ bank 2022 = 38.844/1,24/25,01 im Sweep ergänzt (socgen_2022 S.145, doppelt belegt via socgen_2023 S.147). sovereign 2022 ist jetzt als Quellenzeile erfasst (EAD 271.679), aber wegen wörtlicher „0"-Platzhalter für PD/LGD mit `include_in_backtest=0` aus der Modellrechnung ausgeschlossen. Keine Ableitung aus RWA/Nachbarjahren. |
| UniCredit | 6/7 (ohne mortgage) | 2021–2024 | ✅ alle 4 Jahre komplett. 2021 sovereign + other_retail via Verifikation aus Quelle ergänzt (p134/p138, dichte-verifiziert). „sovereign" = Central-gov-Zeile (relabel) |
| Groupe BPCE | 7/7 | 2021–2024 | ✅ **komplett** (Sweep: corporate/sme/qrre 2021–2023 + mortgage 2021 via Block-Anker-Matching; 2021 aus H1-2022-Update S.81–85 mit umbrochenen Labels, 2022 doppelt belegt via bpce_2023 S.157–161, 2023 aus bpce_2023 S.149–153). Corporate-PD 17,90 (2022) → 4,07 (2023) wie-gemeldet in beiden Publikationen (Note in CSV) |
| Crédit Mutuel | 5/5 (2022–2024) · 4/5 (2021) | 2021–2024 | 🟢 Sweep: bank 2021/2022/2023 (F-IRB-„Etablissements"-Zeile = Serienkonvention, CM meldet Institute nur F-IRB; 2024-Anker 30.410 kontinuierlich), **sme 2021** (A-IRB-Entreprises-Zeile 134.578/4,17/29,00, S.52), sme 2022, other_retail 2021/2022 („Clientèle de détail"-Subtotal). **corporate 2021 = einzige verbleibende Quellgrenze**: keine zur 2022–2024-Konvention passende F-IRB-PD/LGD-Zeile veröffentlicht. Kein IRB-sovereign (Standardansatz). |
| BNP Paribas | 7/7 | **2021–2024** | ✅ **komplett** (Sweep: alle 7 Klassen 31.12.2021 aus den Vergleichsspalten der URD 2022, S.411–423, mit Dezimal-PDs — löst die Ganze-%-Rundung der 2021-Primärseiten). `sme_corporate` korrigiert (echtes „SME corporates", p441-verifiziert) |
| Crédit Agricole S.A. | 7/7 | 2021–2024 | ✅ **CA S.A.** (LEI F0HUI…, im Transparency-Panel; NICHT CA Group). Sweep: qrre 2022 aus ca_sa_2023h1 S.42 (31.12.2022-Komparativ). sme 2021 ist als Quellenzeile erfasst (3.074/26,43/33,29, ca_sa_2022h1 S.40), aber wegen echtem A-IRB-Perimeterbruch mit `include_in_backtest=0` aus der Modellrechnung ausgeschlossen. |
| Banco Santander | 6/6 IRB | 2021–2024 | ✅ 4 Jahre komplett (keine sovereign = Standardised); label+jahr-getrieben, Dichte-verifiziert. bank/corporate-AIRB-EAD schrumpft 2022→2023 (realer IRB-Rollback zu FIRB, verifiziert) |
| Rabobank | 6/6 | 2021–2024 | ✅ **komplett** (Sweep: bank 2024 = 55/2,23/10,65, A-IRB-Residuum nach Rollback 5.783→465→55, S.76 wörtlich + dichte-verifiziert; other_retail 2024 = 1.771/5,38/24,00, S.82 — Parse-Lücke geschlossen) |

**Verbleibende echte Quellgrenze:** Crédit Mutuel Corporate 2021. Der Report
veröffentlicht keine zur 2022–2024-Serienkonvention passende F-IRB-PD/LGD-Zeile;
es wird kein Wert aus RWA oder Nachbarjahren abgeleitet. Zwei weitere
Sonderfälle (SocGen Sovereign 2022, CA SME 2021) sind quellenbelegt in der CSV,
aber mit `include_in_backtest=0` aus der Modellrechnung ausgeschlossen.

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
Der Lückenschluss-Sweep (Juli 2026, Multi-Agenten-Workflow + unabhängiger
Stichproben-Audit von 8 Werten gegen die Quell-PDFs) hat die damals offenen
Lücken bis auf die eine dokumentierte Quellgrenze geschlossen;
ING-Retail-Other-non-SME bleibt strukturell nicht separat erfasst
(zählt nicht als Lücke im Meldeumfang).

## Wichtiger Befund
Für etliche Banken (UniCredit, SocGen-corporate, …) sind die **kuratierten
FY2024-CSV-Werte nicht identisch mit den rohen EU-CR6-Subtotals** (offenbar beim
CSV-Aufbau abgeleitet/proxied). Deshalb diese separate **rohe** Reihe für den
Backtest (konsistent über die Zeit), während das Live-Modell die kuratierten
Werte behält.
