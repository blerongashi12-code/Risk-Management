# MODEL_ASSUMPTIONS · EU Banking Credit Stress Cockpit

> **Single Source of Truth** für alle Modell-Annahmen, Datenquellen,
> mathematischen Formeln und ihre ökonomische Begründung.
> Stand: Juli 2026 · Version 2.1 (2-Faktor-Modell + CET1-Walk-Forward-Backtest)
> Kuratierte Abgabefassung (Word, mit Schaubildern):
> `Abgabe-Files/Abgabedokumente/Modellannahmen.docx`

---

## 1. Modell-Übersicht

Das Cockpit quantifiziert die Wirkung makroökonomischer Schocks auf die
regulatorische Eigenkapitalquote (CET1) der 10 größten EU-IRB-Banken
über zwei separate Risiko-Channels.

### Zwei separate Macro-Faktoren

- **ΔBrent (log-Return)** — Ölpreis-Schock, repräsentiert
  Energie-/Inflations-/Kosten-Wirkung auf Schuldner-Cashflows
- **Δr_10y (Prozentpunkte)** — 10-Jahres-Zins-Schock, repräsentiert
  Refinanzierungs-/Affordability-/Discount-Wirkung

**Empirisch validiert (5-Jahres-Sample, 1242 Handelstage):**
- Pearson ρ(ΔBrent, Δr_10y) = +0.07 (95%-CI [+0.01, +0.12])
- OLS R² = 0.005 — keine erklärende Varianz
- **Schlussfolgerung:** Faktoren empirisch nahezu unabhängig — separate
  Modellierung ökonomisch und statistisch gerechtfertigt
- Quelle: `backend/factor_correlation.py` über Brent (ICE) +
  Bundesbank-Svensson

### Zwei Stress-Channels

| # | Channel | Mechanik |
|---|---------|----------|
| 1 | **Loan Book** | Sektor-differenzierte PD/LGD-Transmission via β-Sensitivitäten |
| 2 | **Sovereign Book** | Modified-Duration-MtM auf Δr_10y, CET1-wirksam nur der bank-individuell gemeldete HfT/FVTPL/FVOCI-Anteil (EBA-IFRS-9-Split) |

*(Der frühere dritte Trading-Book-Channel wurde entfernt — siehe A-07.)*

---

## 2. Modell-Annahmen · Konfidenz-Klassen

Jede Annahme wird transparent klassifiziert:

- **[published]** = direkt veröffentlichte Messung
- **[estimate]** = statistische Schätzung aus Literatur oder Eigen-Regression
- **[approximation]** = strukturelle Vereinfachung
- **[assumption]** = hardcoded Annahme

---

## 3. Annahmen-Inventar

### A-01 · PD-Quelle: bank-spezifische Pillar-3 EU-CR6 (Stichtag 31.12.2024) [published]

**Was:** Bank-spezifische, EAD-gewichtete A-IRB-Average-PDs pro IRB-
Exposure-Klasse, gezogen aus den EU-CR6-Tabellen im Pillar-3-Report
jeder einzelnen Bank. **Einheitlicher Stichtag 31.12.2024** für alle
10 Banken (vom Loader-Test `_test_vintage_consistency` erzwungen).

**Formel:**
```
PD_(bank,class) = Pillar-3-EU-CR6-SubTotal_{bank, class, 31.12.2024}
```

**Coverage:** 10 / 10 Banken und alle 69 IRB-fähigen
Bank-Klassen-Kombinationen stammen direkt aus bank-spezifischen
Pillar-3-EU-CR6-Tabellen. Pro
Bank-Klasse werden PD und LGD gemeinsam aus derselben Sub-total-Zeile
übernommen. Nur Santander Sovereign besitzt keinen IRB-CR6-Wert, weil
Santander diese Exposures dauerhaft im Standardansatz führt. Die Position
wird deshalb aus dem IRB-Kreditbuchkanal ausgeschlossen, bleibt aber im
separaten Sovereign-Marktwertkanal vollständig enthalten.

**Warum ökonomisch:** Diese PDs sind die bank-internen Schätzungen
unter CRR Art. 180 — forward-looking 1-Jahres-Ausfallwahrscheinlichkeiten,
kohorten-basiert berechnet durch die Banken selbst, von der EZB-Aufsicht
auditiert. Sie sind das regulatorische Maß für PD und gemäß CRR
Art. 431–455 + EBA ITS/2020/04 disclosure-pflichtig.

**Warum mathematisch:** Exposure-gewichtete Aggregation pro Asset-Klasse
gemäß COREP-Reporting-Standard C 9.02:
```
PD_class_aggregate = Σ(EAD_i · PD_i) / Σ(EAD_i)  über alle Schuldner i
```

**Quelle:** Jede Zeile der CSV `data/pillar3_bank_pd_lgd.csv` enthält
die Quellen-URL, Seitenzahl und Tabellen-Name der jeweiligen Bank-
Pillar-3-Disclosure (Deutsche Bank Pillar 3 Q4 2024 p.108-113, ING
Additional Pillar III 2024 p.44-51, etc.).

**Adressiert Professor-Kritik:** Punkt 1 (Mengen- vs Wertgrößen),
Punkt 2 (Restschuld), Punkt 3 (1-Jahres-Betrachtung), Punkt 4
(Neugeschäft eliminiert) — durch Verwendung regulatorisch publizierter,
kohorten-basierter, **bank-spezifischer** A-IRB-PDs (keine Country-
Aggregate-Proxy mehr).

---

### A-02 · LGD-Quelle: bank-spezifische Pillar-3 EU-CR6 (Stichtag 31.12.2024) [published]

**Was:** Bank-spezifische, EAD-gewichtete A-IRB-LGDs aus denselben
EU-CR6-Tabellen wie die PDs (A-01).

**Formel:**
```
LGD_(bank,class) = Pillar-3-EU-CR6-SubTotal_LGD_{bank, class, 31.12.2024}
```

Für Klassen, in denen eine Bank keine IRB-Sub-total publiziert
(z. B. Crédit Mutuel Institutions), F-IRB-Default LGD = 45 %
(CRR Art. 161).

**Warum ökonomisch:** Die LGD ist der Anteil der Forderung, der bei
Default nicht zurückkommt (nach Recovery, Sicherheiten-Verwertung,
Insolvenzverfahren). Bank-spezifische A-IRB-LGDs berücksichtigen die
tatsächlichen Sicherheiten der jeweiligen Bank und liegen daher meist
unter den F-IRB-Standardwerten von 45 %.

**Quelle:** Pillar-3 EU-CR6 jeder Bank, dokumentiert pro Zeile in
`data/pillar3_bank_pd_lgd.csv`.

---

### A-02c · PD/LGD-Zeitreihe für den Walk-Forward-Backtest (nur Pillar-3) [published]

**Was:** Für die Out-of-Sample-Validierung (Walk-Forward) muss das
Portfolio an jedem historischen Stichtag T0 mit den *damals gültigen*
PD/LGD eingefroren werden — **kein Look-ahead** aus dem 2024-Snapshot.
Dafür existiert eine eigene **Roh-Reihe `data/pillar3_backtest_pdlgd.csv`**:
**10 Banken × FY2021–FY2024, 225 Datenpunkte**, je Zeile
`LEI × vasicek_class × vintage_date` mit **PD, LGD und EAD aus derselben
EU-CR6-Vintage**. Das Live-Modell nutzt weiterhin den kuratierten
31.12.2024-Snapshot (`data/pillar3_bank_pd_lgd.csv`) — beide Reihen sind
bewusst getrennt (kuratierte Werte ≠ rohe Sub-totals bei einigen Banken).

**Harte Quellen-Vorgabe:** Diese historischen PD/LGD stammen **ausschließlich
aus den Pillar-3-EU-CR6-Tabellen** der jeweiligen Bank — identische Logik wie
A-01/A-02, nur für frühere Stichtage. Es werden **keine** PD/LGD aus
EBA-Transparency-Ausfallquoten, NPL-Quoten oder sonstigen Größen abgeleitet.
EBA-Transparency dient im Backtest ausschließlich der *realisierten*
Vergleichsseite (RWA, CET1, EAD), nie als PD/LGD-Input.

**Stichtags-Zuordnung (no-look-ahead, hold-flat):** Zum Stichtag eines Quartals
im Jahr Y nutzt der Walk-Forward das jüngste *bereits veröffentlichte*
Pillar-3-Jahresende — das ist **31.12.(Y−1)** (FY_Y wird erst ~Q1 von Y+1
publiziert). Das eingefrorene Portfolio verwendet damit nur Information, die zu
T0 real verfügbar war (`vintage_for_period`, `lag_years=1`). Der gewählte
Jahresend-Wert gilt flach über alle vier Quartale des Jahres; eine feinere
Interpolation wäre nicht quellengestützt.

**Frozen-Portfolio = vollständig Pillar-3, Transmission = 2-Faktor:** PD, LGD
**und** EAD je Klasse stammen aus derselben EU-CR6-Vintage der Roh-Reihe; die
Restlaufzeit ist dort nicht offengelegt → Basel-Default 2,5 Jahre (konstant
über alle Jahrgänge, daher zeitvergleichs-neutral). Die gesamte *Input-Seite*
des eingefrorenen Portfolios ist Pillar-3; EBA-Transparency liefert nur die
*realisierte* Vergleichsseite (CET1, RWA) und die Markt-Historie den Schock.
Der realisierte Jahres-Schock (ΔBrent, Δr_10y) läuft durch **dasselbe
2-Faktor-Modell wie das Live-Cockpit** (`two_factor_stress.
capital_bridge_2factor` via `frozen_2factor_delta`; sektor-differenzierte β,
kein Single-Faktor-Aggregat) und durch die IRB-K-Formel (`vasicek.py`); der
relative Effekt wird auf die *gemeldeten* Größen skaliert
(`build_pdlgd_panel`, `build_cet1_backtest`).

**Validierungs-Befund (CET1-Kern-Test, 29 Bank-Jahre 2022–2024):** Die
**Zielgröße des Backtests ist die CET1-Quote** (Solvenz-Kennzahl), nicht die
RWA-Änderung. Methode: realer Jahres-Schock ins Ende-Vorjahr eingefrorene
Portfolio → gestresste CET1-Quote über zwei Kanäle (Nenner: RWA_total +
ΔRWA_credit; Zähler: CET1 − ΔEL) → Vergleich mit der tatsächlich gemeldeten
CET1-Quote. Bewusst **konservative Abwärts-Sicht**: Gewinnthesaurierung und
Zinsüberschuss (NII) werden *nicht* gegengerechnet. **Ergebnis:** MAE ≈ 1,3 pp
(auf ~15 %-Niveau ≈ 9 % relativ), 65 % der Bank-Jahre ≤ 1 pp Abstand, 72 %
konservativ (Prognose ≤ Ist), Bias −1,0 pp; im Zinsschock-Jahr 2022 ~2,7 pp
zu konservativ, weil Banken am Zinsanstieg verdienten (NII-Gegeneffekt,
modellseitig bewusst ausgeklammert).

**Ehrliche Grenze (PIT vs. TTC):** Die einzelnen Kanäle (gemeldete PD,
Kredit-RWA) sind **nicht punktprognostizierbar** (Richtungs-Trefferquote ≈
Zufall). Ursache ist eine Daten-Eigenschaft, kein Modellfehler: das Modell
projiziert eine **Point-in-Time**-Reaktion, die gemeldeten A-IRB-Parameter
sind **Through-the-Cycle** — regulatorisch geglättet und antizyklisch
(CRR Art. 180) sowie management-getrieben (CRM, Rekalibrierung, IRB↔SA).
Beleg 2022: Zins +2,8 pp → Modell-PD +0,5 pp, gemeldete PD −0,2 pp. Ein
sauberer PIT-Test bräuchte realisierte Ausfallraten je Segment (bank-intern,
nicht offengelegt; EBA-NPE-Panel erst ab 2024Q3). Das Modell ist damit als
**konservatives Solvenz-/Frühwarn-Instrument** validiert (im CET1-Niveau nah
und auf der sicheren Seite), nicht als Punktprognose einzelner Melde-Kanäle.

**Default-Band inklusive (geflaggt):** Die publizierte EU-CR6-Sub-total-Ø-PD
**enthält das 100 %-(Default)-Band** — exakt dieselbe Definition wie der
2024-Baseline. In Stressjahren kann ein ausgefallenes Exposure die Ø-PD
sichtbar anheben (z. B. Deutsche Bank *Institutions* FY2021/2022: PD ≈ 9–12 %
durch ein voll wertberichtigtes Default-Exposure von ~€2,5 Mrd.; RWA-Dichte
bleibt niedrig). Solche Werte werden **übernommen** (konsistent) und in der
Spalte `note` mit `DEFAULT-BAND` markiert.

**Datenintegrität / Extraktion:** Pro Bank wird der Parser gegen die
hand-verifizierten 2024-Werte **kalibriert** (er muss alle sieben
Klassen-PD/LGD exakt reproduzieren), erst dann werden Vorjahre extrahiert.
Jede Zelle durchläuft einen **Dichte-Cross-Check** (RWA/EAD gegen die
ausgewiesene Density-Spalte); jede Zeile trägt `source_page` und
`source_url`. Kein Wert wird geschätzt — fehlt/scheitert eine Zelle, bleibt
sie leer.

**Format-Grenze (EU-CR6 seit ~2021):** Das EU-CR6-Template in heutiger Form
existiert erst seit der CRR2-Offenlegungs-ITS (Stichtage ab ~Mitte 2021).
Saubere, konsistente Reihen daher für **FY2021–FY2024**; FY2020 (Alt-Template)
ist bewusst ausgeklammert.

**Status (abgeschlossen):** Alle **10 Banken** sind extrahiert — 225
Datenpunkte, **88 % Abdeckung relativ zum bank-eigenen Meldeumfang**
(strukturell nicht geführte Klassen — z. B. kein IRB-Sovereign/QRRE bei
einzelnen Häusern — zählen nicht als Lücke). Die 31 verbleibenden Zellen sind
dokumentierte **Quellgrenzen** (BNP 2021: PD auf ganze % gerundet; SocGen
2022: Bank+Sovereign im PDF-Text verschmolzen; Crédit-Mutuel-/BPCE-Vorjahre:
mehrdeutige Anker) und werden **nicht** mit abgeleiteten Werten gefüllt.
Detail-Protokoll: `tools/pillar3_backfill/STATUS.md` +
`VERIFICATION_REPORT.json` (adversariale 9-Agenten-Quellprüfung).

**Loader:** Backtest-Reihe über `backtesting_walkforward.load_backtest_series()`
(+ `build_pdlgd_panel`, `build_cet1_backtest`, `build_pd_backtest`);
Live-Snapshot unverändert über `load_pd_table(vintage="latest")`
(`backend/eba_pd_loader.py`) — eine Zeile je LEI×Klasse zum 31.12.2024.

---

### A-03 · EAD: EBA Transparency Item 2520522 [published]

**Was:** Exposure-Wert (post-CCF), direkt aus EBA Item 2520522
(Exposure value - by exposure class, SA_and_IRB).

**Warum ökonomisch:** Das ist die regulatorische "Restschuld zum
Bilanzstichtag" — bereits adjustiert um Credit-Conversion-Factor (CCF)
für Off-Balance-Sheet-Positionen. Diese Größe ist die korrekte Basis
für Stress (im Gegensatz zum ursprünglichen Kreditbetrag).

**Quelle:** EBA Transparency Exercise 2025, `tr_cre.csv`, Stichtag
Juni 2025.

**Limitation:** EAD bleibt unter Stress statisch — Drawdown-Risiken auf
Off-Balance-Linien (CCF steigt) sind in V1 nicht modelliert.

---

### A-04 · 2-Faktor-Stress-Transmission [estimate]

**Was:** Die Stress-Wirkung auf PD und LGD wird pro Vasicek-Klasse über
zwei separate Sensitivitäten parametrisiert:

```
ΔPD_class  = β_oil_class  · ΔBrent_log + β_rate_class  · Δr_10y_pp
ΔLGD_class = γ_oil_class  · ΔBrent_log + γ_rate_class  · Δr_10y_pp
```

**β-Werte (Auszug):**

| Klasse | β_oil (PD) | β_rate (PD) | γ_oil (LGD) | γ_rate (LGD) |
|---|---|---|---|---|
| Corporate | +0.30 | +0.20 | +0.50 | +1.00 |
| SME Corporate | +0.60 | +0.40 | +0.45 | +1.10 |
| Mortgage | +0.05 | +0.30 | +0.10 | +1.50 |
| QRRE | +0.40 | +0.15 | +0.30 | +0.50 |
| Other Retail | +0.30 | +0.25 | +0.25 | +0.80 |
| **Bank** | +0.05 | **−0.05** | +0.10 | +0.50 |
| Sovereign | 0 | 0 | 0 | 0 |

*SME-β auf ≈2× Corporate angehoben (Recalibration Juni 2026) — ECB
WP 2897 (2024) zeigt, dass kleine und mittlere Unternehmen deutlich stärker
auf Angebots- und Zinsschocks reagieren als große Unternehmen. ECB WP 3207
(2026) stützt zusätzlich sektorale Unterschiede bei Unternehmens-Ausfällen
im Stress-Test-Kontext.
Die Werte ersetzen den älteren ≈1,5×-Wert aus Drehmann/Juselius (2014).*

**Sektor-Differenzierung adressiert Professor-Kritik Punkt 7** ("Zinsen
hoch ≠ allgemein schlecht"): Bank-Klasse hat β_rate < 0 — steigende
Zinsen erhöhen die Net-Interest-Margin und kompensieren das
Kreditrisiko-Up.

**Warum ökonomisch:** Sektor-spezifische Wirkungen sind in der
Literatur dokumentiert:
- Energie-Schocks treffen energieintensive Industrien stärker (β_oil)
- Floating-Rate-Hypotheken reagieren stark auf Zinsen (β_rate für Mortgage)
- Banken profitieren von steigender Zinskurve über NIM (β_rate negativ)

**Quellen (Recalibration Juni 2026 — Modell-Stichtag 31.12.2024):**
Methodischer Rahmen: EBA *2025 EU-wide Stress Test — Methodological Note*
(11 Nov 2024), Kap. 2 Credit risk, **§2.4.2 ¶122-123**. ¶122 verlangt
grundsätzlich Modellprojektionen für gestresste TR/LGD/LR-Parameter; ¶123
erlaubt sektorale Sensitivitäten auf Portfolio-Projektionen, wenn keine
geeigneten sektoralen Satellite-Modelle verfügbar sind. **¶130** verlangt,
dass LGD/LR-Projektionen sinkende Fair Values von Credit-Risk-Mitigants
berücksichtigen. Schock-Größe: EBA/ESRB *2025 Macro-financial scenario*
(Jan 2025), **§4.1.6** adverser 10y-Pfad. Per-Segment-β:

| Segment | Was die Quelle modelliert | Warum genau dieser Wert | Plausibilitäts-Check |
|---|---|---|---|
| **Corporate** | ECB WP 2897 (2024): makroökonomische Schocks verändern die **Ausfallwahrscheinlichkeit** nichtfinanzieller Unternehmen. Das passt, weil diese Unternehmen die Kreditnehmer hinter unseren Corporate-Exposures sind. ECB WP 3207 (2026): sektorale Unternehmens-Ausfälle unterscheiden sich im Stress-Test stark. | Corporate ist der Anker. Der Ölkanal ist ein **Proxy**: Brent steht für den Energie-/Angebotsschock, nicht für eine veröffentlichte Öl-Beta. Höhere Energiepreise erhöhen Kosten, drücken Margen und schwächen Cashflows. Deshalb β_oil = +0.30. Der Zinskanal ist direkter: höhere Zinsen verteuern Refinanzierung, deshalb β_rate = +0.20. | +1,9 Prozentpunkte Zinsstress × 0,20 = **+0,38 Prozentpunkte** Ausfallwahrscheinlichkeit. Plausibel, weil EBA §2.4.2 ¶123 sektorale Sensitivitäten verlangt, die Richtung und Größenordnung des Szenarios treffen; WP 2897 zeigt für große Unternehmen eine moderate Reaktion. Genau deshalb ist +0,20 als moderater Corporate-Anker plausibel. |
| **SME Corporate** | ECB WP 2897 (2024): kleine und mittlere Unternehmen reagieren deutlich stärker auf Angebots- und Zinsschocks als große Unternehmen. ECB WP 3207 (2026): Stress trifft Sektoren und Unternehmensgruppen unterschiedlich stark. | β_oil = +0.60 und β_rate = +0.40 sind exakt 2× Corporate. Das ist eine vorsichtige Kalibrierung am unteren Rand der in WP 2897 gezeigten stärkeren KMU-Reaktion. | +1,9 × 0,40 = **+0,76 Prozentpunkte**. Plausibel, weil WP 2897 kleine/mittlere Unternehmen empirisch deutlich sensitiver einordnet; 2× ist der vorsichtige untere Rand. Plausibel gemäß EBA ¶123, weil der Aufschlag die Segment-Rangfolge und die Szenario-Größenordnung transparent wahrt. |
| **Mortgage** | ECB WP 3112 (2025): Zinsanstiege erhöhen die Ausfallwahrscheinlichkeit variabel verzinster Hypotheken deutlich stärker als Zinssenkungen sie senken. EBA ¶130 stützt den Sicherheiten-Kanal. | β_rate = +0.30 liegt über Corporate wegen Schuldendienst- und Hauspreis-Kanal, aber unter SME wegen Besicherung. β_oil = +0.05 nahe null, da Öl nur indirekt über Haushaltsbudgets wirkt. | +1,9 × 0,30 = **+0,57 Prozentpunkte**. Plausibel, weil WP 3112 Zinsanstiege bei variablen Hypotheken als starken, asymmetrischen Ausfalltreiber modelliert. Plausibel gemäß EBA ¶130, weil Sicherheitenwerte im Stress zusätzlich berücksichtigt werden müssen. |
| **QRRE** | ECB Financial Stability Review Mai 2024: Haushalte werden durch Schuldendienst, Lebenshaltungskosten und Energiepreise belastet. EBA 2025 Results Fig. 22: Retail hat die höchste projizierte Verlustquote. | Qualifying Revolving Retail Exposures (revolvierende Konsumentenkredite, z. B. Kreditkarten) sind oft unbesichert und einkommensnah. Deshalb β_oil = +0.40. β_rate = +0.15 bleibt unter Mortgage, weil die Zinssätze ohnehin hoch und variabel sind. | +0,50 Brent-log-Schock × 0,40 = **+0,20 Prozentpunkte**. Methodisch plausibel gemäß EBA ¶123, weil die Sensitivität Richtung und Größenordnung des Stress-Szenarios treffen muss. EBA Results Fig. 22 dient hier nur als grober Verlustquotenvergleich nach Exposure-Klasse: Retail liegt dort im Stress hoch. |
| **Other Retail** | Gleiche Quellenebene wie QRRE: Financial Stability Review 2024 für Haushalte, EBA Results 2025 Fig. 22 für Retail als verlustanfälliges Portfolio. | Mischprofil zwischen Kreditkarte/Konsumkredit und Hypothek: β_oil = +0.30, β_rate = +0.25. Der Wert ist plausibilisiert, aber nicht direkt aus einer Quelle abgeschrieben. | +1,9 × 0,25 = **+0,48 Prozentpunkte**. Plausibel, weil der Wert zur hohen Retail-Verlustquote in Fig. 22 passt, aber keine stärker belegte Unterklassen-Beta behauptet. Genau diese Vorsicht macht den Zwischenwert quellenkonform. |
| **Bank** | WP 2897/3207 sind hier **nicht** passend, weil sie nichtfinanzielle Unternehmen modellieren. Referenz ist EBA 2025 Methodology Kap. 4 zu Net Interest Income, also Zinsüberschuss, plus EBA Results 2025 zur Widerstandsfähigkeit des Zinsüberschusses. | β_rate = −0.05 ist eine kleine Expertenannahme: steigende Zinsen können den Bank-Zinsüberschuss stützen, zugleich steigen Kreditqualitäts- und Refinanzierungsrisiken. Deshalb nur leicht negativ. β_oil = +0.05 minimal und indirekt. | +1,9 × −0,05 = **−0,10 Prozentpunkte**. Plausibel nur als kleine Expertenannahme: die Quellen liefern keine Bank-Ausfall-Beta, daher bleibt der Wert nahe null. Das ist quellenkonform, weil kein empirisch nicht belegter starker Bank-PD-Effekt behauptet wird. |
| **Sovereign** | EBA Methodology **¶154** behandelt Sovereign-Default- und Impairment-Flows separat. Der Zinsschock wird im Modell über Duration/Marktbewertung abgebildet. EBA Results Fig. 22 zeigt Public Sector mit niedrigen Verlustquoten im Vergleich der Exposure-Klassen. | β_oil = β_rate = 0, um Doppelzählung zu vermeiden: Δr wirkt im Marktbuch-Kanal, nicht zusätzlich als Sovereign-PD-β. | 0 × jeder Schock = **0**. Plausibel gemäß EBA ¶154, weil Sovereign-Impairments separat behandelt werden und der Zinseffekt bereits im Marktbuch-Kurswertkanal liegt. Zusätzlich passt es zum EBA-Results-Vergleich, weil Public Sector in Fig. 22 niedrig bleibt. |

**All-Segment-Querverankerung:** EBA *2025 EU-wide Stress Test — Results*
(Aug 2025), Fig. 22, liefert einen groben Verlustquotenvergleich nach
Exposure-Klasse (Retail hoch, Public Sector niedrig) am Startpunkt end-2024.

**Wichtig zur Faktor-Struktur:** Der Zins-Kanal ist direkt auf aktuelle,
segment-spezifische Quellen kalibrierbar. Der Öl-Kanal wirkt im EBA/ECB-
Framework *nicht* direkt, sondern als adverser Angebotsschock (so in
WP 2897 identifiziert) — das EBA-2025-Szenario ist selbst öl-/gas-getrieben
(Energiepreis → HICP +3,9 % → Zinsen ↑ → BIP ↓). Bank-Öl bleibt indirekt
(kein direkter per-Segment-Anker), daher klein gehalten.

**Limitation:** Die Quellen ab 2024 liefern keine fertige aufsichtliche Beta-
Tabelle. Sie liefern Richtung, relative Stärke und Plausibilitätsanker:
WP 2897 (2024) für Unternehmens-Ausfallwahrscheinlichkeiten, WP 3112 (2025)
für Hypotheken-Ausfälle, WP 3207 (2026) für sektorale Unternehmens-Ausfälle
im Stress-Test-Kontext, EBA Results 2025 für den groben Verlustquotenvergleich
nach Exposure-Klasse. Baselines
(PD/LGD) = 31.12.2024, Schock-Größe = EBA-2025-Szenario. β sind via Sidebar-
Override veränderbar.

**Was Annahme ist (explizit):** Die Aufsicht gibt nur die *Logik* vor
(§2.4.2 ¶123: sektorale Sensitivitäten, konsistent mit Richtung und
Größenordnung der Szenario-Schocks) — **keine fertige β-Tabelle**. Wir haben
also **nichts abgeschrieben**: die Quellen liefern Vorzeichen, relative
Struktur und Größenordnung; die konkreten Zahlen sind eine **lineare,
überschreibbare Experten-Kalibrierung**.

**Kalibrierungsprinzip — wichtiger Disclaimer:**

| Frage | Einordnung |
|---|---|
| Methodischer Anker | EBA *2025 EU-wide Stress Test — Methodological Note*, §2.4.2 ¶122-123: gestresste Risikoparameter sollen modelliert werden; wenn sektorale Modelle fehlen, sind sektorale Sensitivitäten auf Portfolio-Projektionen zulässig. |
| Empirischer Anker | ECB WP 2897 (2024) schätzt Makro-/Geldpolitik-Schocks auf Corporate-Ausfallwahrscheinlichkeiten; ECB WP 3112 (2025) schätzt Zinsanstiege auf Mortgage-Defaults. |
| Was ist ein β-Wert hier? | Ein transparenter Übersetzungsfaktor: Schockgröße × β = zusätzliche Ausfallwahrscheinlichkeit in Prozentpunkten. |
| Warum ist ein Wert plausibel? | Weil seine Wirkung im EBA-Szenario ökonomisch erklärbar bleibt, die Segment-Logik wahrt und nicht größer wirkt als die Quelle tragen kann. |
| Was behaupten wir nicht? | Nicht: „Die Quelle beweist exakt β = 0,20“. Sondern: „β = 0,20 ist unsere quellenkonforme Basiskalibrierung und per Override sensitivierbar.“ |

**Plausibilitäts-Checks (keine invertierte Quell-Tabelle):**
- **Zinsschock-Skalierung:** β_rate × EBA-Adverse-Zinsschock (+1,9 pp) →
  Large-Corp +0,38 Prozentpunkte / KMU +0,76 Prozentpunkte / Mortgage +0,57
  Prozentpunkte Ausfallwahrscheinlichkeit. Das hält Corporate moderat, KMU
  etwa 2× Corporate und Mortgage zinsdominiert.
- **EBA-Verlustquotenvergleich:** EBA Results Fig. 22 zeigt aggregierte
  Verlustquoten nach Exposure-Klasse, keine bankindividuellen Detailportfolios.
  Der Vergleich stützt nur die grobe Richtung: Retail stärker, Public Sector
  niedrig.
- **Stress-Randbereich:** WP 3207 zeigt, dass Unternehmens-Ausfälle im harten
  Stress deutlich stärker reagieren können als im Normalfall; der KMU-
  Multiplikator 2× ist deshalb bewusst vorsichtig.
- **Doppelzählung:** Sovereign-Zinsrisiko läuft über Marktbuch-Duration/MtM;
  ein zusätzliches Sovereign-PD-β auf Δr würde denselben Schock doppelt zählen.

**Ausblick — portfolio-spezifische β (bewusst nicht im aktiven Modell).**
Die heutige Matrix nutzt sektorweite Default-Werte. Das ist transparent und
prüfbar, aber langfristig nicht perfekt: eine italienlastige Corporate-Bank,
eine deutsche Hypothekenbank und eine französische Universalbank sollten nicht
exakt dieselben Übersetzungsfaktoren verwenden.

Der nächste öffentliche Entwicklungsschritt wäre ein **effektives Bank-β**:

```
β_effektiv(bank, klasse)
  = β_segment
    × Länder-Schockfaktor
    × Portfolio-Gewicht
    × einfacher Qualitätsfaktor
```

Der **Länder-Schockfaktor** würde berücksichtigen, dass das EBA-Szenario je
Land unterschiedlich hart ist. Italien/Spanien haben im adversen 10-Jahres-
Zinspfad stärkere Bewegungen als Deutschland/Niederlande. Eine Bank mit viel
Italien- oder Spanien-Exposure bekäme daher ein höheres effektives Zins-β als
eine Bank mit eher deutschem oder niederländischem Exposure.

Das **Portfolio-Gewicht** würde die echten Exposure-Anteile der Bank nutzen:
Corporate-lastige Banken bekämen ein anderes Beta-Profil als mortgage- oder
retail-lastige Banken. Diese Information liegt teilweise öffentlich in EBA
Transparency und Pillar-3-Berichten vor.

Ein einfacher **Qualitätsfaktor** könnte die bank-spezifische Ausgangs-
Ausfallwahrscheinlichkeit aus Pillar 3 berücksichtigen. Eine Bank mit deutlich
höherem Start-Risiko in einer Klasse bekäme ein leicht höheres effektives β,
eine Bank mit sehr gutem Kreditbuch ein leicht niedrigeres β. Dieser Faktor
müsste eng gekappt werden, damit aus einer plausiblen Anpassung keine
Scheingenauigkeit wird.

Die Plausibilisierung bliebe dieselbe wie heute: β × EBA-Schock muss in einer
verständlichen Größenordnung liegen; der grobe Verlustquotenvergleich sollte
zu EBA Results 2025
passen; und Sovereign-Zinsrisiko darf nicht doppelt gezählt werden.

Prozessual wäre das eine fünfstufige Erweiterung: zuerst Datenbasis aufbauen
(EBA-Transparency-Exposure je Bank × Klasse × Land, Pillar-3-
Ausfallwahrscheinlichkeit und Verlustquote, EBA/ESRB-Länderpfade, EBA-Results-
Peers), dann jede Position auf Exposure-Klasse, Land und Stresskanal mappen,
dann das effektive β rechnen, anschließend gegen EBA-Verlustquotenvergleich,
β × Szenario und Peer-Gruppe plausibilisieren, und erst danach als
dokumentierten Challenger gegen das heutige sektorweite β einsetzen.

Der **Idealweg** wäre ein bankinternes Satellitenmodell im Sinne der EBA-
Methodik: eigene Ausfall- und Rating-Migrationsdaten je Klasse über mindestens
einen Konjunkturzyklus, dann eine Schätzung der Ausfallwahrscheinlichkeit als
Funktion von Brent, Zinsen, Arbeitslosigkeit, Wachstum, Hauspreisen und Lags.
Das wäre methodisch stärker, ist mit öffentlichen Daten aber nicht möglich.
Deshalb bleibt die aktive Version bei sektorweiten β und bietet den Override
als transparenten Sensitivitätstest.

---

### A-05 · IRB-Capital-Formel [published]

**Was:** Die regulatorische Kapitalanforderung K pro Schuldner unter
ASRF (Asymptotic Single Risk Factor, Vasicek 2002 / BCBS 2017):

```
K = LGD · [N((N⁻¹(PD) + √ρ · N⁻¹(α)) / √(1−ρ)) − PD] · MA(M_eff)
RWA = K · 12.5 · EAD
```

mit:
- α = 0.999 (Konfidenz, Basel III)
- ρ = Asset-Korrelation per Klasse (CRR Art. 153/154)
- MA = Maturity-Adjustment für effektive Laufzeit

**Wichtige Klarstellung:** Wir nutzen weiterhin die Vasicek/ASRF-
IRB-Capital-Formel, weil sie der **regulatorische Standard** ist
(Basel III). Was sich geändert hat: die **Inputs** der Formel
(PD, LGD) kommen jetzt aus dem 2-Faktor-Stress (A-04), nicht aus
einem Vasicek-Single-Factor-M-Mapping.

**Quelle:** BCBS (2017). *Basel III: Finalising post-crisis reforms*,
§272-284.

---

### A-06 · Sovereign-Channel: Modified-Duration-MtM [approximation]

**Was:** Mark-to-Market-Verlust auf Sovereign-Exposures unter Δr_10y:

```
ΔFV_bank = − Σ_buckets D_bucket · Δr · EAD_bucket
```

**Wo:**
- D_bucket = Modified Duration (Bucket-Midpoint; geschlossene Buckets
  über den Laufzeitmittelpunkt, offener >10Y-Bucket konservativ mit
  15 Jahren)
- Δr = Δr_10y_pp / 100
- EAD_bucket = EBA Item 2520810

**Warum ökonomisch:** Bond-Preis ist invers zum Zins. Längere Laufzeit
= höhere Duration = stärkere Preisreaktion auf Zinsschock.

**Warum diese Duration-Annahmen:** Die EBA-Transparency-Daten melden
keine bondgenauen Cashflows, sondern Laufzeit-Buckets. Deshalb wird der
Bucket-Midpoint als Cashflow-/Duration-Proxy verwendet. Diese Logik ist
konsistent mit BCBS 368 "Interest rate risk in the banking book" und den
EBA RTS zum standardisierten IRRBB-Ansatz, die bei aggregierten
Zinsrisiko-Daten ebenfalls Zeitbänder bzw. Midpoints für das Slotting
verwenden. Für den offenen >10Y-Bucket wird 15 Jahre gewählt: einfacher
konservativer langer Tenor, angelehnt an das erste lange Basel-/EBA-
Zeitband nach 10 Jahren ("10-15 Jahre"), statt eine nicht beobachtbare
bankindividuelle Cashflow-Verteilung zu erfinden.

**IFRS-9-Filter (Daten, keine Annahme):** Nur HfT, FVTPL und FVOCI
sind CET1-wirksam (via P&L bzw. OCI). AC trägt keinen direkten
CET1-Effekt (latenter Verlust verborgen — vgl. SVB 2023, Jiang et
al. 2023 NBER WP 31048). Der Klassen-Split ist **bank-individuell
gemeldet** (Items 2520812 HfT / 2520813 FVTPL / 2520814 FVOCI /
2520815 AC, granular pro Land × Laufzeit) — keine Stylized-Fact-
Annahme. Duration-gewichtet sind über die 10 Banken ≈ 51 % des
Brutto-MtM CET1-wirksam (Juni 2025).

**Implementierung (Stand 2026-06-10):** Die CET1-Bridge konsumiert
exakt diesen gefilterten MtM via `sovereign_cet1_pnl_lookup()`
(eba_loader). Die frühere V1-Vereinfachung — gesamte Maturity-Ladder
als FVOCI-ähnlich, d. h. 100 % Durchleitung — sowie die noch ältere
60/40-Pauschale sind ersetzt; Tab 1, Tab 3 und Tab 4 rechnen auf
identischer Datenbasis. Regressionstest:
`_test_sovereign_effective_lt_gross` (effective < gross, Ratio
plausibel 30–90 %).

**Limitation:** Parallel-Shift-Annahme (kein Slope/Curvature-Stress
auf Sovereign-MtM); keine Credit-Spread-Risiken; kein Hedging
(Swaps/Futures nicht in EBA-Public-Disclosure).

**Quelle:** EBA Transparency Exercise 2025, `tr_sov.csv` + Items
2520810/812/813/814/815; IFRS 9 (IASB 2014); Tuckman/Serrat (2012),
Kap. 4.

---

### A-07 · Trading-Book-Channel [ENTFERNT, Stand 2026-06-10]

**Was:** Der frühere dritte CET1-Kanal (Market-RWA-Multiplier +
Trading-Book-P&L-Haircut) wurde **vollständig entfernt** — er lief
zuletzt mit hartkodiertem m_factor = 0.0 und damit wirkungslos.

**Begründung:** Die Handelsbücher der 10 überwiegend Retail-/
Corporate-lastigen Banken sind klein, und eine belastbare
FRTB-Sensitivität ließe sich aus den EBA-Bank-Aggregaten (Items
2520210/2520311, keine Issuer-/Tranche-Granularität) nicht sauber
kalibrieren. Ein Kanal, der konstruktionsbedingt 0 beiträgt,
suggeriert nur Schein-Vollständigkeit. Die CET1-Bridge ist seitdem
ein konsistentes **2-Kanal-Modell** (Kreditbuch + Sovereign);
`cet1_ratio_bridge(tb_stress_df=...)` ignoriert den Legacy-Parameter.

---

### A-08 · Universum: 10 IRB-Banken [approximation]

**Was:** Cockpit deckt die 10 größten EU-Banken nach IRB-EAD ab:

1. Groupe Crédit Agricole (FR)
2. BNP Paribas (FR)
3. ING Groep N.V. (NL)
4. Société Générale (FR)
5. Groupe BPCE (FR)
6. Deutsche Bank (DE)
7. Crédit Mutuel (FR)
8. Banco Santander (ES)
9. Coöperatieve Rabobank (NL)
10. UniCredit (IT)

**Coverage:** €8.28 tn IRB-EAD = 56% der gesamten IRB-EAD im EBA-
Datensatz.

**Begründung der Restriktion:** Einheitliche Datenqualität via
EBA-Annex-PDs ist nur für diese 10 Banken via Heimatland-Aggregat
sauber abbildbar. Andere Banken müssten Pillar-3-spezifisch
extrahiert werden (siehe Limitation A-01).

---

### A-09 · Bank-Klassen-Mapping pro Heimatland [approximation]

**Was:** Jeder Bank wird ein Heimatland zugeordnet und alle EBA-Annex-
PDs/LGDs dieses Landes als ihre Counterparty-Sensitivität verwendet.

| Bank | Land | EBA-Country-Row im Annex |
|---|---|---|
| Crédit Agricole, BNP, Soc Gen, BPCE, Crédit Mutuel | FR | France |
| Deutsche Bank | DE | Germany |
| ING, Rabobank | NL | Netherlands |
| Santander | ES | Spain |
| UniCredit | IT | Italy |

**Limitation:** Banken haben in der Realität multinationale Exposures
(BNP hat US-Geschäft, Santander hat LatAm-Anteile, etc.). Unsere
Approximation überschätzt die Heimatland-Konzentration.

---

### A-10 · Faktor-Korrelations-Annahme [estimate]

**Was:** Brent und Δr_10y werden als zwei separate, weitgehend unabhängige
Faktoren modelliert.

**Empirische Belegung:**
- Pearson ρ über 5 Jahre (Mai 2021 – Mai 2026) = +0.07
- 95%-Konfidenzintervall: [+0.01, +0.12]
- OLS R²(Δr ~ ΔBrent) = 0.005
- Spearman-Rangkorrelation = (siehe Wirkungskette Tab 1)
- Rolling-252d-Korrelation bleibt stabil im Korridor [−0.10, +0.20]

**Quelle:** `backend/factor_correlation.py`, basierend auf Brent (ICE
via yfinance) + Bundesbank-Svensson-Zero-Curve, daily.

**Adressiert Professor-Kritik:** Punkt 8 (separate Modellierung) und
Punkt 9 (Korrelations-Analyse).

---

## 4. Methodik im Vergleich zur Vorgänger-Version

| Aspekt | Vorgänger (V1, abgelöst) | Jetzt (V2) |
|---|---|---|
| PD-Quelle | Defaulted/Original-Ratio aus tr_cre.csv | **bank-spezifische Pillar-3 EU-CR6 (31.12.2024), 10/10 Banken verifiziert** |
| LGD-Quelle | F-IRB-Default (CRR Art. 161) | **bank-spezifische Pillar-3 EU-CR6 LGD (31.12.2024)** |
| Stichtag | uneinheitlich (Mix) | **31.12.2024 einheitlich (Loader-Test erzwingt Konsistenz)** |
| Stress-Transmission | Single-Factor-M via hybrid_mapping | 2-Faktor (ΔBrent + Δr_10y, sektor-differenziert) |
| Zinslogik | Pauschal "hoch = schlecht" | Sektor-spezifisch (Bank-Klasse β < 0) |
| Universe | 67 IRB-Banken (gemischte Datenqualität) | 10 IRB-Banken (einheitlich) |
| EAD-Nenner für DR | Original Exposure | EAD post-CCF (Restschuld-konform) |
| Backtest | — (nicht vorhanden) | **CET1-Walk-Forward** auf eigener Pillar-3-Roh-Reihe (10 Banken × 2021–2024, no-look-ahead, PIT-vs-TTC-Grenze dokumentiert) |

---

## 5. Aus Scope ausgeschlossen

- **Operational Risk** konstant unter Stress
- **CVA / Counterparty Credit Risk** nicht im Stress-Szenario
- **Sovereign-Spread-Risk** (Italien-vs-Bund) out of scope
- **Concentration Risk** (HHI) nur als KPI, nicht stress-gewichtet
- **Liquidity-Risk / LCR / NSFR** nicht modelliert
- **IFRS-9 Lifetime-EL** (Stage-2-Migration) out of scope — nur 1Y-Forward
- **Multi-Period-Stress-Pfade** (3-Jahres-EBA-Logik) nicht modelliert
- **Hedging-Effekte** (Swaps, Futures, CDS) nicht rekonstruierbar
- **Bank-individuelle Sektor-Sensitivitäten** (alle gleichen β pro Klasse)

---

## 6. Verwendungs-Scope

ICAAP-Validierungs-Use-Case und Lehr-/Demo-Zwecke. **Keine Investment-
Empfehlungen.** Modell-Output ist eine konservative Schock-Schätzung
unter regulatorischen Standard-Annahmen, nicht eine bank-spezifische
Punkt-Prognose.

---

## 7. Bibliographie

- BCBS (2017). *Basel III: Finalising post-crisis reforms*. Basel
  Committee on Banking Supervision.
- Bandoni, E., Fourné, F. & Jarmulska, B. (2025). *Mortgage loan rates
  and the defaults of variable rate mortgages*. ECB Working Paper 3112.
- EBA (2024). *2025 EU-wide Stress Test — Methodological Note* (11 Nov
  2024); Kap. 2 Credit risk, §2.4.2.
- EBA/ESRB (2025). *2025 EU-wide Stress Test — Macro-financial scenario*
  (Jan 2025); §4.1.6 Long-term rates.
- EBA (2025). *2025 EU-wide Stress Test — Results* (Aug 2025); Fig. 22
  Projected credit risk losses by portfolio.
- EBA (2025). *Report on the 2024 Credit Risk Benchmarking Exercise*
  (März 2025).
- EBA (2025). *EU-wide Transparency Exercise 2025 — Public Disclosure*.
- EBA GL 14 (ICAAP / Stress-Testing).
- ECB (2024). *Financial Stability Review, Mai 2024* (Energie-/
  Lebenshaltungskosten-Schock → Belastung der Haushalte; Sektor-Heterogenität).
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles
  and Practice*, 3. Aufl., Kap. 5.8 (Punktprognose-Evaluation, MAE).
- Konietschke, P., Metzler, J. & Ponte Marques, A. (2026). *A quantile
  probability model for sectoral corporate defaults in Europe*. ECB
  Working Paper 3207.
- Board of Governors / OCC (2011). *SR 11-7: Supervisory Guidance on
  Model Risk Management* (Outcomes Analysis).
- Pesaran, M. H. & Timmermann, A. (1992). *A Simple Nonparametric Test
  of Predictive Performance*. Journal of Business & Economic Statistics.
- Vasicek, O. (2002). *Loan Portfolio Value*. Risk Magazine, Dezember.
- Lo Duca, M., Moccero, D. & Parlapiano, F. (2024). *The impact of
  macroeconomic and monetary policy shocks on credit risk in the euro
  area corporate sector*. ECB Working Paper 2897.
- Reinhart, C. & Rogoff, K. (2009). *This Time Is Different: Eight
  Centuries of Financial Folly*. Princeton UP.
- SR 11-7 — Federal Reserve / OCC Supervisory Guidance on Model Risk
  Management (2011).
- Svensson, L. (1994). *Estimating and Interpreting Forward Interest
  Rates: Sweden 1992-1994*. IMF Working Paper 94/114.
- Tsay, R. (2010). *Analysis of Financial Time Series*, 3rd ed., Wiley.
- Vasicek, O. (2002). *Loan Portfolio Value*. Risk Magazine, Dec 2002.
