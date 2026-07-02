# EU Banking Credit Stress Cockpit · Modell-Dokumentation

**Version 2.1 (Mai 2026)** · 2-Faktor-Stress-Modell auf regulatorisch publizierten A-IRB-Parametern
der zehn größten EU-IRB-Banken.

> **Autor.** Bleron Gashi · HS Düsseldorf · `bleron.gashi@study.hs-duesseldorf.de`
> **Single Source of Truth.** [`docs/MODEL_ASSUMPTIONS.md`](./docs/MODEL_ASSUMPTIONS.md) (Methodik-Detail)
>

Dieses README ist als **Stand-Alone-Dokumentation** geschrieben — alle Formeln, Annahmen,
Quellen und Parameter sind hier zitiert, sodass ein Kollege jeden einzelnen Bestandteil
ohne Cockpit-Zugriff verifizieren kann.

---

## Inhaltsverzeichnis

1. [Überblick · was das Modell macht](#1-überblick--was-das-modell-macht)
2. [Ökonomische Motivation](#2-ökonomische-motivation)
3. [Mathematische Fundierung](#3-mathematische-fundierung)
4. [Datenbasis · Quellen mit URLs](#4-datenbasis--quellen-mit-urls)
5. [Banken-Universum · Top-10-Auswahl](#5-banken-universum--top-10-auswahl)
6. [Sensitivitäts-Kalibrierung · β-/γ-Matrix](#6-sensitivitäts-kalibrierung--β-γ-matrix)
7. [Annahmen-Katalog · jede Annahme mit Quelle](#7-annahmen-katalog--jede-annahme-mit-quelle)
8. [Repository-Struktur](#8-repository-struktur)
9. [Bibliographie · prüfbare Referenzen mit URLs/DOIs](#9-bibliographie--prüfbare-referenzen-mit-urls-und-dois)
10. [Setup, Start, Tests](#10-setup-start-tests)
11. [Bekannte Limitationen](#11-bekannte-limitationen)

---

## 1 · Überblick · was das Modell macht

Das Cockpit liefert eine **regulatorische Credit-Risk-Sicht auf das europäische
Bankensystem**. Zwei voneinander unabhängige Macro-Faktoren werden frei vorgegeben,
und das Modell rechnet daraus live die Wirkung auf die CET1-Quote der zehn größten
EU-IRB-Banken aus.

### 1.1 · Zwei Eingangsfaktoren

| Faktor | Einheit | Wertebereich | Notation |
|---|---|---|---|
| ΔBrent | logarithmischer Return | −1,0 ≤ Δ ≤ +1,5 | Δlog(Brent) |
| Δr_10y | Prozentpunkte (pp) | −3,0 ≤ Δ ≤ +5,0 | Δr |

### 1.2 · Zwei Stress-Kanäle, ein Ziel

```
        ΔBrent     Δr_10y
           ╲         ╱
            ╲       ╱
             ▼     ▼
   ┌─────────────────────────────────────────┐
   │  Sektor-Sensitivitäten β_oil + β_rate    │
   │  pro Exposure-Klasse (Corporate, SME,    │
   │  Mortgage, QRRE, Other Retail, Bank,     │
   │  Sovereign)                              │
   └─────────────────────────────────────────┘
            │                  │
            ▼                  ▼
       ┌─────────┐        ┌──────────┐
       │ Kanal 1 │        │ Kanal 2  │
       │ Kredit­ │        │ Sovereign│
       │ buch    │        │ Book     │
       │         │        │          │
       │ ΔPD +   │        │ ΔFV via  │
       │ ΔLGD →  │        │ Modified │
       │ ΔEL +   │        │ Duration │
       │ ΔRWA    │        │ · IFRS-9-│
       │ via     │        │ Split    │
       │ Basel-  │        │ (gemeldet│
       │ IRB     │        │ je Bank) │
       └─────────┘        └──────────┘
            │                  │
            └────────┬─────────┘
                     ▼
           ┌──────────────────┐
           │   CET1-Quote     │
           │ (Pillar 1 / CCB  │
           │  / SREP-Schwellen)│
           └──────────────────┘
```

*(Ein dritter Trading-Book-Kanal wurde in V2.1 entfernt — kleine
Handelsbücher, keine belastbare FRTB-Kalibrierung aus EBA-Aggregaten.
Markt-Risiko erscheint nicht mehr in der CET1-Bridge.)*

### 1.3 · 5-Tab-Cockpit (Streamlit-App)

| Tab | Inhalt |
|---|---|
| **0 · Intro (Landing-Page)** | 5-Minuten-Tour, Faktor-Begründung, β-Matrix, Stress-Kanäle mit Mini-Beispielen |
| **1 · Faktor-Analyse & Transmission** | 5-Jahres-Korrelations-Analyse + 5-stufige Stress-Bridge |
| **2 · Kreditbuch** | Loan-Book-Kanal: PD/LGD-Matrix pro Bank, Worked Example, Capital-Bridge, Bank-Drilldown |
| **3 · Marktbuch** | 2 Sub-Tabs: Yield-Curve (Input + historische Episoden) · Sovereigns (IFRS-9-Split, Doom-Loop-Map, latente AC-Verluste, Duration/BPV, Δr-Sensitivität) |
| **4 · Eigenkapital** | 2-Kanal-CET1-Waterfall, Pillar-1/CCB/SREP-Threshold-Analyse, Bank-Drilldown |
| **5 · Validierung** | Walk-Forward-Backtest + Annahmen + Methodologie-Disclosure |

---

## 2 · Ökonomische Motivation

### 2.1 · Warum gerade Brent-Öl und der 10-Jahres-Zins?

Beide Faktoren wirken auf die Bilanz einer Bank, aber über **grundverschiedene
Mechanismen**. Sie sind die zwei Macro-Treiber, die in fast allen aufsichtlichen
Stress-Test-Szenarien explizit erscheinen (vgl. EBA Stress Test 2025 Methodology
Note, Sec. 3.2 und 3.4).

| Faktor | Primäre Wirkung auf | Zentrale Transmissions-Kanäle |
|---|---|---|
| **ΔBrent** | Real-Sektor-Cashflows | Input-Kosten energie­intensiver Industrien · Headline-Inflation → EZB-Reaktion · Konsumenten-Disposable-Income via Heiz-/Kraftstoff-Preise |
| **Δr_10y** | Diskontierungs- und Refi-Kosten | Hypotheken-Refi (Mortgage-PD) · Sovereign-Bond-Mark-to-Market via Modified Duration · Bank-Net-Interest-Margin · Corporate-Bond-Spread-Baseline |

**Warum Brent (und nicht WTI)?**
Brent ist der maßgebliche **europäische** Ölpreis-Referenzpunkt — über zwei Drittel
des global gehandelten Rohöls wird gegen Brent gepreist (ICE Futures Europe). Für
eine EU-Banken-Stichprobe ist Brent damit der ökonomisch näherliegende Energie-Schock
als WTI (US-Crude) oder Dubai (Asien). Auch die EBA-Stress-Test-2025-Methodology-Note
verwendet Brent (Sec. 3.2.4).

**Warum der 10-Jahres-Zins?**
Der 10y-Zins ist der zentrale Pricing-Anker für:
- **Hypotheken** (Fixzins-Refinanzierung, Property-Value-Reaktion)
- **Unternehmensanleihen** (Corporate-Spread-Baseline)
- **Staatsanleihen** (Sovereign-Duration)

Kurze Laufzeiten (2y/5y) reagieren primär auf Geldpolitik; sehr lange (30y) sind
illiquide. 10y trifft die typische Bilanz-Duration europäischer Universalbanken.

### 2.2 · Empirische Unabhängigkeit (über 5 Jahre)

Über 1242 Handelstage (≈ 5 Jahre) gilt:

- **Pearson ρ(ΔBrent, Δr_10y) = +0,07** (95 %-CI [+0,01; +0,12])
- **OLS R²(Δr ~ ΔBrent) = 0,005** — keine erklärende Varianz
- **F-Statistik = 6,2** (p ≈ 0,013) — schwach signifikant, aber inhaltlich vernachlässigbar

Konsequenz: **separate Modellierung als zwei unabhängige Risikofaktoren empirisch
gerechtfertigt**. Die volle Diagnostik (R-Style `lm()`-Output mit Coefficients,
t-Werten, p-Werten, F-Statistik und Residual-Standardfehler) ist im
Cockpit-Tab 1 reproduzierbar.

### 2.3 · Warum logarithmische Returns für Brent — und nicht für Zinsen?

| Faktor | Transformation | Begründung |
|---|---|---|
| **Brent** | log-Return | additiv über Zeit, symmetrisch (+0,50 und −0,50 sind gleich groß in der Verteilung), annähernd normalverteilt für Ölpreise → robuste Regressions-Inferenz (vgl. Tsay 2010, §1.4) |
| **Zinsen** | Δ in Prozentpunkten (kein Log) | Zinsänderungen sind bereits linear interpretierbar; Modified-Duration-Bond-Pricing rechnet direkt mit Δy in pp (vgl. Tuckman/Serrat 2012, §4.2) |

---

## 3 · Mathematische Fundierung

### 3.1 · 2-Faktor-Stress-Transmission (pro Exposure-Klasse)

Für jede Exposure-Klasse *c* ∈ {Corporate, SME, Mortgage, QRRE, Other Retail, Bank, Sovereign}:

```
ΔPD_c (pp)  = β^c_oil  · ΔBrent_log  +  β^c_rate · Δr_10y_pp
ΔLGD_c (pp) = γ^c_oil  · ΔBrent_log  +  γ^c_rate · Δr_10y_pp
```

Anschließend mit Floor / Cap geclipt:

```
PD_stress  = clip(PD_base  + ΔPD_c / 100,   3 bp, 50 %)
LGD_stress = clip(LGD_base + ΔLGD_c / 100,  5 %, 100 %)
```

Der 3-bp-PD-Floor ist der Basel-Sovereign-Floor (BCBS d424, Art. 160 (1)).
Der 50-%-PD-Cap ist ein numerisches Sanity-Limit.

### 3.2 · Basel-III-IRB-Capital-Formel (für Corporate, Bank, Sovereign)

Aus BCBS d424 (2017), Art. 153 der Regulation (EU) 575/2013 (CRR):

```
ρ(PD) = 0,12 · (1 − e^(−50·PD)) / (1 − e^(−50))
      + 0,24 · [1 − (1 − e^(−50·PD)) / (1 − e^(−50))]

b(PD) = (0,11852 − 0,05478 · ln(PD))²

K = [ LGD · N( (N⁻¹(PD) + √ρ · N⁻¹(0,999)) / √(1 − ρ) ) − PD · LGD ]
    · (1 − 1,5 · b(PD))⁻¹ · (1 + (M − 2,5) · b(PD))

RWA = K · 12,5 · EAD
```

Wobei:
- **N(·)** = Standard-Normal-CDF, **N⁻¹(·)** = ihre Inverse
- **ρ** = Asset-Korrelation, abhängig von PD (Basel-Funktion)
- **b(PD)** = Maturity-Adjustment-Slope-Faktor
- **M** = Effective Maturity in Jahren (im Modell auf Klassen-spezifischen Wert gesetzt)
- **0,999** = aufsichtsrechtliches Konfidenz-Niveau (99,9 %-Verlust-Quantil)
- **12,5** = 1 / 8 % Mindest-Kapitalquote (Basel-Standardumrechnung)

Für **Mortgage / Retail** entfallen Maturity-Adjustment (M-Term verschwindet),
und die Asset-Korrelation ist fest:
- Mortgage: ρ = 0,15
- QRRE: ρ = 0,04
- Other Retail: ρ-Funktion klassen-spezifisch (CRR Art. 154 (3))

### 3.3 · Modified-Duration-Approximation (Sovereign-Buch)

Für jeden Laufzeit-Bucket *m* ∈ {<3M, 3M–1J, 1–2J, 2–3J, 3–5J, 5–10J, >10J}:

```
ΔFV_m ≈ − D_m · Δy · E_{b,m}
```

mit:
- **D_m** = Modified Duration des Buckets in Jahren
  (approximiert als Bucket-Mittelpunkt: 0,1 · 0,6 · 1,5 · 2,5 · 4,0 · 7,5 · 15,0)
- **Δy** = Zinsänderung in Dezimal (Δr_10y = +2,0 pp = 0,020)
- **E_{b,m}** = Bond-Bestand der Bank *b* im Bucket *m* in EUR

Aggregat pro Bank:
```
ΔFV_bank = Σ_m ΔFV_m
```

Quelle: Tuckman/Serrat (2012), "Fixed Income Securities", 3rd ed., Wiley, §4.2.

**IFRS-9-Filter (gemeldete Daten, keine Annahme):**
```
CET1-Effekt_bank = − Σ_{m, c ∈ {HfT, FVTPL, FVOCI}} D_m · Δy · E_{b,m,c}
```

Der Klassen-Split E_{b,m,c} wird **bank-individuell** aus der EBA
Transparency 2025 gelesen (`tr_sov.csv`, Items 2520812 HfT / 2520813
FVTPL / 2520814 FVOCI / 2520815 AC — granular pro Land × Laufzeit).
Duration-gewichtet sind über die 10 Banken ≈ 51 % des Brutto-MtM
CET1-wirksam (Juni 2025); die frühere Standardannahme f_IFRS9 = 0,60
ist durch die echten Daten ersetzt. Der AC-Anteil bleibt buchhalterisch
unsichtbar, ist aber ökonomisch real (latenter Verlust, vgl. SVB-Krise
2023; Jiang et al. 2023, NBER WP 31048). Implementierung:
`eba_loader.sovereign_cet1_pnl_lookup()` — identische Datenbasis in
Tab 1, Tab 3 und Tab 4.

### 3.4 · Trading-Book (entfernt in V2.1)

Der frühere dritte Kanal (FRTB-Style Market-RWA-Multiplier k = 0,15 +
Trading-P&L-Haircut h = 0,20) wurde **entfernt**: Die Handelsbücher der
10 überwiegend Retail-/Corporate-lastigen Banken sind klein, und eine
belastbare FRTB-Sensitivität ließe sich aus den EBA-Bank-Aggregaten
(keine Issuer-/Tranche-Granularität) nicht sauber kalibrieren. Ein
Kanal ohne belastbare Kalibrierung suggeriert nur Schein-Vollständigkeit.

### 3.5 · CET1-Quoten-Aggregation

```
                CET1_base − ΔEL_loan + ΔFV_sovereign,CET1-wirksam
CET1_stress = ──────────────────────────────────────────────────────
                RWA_base + ΔRWA_credit
```

(ΔFV ist signiert: bei Zinsanstieg negativ. Market- und
Operational-RWA bleiben konstant.)

Vergleich gegen die drei Aufsichts-Schwellen:

| Schwelle | Wert | Aufsichts-Konsequenz |
|---|---|---|
| Pillar 1 | 4,5 % | Mindest-Kapitalquote (CRR Art. 92) |
| Pillar 1 + Capital Conservation Buffer | 7,0 % | Auto-Dividenden-/Bonus-Restriktion bei Unterschreitung |
| Pillar 1 + CCB + SREP-Add-On | 8,0 % | Aufsichtliche Maßnahmen, Pillar-2-Capital-Guidance |

Rechtsgrundlage: Regulation (EU) 575/2013 (CRR) Art. 92, 128, 129;
EBA Guidelines on SREP (EBA/GL/2018/03).

---

## 4 · Datenbasis · Quellen mit URLs

### 4.1 · Bank-spezifische PDs aus Pillar-3-Disclosures (primary)

**Inhalt im Modell:** EAD-gewichtete 1-Jahres-PDs und -LGDs **pro einzelner
Bank × IRB-Exposure-Klasse**, gezogen aus den EU-CR6-Tabellen der jeweiligen
Pillar-3-Reports (regulatorisch publiziert nach CRR Art. 431–455, EBA ITS
on Disclosure ITS/2020/04).

### 4.1a · Einheitlicher Stichtag · 31.12.2024

**Methodische Konvention:** **Alle Baseline-PDs sind auf den Stichtag
31.12.2024 verankert.** Diese Wahl folgt dem EBA-Stress-Test-2025-Standard
und macht den Forward-Stress methodisch sauber: ein einheitlicher
„Heute"-Snapshot ist der Ausgangspunkt, von dem aus der Macro-Schock
(ΔBrent + Δr_10y) in die Zukunft simuliert wird. Vintage-Mix wäre
methodisch unzulässig — manche Banken hätten dann schon 6+ Monate Macro-
Evolution absorbiert, andere nicht.

Der Loader (`backend/eba_pd_loader.py`) enthält einen Test
`_test_vintage_consistency`, der bei jeder Code-Änderung erzwingt, dass
alle 70 CSV-Zeilen denselben `vintage_date` haben.

**Annahme für den 5-Jahres-Backtest-Horizont (2020–2024).** Für die
Walk-Forward-Backtest-Periode unterstellen wir, dass die 31.12.2024-PDs
auch für historische Quartale als Baseline gültig sind. Begründung:
A-IRB-PDs sind **Through-the-Cycle** (TTC) gemäß CRR Art. 180 Abs. 2 —
sie werden bewusst über ≥ 5 Jahre historischer Default-Daten geglättet
und bewegen sich typischerweise nur ±0,1–0,3 pp quartalweise. Die
echte Macro-Dynamik (Brent, Δr_10y) wird in dem Backtest über die zwei
Faktoren modelliert, **nicht über die Baseline-PDs**. Ein synthetisches
historisches PD-Time-Series wäre dieselbe TTC-Glättung wie heute und
würde nichts hinzufügen — die Information sitzt in den β-Sensitivitäten
und Macro-Returns, nicht im PD-Level.

Diese Annahme deckt sich mit der Behandlung in EBA-Stress-Test-Backtests,
die ebenfalls eine fixe Baseline gegen historische Macro-Realisationen
testen.

**Status (Stand Mai 2026):** **10 von 10 Banken** sind aus dem
Pillar-3-Report für **31.12.2024** extrahiert und Cell-by-Cell verifiziert
(`status = "pillar3_verified"`):

| Bank | Quelle | Stichtag | Seiten |
|---|---|---|---|
| Deutsche Bank | [Pillar 3 Report Q4 2024](https://investor-relations.db.com/files/documents/regulatory-reporting/Pillar-3-Report-Q4-2024.pdf) | 31.12.2024 | 108–113 (EU CR6 AIRB) |
| ING Groep | [Additional Pillar III Report 2024](https://ing.com/binaries/content/assets/documents/annual-reports/2024-ing-groep-nv-additional-pillar-iii-report.pdf) | 31.12.2024 | 44–51 (EU CR6 IRB) |
| Société Générale | [Pillar 3 Report 31.12.2024](https://www.societegenerale.com/sites/default/files/documents/2025-03/pillar-3-31122024.pdf) | 31.12.2024 | 134–137 (Table 58 EU CR6 AIRB) |
| Coöperatieve Rabobank | [Pillar 3 Report 2024](https://a.storyblok.com/f/329380/x/0bfc34783d/rabobank_pillar_3-_report-2024.pdf) | 31.12.2024 | 75–82 (EU CR6 AIRB) |
| UniCredit | [Pillar III Disclosure 31.12.2024](https://www.unicreditgroup.eu/content/dam/unicreditgroup-eu/documents/en/investors/third-pillar-basel/2024/UniCredit-Group-Disclosure-Pillar-III-as-at-31-December-2024.pdf) | 31.12.2024 | 36–40 (EU CR6 AIRB) |
| Crédit Mutuel (5 von 7 Klassen) | [Groupe Crédit Mutuel Pilier 3 Bâle III Exercice 2024](https://www.creditmutuel.com/partage/fr/CNCM/telechargements/presse-et-publications/publications/2025/2024-informations-relatives-au-pilier-3-de-bale-III-exercice-2024.pdf) | 31.12.2024 | 50, 53 (Sous-totaux NI/IRB) |
| Groupe BPCE (5 von 7 Klassen) | [Pillar III Risk Report 2024](https://www.groupebpce.com/app/uploads/2025/03/bpce-pillar-iii-2024.pdf) | 31.12.2024 | 153–157 (EU CR6 AIRB) |
| Groupe Crédit Agricole | [Risk Report Pillar 3 H1 2025](https://www.credit-agricole.com/en/pdfPreview/207696) — **31.12.2024-Komparativ-Spalten extrahiert** | 31.12.2024 | 40–42 (EU CR6 AIRB · 31.12.2024 comparative columns) |
| Banco Santander (6 von 7 Klassen) | [Pillar 3 Disclosures 2025](https://www.santander.com/en/shareholders-and-investors/financial-and-economic-information/pillar-3-disclosures-report) — **31.12.2024-Komparativ-Spalten extrahiert** | 31.12.2024 | 94–100 (Tables 27 + 28 EU CR6 AIRB · 31.12.2024 comparative); Sovereign via Standardised-Approach → F-IRB-Default |
| **BNP Paribas** | [Universal Registration Document 2024 (Pillar 3 Chapter 5)](https://invest.bnpparibas/en/document/release-of-the-english-version-of-the-universal-registration-document-and-annual-financial-report-2024) | 31.12.2024 | 440 + 442 + 443 + 448 + 450 + 451 (Tables 38, 39, 41, 42 IRBA EU CR6 SUB-TOTAL rows) |

**Lokale Datei:** `data/pillar3_bank_pd_lgd.csv` (70 Zeilen,
10 Banken × 7 Klassen, mit Audit-Trail pro Cell: `source_url`,
`source_page`, `source_period`, `status`).

### 4.1b · Datenabdeckung: 69 von 69 IRB-Klassen direkt aus Pillar 3

Die Datentabelle enthält **70 Bank-Klassen-Kombinationen**:
10 Banken × 7 Exposure-Klassen. Jede Kombination enthält gemeinsam eine
PD und eine LGD aus derselben regulatorischen Tabellenzeile.

Nach erneuter Prüfung der Originalberichte konnten vier frühere
Übergangswerte ersetzt werden:

| Bank | Klasse | Direkt extrahierter EU-CR6-Sub-total |
|---|---|---|
| Groupe BPCE | Mortgage | PD 14,68 % · LGD 10,70 % |
| Groupe BPCE | QRRE | PD 9,63 % · LGD 33,85 % |
| Crédit Mutuel | QRRE | PD 3,13 % · LGD 33,00 % |
| Crédit Mutuel | Bank (Institutions) | PD 0,12 % · LGD 34,00 % |

Damit stammen **alle 69 IRB-fähigen Kombinationen direkt aus den
jeweiligen Bank-Pillar-3-Berichten**. Die hohen BPCE-Mortgage-PDs werden nicht mehr
durch ein Länderaggregat ersetzt: Sie sind der ausdrücklich publizierte,
EAD-gewichtete EU-CR6-Sub-total einschließlich des regulatorischen
Portfolio-Mixes und werden deshalb konsistent mit den übrigen Banken
verwendet.

Die einzige strukturelle Ausnahme ist **Banco Santander · Sovereign**.
Santander weist Central Governments/Central Banks laut CR6-A vollständig
im Standardansatz aus (100 % SA, 0 % IRB). Eine IRB-PD und IRB-LGD wird
daher regulatorisch nicht veröffentlicht und kann nicht aus Pillar 3
extrahiert werden. Die Position wird daher aus dem IRB-Kreditbuchkanal
ausgeschlossen. Im separaten Sovereign-Marktwertkanal bleibt Santander
auf Basis der EBA-Bestände und des IFRS-9-Splits vollständig enthalten.

### 4.1c · Methodische Innovation · Komparativ-Spalten-Extraktion

Für **Crédit Agricole** und **Banco Santander** konnten wir die 31.12.2024-
Werte über die **Komparativ-Spalten** in den neueren Pillar-3-Berichten
extrahieren:

- **Crédit Agricole**: das H1 2025-Pillar-3 (`pdfPreview/207696`) enthält
  pro EU-CR6-Tabelle eine separate Spalte mit den 31.12.2024-Komparativen
  (Pages 40–42 zeigen explizit „CREDIT RISK EXPOSURES BY PORTFOLIO AND
  PROBABILITY OF DEFAULT (PD) RANGE — ADVANCED INTERNAL RATINGS-BASED
  APPROACH **AT 31 DECEMBER 2024**").
- **Banco Santander**: das 2025-Pillar-3 (`irp-2025-irp-2025-en.pdf`,
  vom User lokal bereitgestellt) hat dieselbe Konvention — pages 94–100
  haben explizit „CR6 - AIRB approach **(31.12.2024)**" als Header.

Diese Komparativ-Werte sind regulatorisch identisch mit den im
Original-Jahresbericht publizierten 31.12.2024-Werten — sie werden
unverändert in den Folgeperioden mitgeführt (EBA-ITS-Disclosure-Pflicht).

**Coverage-Status:** beim Cockpit-Start emittiert der Loader
(`backend/eba_pd_loader.py`) automatisch eine Konsolen-Meldung
`PD/LGD-Tabelle geladen · 70 Zeilen · verified=69 · country_proxy=0 ·
basel_default=0 · standardised_na=1`. Damit ist direkt sichtbar, dass
alle 69 IRB-fähigen Kombinationen bankpubliziert sind und eine weitere
Kombination regulatorisch nicht zum IRB-Kanal gehört.

### 4.1d · Standardansatz-Ausnahme: Santander Sovereign

Santanders offizieller CR6-A-Scope weist Central Governments/Central
Banks zum 31.12.2024 vollständig dem Standardansatz zu. Die zugehörige
CR4-Tabelle publiziert Exposures und Risikogewichte, aber keine IRB-PD
oder IRB-LGD. Deshalb wird kein künstlicher Ersatzwert verwendet:
Santander Sovereign ist im IRB-Kreditbuch nicht anwendbar, bleibt aber
im Marktwert-/Duration-Kanal des Sovereign-Buchs enthalten.

### 4.2 · EBA Transparency Exercise 2025

**Inhalt im Modell:**
- EAD (Exposure at Default) pro Bank × Klasse (Item 2520522)
- Sovereign-Bestände pro Bank × Maturity-Bucket (Items 2520810/812/813/814/815)
- Market-RWA pro Bank (Item 2520210)
- Trading-Net-Income pro Bank (Item 2520311)
- Common Equity Tier 1 Capital pro Bank (Item 2520102)

**Quelle:** European Banking Authority, "EU-wide Transparency Exercise 2025",
Reporting-Stichtag 30. Juni 2025, veröffentlicht Dezember 2025.

**URL:** https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/eu-wide-transparency-exercise

**Lokale Dateien:** `data/tr_cre.csv`, `data/tr_sov.csv`, `data/tr_oth.csv`,
`data/tr_mrk.csv`, `data/TR_Metadata.xlsx`, `data/SDD.xlsx`.

### 4.3 · Bundesbank Svensson-Parameter (10-Jahres-Zins)

**Inhalt im Modell:** Tagesreihe der vier Svensson-Parameter (β₀, β₁, β₂, β₃) der
deutschen Zins-Strukturkurve, daraus berechnet der 10-Jahres-Zero-Coupon-Zins.

**Quelle:** Deutsche Bundesbank, Zeitreihen-Datenbank, Reihe BBSSY01.

**URL:** https://www.bundesbank.de/dynamic/action/de/statistiken/zeitreihen-datenbanken/zeitreihen-datenbank/759778/759778

**Methodologie:** Svensson, L. E. O. (1994), "Estimating and Interpreting Forward
Interest Rates: Sweden 1992-1994", IMF Working Paper 94/114.

**Lokale Datei:** `data/cache/svensson_params.parquet`.

### 4.4 · Brent Crude (ICE Futures Europe)

**Inhalt im Modell:** Tägliche Brent-Schluss-Kurse in USD.

**Quelle:** ICE Futures Europe via yfinance-Ticker `BZ=F`.

**URL:** https://www.theice.com/products/219/Brent-Crude-Futures (Produkt-Spezifikation)

**Lokale Datei:** `data/cache/brent_crude.parquet`. Auto-Update via
`tools/data_fetch/03_fetch_brent_crude.py`.

---

## 5 · Banken-Universum · Top-10-Auswahl

### 5.1 · Auswahl-Logik

Aus den 120 Banken des EBA-Transparency-Universe werden zunächst die 67 IRB-Banken
ausgewählt (Filter 1: nur Banken mit Internal-Ratings-Based-Approach liefern PD/LGD,
Voraussetzung für jede Vasicek-/IRB-basierte Modellierung). Aus diesen werden die
**zehn größten** nach einem kombinierten Score aus Σ EAD und Klassen-Coverage
selektiert.

### 5.2 · Die zehn Banken im Cockpit

| # | Bank | Heimat | Σ EAD (€ Mrd.) | EBA-LEI |
|---|---|---|---:|---|
| 1 | Crédit Agricole | FR | 1 646 | 969500TJ5KRTCJQWXH05 |
| 2 | BNP Paribas | FR | 1 302 | R0MUWSFPU8MPRO8K5P83 |
| 3 | ING Groep | NL | 928 | 549300NYKK9MWM7GGW15 |
| 4 | Société Générale | FR | 791 | O2RNE8IBXP4R0TD8PU41 |
| 5 | Groupe BPCE | FR | 777 | 9695005MSX1OYEMGDF46 |
| 6 | Deutsche Bank | DE | 739 | 7LTWFZYICNSX8D621K86 |
| 7 | Crédit Mutuel | FR | 627 | 9695000CG7B84NLR5984 |
| 8 | Banco Santander | ES | 595 | 5493006QMFDDMYWIAM13 |
| 9 | Coöperatieve Rabobank | NL | 469 | DG3RU1DBUFHT4ZF9WN62 |
| 10 | UniCredit | IT | 408 | 549300TRUWO2CD2G5692 |

**Σ EAD Top-10 = €8 282 Mrd.** = rund **55 %** der gesamten EU-IRB-EAD,
ca. **45 %** der gesamten EU-Banken-Bilanzsumme.

### 5.3 · Begründung der Begrenzung auf 10

1. **Datenqualität.** Diese zehn Banken haben vollständige PD/LGD-Disclosure
   über alle sieben Exposure-Klassen. Bei kleineren IRB-Banken fehlen oft Klassen
   oder Werte sind aggregiert.
2. **Coverage.** Die Top-10 repräsentieren bereits einen substantiellen Anteil
   des EU-IRB-Systems (siehe Σ EAD oben).
3. **Heimatland-Diversifikation.** Die zehn decken die fünf größten EU-Bankenmärkte
   ab: Frankreich (5), Deutschland (1), Niederlande (2), Spanien (1), Italien (1).
4. **Modell-Transparenz.** Pro Bank wird im Cockpit ein Worked Example dargestellt.
   Mit 67 Banken wäre die didaktische Nachvollziehbarkeit nicht mehr gegeben.

---

## 6 · Sensitivitäts-Kalibrierung · β-/γ-Matrix

### 6.1 · Vollständige Matrix

Pro Exposure-Klasse vier Koeffizienten: β_oil (PD), β_rate (PD), γ_oil (LGD),
γ_rate (LGD). Alle in Prozentpunkten pro Einheit des Faktors.

| Klasse | β_oil | β_rate | γ_oil | γ_rate | Ökonomische Logik |
|---|---:|---:|---:|---:|---|
| Corporate | +0,30 | +0,20 | +0,50 | +1,00 | Energie-Input-Kosten + Bond-Refi-Kosten |
| SME Corporate | +0,60 | +0,40 | +0,45 | +1,10 | Kleine und mittlere Unternehmen reagieren ≈2× so stark wie Large-Corp (ECB WP 2897 2024); geringere Diversifikation, kürzere Refi-Profile |
| Mortgage | +0,05 | +0,30 | +0,10 | +1,50 | Floating-Rate-Affordability + Property-Value-Haircut bei Δr |
| QRRE | +0,40 | +0,15 | +0,30 | +0,50 | Revolvierende Konsumentenkredite, z. B. Kreditkarten: Inflation belastet verfügbares Einkommen stark |
| Other Retail | +0,30 | +0,25 | +0,25 | +0,80 | zwischen Mortgage und QRRE |
| **Bank** | +0,05 | **−0,05** | +0,10 | +0,50 | **NIM-Uplift** bei steigenden Zinsen — adressiert die Kritik, dass „steigende Zinsen pauschal schlecht" ökonomisch unkorrekt ist |
| Sovereign | 0 | 0 | 0 | 0 | fiskalisch determiniert, nicht macro-getrieben |

### 6.2 · Quellen pro Wert

Methodischer Rahmen: **EBA 2025 Methodology Note §2.4.2 ¶122-123** (Modelle
für gestresste TR/LGD/LR; ¶123 erlaubt sektorale Sensitivitäten auf
Portfolio-Projektionen, wenn keine geeigneten sektoralen Satellite-Modelle
verfügbar sind). **¶130** verankert den LGD-Kanal über sinkende Fair Values
von Credit-Risk-Mitigants. Schock-Größe: **EBA 2025 Macro-financial scenario
§4.1.6** (adverser 10y-Pfad, Start Dez-2024). Quelle der per-Segment-β:

| Klasse | Quelle für β-Werte (aktuell, Stichtag 31.12.2024) |
|---|---|
| Corporate | ECB WP 2897 (2024): Schocks verändern die **Ausfallwahrscheinlichkeit** nichtfinanzieller Unternehmen; genau diese Firmen sind die Kreditnehmer hinter Corporate-Exposures. ECB WP 3207 (2026): sektorale Unternehmens-Ausfälle unterscheiden sich im Stress-Test stark. Öl ist hier ein **Proxy** für den Energie-/Angebotsschock, keine veröffentlichte Öl-Beta |
| SME Corporate | ECB WP 2897 (2024): kleine und mittlere Unternehmen reagieren deutlich stärker als große Unternehmen; deshalb β = 2× Corporate. ECB WP 3207 (2026): bestätigt sektorale Unterschiede im Stress-Test |
| Mortgage | ECB WP 3112 (2025): Zinsanstiege erhöhen Ausfallwahrscheinlichkeiten variabel verzinster Hypotheken deutlich; Öl nur indirekt via Haushaltsbudgets · LGD: EBA §2.4.2 ¶130 Sicherheitenwert |
| QRRE | ECB Financial Stability Review Mai 2024 — Haushalts-Schuldendienst, Lebenshaltungskosten und Energiepreise; EBA-2025-Results Fig. 22 — Retail höchste Verlustquote |
| Other Retail | ECB Financial Stability Review Mai 2024 + EBA-2025-Results Fig. 22; Mischprofil zwischen revolvierendem Retail/Konsumkredit und Mortgage, keine direkt publizierte Unterklassen-Beta |
| **Bank** | WP 2897/3207 ausdrücklich **nicht** passend, weil sie nichtfinanzielle Unternehmen modellieren. EBA 2025 Methodology Kap. 4 (Net Interest Income = Zinsüberschuss) + EBA Results 2025 zur Zinsüberschuss-Resilienz; β_rate < 0 als kleine Expertenannahme, keine publizierte Bank-Ausfall-Beta |
| Sovereign | EBA §2.4.2 ¶154 (Sovereign-Default-/Impairment-Flows separat); Zinsrisiko via Marktbuch-Duration/Marktbewertung. EBA-2025-Results Fig. 22: Public Sector niedrige Verlustquote. β = 0 verhindert Doppelzählung |

Die γ-Werte (LGD-Stress) sind nach EBA §2.4.2 ¶130 kalibriert (LGD spiegelt
den Sicherheiten-Fair-Value-Verfall). CRR Art. 181 fordert „Downturn-LGD"
für IRB-Banken — die γ-Werte bilden den Downturn-Aufschlag faktor-spezifisch ab.

> **Plausibilitäts-Checks:** β_rate × EBA-Adverse-Zinsschock (+1,9 pp) ergibt
> Large-Corp +0,38 Prozentpunkte / KMU +0,76 Prozentpunkte / Mortgage +0,57
> Prozentpunkte Ausfallwahrscheinlichkeit. Das hält Corporate moderat, KMU
> ≈2× Corporate und Mortgage zinsdominiert. EBA Results Fig. 22 plausibilisiert
> Retail hoch und Public Sector niedrig; WP 3207 stützt, dass Unternehmens-
> Ausfälle im harten Stress deutlich stärker reagieren können als im Normalfall.
> Sovereign β = 0 verhindert Doppelzählung, weil Δr bereits im Marktbuch-Kanal wirkt.

> **Hinweis:** Die Quellen ab 2024 liefern keine fertige Beta-Tabelle. Sie liefern
> Richtung, relative Stärke und Plausibilitätsanker. Die konkreten β-Werte sind
> transparente, überschreibbare Kalibrierungsannahmen auf Basis des Modell-
> Stichtags 31.12.2024 und des EBA-2025-Adverse-Szenarios.

### 6.3 · Override-Möglichkeit für Sensitivitäts-Analysen

Im Cockpit (Tab 0 · Intro · Schritt 3b) können alle β-Werte überschrieben werden.
Die geänderten Werte wirken sofort und global auf alle Charts.

---

## 7 · Annahmen-Katalog · jede Annahme mit Quelle

| ID | Annahme | Wert | Quelle / Begründung |
|---|---|---|---|
| A-01 | PD pro Bank × Klasse | **bank-spezifisch** aus Pillar-3 EU-CR6-Subtotals: 69/69 IRB-fähige Bank-Klassen direkt publiziert; Santander Sovereign ist Standardansatz und aus dem IRB-Kanal ausgeschlossen | CRR Art. 180 + EBA ITS on Disclosure ITS/2020/04 |
| A-01b | **Einheitlicher Stichtag aller PDs/LGDs** | **31.12.2024** für alle 70 CSV-Zeilen — Loader-Test `_test_vintage_consistency` erzwingt diese Konsistenz | EBA Stress Test 2025 Methodology Note; BCBS d155 |
| A-01c | **PD-Baseline für 5-Jahres-Backtest 2020–2024** | 31.12.2024-PDs gelten als Baseline-Proxy für alle historischen Quartale — Macro-Dynamik (Brent, Δr) wird über die zwei Faktoren modelliert, nicht über das PD-Level | CRR Art. 180 Abs. 2 (TTC-Glättung über ≥ 5 Jahre Default-Daten → A-IRB-PDs sind quartalweise quasi-stabil mit ±0,1–0,3 pp Drift); analog zur EBA-Stress-Test-Backtest-Methodik |
| A-02 | LGD pro Bank × Klasse | analog zu A-01 — 69/69 IRB-fähige Kombinationen direkt aus derselben EU-CR6-Zeile wie die PD; Santander Sovereign nicht anwendbar | CRR Art. 181 |
| A-03 | EAD pro Bank × Klasse | EBA Transparency 2025, Item 2520522, post-CCF | EBA Implementing Technical Standards ITS 680/2014 |
| A-04 | β-/γ-Sensitivitäten | siehe Abschnitt 6 | EBA ST 2025 + akademische Literatur |
| A-05 | Asset-Korrelation ρ | Basel-Funktion gemäß CRR Art. 153 (Corporate/Bank/Sovereign), feste Werte für Mortgage (0,15) und QRRE (0,04) | BCBS d424, Art. 153–154 |
| A-06 | Konfidenz-Niveau IRB-Formel | 99,9 % | BCBS d424, Art. 153 (Pillar-1-Standard) |
| A-07 | Modified Duration Sovereign | gewichteter Bucket-Mittelpunkt: 0,1 / 0,6 / 1,5 / 2,5 / 4,0 / 7,5 / 15,0 Jahre | Tuckman/Serrat (2012) §4.2; EBA-Bucket-Definition Items 2520810ff. |
| A-08 | IFRS-9-Mix Sovereign | **keine Annahme — bank-individuell gemeldete Daten**: Split pro Bank × Land × Laufzeit aus EBA Transparency 2025, `tr_sov.csv` Items 2520812 (HfT) / 2520813 (FVTPL) / 2520814 (FVOCI) / 2520815 (AC). CET1-wirksam nur HfT+FVTPL+FVOCI; duration-gewichtet ≈ 51 % im 10-Banken-Mittel (Juni 2025). Frühere Stylized-Fact-Annahme 60/40 ersetzt | EBA Transparency Exercise 2025 (Sovereign-Template); IFRS 9 (IASB 2014); Plausibilitäts-Querprüfung: EBA "Implementation of IFRS 9 by EU Banks" Reports (50–65 % Marktwert-Anteil seit 2020) |
| A-09 | ~~Trading-Book-RWA-Multiplier k~~ | **obsolet** — Trading-Book-Kanal in V1 entfernt (kleine Handelsbücher, keine belastbare FRTB-Kalibrierung aus EBA-Aggregaten); Markt-Risiko nicht mehr Teil der CET1-Bridge | — |
| A-10 | ~~Trading-P&L-Haircut h~~ | **obsolet** — siehe A-09 | — |
| A-11 | Faktor-Unabhängigkeit Brent ⊥ Δr_10y | empirisch ρ = +0,07 über 5 Jahre, R² = 0,005 | eigene Schätzung auf ICE-Brent + Bundesbank-Svensson, im Cockpit Tab 1 reproduzierbar |
| A-12 | PD-Floor / PD-Cap | 3 bp / 50 % | Basel-Sovereign-Floor (BCBS d424 Art. 160 (1)) bzw. numerisches Sanity-Limit |
| A-13 | LGD-Floor / LGD-Cap | 5 % / 100 % | Sanity-Konvention + CRR-Definition |
| A-14 | Stress-Horizont | 1 Jahr | BCBS d424 (IRB-Standardhorizont) |
| A-15 | Sovereign-Floor RWA | 0 % Risikogewicht für EU-Sovereigns in eigener Währung | CRR Art. 114 (4) (Carve-Out für EU-Sovereigns) |

---

## 8 · Repository-Struktur

```text
Modellarchitektur/
├── README.md                          ← dieses File
├── run_clean.py                       ← Streamlit-Start mit Cache-Cleanup
├── config.py                          ← Pfade, Konstanten
│
├── docs/
│   └── MODEL_ASSUMPTIONS.md           ← detaillierte Methodik-Doku (Single Source)
│
├── tools/data_fetch/
│   ├── 03_fetch_brent_crude.py        ← yfinance-Brent-Daten
│   └── 04_fetch_bundesbank_svensson.py ← Svensson-Parameter-Loader
│
├── backend/                           ← die Rechen-Engines
│   ├── two_factor_stress.py           ← β-/γ-Matrix + Stress-Anwendung
│   ├── factor_correlation.py          ← 5-Jahres-Korrelation + R-Style-lm-Output
│   ├── eba_pd_loader.py               ← EBA-Risk-Dashboard-Loader + Top-10-Filter
│   ├── eba_loader.py                  ← EBA-Transparency-CSV-Parsing
│   ├── vasicek.py                     ← IRB-Capital-Formel (Basel-Standard)
│   ├── svensson.py                    ← Zero-Coupon-Curve-Engine
│   ├── macro_factor.py                ← (Legacy) Single-Factor-Mapping
│   ├── backtesting.py                 ← Panel-OLS-Backtest
│   └── backtesting_walkforward.py     ← Quartals-Walk-Forward-Backtest
│
├── streamlit_app/
│   ├── Einführung_in_das_Modell.py    ← Landing-Page (Tab 1) mit 5-Min-Tour
│   ├── static/mckinsey.css            ← Cockpit-Aesthetic
│   ├── components/
│   │   ├── theme.py                   ← Plotly-Template + Breadcrumb
│   │   ├── sidebar.py                 ← 2-Slider-Sidebar
│   │   ├── data_loader.py
│   │   ├── methodology.py
│   │   ├── legacy_views.py
│   │   └── backend_path.py
│   └── pages/
│       ├── 1_Faktor_Analyse.py        ← Korrelations-Analyse + 5-Stufen-Bridge
│       ├── 2_Kreditbuch.py            ← Loan-Book mit Worked Example
│       ├── 3_Marktbuch.py             ← 2 Sub-Tabs (Yield-Curve · Sovereigns)
│       ├── 4_Eigenkapital.py          ← 2-Kanal-CET1
│       └── 5_Validierung.py           ← Walk-Forward-Backtest
│
└── data/
    ├── pillar3_bank_pd_lgd.csv        ← 70 Zeilen, 10 Banken × 7 Klassen (Single Source of Truth)
    ├── top10_irb_banks.csv            ← Top-10-Selektions-Tabelle
    ├── bundesbank_svensson.csv
    ├── tr_cre.csv  (~123 MB, gitignored)
    ├── tr_sov.csv  (~91 MB, gitignored)
    ├── tr_oth.csv  (~14 MB, gitignored)
    ├── tr_mrk.csv  (~3.6 MB, gitignored)
    ├── TR_Metadata.xlsx, SDD.xlsx
    └── cache/
        ├── brent_crude.parquet
        └── svensson_params.parquet
```

---

## 9 · Bibliographie · prüfbare Referenzen mit URLs und DOIs

### 9.1 · Regulatorische Standards

- **BCBS d424 (2017).** *Basel III: Finalising post-crisis reforms.* Basel Committee
  on Banking Supervision, Dezember 2017.
  https://www.bis.org/bcbs/publ/d424.htm

- **BCBS d457 (2019).** *Minimum capital requirements for market risk* (FRTB).
  Basel Committee on Banking Supervision, Januar 2019.
  https://www.bis.org/bcbs/publ/d457.htm

- **BCBS d155 (2009).** *Principles for sound stress testing practices and supervision.*
  Basel Committee on Banking Supervision, Mai 2009.
  https://www.bis.org/publ/bcbs155.htm

- **Regulation (EU) 575/2013 (CRR).** *Capital Requirements Regulation*,
  konsolidierte Fassung.
  https://eur-lex.europa.eu/eli/reg/2013/575/oj

- **IFRS 9 (IASB 2014).** *Financial Instruments.* International Accounting Standards
  Board, Juli 2014, EU-pflichtig seit Januar 2018.
  https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/

- **EBA Guidelines on SREP (EBA/GL/2018/03).**
  https://www.eba.europa.eu/regulation-and-policy/supervisory-review-and-evaluation-srep-and-pillar-2

- **EBA Guidelines on Stress Testing (EBA/GL/2018/04).**
  https://www.eba.europa.eu/regulation-and-policy/supervisory-review-and-evaluation-srep-and-pillar-2

- **EBA ITS 680/2014.** *Implementing Technical Standards on Supervisory Reporting*
  (COREP / FINREP).
  https://www.eba.europa.eu/regulation-and-policy/supervisory-reporting

- **SR 11-7 (Federal Reserve / OCC / FDIC 2011).** *Guidance on Model Risk Management.*
  https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm

### 9.2 · EBA-Veröffentlichungen (Datenbasis)

- **EBA Transparency Exercise 2025.** EU-wide Transparency Exercise, Reporting-Stichtag
  30. Juni 2025, veröffentlicht Dezember 2025.
  https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/eu-wide-transparency-exercise

- **EBA Risk Dashboard — Credit Risk Parameters Annex Q4 2025.** Veröffentlicht
  März 2026.
  https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/risk-dashboard

- **EBA 2025 EU-wide Stress Test — Methodological Note.** Veröffentlicht 11. November
  2024 (Kreditrisiko = Kap. 2, §2.4.2). · **Macro-financial scenario** (Jan 2025,
  §4.1.6 Long-term rates). · **Results** (Aug 2025, Fig. 22 Verlustquoten je Portfolio).
  https://www.eba.europa.eu/regulation-and-policy/stress-testing

- **EBA Reports on the Implementation of IFRS 9 by EU Banks.** Jährlich seit 2018.
  https://www.eba.europa.eu/risk-and-data-analysis/credit-risk/accounting

- **ECB SSM Supervisory Banking Statistics.** Quartalsweise.
  https://www.bankingsupervision.europa.eu/banking/statistics/html/index.en.html

### 9.3 · Akademische Literatur

**Per-Segment-β-Kalibrierung (aktuell, Stichtag 31.12.2024):**

- **Lo Duca, M., Moccero, D. & Parlapiano, F. (2024).** "The impact of macroeconomic
  and monetary policy shocks on credit risk in the euro area corporate sector."
  *ECB Working Paper Series* No 2897. — Corporate/KMU: Angebots- und
  Zinsschocks erhöhen Ausfallwahrscheinlichkeiten; Öl im Modell als Proxy
  für den Energie-/Angebotsschock.
  https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2897~449ca98c99.en.pdf

- **Konietschke, P., Metzler, J. & Ponte Marques, A. (2026).** "A quantile
  probability model for sectoral corporate defaults in Europe." *ECB Working
  Paper Series* No 3207. — Corporate/KMU-Zusatzanker: sektorale Unternehmens-
  Ausfälle unterscheiden sich im Stress-Test deutlich.
  https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3207~4ec5f4abf6.en.pdf

- **Bandoni, E., Fourné, F. & Jarmulska, B. (2025).** "Mortgage loan rates and the
  defaults of variable rate mortgages." *ECB Working Paper Series* No 3112. —
  Hypotheken-Ausfälle ↔ Zins, nichtlinear & asymmetrisch.
  https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3112~d8d7660171.en.pdf

- **ECB (2024).** *Financial Stability Review, Mai 2024.* — Energie-/Cost-of-
  Living-Schock → Haushalts-Arrears/Default (untere Einkommensquintile);
  Sektor-Heterogenität der erwarteten Default-Raten.
  https://www.ecb.europa.eu/press/financial-stability-publications/fsr/html/index.en.html

**Foundational (Öl-Makro-Transmission, Identifikation):**

- **Hamilton, J. D. (1983).** "Oil and the macroeconomy since World War II."
  *Journal of Political Economy* 91(2), S. 228–248.
  https://www.jstor.org/stable/1832055

- **Kilian, L. (2009).** "Not All Oil Price Shocks Are Alike: Disentangling Demand
  and Supply Shocks in the Crude Oil Market." *American Economic Review* 99(3),
  S. 1053–1069. DOI: 10.1257/aer.99.3.1053
  https://doi.org/10.1257/aer.99.3.1053

- **Reinhart, C. M. & Rogoff, K. S. (2009).** *This Time Is Different: Eight Centuries
  of Financial Folly.* Princeton University Press. ISBN 978-0-691-15264-6.

- **Svensson, L. E. O. (1994).** "Estimating and Interpreting Forward Interest Rates:
  Sweden 1992–1994." IMF Working Paper 94/114.
  https://www.imf.org/external/pubs/cat/longres.aspx?sk=1153

- **Tsay, R. S. (2010).** *Analysis of Financial Time Series.* 3rd edition, Wiley.
  ISBN 978-0-470-41435-4.

- **Tuckman, B. & Serrat, A. (2012).** *Fixed Income Securities: Tools for Today's
  Markets.* 3rd edition, Wiley. ISBN 978-0-470-89169-8.

- **Vasicek, O. (2002).** "Loan Portfolio Value." *Risk Magazine* 15(12), S. 160–162.
  Preprint frei verfügbar via Moody's KMV / Internet.

### 9.4 · Daten-Quellen mit Direkt-URLs

- **ICE Brent Crude Futures.** Produktbeschreibung:
  https://www.theice.com/products/219/Brent-Crude-Futures · Tagesdaten via
  yfinance-Ticker `BZ=F`.

- **Deutsche Bundesbank · Zeitreihe BBSSY01 (Svensson-Parameter).**
  https://www.bundesbank.de/dynamic/action/de/statistiken/zeitreihen-datenbanken/zeitreihen-datenbank/759778/759778

---

## 10 · Setup, Start, Tests

### 10.1 · Setup

```bash
pip install streamlit pandas pyarrow numpy scipy openpyxl yfinance plotly \
            python-docx pdfplumber
```

Python ≥ 3.10 empfohlen.

### 10.2 · Daten beschaffen

**Brent Crude (automatisch):**
```bash
python tools/data_fetch/03_fetch_brent_crude.py
```

**Bundesbank Svensson (einmalig):**
Datei von der Bundesbank-URL als CSV exportieren → `data/bundesbank_svensson.csv` →
```bash
python tools/data_fetch/04_fetch_bundesbank_svensson.py
```

**EBA Transparency 2025:** Download der CSV/XLSX-Files von
[eba.europa.eu/risk-and-data-analysis/risk-analysis/eu-wide-transparency-exercise](https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/eu-wide-transparency-exercise)
nach `data/`.

**Bank-spezifische Pillar-3 EU-CR6 (Primary Source, 31.12.2024):**
Bereits enthalten unter `data/pillar3_bank_pd_lgd.csv` (70 Zeilen,
10 Banken × 7 Klassen, alle mit `vintage_date = 2024-12-31`).
Pro Zeile ist die Quelle (Bank-Pillar-3-PDF + Seitenzahl + URL)
dokumentiert. Falls für eine spätere Vintage neu extrahiert wird:
sämtliche 10 Banken konsistent auf den neuen Stichtag bringen (sonst
schlägt der Loader-Test `_test_vintage_consistency` fehl).

**EBA Risk Dashboard Q4 2024 (Plausibilisierung):**
Die Länderaggregate werden weiterhin als externer Plausibilitätsvergleich
vorgehalten, sind aber nicht mehr als aktive PD/LGD-Fallbacks in der
Top-10-Tabelle erforderlich. Der einzige aktive Fallback ist Santander
Sovereign, weil hierfür regulatorisch keine IRB-PD/LGD publiziert wird.

### 10.3 · Cockpit starten

```bash
cd "Modellarchitektur"
python run_clean.py
```

`run_clean.py` räumt vor dem Start stale `__pycache__/` auf (häufigste Fehlerquelle
nach Page-Renames) und startet Streamlit headless auf Port 8501. Browser öffnet
automatisch `http://localhost:8501`.

### 10.4 · Backend Self-Tests

```bash
python backend/two_factor_stress.py        # 5 Tests: Klassen, NIM-Effekt, Stress-Anwendung
python backend/factor_correlation.py       # Synthetic + Real-Data-Smoke
python backend/eba_pd_loader.py            # 3 Tests: CSV-Load, Lookup, Segment-Anreicherung
python backend/svensson.py                 # 6 Tests
python backend/vasicek.py                  # 9 Tests: BCBS-Referenz-Werte
python backend/eba_loader.py               # Synthetic + Real-Loader-Smoke
python backend/backtesting.py              # 3 Tests
python backend/backtesting_walkforward.py  # 2 Tests
```

Erwartete Ausgabe pro Modul: `[PASS] All tests passed.`

---

## 11 · Bekannte Limitationen

Vollständige Liste in `MODEL_ASSUMPTIONS.md`. Highlights:

- **PD/LGD per Heimatland-Aggregat.** Der EBA-Annex aggregiert nach Counterparty-
  Land. Bank-individuelle Abweichungen (etwa BNP Paribas' US-Geschäft oder
  Santander's Lateinamerika-Geschäft) sind nicht abgebildet — die Bank wird hier
  als wäre sie ein reines Heimatland-Portfolio modelliert.
- **β-Koeffizienten aus Literatur, nicht aus Schätzung.** Eine banken- und
  periodenspezifische Re-Schätzung wäre methodisch ideal, aber durch Datenverfüg­
  barkeit eingeschränkt (Pillar-3 ist nur halbjährlich, AnaCredit ist nicht
  öffentlich zugänglich).
- **Sovereign-Buch.** Parallel-Shift-Annahme (kein Slope/Curvature-Stress); kein
  Credit-Spread-Risiko; kein Hedging.
- **Trading-Book.** Als CET1-Kanal entfernt (V2.1): EBA publiziert nur
  Bank-Aggregate ohne Issuer-Granularität oder Asset-Class-Split — keine
  belastbare FRTB-Kalibrierung möglich.
- **Walk-Forward-Backtest.** Zeigt die Vorgänger-Methodik (Single-Factor-M) als
  historische Diagnose, nicht das aktive 2-Faktor-Modell — eine Re-Implementierung
  des Backtests im 2-Faktor-Setup steht noch aus.
- **Kreuz-Korrelationen.** β- und γ-Werte sind als unabhängige Effekte modelliert;
  Kreuz-Terme β_oil × γ_rate etc. sind nicht berücksichtigt.

---

## Projektkontext

Studienprojekt **Quantitatives Credit-Risk-Management** · HS Düsseldorf ·
Sommersemester 2026 · Bleron Gashi · `bleron.gashi@study.hs-duesseldorf.de`.

**Versionierung:**
- **2.1 (Mai 2026)** · Top-10-Universe-Refinement, vollständige Quellen-Verifizierbarkeit
- **2.0 (Mai 2026)** · 2-Faktor-Modell nach Professor-Review
- **1.0 (April 2026)** · Single-Factor-M-Modell (abgelöst)
