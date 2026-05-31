# MODEL_ASSUMPTIONS · EU Banking Credit Stress Cockpit

> **Single Source of Truth** für alle Modell-Annahmen, Datenquellen,
> mathematischen Formeln und ihre ökonomische Begründung.
> Stand: Mai 2026 · Version 2.0 (2-Faktor-Modell nach Professor-Review)

---

## 1. Modell-Übersicht

Das Cockpit quantifiziert die Wirkung makroökonomischer Schocks auf die
regulatorische Eigenkapitalquote (CET1) der 10 größten EU-IRB-Banken
über drei separate Risiko-Channels.

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

### Drei Stress-Channels

| # | Channel | Mechanik |
|---|---------|----------|
| 1 | **Loan Book** | Sektor-differenzierte PD/LGD-Transmission via β-Sensitivitäten |
| 2 | **Sovereign Book** | Modified-Duration-MtM auf Δr_10y |
| 3 | **Trading Book** | FRTB-style Multiplier auf Market-RWA + P&L |

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

**Coverage:** 10 / 10 Banken Pillar-3-verifiziert. 65 von 70 CSV-Zellen
direkt bank-spezifisch (93 %). 5 Zellen verbleiben auf EBA-Risk-
Dashboard-Country-Aggregat bzw. F-IRB-Default, wo eine Bank eine
bestimmte Klasse nicht als IRB-Sub-total publiziert (z. B. Santander
Sovereign unter Standardised Approach).

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
| SME Corporate | +0.45 | +0.35 | +0.40 | +1.20 |
| Mortgage | +0.05 | +0.30 | +0.10 | +1.50 |
| QRRE | +0.40 | +0.15 | +0.30 | +0.50 |
| Other Retail | +0.30 | +0.25 | +0.25 | +0.80 |
| **Bank** | +0.05 | **−0.05** | +0.10 | +0.50 |
| Sovereign | 0 | 0 | 0 | 0 |

**Sektor-Differenzierung adressiert Professor-Kritik Punkt 7** ("Zinsen
hoch ≠ allgemein schlecht"): Bank-Klasse hat β_rate < 0 — steigende
Zinsen erhöhen die Net-Interest-Margin und kompensieren das
Kreditrisiko-Up.

**Warum ökonomisch:** Sektor-spezifische Wirkungen sind in der
Literatur dokumentiert:
- Energie-Schocks treffen energieintensive Industrien stärker (β_oil)
- Floating-Rate-Hypotheken reagieren stark auf Zinsen (β_rate für Mortgage)
- Banken profitieren von steigender Zinskurve über NIM (β_rate negativ)

**Quellen:**
- EBA Stress Test 2025 Methodology Note (Sept 2024), Sec. 5.3.2 (Sektor-
  PD-Elastizitäten) + Sec. 6.1-6.2 (LGD-Stress)
- Drehmann & Juselius (2014). *Evaluating early warning indicators of
  banking crises*. BIS WP 421 (SME-PD-Elastizität)
- Hosszú & Király (2018). MNB Working Papers 2018/2 (Mortgage Rate-
  Sensitivität)
- Castro (2013). *Macroeconomic determinants of credit risk in the
  banking system*. Economic Modelling 31 (Oil-Price-Transmission)
- Reinhart & Rogoff (2009). *This Time Is Different* (Sovereign-Defaults)

**Limitation:** β-Koeffizienten sind aus Literatur kalibriert. Eine
banken- und periodenspezifische Re-Schätzung wäre ideal, aber durch
Datenverfügbarkeit (Pillar-3 vs AnaCredit) eingeschränkt.

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
- D_bucket = Modified Duration (Bucket-Midpoint)
- Δr = Δr_10y_pp / 100
- EAD_bucket = EBA Item 2520810

**Warum ökonomisch:** Bond-Preis ist invers zum Zins. Längere Laufzeit
= höhere Duration = stärkere Preisreaktion auf Zinsschock.

**IFRS-9-Filter:** Nur HfT, FVTPL und FVOCI sind CET1-wirksam
(via P&L bzw OCI). AC und HtM tragen keinen direkten CET1-Effekt
(latenter Verlust verborgen).

**Limitation:** Parallel-Shift-Annahme (kein Slope/Curvature-Stress
auf Sovereign-MtM); keine Credit-Spread-Risiken; kein Hedging
(Swaps/Futures nicht in EBA-Public-Disclosure).

**Quelle:** EBA Transparency Exercise 2025, `tr_sov.csv` + Items
2520810/812/813/814/815.

---

### A-07 · Trading-Book-Channel [approximation]

**Was:** Market-RWA-Multiplier und Trading-Book-P&L-Haircut unter Stress.

**Formel (vereinfacht):**
```
RWA_market_stress = RWA_market_base · (1 + k_brent·|ΔBrent| + k_rate·|Δr_10y|)
```

mit k_brent ≈ k_rate ≈ 0.12 (FRTB-style Sensitivität).

**Quelle:** Item 2520210 (Market-RWA) und 2520311 (TB-P&L) der EBA
Transparency 2025.

**Limitation:** Bank-Aggregat-Werte, keine Issuer- oder Tranche-
Granularität (ABS/MBS in V1 nicht stress-elastisch modelliert).

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
- Castro, V. (2013). *Macroeconomic determinants of the credit risk in
  the banking system: The case of the GIPSI*. Economic Modelling 31.
- Drehmann, M. & Juselius, M. (2014). *Evaluating early warning
  indicators of banking crises*. BIS Working Paper 421.
- EBA (2025). *EU-wide Transparency Exercise 2025 — Public Disclosure*.
- EBA (2026). *Risk Dashboard Credit Risk Parameters Annex Q4 2025*.
- EBA (2024). *EU-wide Stress Test 2025 — Methodology Note*.
- EBA GL 14 (ICAAP / Stress-Testing).
- Hosszú, Zs. & Király, J. (2018). *Default risk in Hungarian household
  credit portfolios*. MNB Working Papers 2018/2.
- Reinhart, C. & Rogoff, K. (2009). *This Time Is Different: Eight
  Centuries of Financial Folly*. Princeton UP.
- SR 11-7 — Federal Reserve / OCC Supervisory Guidance on Model Risk
  Management (2011).
- Svensson, L. (1994). *Estimating and Interpreting Forward Interest
  Rates: Sweden 1992-1994*. IMF Working Paper 94/114.
- Tsay, R. (2010). *Analysis of Financial Time Series*, 3rd ed., Wiley.
- Vasicek, O. (2002). *Loan Portfolio Value*. Risk Magazine, Dec 2002.
