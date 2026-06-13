# MODEL_ASSUMPTIONS · EU Banking Credit Stress Cockpit

> **Single Source of Truth** für alle Modell-Annahmen, Datenquellen,
> mathematischen Formeln und ihre ökonomische Begründung.
> Stand: Mai 2026 · Version 2.0 (2-Faktor-Modell nach Professor-Review)

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
| SME Corporate | +0.60 | +0.40 | +0.45 | +1.10 |
| Mortgage | +0.05 | +0.30 | +0.10 | +1.50 |
| QRRE | +0.40 | +0.15 | +0.30 | +0.50 |
| Other Retail | +0.30 | +0.25 | +0.25 | +0.80 |
| **Bank** | +0.05 | **−0.05** | +0.10 | +0.50 |
| Sovereign | 0 | 0 | 0 | 0 |

*SME-β auf ≈2× Corporate angehoben (Recalibration Juni 2026) — ECB
WP 2897 (2024) misst die KMU-Größen-Heterogenität direkt (≈2× bei
Angebots-, ≈3× bei Geldpolitik-Schock); ersetzt den älteren ≈1,5×-Wert
aus Drehmann/Juselius (2014).*

**Sektor-Differenzierung adressiert Professor-Kritik Punkt 7** ("Zinsen
hoch ≠ allgemein schlecht"): Bank-Klasse hat β_rate < 0 — steigende
Zinsen erhöhen die Net-Interest-Margin und kompensieren das
Kreditrisiko-Up.

**Warum ökonomisch:** Sektor-spezifische Wirkungen sind in der
Literatur dokumentiert:
- Energie-Schocks treffen energieintensive Industrien stärker (β_oil)
- Floating-Rate-Hypotheken reagieren stark auf Zinsen (β_rate für Mortgage)
- Banken profitieren von steigender Zinskurve über NIM (β_rate negativ)

**Quellen (Recalibration Juni 2026 — aktuell, Stichtag 31.12.2024):**
Methodischer Rahmen: EBA *2025 EU-wide Stress Test — Methodological Note*
(11 Nov 2024), Kap. 2 Credit risk, **§2.4.2 ¶120-128** (¶121: „sectoral
sensitivities applied to portfolio-level projections", konsistent mit
Richtung *und* Größenordnung der Szenario-Schocks; ¶128: LGD spiegelt den
Sicherheiten-Fair-Value-Verfall). Schock-Größe: EBA/ESRB *2025 Macro-
financial scenario* (Jan 2025), **§4.1.6** adverser 10y-Pfad. Per-Segment-β:
- **Corporate / SME** ← ECB WP 2897 (Lo Duca/Moccero/Parlapiano 2024):
  Makro- & Geldpolitik-Schocks → Corporate-PD (DE/FR/IT/ES); KMU/Mikro
  ≈2× (Angebot) bzw. ≈3× (Geldpolitik) Large-Corp. Deckt beide Faktoren ab
  (Supply-Schock ≈ Öl, Monetary ≈ Zins).
- **Mortgage** ← ECB WP 3112 (Bandoni/Fourné/Jarmulska 2025): Variable-
  Rate-Mortgage-Defaults ↔ Zins (nichtlinear/asymmetrisch); Öl-Kanal via
  ECB FSR Mai 2024 (Energie → Haushalts-Default).
- **QRRE / Other Retail** ← ECB FSR Mai 2024 (Haushalts-Schuldendienst ↔
  Zins; Energie → Konsumenten-Arrears, untere Einkommensquintile);
  EBA-2025-Results Fig. 22 (Retail = höchste projizierte Verlustquote).
- **Bank** ← EBA-2025-Methodik Kap. 4 (NII): NIM-Uplift bei +Δr → β_rate<0.
- **Sovereign** ← macro-orthogonal; Zins wirkt über den separaten Marktbuch-
  MtM-Kanal (EBA §2.4.2 ¶153). Empirisch aktuell bestätigt durch EBA-2025-
  Results Fig. 22 (Public sector = niedrigste Verlustquote). Reinhart &
  Rogoff (2009) für die fiskalische Natur von Sovereign-Defaults.
- **All-Segment-Querverankerung:** EBA *2025 EU-wide Stress Test — Results*
  (Aug 2025), Fig. 22, bestätigt die relative Sensitivitäts-Rangfolge.

**Wichtig zur Faktor-Struktur:** Der Zins-Kanal ist direkt auf aktuelle,
segment-spezifische Quellen kalibrierbar. Der Öl-Kanal wirkt im EBA/ECB-
Framework *nicht* direkt, sondern als adverser Angebotsschock (so in
WP 2897 identifiziert) — das EBA-2025-Szenario ist selbst öl-/gas-getrieben
(Energiepreis → HICP +3,9 % → Zinsen ↑ → BIP ↓). Bank-Öl bleibt indirekt
(kein direkter per-Segment-Anker), daher klein gehalten.

**Limitation:** Die β-PUBLIKATIONEN sind 2024/2025; ihre ökonometrischen
Schätzfenster sind historisch (WP 2897 und WP 3112: 2014-2019) — methodisch
unvermeidbar, da die Messung von Default-Sensitivitäten einen Default-Zyklus
mit hinreichend Ausfällen braucht. Baselines (PD/LGD) = 31.12.2024,
Schock-Größe = EBA-2025-Szenario. β sind via Sidebar-Override veränderbar.

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
- ECB/ESRB (2024). *Financial Stability Review, Mai 2024* (Energie-/
  Cost-of-Living-Schock → Haushalts-Default; Sektor-Heterogenität).
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
