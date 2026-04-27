# Modell-Prämissen · DAX Credit Stress Cockpit

Dieses Dokument fasst alle modell-relevanten Annahmen zusammen, damit Reviewer, Auditoren und zukünftige Bearbeiter das Verhalten des Modells nachvollziehen und reproduzieren können. Jede Prämisse ist zu ihrer Quelle (Code-Ort + akademische Referenz) verlinkt.

Stand: 2026-04-26

---

## §1 Modell-Architektur

| Schicht | Modul | Methode |
|---|---|---|
| Risk-free rate | `backend/svensson.py` | Svensson (1994), Bundesbank-Parameter, tagesaktuell |
| Strukturmodell | `backend/merton.py` | Merton (1974) mit KMV-Iteration nach Crosbie/Bohn (2003) |
| Faktor-Modell | `backend/factor_model.py` | 2-Faktor-OLS: Brent Crude + Δr_10y |
| Stress-Engine | `backend/monte_carlo.py` | Multivariate-Normal Shocks, historische Korrelations-Matrix |
| Aggregation | `backend/portfolio.py` | Portfolio-EL, VaR/CVaR, HHI, Sektor-Breakdown |
| Frontend | `streamlit_app/` *(noch nicht implementiert)* | Streamlit-Cockpit |

---

## §2 Datenquellen

| Datentyp | Quelle | Modul | Update-Frequenz |
|---|---|---|---|
| DAX-40 Aktienkurse | yfinance (`*.DE`) | `01_fetch_dax40_prices.py` | täglich |
| DAX-40 Bilanzdaten | yfinance | `02_fetch_dax40_fundamentals.py` | quartalsweise |
| Brent Crude Futures | yfinance (`BZ=F`) | `03_fetch_brent_crude.py` | täglich |
| Svensson-Parameter | Bundesbank Zeitreihen-DB | `04_fetch_bundesbank_svensson.py` | täglich |
| DAX-Index | yfinance (`^GDAXI`) | `05_fetch_market_proxy.py` | täglich |

Lookback-Zeitraum: 6 Jahre (`config.LOOKBACK_YEARS = 6`).

---

## §3 Default Point (DPT)

**Prämisse:** Die Default-Schwelle einer Firma ist *nicht* die gesamte Verschuldung, sondern der kurzfristig fällige Anteil zuzüglich eines Anteils der langfristigen Schulden:

$$\text{DPT} = \text{ShortTermDebt} + \alpha \cdot \text{LongTermDebt}, \qquad \alpha = 0{,}5$$

| Parameter | Wert | Konfiguration |
|---|---|---|
| `α` (LTD-Gewicht) | **0.5** | `config.DPT_LTD_WEIGHT` |

**Begründung:** Langfristige Schulden lösen nicht akut einen Default aus — nur ein Anteil zählt zur tatsächlichen Default-Schwelle. Dieser Standard geht auf Moody's KMV CreditEdge zurück und ist in der akademischen Literatur etabliert.

**Quellen:**
- Crosbie, P. & Bohn, J. (2003). *Modeling Default Risk*. Moody's KMV Corporation.
- Bharath, S.T. & Shumway, T. (2008). *Forecasting Default with the Merton Distance to Default Model.* Review of Financial Studies 21(3), 1339–1369.

**Fallback-Logik** (`merton._compute_default_point`):
1. Wenn `ShortTermDebt` und `LongTermDebt` beide vorhanden → DPT-Formel.
2. Sonst → `TotalDebt` als konservative Obergrenze (im Output mit `DebtSource = "TotalDebt"` markiert).
3. Sonst → Skip mit `DebtSource = "skipped: no debt"`.

---

## §4 Sektor-σ_V-Multiplier

**Prämisse:** Standard-Merton untertreibt die PD bei stark geleveragten Sektoren (Banken, REITs), weil die beobachtete Equity-Volatilität strukturell gedämpft ist (Einlagensicherung, Notenbank-Liquidität, regulatorische Eigenkapital-Floor). Korrektur: nach KMV-Lösung wird die Asset-Volatilität sektor-spezifisch multipliziert, DD und PD werden mit der angepassten Vola neu berechnet.

$$\sigma_V^{\text{adj}} = m_{\text{sector}} \cdot \sigma_V$$

| Sektor (Yahoo) | Multiplier `m` | Konfiguration |
|---|---|---|
| Financial Services | **1.5** | `config.SECTOR_VOL_MULTIPLIER["Financial Services"]` |
| Real Estate | **1.2** | `config.SECTOR_VOL_MULTIPLIER["Real Estate"]` |
| alle anderen | **1.0** | `config.DEFAULT_SECTOR_VOL_MULTIPLIER` |

**Begründung der Werte:**
- *Financial Services (1.5):* In US-Bank-Stresstests (Hovakimian/Kane/Laeven 2012) und Moody's KMV-CreditEdge-Kalibrierung sind Bank-Asset-Volas typisch 30–60% höher als das Merton-Modell direkt liefert.
- *Real Estate (1.2):* REITs zeigen ähnliche Bilanz-Hebelung, aber weniger ausgeprägt als bei Banken.
- *Alle anderen Sektoren (1.0):* Standard-Merton, kein Multiplier.

**Quellen:**
- Hovakimian, A., Kane, E.J. & Laeven, L. (2012). *Variation in systemic risk at US banks during 1974–2010.* NBER WP 18043.
- Crosbie & Bohn (2003), KMV CreditEdge Methodology Document.

**Effekt im DAX-40-Run (Stand 2026-04-25):**

| Ticker | Name | PD ohne Multiplier | PD mit Multiplier (1.5) |
|---|---|---|---|
| CBK.DE | Commerzbank | 0.0455% | **1.594%** |
| DBK.DE | Deutsche Bank | 0.0048% | **0.538%** |

Die korrigierten Werte liegen im plausiblen Bereich für 1Y-Default-Rates von Banken im BBB-Spektrum.

---

## §4a Faktor-Modell (Brent + Δr_10y)

**Spezifikation:** Pro Firma wird eine OLS-Regression über die Equity-Log-Returns geschätzt:

$$r_{\text{equity},t} = \alpha + \beta_E \cdot r_{\text{Brent},t} + \beta_R \cdot \Delta r_{10y,t} + \varepsilon_t$$

| Parameter | Wert | Konfiguration |
|---|---|---|
| Lookback | 252 Handelstage (≈ 1 Jahr) | `config.FACTOR_LOOKBACK_DAYS` |
| Δr-Maturity | 10 Jahre | `config.FACTOR_MATURITY` |
| Min-Obs | 60 | `config.FACTOR_MIN_OBS` |
| Lookback-Konsistenz | identisch zu Merton (§5) | bewusst gleich gewählt |
| OLS-Engine | `numpy.linalg.lstsq` | kein scipy/statsmodels |

**Faktor-Konstruktion** (`factor_model.factor_returns`):
- `r_brent`: tägliche Log-Returns aus `brent_crude.parquet` (Spalte `Return_log`, bereits in der Data-Layer-Pipeline berechnet)
- `Δr_10y`: $r_{10y}(t) - r_{10y}(t-1)$, ausgewertet mit der Svensson-Funktion bei τ = 10 Jahren, in Prozentpunkten

**Sektor-Multiplier auf β_E:** Siehe §4.5.

## §4.5 Energy-Beta-Multiplier (Sektor-Skalierung von β_E)

**Prämisse:** Die OLS-Schätzung von β_E (Brent-Sensitivität) auf einer 252-Tages-Stichprobe enthält statistisches Rauschen und reflektiert nur den realisierten Stress-Pfad. Für ein **Stress-Cockpit** ist eine fachlich fundierte Skalierung sinnvoll, die die strukturelle Energie-Exposition einer Branche abbildet.

**Methodik:**

$$\text{EnergyMul}_{\text{sector}} = \text{clip}\!\left( \frac{(E/U)_{\text{sector}}}{(E/U)_{\text{Industrials}}}, \; 0{,}1, \; 4{,}0 \right)$$

mit $E/U$ = Energieaufwand / Umsatz beim DAX-Benchmark-Unternehmen des Sektors. Industrials (Siemens, $E/U \approx 5\%$) dient als Referenz mit Multiplier 1.0. Cap [0.1, 4.0] verhindert Extremwerte bei sehr energieintensiven Sektoren (Utilities, Energy).

**Anwendung im Code** (`factor_model.run_dax40`):
```python
beta_brent_adjusted = beta_brent_raw * EnergyMul[sector]
```
Beide Werte (`BetaBrentRaw` und `BetaBrentAdjusted`) erscheinen im Output, damit der Multiplier-Effekt in der Streamlit-Frontend transparent dargestellt werden kann.

**First-Cut-Werte** (Stand: 2026-04-26):

| Yahoo-Sektor | DAX-Benchmark | $E/U$ (geschätzt) | Multiplier | Quelle / Begründung |
|---|---|---|---|---|
| Utilities | RWE | ~35 % | **4.0** (cap) | RWE GB, Brennstoff-/Energiebezugskosten relativ Umsatz; reine Stromerzeugung ist 1.-energieintensivste Branche |
| Energy | (keine reine in DAX) | ~50 % | **4.0** (cap) | Branchenstandard EIA / IEA |
| Basic Materials | BASF | ~12 % | **2.4** | VCI Energiekostenmonitor; Chemie ist 2.-energieintensivste Branche |
| Industrials | Siemens | ~5 % | **1.0** | Siemens GB; **Referenz-Sektor** |
| Consumer Cyclical | BMW | ~4 % | **0.8** | BMW GB, Material- + Energieaufwand |
| Communication Services | Deutsche Telekom | ~3 % | **0.6** | DT GB, Energiekosten Rechenzentren |
| Healthcare | Merck | ~2.5 % | **0.5** | Merck GB |
| Real Estate | Vonovia | ~2 % | **0.4** | Vonovia GB; Heizenergie wird grossteils umgelegt |
| Consumer Defensive | Henkel | ~2 % | **0.4** | Henkel GB |
| Technology | SAP | ~1.5 % | **0.3** | SAP GB, Rechenzentrum-Strom |
| Financial Services | Deutsche Bank | ~0.5 % | **0.1** | DB GB, reine Verwaltungs-/RZ-Stromkosten |

> ⚠️ **Status: First-Cut.** Die $E/U$-Werte sind aus öffentlich bekannten Sektorstudien und Geschäftsberichts-Auszügen abgeleitet, **noch nicht** durch direkte Recherche der spezifischen DAX-Geschäftsberichte verifiziert. Spätere Kalibrierung ist ausdrücklich vorgesehen — alle Werte sind in `config.SECTOR_ENERGY_MUL` zentral änderbar, ohne Code-Anpassung.

**Vorgehen für die Kalibrierung** (für zukünftige Bearbeiter):
1. Aus dem aktuellen Geschäftsbericht des Benchmark-Unternehmens den Posten *Materialaufwand → Energiekosten* (oder *Sonstige Aufwendungen → Energiebezug*) extrahieren.
2. $E/U$ = Energiekosten / Umsatzerlöse berechnen.
3. Multiplier = $\text{clip}( E/U_{\text{sector}} / 0.05, 0.1, 4.0 )$.
4. Wert in `config.SECTOR_ENERGY_MUL` aktualisieren, Quelle (GB-Seite) im Kommentar dokumentieren.
5. `python backend/factor_model.py` zur Verifikation laufen lassen.

## §5 Equity-Volatilitäts-Schätzung

**Methode:** Annualisierte Standardabweichung der täglichen Log-Returns über die letzten `lookback` Handelstage.

$$\sigma_E = \text{std}\!\left(\ln \frac{P_t}{P_{t-1}}\right) \cdot \sqrt{252}$$

| Parameter | Wert | Konfiguration |
|---|---|---|
| Lookback | 252 Handelstage (≈ 1 Jahr) | `merton.equity_vol_from_prices(lookback=…)` |
| Annualisierungs-Faktor | √252 | `merton.equity_vol_from_prices(trading_days_per_year=…)` |
| Stichproben-σ | `ddof=1` (unbiased) | hardcoded |

**Bekannte Vereinfachung:** Es wird *nicht* GARCH/EWMA modelliert. Bei Bedarf ist das Modul leicht erweiterbar.

---

## §6 Risk-free rate für Merton

**Methode:** Svensson-Zero-Rate auf den letzten verfügbaren Bundesbank-Handelstag, ausgewertet bei der Modell-Horizon `T`.

```python
params = svensson.historical_curve(as_of, svensson_df, method="ffill")
r      = svensson.zero_rate(T, params, as_decimal=True)
```

| Parameter | Wert | Konfiguration |
|---|---|---|
| Horizon `T` | 1 Jahr | `config.DEFAULT_HORIZON` |
| `method` | `"ffill"` (letzter Handelstag) | `merton.run_dax40` |
| Validierung | 48 060 Punkte gegen Excel-Sheet | max abs error 2.66e-14 % |

---

## §6a Monte-Carlo-Stress-Engine

**Ziel:** Statistische Verteilung der Stress-PD pro Firma über N stochastische Pfade. Im Gegensatz zu deterministischen Szenarien (Corona/Ukraine — kommt in `scenarios.py`) sind hier **alle Pfade Realisationen aus einer multivariaten Normal-Verteilung**, geschätzt aus der historischen Faktor-Statistik.

### §6a.1 Schock-Vektor und Verteilung

$$\mathbf{z}_t = \begin{bmatrix} r_{\text{Brent}, t} \\ \Delta r_{10y, t} \end{bmatrix} \sim \mathcal{N}\bigl(\boldsymbol{\mu}, \boldsymbol{\Sigma}\bigr)$$

| Komponente | Konstruktion | Quelle |
|---|---|---|
| $\boldsymbol{\mu}$ | tägliches Mean der Faktor-Returns | `factor_model.factor_returns` über letzte 252 Tage |
| $\boldsymbol{\Sigma}$ | tägliche Kovarianzmatrix | dito |
| $\mathbf{u}_t$ | iid Standard-Normal | `numpy.random.Generator` |
| $L L^T = \boldsymbol{\Sigma}$ | Cholesky | `np.linalg.cholesky` |
| $\mathbf{z}_t = \boldsymbol{\mu} + L \mathbf{u}_t$ | korrelierte tägliche Schocks | vektorisiert |

**Pfad am Horizont H** (Summen über H iid Tage):
$$\mathbf{Z}_H = \sum_{t=1}^H \mathbf{z}_t \quad \Rightarrow \quad \mathbf{Z}_H \sim \mathcal{N}(H\boldsymbol{\mu}, H\boldsymbol{\Sigma})$$

### §6a.2 Anwendung pro Firma & Pfad

Für jede Firma mit Baseline-Inputs $(E_0, \sigma_E, L, r_0, T)$ und Faktor-Modell-Outputs $(\alpha, \beta_E^{\text{adj}}, \beta_R, \sigma_\varepsilon)$ wird pro Pfad $i$ folgendes berechnet:

$$\log r_{E,H}^{(i)} = \alpha \cdot H + \beta_E^{\text{adj}} \cdot Z_{\text{Brent}}^{(i)} + \beta_R \cdot Z_{\Delta r}^{(i)} + \varepsilon^{(i)} \cdot \sqrt{H}$$

mit $\varepsilon^{(i)} \sim \mathcal{N}(0, \sigma_\varepsilon)$ wenn `MC_INCLUDE_IDIO=True`.

**Stressed Inputs für KMV:**
- $E_{\text{stress}}^{(i)} = E_0 \cdot \exp\bigl(\log r_{E,H}^{(i)}\bigr)$
- $r_{\text{stress}}^{(i)} = r_0 + Z_{\Delta r}^{(i)} / 100$ (pp → Dezimal)
- $\sigma_E$ und $L$ bleiben unverändert (Vola-Stress in V2)

→ Vektorisiertes KMV (`monte_carlo._kmv_vec`) löst $V^{(i)}, \sigma_V^{(i)}$ über alle Pfade gleichzeitig.

→ Anschließend Sektor-σ_V-Multiplier wie in §4 (Banken 1.5, REITs 1.2): $\sigma_V^{\text{adj}} = m_{\text{sector}} \cdot \sigma_V$.

→ Final $\text{DD}^{(i)} = \frac{\ln(V^{(i)}/L) + (r_{\text{stress}}^{(i)} - \tfrac{1}{2}\sigma_V^{\text{adj},(i)2}) T}{\sigma_V^{\text{adj},(i)} \sqrt{T}}$, $\text{PD}^{(i)} = N(-\text{DD}^{(i)})$.

### §6a.3 Konsistenz mit Baseline

Die Multiplier (Sektor-Energy auf $\beta_E$, Sektor-Vola auf $\sigma_V$) werden **identisch** zur Baseline angewendet — sonst wäre das Stress-Ergebnis inkonsistent zwischen den beiden Pipeline-Pfaden.

### §6a.4 Mean vs. Quantile — wichtige Interpretations-Anmerkung

Das **Mean** der Stress-PD-Verteilung ist **kein adverser Stress-Indikator** — bei positivem historischem μ kann Mean(Stress-PD) sogar unter der Baseline-PD liegen (Drift-Effekt). Die **adversen Indikatoren** sind die **Quantile**:

| Statistik | Bedeutung |
|---|---|
| `StressPDMean` | Erwartungswert über alle Pfade — zentrale Tendenz |
| `StressPDp50` | Median |
| `StressPDp95` | 95%-Quantil — moderater Stress |
| `StressPDp99` | **99%-Quantil — adverser 1%-Worst-Case-Pfad** |
| `StressPDMax` | Maximum über N Pfade |

Beim DAX-40-Run (Stand 2026-04-26):

| Ticker | Baseline-PD | Stress-p99 | Δ(p99) |
|---|---|---|---|
| CBK.DE | 1.59 % | 2.04 % | **+0.45 pp** |
| DBK.DE | 0.54 % | 0.71 % | **+0.17 pp** |
| VNA.DE | 0.003 % | 0.020 % | +0.017 pp |

Die p99-Werte zeigen die richtige Stress-Asymmetrie.

### §6a.5 Konfiguration und Reproduzierbarkeit

| Parameter | Wert | Konfiguration |
|---|---|---|
| `MC_N_SIMS` | 10 000 | `config.py` |
| `MC_HORIZON_DAYS` | 252 (1Y) | `config.py` |
| `MC_SEED` | 42 | `config.py` |
| `MC_INCLUDE_IDIO` | True | `config.py` |
| **Reproduzierbarkeit** | bit-identisch bei gleichem Seed | Test [3] in `monte_carlo.py` |
| **Performance** | ~0.5 s für 38 Firmen × 10k Pfade × H=252 | vektorisiert via Cholesky + KMV-vec |

### §6a.6 Bekannte Limitierungen

| Limit | Status |
|---|---|
| Vola-Schock auf $\sigma_E$ | nicht modelliert (V2) |
| Fat-Tails (t-Verteilung) | nicht modelliert; Standard-MVN |
| Regime-Wechsel | keine Markov-Chain |
| Asymmetrische Schocks | keine GARCH/Sprung-Komponenten |
| Drift-freie Variante | Toggle in V2 möglich (`drift_mode='zero'`) |

## §6b Portfolio-Aggregation

Aggregiert Einzelfirmen-PDs zu Portfolio-Risiko-Metriken.

### §6b.1 Exposure-at-Default (EAD)

$$\text{EAD}_i = \text{DPT}_i = \text{ShortTermDebt}_i + 0.5 \cdot \text{LongTermDebt}_i$$

**Begründung:** Konsistent mit der Default-Schwelle aus §3 — was im Default ökonomisch verloren geht, ist die Default-Punkt-Verschuldung. Konfigurierbar über `ead_col`-Parameter (alternativ: `MarketCap`, `TotalDebt`).

### §6b.2 Loss Given Default (LGD)

$$\text{LGD} = 0{,}45$$

**Quelle:** Basel-II/III-Standard für Senior Unsecured Corporate Debt. Konstant für alle Sektoren in V1; sektor-spezifische LGD ist in V2 vorgesehen.

### §6b.3 Expected Loss

$$\text{EL} = \sum_i \text{EAD}_i \cdot \text{LGD} \cdot \text{PD}_i$$

Ausgewertet drei Mal: mit Baseline-PD, Stress-Mean-PD und Stress-p99-PD. Letzteres ist der **adverse-Stress-Indikator**.

### §6b.4 Loss-Distribution (Conditional EL pro Pfad)

$$L^{(i)} = \sum_j \text{EAD}_j \cdot \text{LGD} \cdot \text{PD}_j^{(i)}$$

mit $\text{PD}_j^{(i)}$ = realisierte PD von Firma $j$ auf MC-Pfad $i$ (aus `monte_carlo.stress_one_firm` mit `keep_samples=True`).

**Korrelation zwischen Firmen** entsteht **automatisch** über die gemeinsamen Faktoren (Brent, Δr_10y) im MC — keine zusätzliche Copula nötig.

**Optional: Bernoulli-Sampling** (`sample_defaults=True`) erzeugt zusätzlich diskrete Default-Events $D_j^{(i)} \sim \text{Bernoulli}(\text{PD}_j^{(i)})$. Liefert echte Loss-Sprünge (höhere Varianz im Tail), aber rauschiger. Default ist die **Conditional-EL-Variante**.

### §6b.5 VaR und CVaR

$$\text{VaR}_\alpha = \text{Quantil}_\alpha(L), \qquad \text{CVaR}_\alpha = \mathbb{E}[L \mid L \geq \text{VaR}_\alpha]$$

Berechnet für $\alpha \in \{0.95, 0.99\}$. CVaR (= Expected Shortfall) ist die kohärente Risiko-Metrik nach Basel III FRTB.

### §6b.6 Konzentrations-Metriken

$$\text{HHI} = \sum_i w_i^2, \quad w_i = \text{EAD}_i / \sum_j \text{EAD}_j$$

$$N_{\text{eff}} = 1/\text{HHI}$$

Bei perfekter Gleichgewichtung: $N_{\text{eff}} = N$. Bei vollständiger Konzentration: $N_{\text{eff}} = 1$.

### §6b.7 Sektor-Breakdown

Pro Yahoo-Sektor werden aggregiert: Anzahl Firmen, EAD, EAD-Anteil, EL-Baseline, EL-Stress-Mean, EL-Stress-p99. Ermöglicht Sektor-Konzentrations-Analyse im Streamlit-Frontend.

### §6b.8 Numerische Resultate (DAX-40, Stand 2026-04-26)

| Metrik | Wert |
|---|---|
| Konvergierte Firmen | 36 / 38 |
| Total EAD (Σ DPT) | 650.2 bn EUR |
| LGD | 45 % |
| **EL Baseline** | **611 m EUR (0.094 %)** |
| EL Stress (Mean) | 365 m EUR (0.056 %) |
| **EL Stress (p99)** | **794 m EUR (0.122 %)** |
| VaR 95 | 581 m EUR |
| CVaR 95 | 636 m EUR |
| VaR 99 | 671 m EUR |
| **CVaR 99** | **719 m EUR** |
| HHI | 0.0943 |
| Effective N | 10.6 (von 36) |

**Sektor-Konzentration (Top-3 nach EAD):**

| Sektor | N | EAD-Anteil | EL Baseline | EL p99 |
|---|---|---|---|---|
| Consumer Cyclical | 8 | 39.0 % | 0.6 m | 2.2 m |
| Financial Services | 6 | 30.2 % | **609.9 m** | **789.7 m** |
| Communication Services | 1 | 8.6 % | 0.0 m | 0.0 m |

**Beobachtung:** Trotz dass Consumer Cyclical der EAD-grösste Sektor ist (39 %), trägt **Financial Services 99.8 % des EL** (610 m von 611 m). Das ist die Konsequenz aus den Bank-Multipliern (§4) — fachlich korrekt: der Stress-Test misst Risiko, nicht Größe. Banken haben strukturell höhere PDs als Industriefirmen.

## §6c Szenario-Library (deterministische Stress-Pfade)

Im Gegensatz zur stochastischen MC-Engine (§6a) werden hier **deterministische Schock-Vektoren** auf die gleiche Faktor → Equity → KMV-Pipeline angewendet. Jedes Szenario reduziert sich auf einen **Single-Pfad**:

$$\mathbf{z}_{\text{scenario}} = \begin{bmatrix} \Delta\!\log P_{\text{Brent}} \\ \Delta r_{10y} \;[\text{pp}] \end{bmatrix}, \quad H \;[\text{Tage}]$$

### §6c.1 Szenario-Quellen-Hierarchie

| `source` | Bedeutung | Anwendung |
|---|---|---|
| `historical` | Kalibrierung aus realen Brent + Bundesbank-Daten über Periode `(start, end)` | für Episoden, die im Daten-Lookback liegen |
| `literature` | Hartkodierte Werte aus akademischer Stress-Test-Literatur | für Episoden **vor** dem Daten-Lookback (z.B. Corona-Crash 2020-Q1, Brent-Daten beginnen erst 2020-04-23) |
| `hypothetical` | Forward-looking Szenarien | What-if-Analysen |

### §6c.2 Library-Inhalt (Stand 2026-04-27)

| Szenario | Source | ΔBrent_log | Δr_10y | H | Bemerkung |
|---|---|---|---|---|---|
| `corona_2020` | literature | −1.20 (≈ −70%) | −65 bp | 60 d | Brent ~$66 → ~$20 (Jan-Apr 2020), 10y Bund −65 bp |
| `ukraine_2022` | historical | **+0.12** | **+67 bp** | 66 d | aus Bundesbank/Brent-Daten 2022-02-22 bis 2022-04-30 |
| `ukraine_2022_peak` | historical | **+0.28** | **−15 bp** | 14 d | Initial-Peak Brent bis 2022-03-08 |
| `iran_2026` | hypothetical | +0.55 (≈ +73%) | +50 bp | 30 d | Supply-Schock + ECB cautious |

### §6c.3 Anwendung pro Firma

Identisch zu `monte_carlo.stress_one_firm` mit `Z_brent = ΔBrent_log` und `Z_Δr = Δr_pp`, **ohne** ε-Drift (deterministisch):

$$\log r_E^{(\text{scen})} = \alpha \cdot H + \beta_E^{\text{adj}} \cdot \Delta\!\log P_{\text{Brent}} + \beta_R \cdot \Delta r_{10y}$$

Das Output-DataFrame enthält die **Decomposition** (`AlphaContrib`, `BrentContrib`, `RateContrib`) für eine Waterfall-Visualisierung im Streamlit.

### §6c.4 Bekannte Modell-Limitation bei Bank-Szenarien

**Beobachtung (Stand 2026-04-27):** Im DAX-40-Run zeigen Banken (CBK, DBK) bei Corona/Ukraine **niedrigere** Stress-PDs als Baseline.

| Ticker | Baseline | corona_2020 | ukraine_2022 | iran_2026 |
|---|---|---|---|---|
| CBK.DE | 1.59 % | 1.39 % | 1.39 % | 1.52 % |
| DBK.DE | 0.54 % | 0.47 % | 0.47 % | 0.51 % |

**Mechanik:**
1. $\beta_R > 0$ für Banken (Zinsanstieg → Margenexpansion → höherer Aktienkurs).
2. $\beta_E^{\text{adj}}$ für Banken ist klein ($\text{mul} = 0.1$, §4.5) — Brent-Schock ist quasi wirkungslos.
3. Direkt-Effekt im Merton: bei steigendem $r$ sinkt $L \cdot e^{-rT}$ → $V/L$ erhöht sich → PD sinkt.

**Was das Modell NICHT erfasst** (Banken-spezifisch):
- Kreditausfall-Wellen bei Stress (Corporate Default Domino)
- Counterparty- und Liquiditätsrisiken
- Vola-Sprünge bei Krisen (Equity-Vola steigt typisch um 2–3× im Stress)
- Regulatorische Eingriffe / Notenbank-Liquiditäts-Push

**Konsequenz:** Bank-Szenarien sind im Cockpit als **modell-konsistente Markt-Reaktion** zu lesen, nicht als „echtes" Bankenkrisen-Szenario. Für letzteres wäre ein anderes Modell (z.B. SRISK oder Bank-spezifisches CCAR-Framework) nötig — bewusst ausserhalb des Scope dieses Cockpits.

## §7 KMV-Iteration

**Fixpunkt-System** (für jede Firma einzeln gelöst):

$$E = V \cdot N(d_1) - L \cdot e^{-rT} \cdot N(d_2), \qquad \sigma_E \cdot E = \sigma_V \cdot V \cdot N(d_1)$$

mit $d_1 = \frac{\ln(V/L) + (r + \tfrac{1}{2}\sigma_V^2)T}{\sigma_V\sqrt{T}}$, $d_2 = d_1 - \sigma_V\sqrt{T}$.

| Parameter | Wert |
|---|---|
| Init | `V₀ = E + L`, `σ_V0 = σ_E · E / V₀` |
| Konvergenz-Toleranz | `tol = 1e-8` (relative Änderung in V und σ_V) |
| Max-Iterationen | `max_iter = 200` |
| Beobachtete Konvergenz-Rate | < 50 Iter für realistische Inputs |

**Output:**
$$\text{DD} = \frac{\ln(V/L) + (r - \tfrac{1}{2}\sigma_V^2)T}{\sigma_V\sqrt{T}}, \qquad \text{PD} = N(-\text{DD})$$

---

## §8 Bekannte Limitationen

| Limit | Beschreibung | Mitigation |
|---|---|---|
| **Banken-Eignung** | Merton ist für Banken konzeptuell ungeeignet (Einlagen ≠ klassische Schuldverpflichtungen). | Sektor-Multiplier 1.5 — pragmatischer Workaround, kein Ersatz für ein bankspezifisches Modell. |
| **yfinance Datenqualität** | Lückenhafte Bilanzdaten; einzelne Firmen ohne sauberen `TotalDebt`-Eintrag. | Skip-Mechanismus mit Tagging in `DebtSource`. Aktuell betroffen (2026-04): EOAN.DE, 1COV.DE. |
| **Konstante Vola** | Kein zeitvariabler Vola-Prozess (GARCH/EWMA). | Sample-σ über 252 Tage als robuste Approximation. |
| **Single-Snapshot** | Bilanzdaten sind quartalsweise; PD wird gegen aktuelle Marktdaten gerechnet → Mismatch in Stichtagen. | Akzeptiert, weil die Stress-Szenarien dieselbe Asymmetrie haben. |
| **Fehlende Recovery-Modellierung** | LGD ist konstant (Basel-Standard 0.45); kein Sektor-spezifisches Recovery. | `config.DEFAULT_LGD` ist override-bar im Portfolio-Aggregator. |
| **Keine Korrelationsstruktur firmenseitig** | Merton wird firma-individuell gelöst, ohne zwischen-Firmen-Korrelationen. | Korrelationen kommen über das Faktor-Modell (Brent + Δr) und MC-Engine. |

---

## §9 Reproduzierbarkeit

Jedes Backend-Modul hat einen `__main__`-Block mit Validierungstests, der ohne Argumente läuft:

```bash
cd "Risk management"
python backend/svensson.py     # 6 Test-Blöcke (Spot, Limits, Shifts, Lookup, Excel, Round-Trip)
python backend/merton.py       # 6 Test-Blöcke (Round-Trip, Hull, Monoton, Boundary, Multiplier, DAX)
```

Erwartete Ausgabe: `[PASS] Alle N Test-Blöcke bestanden.`

**Excel-Validierung Svensson** (`Risk Free Rates`-Sheet):
- 48 060 Datenpunkte verglichen
- Max abs error: 2.66e-14 %  → mathematisch identisch zur Excel-Formel

**Hull-Spot-Test Merton** (Hull, *Risk Management & Financial Institutions*, 4. Aufl.):
- E=3, L=10, σ_E=0.80, r=0.05, T=1
- Erwartung: V≈12.40, PD ≈ 12–13 %
- Beobachtet: V=12.395, PD=12.70 % ✓

---

## §10 Änderungs-Log

| Datum | Änderung | Begründung |
|---|---|---|
| 2026-04-25 | `svensson.py` initial | Excel-Sweep validiert |
| 2026-04-25 | `merton.py` initial (TotalDebt, kein Multiplier) | Erste KMV-Implementierung |
| 2026-04-26 | DPT-Switch (Moody's KMV Standard) + Sektor-σ_V-Multiplier (1.5 Banken / 1.2 REITs) | Akademisch sauberere Default-Schwelle; Banken-PDs realistisch nach §4 |
| 2026-04-26 | `factor_model.py` initial (2-Faktor: Brent + Δr_10y) + Sektor-Energy-Multiplier nach E/U-Methodik (§4.5) | Stress-Cockpit braucht strukturelle Energie-Exposition pro Sektor |
| 2026-04-26 | `monte_carlo.py` initial (MVN Pfade × vektorisierter KMV) | Stochastische Stress-Distribution mit Konsistenz zu Sektor-Multipliern |
| 2026-04-27 | `portfolio.py` initial (EL/VaR/CVaR/HHI + Sektor-Breakdown, EAD = DPT, LGD = 45 %) | Aggregations-Layer: Einzelfirmen-PDs zu Portfolio-Metriken |
| 2026-04-27 | `scenarios.py` initial (Library: corona_2020, ukraine_2022, ukraine_2022_peak, iran_2026; historical/literature/hypothetical Quellen) | Deterministische Stress-Pfade mit Waterfall-Decomposition; Modell-Limitation bei Banken in §6c.4 dokumentiert |
