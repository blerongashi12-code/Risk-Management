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
| Stress-Engine | `backend/monte_carlo.py` *(noch nicht implementiert)* | Multivariate-Normal Shocks, historische Korrelations-Matrix |
| Aggregation | `backend/portfolio.py` *(noch nicht implementiert)* | Portfolio-PD, Expected Loss, Concentration |
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
