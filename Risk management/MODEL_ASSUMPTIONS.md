# Model Assumptions · EU Banking Credit Stress Cockpit

This document records every assumption, parameter, formula and known limitation of the cockpit, with source references. It is the single source of truth for model behaviour. Each premise links to its code location and the academic / regulatory anchor.

Vintage: 2026-04-30

---

## §1 Architecture

| Layer | Module | Method |
|---|---|---|
| Risk-free curve | `backend/svensson.py` | Svensson (1994), daily Bundesbank parameters |
| Macro factor mapping | `backend/macro_factor.py` | (Brent, Δr_10y) → Vasicek systematic factor M |
| Loan-book engine | `backend/vasicek.py` | Vasicek single-factor (ASRF), Basel-III IRB capital formulas |
| EBA universe + sovereign | `backend/eba_loader.py` | Parses `tr_cre.csv` (loan-book IRB) + `tr_sov.csv` (sovereign book) |
| Frontend | `streamlit_app/` | Streamlit cockpit · McKinsey aesthetic |

Data inputs:

| Type | Source | Frequency |
|---|---|---|
| Brent Crude futures | yfinance (`BZ=F`) | daily |
| Svensson zero-curve | Bundesbank time-series database | daily |
| EBA Transparency Exercise 2025 | EBA public CSV (`tr_cre.csv`, `tr_sov.csv`, `TR_Metadata.xlsx`, `SDD.xlsx`) | annual (Dec release) |

---

## §2 Bundesbank Svensson Curve

**Function:** Svensson (1994) extension of the Nelson-Siegel zero-coupon curve:

$$y(\tau) = \beta_0 + \beta_1 \cdot \frac{1 - e^{-\tau/\tau_1}}{\tau/\tau_1} + \beta_2 \cdot \!\left[\frac{1 - e^{-\tau/\tau_1}}{\tau/\tau_1} - e^{-\tau/\tau_1}\right] + \beta_3 \cdot \!\left[\frac{1 - e^{-\tau/\tau_2}}{\tau/\tau_2} - e^{-\tau/\tau_2}\right]$$

Parameters $\beta_0, \beta_1, \beta_2, \beta_3, \tau_1, \tau_2$ are published daily by the Bundesbank.

**Validation:** 48 060 historical observations cross-checked against an Excel-sheet reference; max absolute error $2{,}66 \times 10^{-14}\%$ — mathematically identical to the published formula.

**Use in the cockpit:**

| Use | Module |
|---|---|
| Δr_10y daily diff over 252 days | `macro_factor.factor_returns` |
| Cumulative Δr_10y under live β-shifts | `data_loader.delta_r_from_beta_shifts` |
| Multi-maturity rate KPI strip | Yield-Curve page |
| Discount factor in Sovereign duration P&L | implicit in modified-duration approximation |

---

## §3 Macro Factor Mapping (Macro → M)

**Bridge:** the (ΔBrent, Δr_10y) shock vector is mapped onto the Vasicek systematic factor $M \sim \mathcal{N}(0,1)$ that drives §4's conditional PDs. Two methodologically independent routes are computed and a hybrid is the default.

### §3.1 Anchor route — primary, regulator-citable

Use the EBA 2025 stress-test adverse anchor:

$$\bigl(\Delta\!\log P_{\text{Brent}}^{\text{anchor}}, \Delta r_{10y}^{\text{anchor}}, M^{\text{anchor}}\bigr) = (+0{,}47, +200\,\text{bp}, -2{,}5)$$

A new shock vector projects onto the anchor:

$$M_{\text{anchor}} = M^{\text{anchor}} \cdot \cos(\angle\,\text{shock, anchor}) \cdot \frac{\|\text{shock}\|}{\|\text{anchor}\|}$$

Properties:
- self-consistency: shock = anchor → $M = M^{\text{anchor}}$
- linearity: 2× shock → 2× $M$
- sign-flip: anti-aligned shock (Brent crash + rate cut) → $M > 0$ (benign)

### §3.2 Data route — empirical, validation

Mahalanobis distance of the shock in the empirical 252-day Brent + Δr_10y covariance, signed by alignment with the adverse direction $(+1, +1)/\sqrt{2}$:

$$M_{\text{data}} = -\sqrt{\text{shock}^\top \boldsymbol{\Sigma}_h^{-1}\text{shock}} \cdot \cos(\angle\,\text{shock, adverse})$$

where $\boldsymbol{\Sigma}_h = H \cdot \boldsymbol{\Sigma}_{\text{daily}}$ is the daily covariance scaled to the horizon.

### §3.3 Hybrid route — default

$M_{\text{hybrid}} = \tfrac{1}{2}(M_{\text{anchor}} + M_{\text{data}})$. The cockpit displays all three values and the consistency $|M_{\text{anchor}} - M_{\text{data}}|$ as a diagnostic.

**Empirical baseline:** with 252-day Bundesbank + Brent data, $\hat\rho(\text{Brent}, \Delta r_{10y}) = +0{,}20$. The anchor and data routes agree to within ~0.5σ on the EBA adverse anchor itself.

---

## §4 Vasicek / ASRF Engine

### §4.1 Asset model

$$A_i = \sqrt{\rho}\,M + \sqrt{1-\rho}\,\varepsilon_i, \quad M, \varepsilon_i \stackrel{iid}{\sim} \mathcal{N}(0,1)$$

Default occurs if $A_i < N^{-1}(\text{PD}_i)$.

### §4.2 Conditional PD given systematic factor

$$\text{PD}(z) = N\!\left(\frac{N^{-1}(\text{PD}) - \sqrt{\rho}\,z}{\sqrt{1-\rho}}\right)$$

Sign convention: $z<0$ is adverse stress, $z>0$ is benign.

### §4.3 ASRF loss quantile (Vasicek 2002, large-pool limit)

$$L_\alpha = \text{LGD} \cdot N\!\left(\frac{N^{-1}(\text{PD}) + \sqrt{\rho}\,N^{-1}(\alpha)}{\sqrt{1-\rho}}\right)$$

This is the direct mathematical source of the Basel III IRB capital formula.

### §4.4 Asset correlation ρ (Basel III, BCBS §272 ff.)

| Exposure class | Formula | Range |
|---|---|---|
| Corporate / Bank / Sovereign | $\rho = 0{,}12 R + 0{,}24(1-R)$, $R = \frac{1-e^{-50\,\text{PD}}}{1-e^{-50}}$ | 0.12–0.24 |
| SME-Corporate ($S \in [5, 50]$ m EUR) | $\rho_{\text{Corp}} - 0{,}04\bigl(1 - (S-5)/45\bigr)$ | up to −0.04 |
| Residential Mortgage | $\rho = 0{,}15$ (constant) | 0.15 |
| QRRE (Qualifying Revolving Retail) | $\rho = 0{,}04$ (constant) | 0.04 |
| Other Retail | $\rho = 0{,}03 R + 0{,}16(1-R)$, $R = \frac{1-e^{-35\,\text{PD}}}{1-e^{-35}}$ | 0.03–0.16 |

### §4.5 Maturity adjustment (corporate only)

$$b(\text{PD}) = (0{,}11852 - 0{,}05478 \cdot \ln \text{PD})^2$$

$$\text{MA}(M) = \frac{1 + (M - 2{,}5) \cdot b(\text{PD})}{1 - 1{,}5 \cdot b(\text{PD})}$$

with effective maturity $M$ capped to $[1{,}0; 5{,}0]$ years. Retail / mortgage have no MA.

### §4.6 IRB capital charge

$$K = \bigl[L_\alpha - \text{PD} \cdot \text{LGD}\bigr] \cdot \text{MA}(M), \quad \alpha = 0{,}999$$

$$\text{RWA} = K \cdot 12{,}5 \cdot \text{EAD}$$

Reproduced in tests against the BCBS reference example: PD = 1%, LGD = 45%, M = 2.5 y → $K \approx 0{,}084$.

---

## §5 EBA Transparency Loader

### §5.1 Source files

EBA EU-wide Transparency Exercise 2025 (~120 banks). Reporting period: June 2025 (file vintage Dec 2024 release).

| File | Content | Size |
|---|---|---|
| `tr_cre.csv` | Credit risk (IRB + SA) — long format | ~123 MB |
| `tr_sov.csv` | Sovereign debt exposures | ~91 MB |
| `tr_oth.csv` | Capital, leverage, P&L, NPE | ~15 MB |
| `tr_mrk.csv` | Market risk | ~3.8 MB |
| `TR_Metadata.xlsx` | LEI ↔ name + dimensional dictionaries | ~2.8 MB |
| `SDD.xlsx` | Single Data Dictionary (item codes) | ~58 KB |

The cockpit currently consumes only `tr_cre.csv` and `tr_sov.csv` plus the metadata workbooks.

### §5.2 Loan-book parsing (`parse_credit_risk_csv`)

Streams `tr_cre.csv` in 500 k-row chunks and filters to the IRB items:

| Item | Meaning |
|---|---|
| `2520502` | Original Exposure (SA + IRB) |
| `2520512` | of which DEFAULTED (only published under Status = 2) |
| `2520522` | Exposure Value ≈ EAD (post-CCF) |
| `2520532` | Risk-Exposure Amount (RWA) |
| `2520552` | Value adjustments and provisions |

Filters: `Portfolio = 2 (IRB)`, `Country = 0 (counterparty-aggregate)`, `Status ∈ {0, 2}`, `Perf_Status = 0`.

### §5.3 EBA exposure-class → Vasicek class mapping

`eba_loader.EXPOSURE_TO_VASICEK_CLASS` maps 30+ EBA codes onto the seven Vasicek classes (`sovereign`, `bank`, `corporate`, `sme_corporate`, `mortgage`, `qrre`, `other_retail`). Items outside this set (601 default flag, 604–607 securitisations / equity, 800 other) are excluded as ancillary.

### §5.4 Implied PD

Since the Transparency Exercise does not publish risk parameters directly, PD is derived from the observed default ratio:

$$\text{PD}_{\text{implied}}^{(\text{class},\text{bank})} = \frac{\text{Defaulted Exposure (Item 2520512, Status = 2)}}{\text{Original Exposure (Item 2520502, Status = 0)}}$$

Floor 3 bp (Basel sovereign floor); cap 50% for numerical stability. This is a *backward-looking* point estimate of the realised default rate, not a forward-looking 1Y PD. Stress sensitivity (§4.2) is robust to this choice — the macro shock drives the *change* in PD, not its absolute level.

### §5.5 LGD assumptions (Basel F-IRB defaults)

| Vasicek class | LGD | Source |
|---|---|---|
| Sovereign / Bank / Corporate / SME-Corp | 45% | F-IRB senior unsecured standard |
| Mortgage | 20% | Basel III residential mortgage LGD floor |
| QRRE | 65% | Unsecured revolving standard |
| Other Retail | 45% | F-IRB retail other |

A-IRB banks use internal LGD models that are not published by the EBA, so the F-IRB defaults serve as a regulator-citable workaround.

### §5.6 Sovereign parsing (`parse_sovereign_csv`)

Streams `tr_sov.csv`. Schema dimensions: Bank × Counterparty-Country × Maturity-Bucket × Accounting-Portfolio. Items kept:

| Item | Meaning |
|---|---|
| `2520810` | On-balance gross carrying amount (primary exposure measure) |
| `2520822` | RWA on sovereign exposures |

**Filter subtlety:** unlike `tr_cre.csv`, the sovereign file does **not** publish a `Country = 0` aggregate; aggregation across counterparty countries must be done by summing the specific-country rows (`Country > 0`).

### §5.7 Modified-duration approximation per maturity bucket

The 7 EBA maturity buckets are approximated by their midpoints as Macaulay duration:

| Bucket | Duration (years) |
|---|---|
| `< 3M` | 0.125 |
| `3M – 1Y` | 0.625 |
| `1 – 2Y` | 1.5 |
| `2 – 3Y` | 2.5 |
| `3 – 5Y` | 4.0 |
| `5 – 10Y` | 7.5 |
| `> 10Y` | 15.0 |

Per-bank Mark-to-Market under parallel rate shock $\Delta r$ (in pp), $\Delta y = \Delta r / 100$:

$$\Delta P_b = -\sum_{m \in \text{buckets}} D_m \cdot \Delta y \cdot E_{b,m}$$

### §5.8 Adverse-scenario anchors

Hardcoded from the EBA stress-test methodology notes (the Excel templates are password-protected and not parsed):

| Vintage | ΔBrent log | Δr_10y | GDP shock | $M^{\text{anchor}}$ |
|---|---|---|---|---|
| EBA 2025 | +0.47 | +200 bp | −6.0 pp | −2.5 |
| EBA 2023 | +0.55 | +250 bp | −6.0 pp | −2.7 |

---

## §6 Numerical Resultate (Baseline · 2026-04-30)

### §6.1 Loan book — Top-10 EU banks by IRB EAD (June 2025)

| Bank | EAD bn | EL bn | RWA bn | RWA-Density | EL % |
|---|---|---|---|---|---|
| Groupe Crédit Agricole | 1 646 | 9.6 | 1 038 | 63% | 0.59% |
| BNP Paribas | 1 302 | 6.7 | 865 | 66% | 0.51% |
| ING Groep | 928 | 4.7 | 679 | 73% | 0.50% |
| Société Générale | 791 | 3.7 | 487 | 62% | 0.46% |
| Groupe BPCE | 777 | 7.9 | 603 | 78% | 1.02% |
| Deutsche Bank | 739 | 6.1 | 707 | 96% | 0.82% |
| Crédit Mutuel | 627 | 5.3 | 451 | 72% | 0.84% |
| Banco Santander | 595 | 4.8 | 444 | 75% | 0.80% |
| Rabobank | 469 | 4.0 | 391 | 83% | 0.85% |
| UniCredit | 408 | 3.2 | 366 | 90% | 0.79% |

Aggregate: Σ EAD 8 282 bn EUR, Σ EL 55.9 bn EUR, Σ RWA 6 031 bn EUR (density 73%).

### §6.2 Sovereign book — total system

Σ Sovereign exposure 4.01 tn EUR. Under a parallel +100 bp shock:

| Metric | Value |
|---|---|
| System Mark-to-Market P&L | **−247 bn EUR** |
| Hardest hit | BNP Paribas (−21 bn) |
| #2 / #3 | Société Générale (−15) / Deutsche Bank (−13) |

Doom-loop concentration is most pronounced for French (BPCE 70% domestic), Italian (UniCredit ~70%), Spanish (CaixaBank, BBVA ~70%) and Polish banks.

---

## §7 Reproducibility & Validation

Each backend module has a `__main__` validation block:

```bash
cd "Risk management"
python backend/svensson.py        # Svensson Excel cross-check + 6 test blocks
python backend/vasicek.py         # 6 ASRF / IRB tests (incl BCBS reference)
python backend/eba_loader.py      # Synthetic + real-loader smoke tests
python backend/macro_factor.py    # 6 anchor + data-route tests
```

Expected output: `[PASS] All tests passed.`

Every page also boots cleanly under `streamlit.testing.v1.AppTest`.

---

## §8 Known Limitations

| Limit | Description | Mitigation |
|---|---|---|
| Implied PD = backward-looking default ratio | Stock measure of current default quote, not forward-looking 1Y PD | Stress sensitivity remains correct; absolute level conservative |
| F-IRB LGD assumptions | A-IRB banks use internal LGD models that EBA does not publish | F-IRB defaults are a regulator-citable workaround |
| Single-factor Vasicek | One systematic factor — no sector / region clusters | Multi-factor CreditRisk+ would be V3 |
| Constant LGD across stress | LGD held fixed under shock | Downturn-LGD add-on is V2 |
| Anchor-only adverse scenario | EBA anchor is a single point — no distribution information | Hybrid with Mahalanobis covers data side |
| Sovereign: parallel-shift only | Δy uniform across maturities — no slope / curvature stress on sovereign book | Slope / curvature shifts in sidebar already drive the loan-book channel via Δr_10y |
| Sovereign: Mark-to-Market only | Realized losses for FVTPL / HfT only; HtM stays at amortised cost | EBA aggregate Accounting_portfolio = 0 only (no breakdown) |
| Sovereign: no credit-spread component | Pure rate sensitivity, no Italy-vs-Bund spread risk | Multi-country yield curves would be V3 |
| EBA stress-test 2025 file password-protected | Anchor values hardcoded from methodology note | Acceptable single source of truth |

---

## §9 Change Log

| Date | Change | Reason |
|---|---|---|
| 2026-04-29 | Vasicek/ASRF engine (`vasicek.py`), EBA loader scaffold (`eba_loader.py`), macro→M mapping (`macro_factor.py`) — initial Tier-2 stack | Regulatory tier alongside the original structural model |
| 2026-04-29 | Real EBA loader: streams `tr_cre.csv` (123 MB) chunk-wise, pivots IRB items to Vasicek classes, derives implied PD from the default ratio | Migration off synthetic universe — real EBA Transparency 2025 |
| 2026-04-29 | Empirical factor Σ in macro→M mapping (was synthetic 2×2). Anchor / data routes now agree within ~0.5σ instead of ~8σ | Hybrid mapping is now mutually validating |
| 2026-04-29 | Sovereign loader (`tr_sov.csv`), doom-loop heatmap, modified-duration P&L | Second risk channel — same Δr_10y drives both loan-book Vasicek M and sovereign Mark-to-Market |
| 2026-04-29 | McKinsey-aesthetic UI: navy / bright-blue / crimson palette, Source-Serif / Inter typography, Plotly custom template, click-through page-link cards | High-end consulting visual standard |
| 2026-04-30 | Tier-1 cleanup: removed Merton/KMV, factor model, Monte Carlo, scenario library, reverse stress, portfolio aggregator, equity-fetchers and equity-cache. Renumbered pages. Trimmed `config.py`, `data_loader.py`, `sidebar.py`. | Cockpit focuses on the regulatory tier only |
