# DAX Credit Stress · Streamlit Frontend

Interaktives Cockpit für das Backend-Modell.

## Setup

```bash
# 1) Daten-Layer einmalig bauen (vom Repo-Root, nicht hier!)
cd ..
python 00_build_data_layer.py

# 2) Streamlit-Abhängigkeiten installieren
pip install -r streamlit_app/requirements.txt

# 3) App starten
streamlit run streamlit_app/app.py
```

App läuft auf http://localhost:8501.

## Struktur

```
streamlit_app/
├── app.py                          # Welcome-Page + globale Sidebar
├── requirements.txt
├── components/
│   ├── backend_path.py             # sys.path-Setup für backend/
│   ├── sidebar.py                  # globale Stress-Slider
│   ├── data_loader.py              # cached Backend-Calls
│   ├── stress_engine.py            # Custom-Scenario, MC, Decomposition
│   └── charts.py                   # Plotly-Helpers
└── pages/
    ├── 1_📊_Overview.py            # KPIs, Waterfall, Heatmaps, Top-N
    ├── 2_🔍_Firm_Drilldown.py     # Stub
    ├── 3_🔥_Scenario_Heatmap.py   # Library-Heatmap
    ├── 4_⚡_Reverse_Stress.py      # Stub
    ├── 5_📈_Yield_Curve.py        # Live-Svensson + Slider-Shifts
    └── 6_📚_Methodology.py        # rendert MODEL_ASSUMPTIONS.md
```

## Live-Slider (Sidebar)

| Slider | Bereich | Bedeutung |
|---|---|---|
| ΔBrent | -2.0 … +2.0 | log-Return Brent kumuliert |
| Δβ₀ … Δβ₃ | -3.0 … +3.0 | Svensson-Parameter-Shifts (Level/Slope/Curvature) |
| Korrelation Brent ↔ Δr | -0.95 … +0.95 | Override historische Σ |
| Time Horizon | 30 … 504 d | MC-Pfad-Länge |
| Monte-Carlo-Pfade | 1k … 100k | N für Quantil-Stabilität |

Quick-Presets: Corona 2020, Ukraine 2022, Iran 2026.

## Performance

- Statische Daten + Baseline werden via `@st.cache_data(ttl=24h)` gecached.
- Custom-Scenario: ~0.3 s
- Custom-MC mit 10k Pfaden: ~0.5 s
- 100k Pfaden: ~5 s
- Slider-Move triggert Re-Compute der abhängigen Schritte (cached, schnell).

## Bekannte V1-Limitationen

- Pages 2 (Drilldown) und 4 (Reverse Stress) sind Stubs.
- Sektor-Heatmap nutzt gleichgewichteten Sektor-Mittel statt EAD-Gewichtung.
- Korrelations-Override greift nur in `run_custom_mc`, nicht in `run_scenarios`.
