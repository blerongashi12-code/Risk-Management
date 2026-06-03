# EU Banking Credit Stress Cockpit · Streamlit Frontend

Interaktives Cockpit für das Vasicek/ASRF + EBA-Transparency Credit-Risk-Modell.

## Setup

```bash
# Aus dem 'Risk management/'-Verzeichnis:

# 1) Dependencies installieren
pip install -r streamlit_app/requirements.txt

# 2) Cockpit starten — robuste Variante (räumt stale __pycache__ auf)
python run_clean.py

# Alternativ direkt:
# streamlit run "streamlit_app/Einführung_in_das_Modell.py"
```

App läuft auf <http://localhost:8501>.

## Struktur

```
streamlit_app/
├── Einführung_in_das_Modell.py     # Landing-Page (Tab 1) + globale Sidebar
├── requirements.txt
├── .streamlit/config.toml          # Theme-Primaries
├── static/
│   └── mckinsey.css                # Konsulting-Stil-CSS (Navy / Crimson)
├── components/
│   ├── backend_path.py             # sys.path-Setup für backend/
│   ├── sidebar.py                  # globale Macro-Stress-Slider
│   ├── data_loader.py              # cached Brent + Svensson
│   ├── theme.py                    # apply_theme + Plotly-Template
│   └── methodology.py              # collapsible Methodik-Boxen
└── pages/
    ├── 1_Credit_Risk.py            # Vasicek IRB + NPL + CET1-Strip
    ├── 2_Bonds.py                  # 3 Sub-Tabs: Sov · BB-Bonds · TB+ABS
    ├── 3_Capital_Adequacy.py       # 3-Channel CET1-Ratio
    ├── 4_Yield_Curve.py            # Bundesbank Svensson + β-Shifts
    ├── 5_Backtesting.py            # Forecast vs Realized (22 Quartale)
    ├── 6_Annahmen.py               # Governance-Doku (3 Layer)
    └── 7_Methodology.py            # Vollständige MODEL_ASSUMPTIONS.md
```

## Live-Slider (Sidebar)

| Slider | Bereich | Bedeutung |
|---|---|---|
| ΔBrent (log) | −2.0 … +2.0 | Kumulativer Brent log-Return |
| Δβ₀ … Δβ₃ | −3.0 … +3.0 | Svensson-Parameter-Shifts (Level/Slope/Curvature) |
| Quick-Scenarios | — | Corona 2020 · Ukraine 2022 · Iran 2026 · EBA 2025 adverse |

Implied Δr_10y wird live aus Δβ-Shifts via Svensson-Funktion berechnet und unten in der Sidebar angezeigt.

## Performance

- Statische Daten via `@st.cache_data(ttl=24h)` gecached.
- Initialer EBA-Load (tr_cre + tr_sov + tr_oth, ~150 MB): ~5–10 s.
- Slider-Move triggert nur die abhängigen Compute-Schritte (~0.3–1 s).

## Falls Pages nicht öffnen / "nicht erreichbar"

In Reihenfolge der Wahrscheinlichkeit:

1. **Stale `__pycache__/`** (häufigste Ursache nach Page-Renamings)
   ```bash
   python run_clean.py
   ```
   räumt automatisch auf und startet.

2. **Port 8501 belegt**
   ```bash
   streamlit run "streamlit_app/Einführung_in_das_Modell.py" --server.port 8502
   ```

3. **Firewall / VPN blockt localhost** — temporär deaktivieren.

4. **Streamlit nicht installiert oder veraltet**
   ```bash
   pip install --upgrade streamlit pandas pyarrow plotly
   ```

5. **Browser-Cache mit alten URLs** — `Ctrl+F5` für Hard-Refresh.
