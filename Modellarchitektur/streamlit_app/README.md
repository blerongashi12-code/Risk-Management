# Streamlit Frontend

Interaktives Cockpit für das EU Banking Credit Stress Modell.

## Start

Empfohlen aus dem Hauptordner des Projekts:

- Windows: `Start-Cockpit-Windows.bat`
- macOS: `Start-Cockpit-Mac.command`

Manuell aus `Modellarchitektur/`:

```bash
pip install -r requirements.txt
python run_clean.py
```

Die App läuft anschließend auf <http://localhost:8501>.

## Struktur

```text
streamlit_app/
├── Einführung_in_das_Modell.py      # Landing Page / Intro
├── .streamlit/config.toml           # Streamlit Theme
├── static/mckinsey.css              # visuelles Styling
├── components/
│   ├── theme.py                     # Header, Breadcrumb, Plotly-Template
│   ├── sidebar.py                   # globale Makro-Stress-Slider
│   ├── data_loader.py               # Brent + Svensson Cache
│   ├── methodology.py               # Methodik-Boxen
│   ├── legacy_views.py              # entfernte Alt-Views / Hinweise
│   └── backend_path.py              # sys.path-Setup für backend/
└── pages/
    ├── 1_Faktor_Analyse.py          # Faktoren + Stress-Transmission
    ├── 2_Kreditbuch.py              # Loan Book, PD/LGD, EL/RWA
    ├── 3_Marktbuch.py               # Sovereign-/Zinskanal
    ├── 4_Eigenkapital.py            # CET1-Bridge
    └── 5_Validierung.py             # Walk-Forward-Backtest
```

## Falls die App nicht startet

1. Python 3.10+ installieren.
2. Aus `Modellarchitektur/` ausführen:
   ```bash
   pip install -r requirements.txt
   python run_clean.py
   ```
3. Falls Port 8501 belegt ist:
   ```bash
   streamlit run "streamlit_app/Einführung_in_das_Modell.py" --server.port 8502
   ```
