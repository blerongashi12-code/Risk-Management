# Risk-Management

**EU Banking Credit Stress Cockpit** — 2-Faktor-Kreditstress (Basel-III-IRB) + EBA Transparency, mit Pillar-3-Walk-Forward-Backtest.

## Struktur

| Ordner | Inhalt |
|---|---|
| [`Modellarchitektur/`](./Modellarchitektur/) | Das komplette Modell: Engines (`backend/`), Streamlit-Cockpit (`streamlit_app/`), Datenbasis (`data/`), Extraktions-Tools (`tools/`), Annahmen-Dokumentation (`docs/`) |
| [`Praesentation_Abgabe/`](./Praesentation_Abgabe/) | Finale Präsentation (Ablageort) + Abgabedokumente (Word) |

## Start

| System | Aufruf |
|---|---|
| **Windows** | Doppelklick auf `Start_Cockpit.bat` |
| **macOS** | Doppelklick auf `Start_Cockpit.command` *(beim ersten Mal: Rechtsklick → „Öffnen" wegen Gatekeeper)* |

Beide Launcher installieren beim ersten Start die Pakete (`requirements.txt`), beenden alte
Cockpit-Prozesse, leeren den Bytecode-Cache und öffnen das Cockpit im Browser.

→ Setup, 5-Tab-Walkthrough, Backtesting-Konzept und Detail-Struktur:
[`Modellarchitektur/README.md`](./Modellarchitektur/README.md)
