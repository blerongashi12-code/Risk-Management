# Risk-Management

**EU Banking Credit Stress Cockpit** — 2-Faktor-Kreditstress (Basel-III-IRB) + EBA Transparency, mit Pillar-3-Walk-Forward-Backtest.

## Struktur

| Ordner | Inhalt |
|---|---|
| [`Modellarchitektur/`](./Modellarchitektur/) | Das komplette Modell: Engines (`backend/`), Streamlit-Cockpit (`streamlit_app/`), Datenbasis (`data/`), Extraktions-Tools (`tools/`), Annahmen-Dokumentation (`docs/`) |
| [`Abgabe-Files/`](./Abgabe-Files/) | Finale Präsentation (Ablageort) + Abgabedokumente (Word) |

## Start

**Doppelklick auf `START_COCKPIT`** im Hauptordner — Windows nutzt automatisch
`START_COCKPIT.bat`, macOS `START_COCKPIT.command` *(dort beim ersten Mal:
Rechtsklick → „Öffnen" wegen Gatekeeper)*. Bei ausgeblendeten Datei-Endungen
erscheint auf beiden Systemen schlicht „START_COCKPIT".

Beide Launcher installieren beim ersten Start die Pakete (`requirements.txt`), beenden alte
Cockpit-Prozesse, leeren den Bytecode-Cache und öffnen das Cockpit im Browser.

→ Setup, 5-Tab-Walkthrough, Backtesting-Konzept und Detail-Struktur:
[`Modellarchitektur/README.md`](./Modellarchitektur/README.md)
