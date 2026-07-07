# Abgabe-Files

Dieser Ordner enthält die finalen Abgabe-Artefakte. Die technische
Modellarchitektur liegt getrennt in `../Modellarchitektur/`.

## Schnelle Orientierung

| Pfad | Zweck |
|---|---|
| `Abgabedokumente/Modellannahmen.docx` | Kuratierte Modelldokumentation für die Abgabe: Methodik, Annahmen, Quellen, Backtest-Grenzen. |
| `Abgabedokumente/Schaubilder/` | Exportierte Grafiken aus der Modelldokumentation. |
| `Praesentation/` | Präsentationsdateien und Visuals für die mündliche/folienbasierte Abgabe. |

## Cockpit starten

Bitte nicht aus diesem Ordner starten, sondern eine Ebene höher:

- Windows: `Start-Cockpit-Windows.bat`
- macOS: `Start-Cockpit-Mac.command`

Beide Starter installieren beim ersten Start die benötigten Python-Pakete aus
`Modellarchitektur/requirements.txt` und öffnen anschließend das Streamlit-
Cockpit im Browser.

## Quellen der Methodik

Die Word-Datei `Abgabedokumente/Modellannahmen.docx` ist die lesbare
Abgabefassung. Die technische Single Source of Truth bleibt:

`../Modellarchitektur/docs/MODEL_ASSUMPTIONS.md`
