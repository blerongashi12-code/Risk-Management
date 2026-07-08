# Abgabe-Files

Dieser Ordner enthält nur die finalen Unterlagen für die Abgabe. Die
technische Modellarchitektur liegt getrennt in `../Modellarchitektur/`.

## Finale Abgabeunterlagen

| Pfad | Zweck |
|---|---|
| `Praesentation/EU-Banking-Credit-Stress-Cockpit-Praesentation.pptx` | Finale Präsentation, inkl. Foliennummerierung unten rechts. |
| `Abgabedokumente/Modellannahmen.docx` | Modelldokumentation: Methodik, Annahmen, Quellen, Backtest und Modellgrenzen. |

## Cockpit starten

Bitte nicht aus diesem Ordner starten, sondern eine Ebene höher:

- Windows: `Start-Cockpit-Windows.bat`
- macOS: `Start-Cockpit-Mac.command`

Beide Starter installieren beim ersten Start die benötigten Python-Pakete aus
`Modellarchitektur/requirements.txt` und öffnen anschließend das Streamlit-
Cockpit im Browser.

## Quellen der Methodik

Die Word-Datei `Abgabedokumente/Modellannahmen.docx` ist die lesbare
Abgabefassung. Die technische Referenz bleibt:

`../Modellarchitektur/docs/MODEL_ASSUMPTIONS.md`
