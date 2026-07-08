# EU Banking Credit Stress Cockpit

Dieses Repository enthaelt das finale Risk-Management-Modell, die
Modellarchitektur und die Abgabeunterlagen.

## Ordnerstruktur

| Pfad | Inhalt |
|---|---|
| `Abgabe-Files/` | Finale Praesentation und Modellannahmen-Dokumentation |
| `Modellarchitektur/` | Datenbasis, Backend, Streamlit-Cockpit und technische Dokumentation |
| `Start-Cockpit-Windows.bat` | Startdatei fuer Windows |
| `Start-Cockpit-Mac.command` | Startdatei fuer macOS |

## Start unter Windows

1. GitHub-ZIP herunterladen.
2. Im Windows-Explorer auf `Alle extrahieren` klicken.
3. Den entpackten Ordner oeffnen.
4. `Start-Cockpit-Windows.bat` doppelklicken.

Der Windows-Hinweis beim Start aus dem ZIP ist normal: Das Modell sollte vor
dem Start entpackt werden, weil die Anwendung auf mehrere Dateien und Ordner
zugreift.

## Start unter macOS

1. GitHub-ZIP herunterladen und entpacken.
2. Den entpackten Ordner im Terminal oeffnen.
3. Einmalig ausfuehren:

```bash
chmod +x Start-Cockpit-Mac.command
xattr -dr com.apple.quarantine .
```

4. Danach `Start-Cockpit-Mac.command` per Doppelklick starten.

Falls macOS den Start weiterhin blockiert: Rechtsklick auf
`Start-Cockpit-Mac.command`, dann `Oeffnen` waehlen. Python 3.11 oder neuer muss
installiert sein.

## Hinweis

Beim ersten Start installiert das Script die benoetigten Python-Pakete aus
`Modellarchitektur/requirements.txt`. Danach oeffnet sich das Cockpit lokal im
Browser.
