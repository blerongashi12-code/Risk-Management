@echo off
REM ============================================================
REM  Start_Cockpit.bat - startet das Credit Stress Cockpit lokal
REM  Einfach doppelklicken. Der Browser oeffnet sich automatisch.
REM  Beim allerersten Start werden fehlende Pakete installiert.
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- 1) Python vorhanden? ---
where python >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python wurde nicht gefunden.
    echo Bitte Python 3.11+ von https://www.python.org installieren
    echo und beim Setup "Add python.exe to PATH" anhaken.
    pause
    exit /b 1
)

REM --- 2) Abhaengigkeiten nur installieren, falls Streamlit fehlt ---
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [Setup] Installiere benoetigte Pakete ^(nur beim ersten Start^) ...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

REM --- 3) Entry-Datei per Muster finden ------------------------
REM  Der Dateiname enthaelt einen Umlaut (Einfuehrung_in_das_Modell).
REM  Damit die .bat unabhaengig von der Windows-Codepage laeuft, wird
REM  die Datei ueber ein Wildcard-Muster gesucht statt fest getippt
REM  (ein direkt getippter Umlaut wuerde je nach Codepage scheitern).
cd "Risk management"
set "ENTRY="
for %%F in ("streamlit_app\Einf*_Modell.py") do set "ENTRY=%%~fF"

if not defined ENTRY (
    echo [FEHLER] Entry-Datei nicht gefunden ^(streamlit_app\Einf*_Modell.py^).
    pause
    exit /b 1
)

REM --- 4) App starten (oeffnet automatisch den Browser) ---
echo.
echo [Start] Credit Stress Cockpit wird gestartet - der Browser oeffnet automatisch.
echo         Zum Beenden: dieses Fenster schliessen oder Strg+C druecken.
echo.
python -m streamlit run "%ENTRY%"

pause
