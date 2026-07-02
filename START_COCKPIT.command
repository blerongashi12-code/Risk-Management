#!/bin/bash
# ============================================================
#  Start_Cockpit.command - startet das Credit Stress Cockpit (macOS)
#  Einfach doppelklicken. Beim ersten Mal: Rechtsklick -> "Oeffnen"
#  (Gatekeeper). Der Browser oeffnet sich automatisch.
#  Gleiche Logik wie Start_Cockpit.bat (Windows).
# ============================================================
set -u
cd "$(dirname "$0")"

# --- 1) Python vorhanden? ---
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
    echo "[FEHLER] Python wurde nicht gefunden."
    echo "Bitte Python 3.11+ installieren: https://www.python.org"
    read -r -p "Enter zum Beenden..." _
    exit 1
fi

# --- 2) Abhaengigkeiten nur installieren, falls Streamlit fehlt ---
if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
    echo "[Setup] Installiere benoetigte Pakete (nur beim ersten Start) ..."
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
fi

# --- 2b) Alte Cockpit-Server beenden + Bytecode-Cache leeren ---
#  Ein noch laufender Server haelt nach Code-Updates das alte Modul im
#  Speicher (ImportError beim Neu-Laden). Darum vor jedem Start aufraeumen.
echo "[Reset] Beende evtl. laufende Cockpit-Server und leere den Bytecode-Cache ..."
pkill -f "streamlit run" >/dev/null 2>&1 || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# --- 3) Entry-Datei per Muster finden (Umlaut-sicher) ---
cd "Modellarchitektur"
ENTRY="$(ls streamlit_app/Einf*_Modell.py 2>/dev/null | head -1)"
if [ -z "$ENTRY" ]; then
    echo "[FEHLER] Entry-Datei nicht gefunden (streamlit_app/Einf*_Modell.py)."
    read -r -p "Enter zum Beenden..." _
    exit 1
fi

# --- 4) App starten (oeffnet automatisch den Browser) ---
echo ""
echo "[Start] Credit Stress Cockpit wird gestartet - der Browser oeffnet automatisch."
echo "        Zum Beenden: dieses Fenster schliessen oder Ctrl+C druecken."
echo ""
"$PY" -m streamlit run "$ENTRY"
