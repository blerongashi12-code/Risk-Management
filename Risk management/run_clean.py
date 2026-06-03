"""run_clean.py — robuster Streamlit-Starter mit Pycache-Cleanup + Live-Healthcheck.

Aufruf aus dem `Risk management/`-Verzeichnis:

    python run_clean.py

Was passiert:
  1. Pre-flight Checks (Python, Streamlit, EBA-Files, Cache).
  2. Cleanup aller `__pycache__/`-Verzeichnisse.
  3. Streamlit-Server-Start auf einem freien Port (8501, fallback 8502 ...).
  4. Healthcheck: nach 6 Sekunden wird HTTP-GET auf localhost gesendet.
     Falls der Server NICHT antwortet, wird der Fehler hervorgehoben.
  5. Streamlit läuft im Vordergrund. **Terminal MUSS offen bleiben** —
     wenn du das Fenster schließt, stirbt der Server. URL im Browser
     manuell öffnen.

Beim Schließen des Terminals oder Ctrl+C wird Streamlit sauber beendet.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OK   = "[OK]"
WARN = "[!!]"
ERR  = "[XX]"


# =====================================================================
# Pre-flight
# =====================================================================
def _check_python() -> bool:
    if sys.version_info < (3, 10):
        print(f"{ERR} Python {sys.version_info.major}.{sys.version_info.minor} zu alt — mindestens 3.10 nötig.")
        return False
    print(f"{OK} Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"     Pfad: {sys.executable}")
    return True


def _check_streamlit() -> bool:
    try:
        import streamlit
        ver = streamlit.__version__
        print(f"{OK} Streamlit {ver}")
        return True
    except ImportError:
        print(f"{ERR} Streamlit nicht installiert (in dieser Python-Umgebung).")
        print(f"     Lösung: {sys.executable} -m pip install streamlit pandas pyarrow plotly numpy scipy openpyxl")
        return False


def _check_data_files(here: Path) -> bool:
    candidates = [here / "data"]
    parts = here.parts
    if ".claude" in parts and "worktrees" in parts:
        try:
            wt_idx = parts.index(".claude")
            candidates.append(Path(*parts[:wt_idx]) / "Risk management" / "data")
        except ValueError:
            pass
    for d in candidates:
        if (d / "tr_cre.csv").exists():
            print(f"{OK} EBA-Files in {d}")
            return True
    print(f"{WARN} EBA-Dateien (tr_cre.csv ...) nicht gefunden — einige Pages werden Fehler zeigen.")
    return False


def _check_cache_files(here: Path) -> bool:
    cache_dir = here / "data" / "cache"
    needed = ["brent_crude.parquet", "svensson_params.parquet"]
    missing = [f for f in needed if not (cache_dir / f).exists()]
    if missing:
        print(f"{WARN} Cache-Files fehlen: {missing}")
        print("     Lösung: python 03_fetch_brent_crude.py + python 04_fetch_bundesbank_svensson.py")
        return False
    print(f"{OK} Cache (Brent + Svensson) vorhanden")
    return True


# =====================================================================
# Cleanup
# =====================================================================
def _clean_pycache(root: Path) -> int:
    count = 0
    for pc in root.rglob("__pycache__"):
        if pc.is_dir():
            shutil.rmtree(pc, ignore_errors=True)
            count += 1
    return count


# =====================================================================
# Free-port detection
# =====================================================================
def _find_free_port(start: int = 8501, attempts: int = 20) -> int | None:
    for offset in range(attempts):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


# =====================================================================
# Healthcheck (background thread)
# =====================================================================
def _wait_for_server(url: str, max_wait_seconds: float = 12.0) -> None:
    """In background: probes URL until alive, then prints visible 'READY' message.

    On Windows, Streamlit can take 4-8 seconds to fully boot. We poll
    the health endpoint every 0.5s and print the green message as soon
    as it answers.
    """
    health_url = url.rstrip("/") + "/_stcore/health"
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    print()
                    print("=" * 60)
                    print(f" {OK}  COCKPIT IST BEREIT — öffne im Browser:")
                    print(f"      {url}")
                    print(f"      (Terminal offen lassen! Server stoppt mit Ctrl+C.)")
                    print("=" * 60)
                    print()
                    return
        except Exception:
            pass
        time.sleep(0.5)
    print()
    print("=" * 60)
    print(f" {WARN}  Healthcheck nach {max_wait_seconds:.0f}s ohne Antwort.")
    print(f"      Versuche trotzdem im Browser: {url}")
    print(f"      Falls keine Antwort, bitte Streamlit-Output oben prüfen.")
    print("=" * 60)
    print()


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    here = Path(__file__).resolve().parent
    app_path = here / "streamlit_app" / "Einführung_in_das_Modell.py"

    print("=" * 60)
    print(" EU Banking Credit Stress Cockpit · run_clean.py")
    print("=" * 60)

    if not app_path.exists():
        print(f"{ERR} Streamlit-App nicht gefunden unter {app_path}.")
        print("     Bitte aus dem 'Risk management/'-Verzeichnis starten.")
        sys.exit(1)

    print("\n>> Pre-flight checks:")
    ok_py    = _check_python()
    ok_st    = _check_streamlit()
    ok_data  = _check_data_files(here)
    ok_cache = _check_cache_files(here)

    if not (ok_py and ok_st):
        print(f"\n{ERR} Critical dependencies fehlen — siehe oben. Abbruch.")
        sys.exit(1)

    if not (ok_data and ok_cache):
        print(f"\n{WARN} Cockpit startet trotzdem — einige Pages werden Fehler zeigen.")

    print("\n>> Cleanup stale __pycache__ …")
    n = _clean_pycache(here)
    print(f"   {n} pycache-Verzeichnisse entfernt.")

    port = _find_free_port(8501)
    if port is None:
        print(f"{ERR} Kein freier Port zwischen 8501-8520 gefunden.")
        sys.exit(1)
    if port != 8501:
        print(f"\n{WARN} Port 8501 belegt — verwende stattdessen Port {port}.")

    url = f"http://localhost:{port}"

    print(f"\n>> Starte Streamlit auf {url}")
    print("   Der Server hat 4-8 Sekunden Boot-Zeit, dann erscheint")
    print("   die READY-Meldung.\n")

    # Start background healthcheck
    health_thread = threading.Thread(
        target=_wait_for_server, args=(url,), daemon=True,
    )
    health_thread.start()

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--server.runOnSave", "false",
        "--browser.gatherUsageStats", "false",
    ]

    try:
        # Run streamlit IN FOREGROUND. Streamlit's stdout flows through.
        result = subprocess.run(cmd, cwd=str(here))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n>> Streamlit beendet.")
        sys.exit(0)
    except FileNotFoundError:
        print(f"\n{ERR} 'streamlit' Executable nicht gefunden.")
        print(f"     Lösung: {sys.executable} -m pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
