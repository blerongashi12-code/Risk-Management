"""run_clean.py — robuster Streamlit-Starter mit Pycache-Cleanup + Diagnose.

Aufruf aus dem `Risk management/`-Verzeichnis:

    python run_clean.py

Was passiert:
  1. Pre-flight Checks: Python-Version, Streamlit-Installation,
     Daten-Files (Bundesbank, EBA), Cache-Files.
  2. Cleanup aller `__pycache__/`-Verzeichnisse (häufigste Ursache
     für "Pages öffnen nicht" nach Page-Renamings).
  3. Streamlit-Server-Start auf einem freien Port (default 8501,
     fallback 8502, 8503, ...).
  4. Browser wird automatisch geöffnet auf der App-URL.

Falls etwas fehlschlägt, gibt das Script eine konkrete Fehlermeldung
mit Lösungsvorschlag aus, statt stumm zu enden.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows console; fall back silently if not supported.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ASCII-only status glyphs to avoid encoding issues across terminals.
OK   = "[OK]"
WARN = "[!!]"
ERR  = "[XX]"


# =====================================================================
# Pre-flight checks
# =====================================================================
def _check_python() -> bool:
    if sys.version_info < (3, 10):
        print(f"[ERROR] Python {sys.version_info.major}.{sys.version_info.minor} "
              f"zu alt. Mindestens Python 3.10 nötig.")
        return False
    print(f"  [OK]Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def _check_streamlit() -> bool:
    try:
        import streamlit  # noqa: F401
        ver = streamlit.__version__
        print(f"  [OK]Streamlit {ver}")
        return True
    except ImportError:
        print("[ERROR] Streamlit nicht installiert.")
        print("        Lösung: pip install streamlit pandas pyarrow plotly numpy scipy openpyxl")
        return False


def _check_data_files(here: Path) -> bool:
    """Prüft ob die EBA-Files vorliegen (Worktree → Main-Repo Resolver)."""
    candidates = [here / "data"]
    parts = here.parts
    if ".claude" in parts and "worktrees" in parts:
        try:
            wt_idx = parts.index(".claude")
            candidates.append(Path(*parts[:wt_idx]) / "Risk management" / "data")
        except ValueError:
            pass

    found_dir = None
    for d in candidates:
        if (d / "tr_cre.csv").exists():
            found_dir = d
            break
    if found_dir is None:
        print("[WARN] EBA-Dateien (tr_cre.csv ...) nicht gefunden.")
        print("        Pages 1, 2, 3, 5 werden Fehler zeigen.")
        print("        Lösung: tr_*.csv + TR_Metadata.xlsx + SDD.xlsx nach data/ legen.")
        return False
    print(f"  [OK]EBA-Files in {found_dir}")
    return True


def _check_cache_files(here: Path) -> bool:
    """Cache muss Brent + Svensson enthalten."""
    cache_dir = here / "data" / "cache"
    needed = ["brent_crude.parquet", "svensson_params.parquet"]
    missing = [f for f in needed if not (cache_dir / f).exists()]
    if missing:
        print(f"[WARN] Cache-Files fehlen: {missing}")
        print("        Lösung: python 03_fetch_brent_crude.py")
        print("                python 04_fetch_bundesbank_svensson.py")
        return False
    print(f"  [OK]Cache (Brent + Svensson) vorhanden")
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
# Free port detection
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
# Main
# =====================================================================
def main() -> None:
    here = Path(__file__).resolve().parent
    app_path = here / "streamlit_app" / "app.py"

    print("=" * 60)
    print(" EU Banking Credit Stress Cockpit · run_clean.py")
    print("=" * 60)

    if not app_path.exists():
        print(f"[ERROR] Streamlit-App nicht gefunden unter {app_path}.")
        print("        Bitte aus dem 'Risk management/'-Verzeichnis starten.")
        sys.exit(1)

    print("\n>> Pre-flight checks:")
    ok_py    = _check_python()
    ok_st    = _check_streamlit()
    ok_data  = _check_data_files(here)
    ok_cache = _check_cache_files(here)

    if not (ok_py and ok_st):
        print("\n[ABORT] Critical dependencies fehlen — siehe oben.")
        sys.exit(1)

    if not (ok_data and ok_cache):
        print("\n[NOTE] Cockpit startet trotzdem — einige Pages werden aber Fehler zeigen.")

    print("\n>> Cleanup stale __pycache__ …")
    n = _clean_pycache(here)
    print(f"   {n} pycache-Verzeichnisse entfernt.")

    port = _find_free_port(8501)
    if port is None:
        print("[ERROR] Kein freier Port zwischen 8501-8520 gefunden.")
        sys.exit(1)
    if port != 8501:
        print(f"\n[NOTE] Port 8501 belegt — verwende stattdessen Port {port}.")

    url = f"http://localhost:{port}"
    print(f"\n>> Starte Streamlit auf {url}")
    print(f"   Befehl: streamlit run {app_path.relative_to(here)}\n")

    # --server.headless=true verhindert dass Streamlit einen Browser
    # zu öffnen versucht (was unter manchen Windows-Setups einen
    # Dialog auslöst und den Server-Start blockiert). Der User öffnet
    # die URL manuell — sie wird in der Streamlit-Console ausgegeben.
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.runOnSave", "false",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"   Browser manuell öffnen: {url}", flush=True)
    print(f"   Stop: Ctrl+C\n", flush=True)

    try:
        # Hand off entirely — no subprocess management. Streamlit handles
        # its own browser-open and terminal output.
        result = subprocess.run(cmd, cwd=str(here))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n>> Streamlit beendet.")
        sys.exit(0)
    except FileNotFoundError:
        print("[ERROR] 'streamlit' Befehl nicht gefunden.")
        print("        Lösung: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
