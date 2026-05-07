"""run_clean.py — robuster Streamlit-Starter mit Pycache-Cleanup.

Wird aus dem `Risk management/`-Verzeichnis aufgerufen:
    python run_clean.py

Räumt vor dem Start alle stale `__pycache__/`-Verzeichnisse weg
(häufigste Ursache für "Pages öffnen nicht" nach Page-Renamings) und
startet dann `streamlit run streamlit_app/app.py`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _clean_pycache(root: Path) -> int:
    """Löscht alle `__pycache__`-Verzeichnisse rekursiv. Liefert Anzahl."""
    count = 0
    for pc in root.rglob("__pycache__"):
        if pc.is_dir():
            shutil.rmtree(pc, ignore_errors=True)
            count += 1
    return count


def main() -> None:
    here = Path(__file__).resolve().parent
    app_path = here / "streamlit_app" / "app.py"
    if not app_path.exists():
        print(f"[ERROR] Streamlit-App nicht gefunden unter {app_path}.")
        print("        Bitte aus dem 'Risk management/'-Verzeichnis starten.")
        sys.exit(1)

    print(">> Cleanup stale __pycache__ …")
    n = _clean_pycache(here)
    print(f"   {n} pycache-Verzeichnisse entfernt.")

    print(">> Streamlit starten:")
    print(f"   streamlit run {app_path.relative_to(here)}")
    print()

    # Hand off to streamlit; user can Ctrl+C to stop normally
    cmd = [sys.executable, "-m", "streamlit", "run",
           str(app_path), "--server.runOnSave", "false"]
    try:
        subprocess.run(cmd, cwd=str(here), check=False)
    except KeyboardInterrupt:
        print("\n>> Streamlit beendet.")


if __name__ == "__main__":
    main()
