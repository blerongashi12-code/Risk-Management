"""Helper: makes backend/ and the repo root importable for the pages.

Used at the top of every Streamlit page:
    from components.backend_path import setup
    setup()
    from vasicek import asset_correlation
"""
import sys
from pathlib import Path


def setup() -> Path:
    """Fügt backend/ und parent-Ordner zu sys.path hinzu.

    Returns
    -------
    repo_root : Path
        Das Verzeichnis mit `config.py`, `backend/`, `data/`.
    """
    # streamlit_app/components/backend_path.py
    # → parent.parent.parent = "Risk management/"
    repo_root = Path(__file__).resolve().parent.parent.parent
    backend = repo_root / "backend"

    for p in (str(repo_root), str(backend)):
        if p not in sys.path:
            sys.path.insert(0, p)

    return repo_root
