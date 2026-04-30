"""
============================================================================
 config.py · Central configuration for the Credit Stress Cockpit
============================================================================
 Tier 2 (Vasicek/ASRF + EBA Sovereign) configuration only:
   - Filesystem paths (data, cache, EBA raw files, output)
   - Bundesbank Svensson curve (still drives the macro shock)
   - Brent crude proxy
   - Vasicek/Basel-III IRB constants
   - EBA macro→M routing
============================================================================
"""

from pathlib import Path
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

DATA_DIR              = BASE_DIR / "data"
CACHE_DIR             = DATA_DIR / "cache"
EBA_DIR               = DATA_DIR / "eba"            # parsed parquet outputs
EBA_TRANSPARENCY_DIR  = EBA_DIR / "transparency_2025"
EBA_STRESS_DIR        = EBA_DIR / "stress_test_2025"
OUTPUT_DIR            = BASE_DIR / "output"
for p in (DATA_DIR, CACHE_DIR, EBA_DIR, EBA_TRANSPARENCY_DIR,
          EBA_STRESS_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 1a. EBA raw-data resolution
# ----------------------------------------------------------------------
# The raw EBA Transparency CSVs (tr_cre.csv, tr_sov.csv, tr_oth.csv,
# tr_mrk.csv) plus metadata (TR_Metadata.xlsx, SDD.xlsx) are large and
# kept out of the repo. The loader looks for them in:
#   1. BASE_DIR / "data"  (worktree-local, if present)
#   2. The main repo's data/ when running from a git worktree
def _resolve_eba_raw_dir() -> Path:
    """Find the directory containing the raw EBA CSVs."""
    candidates = [DATA_DIR]
    parts = BASE_DIR.parts
    if ".claude" in parts and "worktrees" in parts:
        try:
            wt_idx = parts.index(".claude")
            main_data = Path(*parts[:wt_idx]) / "Risk management" / "data"
            candidates.append(main_data)
        except ValueError:
            pass
    for d in candidates:
        if (d / "tr_cre.csv").exists():
            return d
    return DATA_DIR

EBA_RAW_DIR = _resolve_eba_raw_dir()


# ----------------------------------------------------------------------
# 2. Macro proxies
# ----------------------------------------------------------------------
ENERGY_PROXY = "BZ=F"      # ICE Brent Crude Futures (USD/bbl)
FX_PROXY     = "EURUSD=X"  # USD → EUR conversion if needed


# ----------------------------------------------------------------------
# 3. Lookback window (for empirical factor covariance)
# ----------------------------------------------------------------------
LOOKBACK_YEARS = 6
START_DATE = (datetime.today() - timedelta(days=int(LOOKBACK_YEARS * 365.25))).strftime("%Y-%m-%d")
END_DATE   = datetime.today().strftime("%Y-%m-%d")


# ----------------------------------------------------------------------
# 4. Vasicek / Basel III IRB
# ----------------------------------------------------------------------
# Confidence level for the ASRF loss quantile (Basel III: 99.9%).
VASICEK_CONFIDENCE = 0.999

# Default effective maturity for corporate exposures (Basel cap [1.0, 5.0]).
VASICEK_DEFAULT_MATURITY_YEARS = 2.5

# Default LGD for the bank-portfolio aggregate (override per segment).
VASICEK_DEFAULT_LGD = 0.45


# ----------------------------------------------------------------------
# 5. EBA macro→M routing
# ----------------------------------------------------------------------
# Anchor vintage used by macro_factor.anchor_from_eba()
EBA_VINTAGE_PRIMARY = "2025"

# Macro-shock → systematic-factor M mapping route:
#   "anchor"  : EBA stress-test anchor only (regulator-citable)
#   "data"    : empirical Mahalanobis only (data-driven)
#   "hybrid"  : average of both — anchor as primary, data as sanity
MACRO_FACTOR_ROUTE = "hybrid"


# ----------------------------------------------------------------------
# 6. Bundesbank CSV layout
# ----------------------------------------------------------------------
BBK_COLUMNS     = ["Date", "Beta0", "Beta1", "Beta2", "Beta3", "Tau1", "Tau2"]
BBK_RAW_INDICES = [0, 1, 3, 5, 7, 9, 11]


def show_config():
    """Prints the current configuration."""
    print("=" * 60)
    print("Credit Stress Cockpit · Configuration")
    print("=" * 60)
    print(f"  Base Dir:        {BASE_DIR}")
    print(f"  Data Dir:        {DATA_DIR}")
    print(f"  Cache Dir:       {CACHE_DIR}")
    print(f"  EBA Dir:         {EBA_DIR}")
    print(f"  EBA Raw Dir:     {EBA_RAW_DIR}")
    print(f"  Lookback:        {START_DATE} -> {END_DATE}")
    print(f"  Vasicek conf:    {VASICEK_CONFIDENCE}")
    print(f"  Vasicek default LGD: {VASICEK_DEFAULT_LGD}")
    print(f"  EBA vintage:     {EBA_VINTAGE_PRIMARY}")
    print(f"  Macro route:     {MACRO_FACTOR_ROUTE}")
    print("=" * 60)


if __name__ == "__main__":
    show_config()
