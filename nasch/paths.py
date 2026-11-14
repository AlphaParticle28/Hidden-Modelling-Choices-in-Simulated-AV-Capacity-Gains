"""Folders used by every script, defined relative to the repository root so that nothing
depends on machine-specific settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"              # PeMS .txt.gz downloads (not in git)
DATA_PROCESSED = ROOT / "data" / "processed"  # cleaned per station-day tables
RESULTS = ROOT / "results"                    # summary tables
CACHE = RESULTS / "cache"                     # large regenerable outputs (not in git)
FIGURES = ROOT / "figures"


def ensure_dirs():
    for d in (DATA_PROCESSED, RESULTS, CACHE, FIGURES):
        d.mkdir(parents=True, exist_ok=True)

# The analysis uses these five weekdays. Two further days are kept aside, never used in fitting,
# and only used to test whether a site's calibration carries over to a day it has not seen.
ANALYSIS_DAYS = ("2026_08_27", "2026_09_15", "2026_09_16", "2026_09_17", "2026_09_22")
TEST_DAYS = ("2026_09_25", "2026_09_29")


def raw_files(days):
    return [p for d in days for p in sorted(DATA_RAW.glob(f"*_text_station_5min_{d}.txt*"))]
