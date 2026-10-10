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
