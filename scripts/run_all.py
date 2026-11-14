"""Run the whole analysis, from raw PeMS files to every table and figure.

    python scripts/run_all.py                  # everything (about 6.5 hours on a 4-core laptop)
    python scripts/run_all.py --skip-checks    # skip the 1.5 h robustness checks (about 5 hours)
    python scripts/run_all.py --from 05        # start at a given step (earlier outputs must exist)

Steps 04 and 06 save progress as they go, so an interrupted run can simply be started again.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("01", "01_validate_simulator.py", "check the simulator against exact results", "1 min"),
    ("02", "02_capacity_definitions.py", "compare ways of measuring capacity", "2 min"),
    ("03", "03_prepare_data.py", "screen the PeMS files and keep usable station-days", "1 min"),
    ("04", "04_build_library.py", "simulate every candidate driver once", "2.7 h"),
    ("05", "05_calibrate.py", "calibrate every station-day, with bootstrap", "20 min"),
    ("06", "06_check_calibration.py", "recovery, Monte Carlo and block-length checks", "1.5 h"),
    ("08", "08_density_sensitivity.py", "refit with density from occupancy", "70 min"),
    ("07", "07_site_analysis.py", "detector-to-detector analysis, AV gains, headline figure", "1 min"),
    ("09", "09_simple_baseline.py", "congested-only skill vs a triangle and a model-free lookup", "15 min"),
    ("10", "10_transfer.py", "predict two days never used in fitting", "20 min"),
    ("11", "11_bottlenecks.py", "calibrated capacity vs discharge measured at active bottlenecks", "2 min"),
    ("16", "16_lane2.py", "recalibrate lane 2 at the same detectors", "10 min"),
    ("15", "15_balanced_fit.py", "refit with free flow and congestion weighted equally", "15 min"),
    ("12", "12_error_budget.py", "how far each modelling choice moves the AV gain", "2 min"),
    ("13", "13_misspecification.py", "recover a known capacity from a different model's data", "3 min"),
    ("14", "14_attribution.py", "what explains detector-to-detector differences (incl. position)", "2 min"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="01", help="first step to run, e.g. 05")
    ap.add_argument("--skip-checks", action="store_true", help="skip step 06")
    args = ap.parse_args()

    if not list((ROOT / "data" / "raw").glob("*_text_station_5min_*.txt*")):
        sys.exit("No PeMS files in data/raw. See data/raw/README.md for how to download them.")

    order = [s[0] for s in STEPS]
    if args.start not in order:
        sys.exit(f"--from must be one of {', '.join(order)}")
    todo = STEPS[order.index(args.start):]
    if args.skip_checks:
        todo = [s for s in todo if s[0] != "06"]

    t_all = time.time()
    for num, script, what, eta in todo:
        print(f"\n[{num}] {what}  (about {eta})", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"[{num}] {script} failed. Fix it and resume with: "
                     f"python scripts/run_all.py --from {num}")
        print(f"[{num}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nAll steps finished in {(time.time() - t_all) / 3600:.1f} h. "
          f"Figures are in figures/, tables in results/.")


if __name__ == "__main__":
    main()
