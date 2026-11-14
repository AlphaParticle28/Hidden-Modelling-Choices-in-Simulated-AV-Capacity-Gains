"""Does the calibrated capacity depend on how density is computed from the detector?

Refits every kept station-day (point estimates only) with density = occupancy / g for a fixed
effective vehicle length g = 18, 20, 22, 25 and 28 ft (PeMS's own median is ~20 ft), instead of
PeMS's flow / speed. Already computed values of g are reused.

    python scripts/08_density_sensitivity.py   -> results/density_sensitivity.csv   (~15 min)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.calibration import fit_station_day, load_library  # noqa: E402
from nasch.pems import density_from_occupancy  # noqa: E402

lib = load_library(paths.CACHE / "fd_library.npz")
discharge = pd.read_csv(paths.RESULTS / "human_discharge.csv")["discharge"].to_numpy()
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
kept = cal[cal["kept_for_av_gain"]]
data = pd.read_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", dtype={"time": str})
groups = {key: g.sort_values("time") for key, g in data.groupby(["station", "date"])}

G_VALUES = (18, 20, 22, 25, 28)
out_file = paths.RESULTS / "density_sensitivity.csv"
old = pd.read_csv(out_file) if out_file.exists() else None
todo = [g for g in G_VALUES if old is None or f"capacity_g{g}" not in old.columns]
rows = []
for n, r in enumerate(kept.to_dict("records"), 1):
    d = groups[(r["station"], r["date"])]
    res = dict(station=r["station"], date=r["date"], capacity_qv=r["capacity_vph"])
    for g in todo:
        k = density_from_occupancy(d["occupancy"].to_numpy(), g)
        out, _ = fit_station_day(lib, discharge, k, d["flow_vph"].to_numpy(), d["time"].tolist(), n_boot=2)
        res[f"capacity_g{g}"] = out["capacity_vph"]
        res[f"geh_g{g}"] = out["geh_all"]
    rows.append(res)
    if n % 100 == 0:
        print(f"{n}/{len(kept)}", flush=True)
new = pd.DataFrame(rows)
if old is not None:
    new = old.merge(new.drop(columns="capacity_qv"), on=["station", "date"])
cols = ["station", "date", "capacity_qv"] + [c for g in G_VALUES for c in (f"capacity_g{g}", f"geh_g{g}")]
new[cols].to_csv(out_file, index=False, float_format="%.3f")
print("saved results/density_sensitivity.csv")
