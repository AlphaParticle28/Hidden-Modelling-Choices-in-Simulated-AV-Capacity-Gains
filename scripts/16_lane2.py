"""Is the detector-to-detector spread about the detector's position, or about lane 1's own loop?

Calibrate lane 2 at the same stations and days, then compare. A loop-specific error in lane 1
would not show up in lane 2. A position effect (the downstream bottleneck, ramps) is shared by both
lanes, so the two lanes would rank detectors alike.

    python scripts/16_lane2.py   -> results/lane2_check.csv, results/lane2_summary.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.calibration import fit_station_day, load_library  # noqa: E402
from nasch.pems import (LANE1_COLUMNS, STATION_COLUMNS, add_physical_units, day_from_filename,  # noqa: E402
                        quality_mask, screen_station_day)

lib = load_library(paths.CACHE / "fd_library.npz")
discharge = pd.read_csv(paths.RESULTS / "human_discharge.csv")["discharge"].to_numpy()
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
kept = cal[cal["kept_for_av_gain"]]

lane2 = ["l2_samples", "l2_flow", "l2_occupancy", "l2_speed_mph", "l2_observed"]
rows = []
for f in paths.raw_files(paths.ANALYSIS_DAYS):
    date = day_from_filename(f)
    want = set(kept.loc[kept["date"] == date, "station"])
    df = pd.read_csv(f, header=None, usecols=range(22), names=STATION_COLUMNS + LANE1_COLUMNS + lane2)
    df = df[(df["lane_type"] == "ML") & df["station"].isin(want)].drop(columns=LANE1_COLUMNS)
    df = df.rename(columns=dict(zip(lane2, LANE1_COLUMNS)))          # reuse the lane-1 machinery
    df["time"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M:%S").dt.strftime("%H:%M")
    df = add_physical_units(df)
    df = df[quality_mask(df)].sort_values("time")
    for st, g in df.groupby("station"):
        if screen_station_day(g)["reason"]:
            continue
        out, _ = fit_station_day(lib, discharge, g["density_vpk"].to_numpy(), g["flow_vph"].to_numpy(),
                                 g["time"].tolist(), n_boot=2)
        rows.append(dict(station=st, date=date, capacity_lane2=out["capacity_vph"]))
    print(f"{f.name}: {len(rows)} lane-2 station-days so far", flush=True)

l2 = pd.DataFrame(rows).merge(kept[["station", "date", "capacity_vph", "freeway", "direction"]], on=["station", "date"])
l2.to_csv(paths.RESULTS / "lane2_check.csv", index=False, float_format="%.2f")
det = l2.groupby("station").agg(c1=("capacity_vph", "mean"), c2=("capacity_lane2", "mean"),
                                corridor=("freeway", "first"), d=("direction", "first"))
det["corridor"] = det["corridor"].astype(str) + det["d"]
rho_all = spearmanr(det["c1"], det["c2"])[0]
dm = det[["c1", "c2"]] - det.groupby("corridor")[["c1", "c2"]].transform("mean")
multi = det.groupby("corridor")["c1"].transform("size") >= 2
rho_within = spearmanr(dm.loc[multi, "c1"], dm.loc[multi, "c2"])[0]
summary = pd.Series(dict(station_days=len(l2), detectors=len(det),
                         spearman_lane1_vs_lane2=rho_all, spearman_within_corridor=rho_within,
                         sd_lane1=det["c1"].std(), sd_lane2=det["c2"].std(),
                         median_lane2_over_lane1=(det["c2"] / det["c1"]).median()))
summary.to_csv(paths.RESULTS / "lane2_summary.csv", header=["value"])
print(summary.round(3).to_string())
