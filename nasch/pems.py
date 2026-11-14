"""Reading and screening Caltrans PeMS "Station 5-Minute" files (lane 1 only).

Each row of a raw file is one station for one 5-minute interval: 12 station-level columns, then
5 columns per lane (samples, flow, average occupancy, average speed, observed flag). Only mainline
("ML") stations and lane 1 are used. Lane 1 is used because the simulator is single-lane and
flow and occupancy must describe the same lane. (Station totals add flow across lanes but average
occupancy across lanes, so mixing them describes different numbers of lanes.)

Density. A single loop measures occupancy, not density. Two conversions are provided:
  * k = q / v           (primary): uses PeMS's own speed for the lane. For single-loop detectors
                         that speed is itself PeMS's estimate, based on a vehicle length PeMS
                         learns for each detector, so this is "PeMS's station-specific length"
                         rather than a direct measurement.
  * k = occupancy / g    (sensitivity): one fixed effective vehicle length g for every station.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

MPH_TO_KMH = 1.609344
FT_PER_MILE = 5280.0
INTERVALS_PER_DAY = 288

STATION_COLUMNS = ["timestamp", "station", "district", "freeway", "direction", "lane_type",
                   "station_length", "samples", "pct_observed", "total_flow",
                   "avg_occupancy", "avg_speed"]
LANE1_COLUMNS = ["l1_samples", "l1_flow", "l1_occupancy", "l1_speed_mph", "l1_observed"]

# interval-level quality rules
G_RANGE_FT = (12.0, 60.0)       # implied effective vehicle length must be physically possible

# station-day selection rules (the day must show free flow, capacity and congestion)
MIN_USABLE_INTERVALS = 250      # of 288
CAPACITY_BAND_VPH = (1700.0, 2500.0)
FREE_SPEED_BAND_KMH = (75.0, 125.0)
MIN_CONGESTED_INTERVALS = 15    # intervals above 1.5 x the density at which flow peaked


def day_from_filename(path):
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", Path(path).name)
    if not m:
        raise ValueError(f"cannot find a date in {path}")
    return "-".join(m.groups())


def read_lane1(path):
    """All mainline stations' lane-1 data from one raw file, before any quality checks."""
    cols = STATION_COLUMNS + LANE1_COLUMNS
    df = pd.read_csv(path, header=None, usecols=range(len(cols)), names=cols,
                     compression="infer")
    df = df[df["lane_type"] == "ML"].copy()
    df["date"] = day_from_filename(path)
    df["time"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M:%S").dt.strftime("%H:%M")
    return df


def add_physical_units(df):
    """flow (veh/h), speed (km/h), density (veh/km) and implied effective vehicle length (ft)."""
    out = df.copy()
    out["flow_vph"] = 12.0 * out["l1_flow"]                          # 5-minute count -> per hour
    out["speed_kmh"] = out["l1_speed_mph"] * MPH_TO_KMH
    out["occupancy"] = out["l1_occupancy"]                           # fraction of time, 0-1
    with np.errstate(divide="ignore", invalid="ignore"):
        out["density_vpk"] = out["flow_vph"] / out["speed_kmh"]      # k = q / v
        out["g_ft"] = FT_PER_MILE * out["occupancy"] * out["l1_speed_mph"] / out["flow_vph"]
    return out


def density_from_occupancy(occupancy, g_ft):
    """Density (veh/km) from occupancy with a fixed effective vehicle length g (feet)."""
    return np.asarray(occupancy) * (FT_PER_MILE / g_ft) / MPH_TO_KMH


def quality_mask(df):
    """Keep an interval only if lane 1 was actually observed and the numbers are physical."""
    g_ok = df["g_ft"].between(*G_RANGE_FT)
    return ((df["l1_observed"] == 1) & (df["l1_samples"] > 0) & (df["l1_flow"] > 0)
            & (df["l1_speed_mph"] > 0) & (df["occupancy"] > 0) & g_ok)


def screen_station_day(day):
    """Selection metrics and verdict for one station on one day (`day` = QC-passed intervals).

    Returns a dict. `reason` is "" when the station-day qualifies, otherwise the first rule failed.
    """
    n = len(day)
    res = dict(usable_intervals=n, q_max_vph=np.nan, k_at_q_max=np.nan, free_speed_kmh=np.nan,
               congested_intervals=0, k_max=np.nan, reason="")
    if n < MIN_USABLE_INTERVALS:
        res["reason"] = "too few usable intervals"
        return res
    k = day["density_vpk"].to_numpy()
    q = day["flow_vph"].to_numpy()
    v = day["speed_kmh"].to_numpy()
    i = int(np.argmax(q))
    res.update(q_max_vph=q[i], k_at_q_max=k[i], k_max=k.max(),
               free_speed_kmh=float(np.median(v[k <= np.quantile(k, 0.25)])),
               congested_intervals=int((k > 1.5 * k[i]).sum()))
    if not CAPACITY_BAND_VPH[0] <= res["q_max_vph"] <= CAPACITY_BAND_VPH[1]:
        res["reason"] = "peak flow outside 1700-2500 veh/h"
    elif not FREE_SPEED_BAND_KMH[0] <= res["free_speed_kmh"] <= FREE_SPEED_BAND_KMH[1]:
        res["reason"] = "free-flow speed outside 75-125 km/h"
    elif res["congested_intervals"] < MIN_CONGESTED_INTERVALS:
        res["reason"] = "not enough congestion"
    return res


def screen_file(path):
    """Screen every mainline station in one raw file.

    Returns (screening table with one row per station, QC-passed intervals of the qualifying
    stations)."""
    df = add_physical_units(read_lane1(path))
    df["qc_ok"] = quality_mask(df)
    rows, keep = [], []
    for (station, fwy, direction), g in df.groupby(["station", "freeway", "direction"]):
        good = g[g["qc_ok"]]
        res = screen_station_day(good)
        rows.append(dict(date=g["date"].iloc[0], station=station, freeway=fwy,
                         direction=direction, rows_in_file=len(g), **res))
        if res["reason"] == "":
            keep.append(good)
    screening = pd.DataFrame(rows)
    cols = ["date", "station", "freeway", "direction", "time", "flow_vph", "speed_kmh",
            "occupancy", "density_vpk", "g_ft"]
    data = pd.concat(keep)[cols] if keep else pd.DataFrame(columns=cols)
    return screening, data
