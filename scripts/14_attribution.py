"""What explains the detector-to-detector differences in calibrated capacity?

Covariates from the raw files themselves:
  * corridor (freeway + direction)
  * number of mainline lanes at the detector
  * lane 1's share of the station's total flow while congested (speed < 60 km/h)
  * calibrated free-flow speed
  * exploratory, for HOV lanes: lane 1's median speed advantage over lane 2 while the station is
    congested (station speed < 40 mph); the text metadata has no HOV flag
Position covariates from the PeMS station metadata (if it is in data/raw/), fixed in advance:
  * the metadata lane count
  * distance to the nearest on-ramp upstream and to the nearest off-ramp downstream (km, capped at 3)
  * whether the detector is next to a recurrent active bottleneck (scripts/11, active >= 3 days)

Reported as the share of detector-to-detector variance (detector means over days) explained by
linear models. The position test is the gain in adjusted R^2 from adding the position covariates to
corridor-only; "position explains a meaningful part" needs a gain of at least 0.10.

    python scripts/14_attribution.py   -> results/attribution.csv, results/detector_covariates.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.bottlenecks import load_metadata, ramp_distances  # noqa: E402

cal = pd.read_csv(paths.RESULTS / "calibration.csv")
kept = cal[cal["kept_for_av_gain"]]
stations = set(kept["station"])

lane_cols = []
for lane in range(1, 9):
    lane_cols += [f"l{lane}_samples", f"l{lane}_flow", f"l{lane}_occ", f"l{lane}_speed", f"l{lane}_obs"]
cols = ["timestamp", "station", "district", "freeway", "direction", "lane_type", "station_length",
        "samples", "pct_observed", "total_flow", "avg_occupancy", "avg_speed"] + lane_cols
rows = []
for f in paths.raw_files(paths.ANALYSIS_DAYS):
    df = pd.read_csv(f, header=None, names=cols, usecols=range(len(cols)))
    df = df[(df["lane_type"] == "ML") & df["station"].isin(stations)]
    date = "-".join(f.name.split("_")[-3:]).split(".")[0]
    for st, g in df.groupby("station"):
        lanes = int(sum(g[f"l{i}_samples"].fillna(0).gt(0).any() for i in range(1, 9)))
        cong = g[(g["l1_speed"] * 1.609344 < 60) & (g["total_flow"] > 0)]
        share = cong["l1_flow"].sum() / cong["total_flow"].sum() if len(cong) else np.nan
        jam = g[(g["avg_speed"] < 40) & (g["l1_speed"] > 0) & (g["l2_speed"] > 0)]
        hov = float((jam["l1_speed"] - jam["l2_speed"]).median()) if len(jam) >= 6 else np.nan
        rows.append(dict(station=st, date=date, lanes=lanes, lane1_share_congested=share,
                         lane1_speed_advantage_mph=hov))
cov = pd.DataFrame(rows)
k = kept.merge(cov, on=["station", "date"], how="left")
det = k.groupby("station").agg(C=("capacity_vph", "mean"), lanes=("lanes", "median"),
                               share=("lane1_share_congested", "mean"), vf=("free_speed_kmh", "mean"),
                               hov=("lane1_speed_advantage_mph", "median"),
                               corridor=("freeway", "first"), direction=("direction", "first"))
det = det.dropna(subset=["C", "lanes", "share", "vf"])
det["corridor"] = det["corridor"].astype(str) + det["direction"]
meta = load_metadata()
if meta is not None:
    ml = meta[meta["type"] == "ML"].set_index("station")
    det["lanes_meta"] = det.index.map(ml["lanes"])
    det = det.join(ramp_distances(meta).set_index("station"))
    bn = pd.read_csv(paths.RESULTS / "bottlenecks.csv")
    recurrent = bn[bn["days_active"] >= 3]
    det["next_to_bottleneck"] = det.index.isin(set(recurrent["upstream"]) | set(recurrent["downstream"])).astype(float)
det.to_csv(paths.RESULTS / "detector_covariates.csv", float_format="%.4f")


def r2(X, d=None):
    d = det if d is None else d
    X = np.column_stack([np.ones(len(d))] + X)
    beta, *_ = np.linalg.lstsq(X, d["C"].to_numpy(), rcond=None)
    resid = d["C"].to_numpy() - X @ beta
    n, p = X.shape
    r2_ = 1 - resid.var() / d["C"].var(ddof=0)
    return r2_, 1 - (1 - r2_) * (n - 1) / (n - p)


corr = pd.get_dummies(det["corridor"], drop_first=True).to_numpy(float)
lanes = pd.get_dummies(det["lanes"].astype(int), drop_first=True).to_numpy(float)
models = {
    "corridor": [corr],
    "lanes": [lanes],
    "lane-1 share": [det[["share"]].to_numpy()],
    "lanes + lane-1 share + free speed": [lanes, det[["share", "vf"]].to_numpy()],
    "corridor + lanes + lane-1 share + free speed": [corr, lanes, det[["share", "vf"]].to_numpy()],
}
res = pd.DataFrame([(name, *r2(X)) for name, X in models.items()], columns=["model", "r2", "adj_r2"])
res["n_detectors"] = len(det)
if meta is not None:
    pd_ = det.dropna(subset=["lanes_meta", "on_ramp_upstream_km", "off_ramp_downstream_km"])
    corr_p = pd.get_dummies(pd_["corridor"], drop_first=True).to_numpy(float)
    lanes_p = pd.get_dummies(pd_["lanes_meta"].astype(int), drop_first=True).to_numpy(float)
    ramps = pd_[["on_ramp_upstream_km", "off_ramp_downstream_km"]].to_numpy()
    nb = pd_[["next_to_bottleneck"]].to_numpy()
    position = [lanes_p, ramps, nb]
    meta_models = {
        "corridor (metadata sample)": [corr_p],
        "corridor + metadata lanes": [corr_p, lanes_p],
        "corridor + ramp distances": [corr_p, ramps],
        "corridor + next to bottleneck": [corr_p, nb],
        "corridor + position (lanes, ramps, bottleneck)": [corr_p] + position,
    }
    extra = pd.DataFrame([(name, *r2(X, pd_)) for name, X in meta_models.items()], columns=["model", "r2", "adj_r2"])
    extra["n_detectors"] = len(pd_)
    hv = pd_.dropna(subset=["hov"])
    hov_models = {"corridor (HOV sample)": [pd.get_dummies(hv["corridor"], drop_first=True).to_numpy(float)],
                  "corridor + lane-1 speed advantage": [pd.get_dummies(hv["corridor"], drop_first=True).to_numpy(float),
                                                        hv[["hov"]].to_numpy()]}
    extra_h = pd.DataFrame([(name, *r2(X, hv)) for name, X in hov_models.items()], columns=["model", "r2", "adj_r2"])
    extra_h["n_detectors"] = len(hv)
    res = pd.concat([res, extra, extra_h], ignore_index=True)
    a = res.set_index("model")["adj_r2"]
    gain = a["corridor + position (lanes, ramps, bottleneck)"] - a["corridor (metadata sample)"]
    print(f"position test: adjusted R2 gain over corridor-only = {gain:.3f} "
          f"({'meaningful' if gain >= 0.10 else 'below the pre-set 0.10'})")
    print(f"next to a recurrent bottleneck: {int(pd_['next_to_bottleneck'].sum())} of {len(pd_)} detectors; "
          f"lane-1 speed advantage while congested: median {hv['hov'].median():.1f} mph, "
          f"share > 10 mph {np.mean(hv['hov'] > 10):.2f}")
slope = np.polyfit(det["share"], det["C"], 1)[0]
res.to_csv(paths.RESULTS / "attribution.csv", index=False, float_format="%.3f")
print(f"{len(det)} detectors, {det['corridor'].nunique()} corridors")
print(res.round(3).to_string(index=False))
print(f"capacity vs lane-1 share: {slope / 100:.1f} veh/h per percentage point; "
      f"lane-1 share median {det['share'].median():.3f}, lanes median {det['lanes'].median():.0f}")
