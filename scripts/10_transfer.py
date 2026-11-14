"""Out-of-sample test on days the analysis never saw (25 and 29 Sep 2026).

For every calibrated site that also shows congestion on a test day, predict that day's CONGESTED
intervals (speed < 60 km/h; free flow is near-identity because density = flow / speed) five ways,
scored with GEH:

  own day   calibrate on the test day itself (the best this model could possibly do)
  site      reuse the site's calibrations from the analysis days (no refitting)
  triangle  reuse triangles fitted to the site's analysis days
  generic   one parameter set for every site (the best single compromise over all analysis days)
  history   no model: median flow at a similar density (+/- 5 veh/km) on the site's analysis days

If "site" is close to "own day" and clearly better than "generic", the calibration captures
something real and lasting about the location. Capacity is compared the same way.

    python scripts/10_transfer.py   -> results/transfer_validation.csv, results/transfer_summary.csv,
                                       figures/07_transfer.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.baseline import fit_triangle, predict_triangle  # noqa: E402
from nasch.calibration import (CELL_KM, DT_GRID, fit_station_day, geh, geh_matrix, load_library,  # noqa: E402
                               model_flow, split_column)
from nasch.pems import screen_file  # noqa: E402
from nasch.plotting import AQUA, BLUE, INK_2, ORANGE, plt, save, use_style  # noqa: E402

use_style()
lib = load_library(paths.CACHE / "fd_library.npz")
discharge = pd.read_csv(paths.RESULTS / "human_discharge.csv")["discharge"].to_numpy()
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
kept = cal[cal["kept_for_av_gain"]]
data = pd.read_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", dtype={"time": str})
groups = {key: g.sort_values("time") for key, g in data.groupby(["station", "date"])}

# ---- the best single parameter set over all analysis station-days ("generic")
pooled_file = paths.CACHE / "pooled_mean_geh.npy"
if pooled_file.exists():
    pooled = np.load(pooled_file)
else:
    pooled = np.zeros(len(lib.grid) * len(DT_GRID))
    for n, r in enumerate(kept.to_dict("records"), 1):
        d = groups[(r["station"], r["date"])]
        G = geh_matrix(model_flow(lib, d["density_vpk"].to_numpy() * CELL_KM), d["flow_vph"].to_numpy())
        pooled += G.mean(axis=0)
        if n % 200 == 0:
            print(f"generic fit {n}/{len(kept)}", flush=True)
    np.save(pooled_file, pooled)
g_param, g_dt = split_column(int(pooled.argmin()))
g_param, g_dt = int(g_param), float(DT_GRID[g_dt])
generic_capacity = 3600 * discharge[g_param] / g_dt


def empirical(k_ref, q_ref, k_new, width=5.0):
    out = np.empty(len(k_new))
    for i, kk in enumerate(k_new):
        near = np.abs(k_ref - kk) <= width
        out[i] = np.median(q_ref[near]) if near.any() else q_ref[np.argmin(np.abs(k_ref - kk))]
    return out


def ca_predict(param, dt, k):
    return 3600 * model_flow(lib.subset([int(param)]), k * CELL_KM)[:, 0] / dt


rows = []
for f in paths.raw_files(paths.TEST_DAYS):
    _, test = screen_file(f)
    test = test[test["station"].isin(kept["station"])]
    for (station, date), d in test.groupby(["station", "date"]):
        d = d.sort_values("time")
        k, q, times = d["density_vpk"].to_numpy(), d["flow_vph"].to_numpy(), d["time"].tolist()
        c = d["speed_kmh"].to_numpy() < 60
        if c.sum() < 5:
            continue
        site = kept[kept["station"] == station]
        own, _ = fit_station_day(lib, discharge, k, q, times, n_boot=2)
        own_pred = ca_predict(own["param_index"], own["dt"], k)
        site_geh = [geh(ca_predict(s.param_index, s.dt, k[c]), q[c]).mean() for s in site.itertuples()]
        tris = [fit_triangle(groups[(station, s.date)]["density_vpk"], groups[(station, s.date)]["flow_vph"])
                for s in site.itertuples()]
        tris = [t for t in tris if t is not None]
        hist = pd.concat([groups[(station, s.date)] for s in site.itertuples()])
        hist = hist[hist["speed_kmh"] < 60]
        hist_geh = (float(geh(empirical(hist["density_vpk"].to_numpy(), hist["flow_vph"].to_numpy(), k[c]), q[c]).mean())
                    if len(hist) >= 5 else np.nan)
        rows.append(dict(
            station=station, date=date, analysis_days=len(site), congested_intervals=int(c.sum()),
            geh_own=float(geh(own_pred[c], q[c]).mean()), geh_site=float(np.mean(site_geh)),
            geh_triangle=float(np.mean([geh(predict_triangle(t, k[c]), q[c]).mean() for t in tris])) if tris else np.nan,
            geh_generic=float(geh(ca_predict(g_param, g_dt, k[c]), q[c]).mean()),
            geh_history=hist_geh,
            capacity_own=own["capacity_vph"], capacity_site=site["capacity_vph"].mean(),
            capacity_generic=generic_capacity))
    print(f"{f.name}: {len(rows)} test station-days so far", flush=True)

t = pd.DataFrame(rows)
t.to_csv(paths.RESULTS / "transfer_validation.csv", index=False, float_format="%.3f")
err_site = t["capacity_site"] - t["capacity_own"]
err_gen = t["capacity_generic"] - t["capacity_own"]
summary = pd.Series(dict(
    test_station_days=len(t), sites=t["station"].nunique(),
    geh_own_median=t["geh_own"].median(), geh_site_median=t["geh_site"].median(),
    geh_triangle_median=t["geh_triangle"].median(), geh_generic_median=t["geh_generic"].median(),
    geh_history_median=t["geh_history"].median(),
    geh_site_median_tuesday=t.loc[t["date"] == "2026-09-29", "geh_site"].median(),
    geh_site_median_friday=t.loc[t["date"] == "2026-09-25", "geh_site"].median(),
    share_site_better_than_history=(t["geh_site"] < t["geh_history"]).mean(),
    share_site_below_5=(t["geh_site"] < 5).mean(),
    capacity_error_site_rmse=np.sqrt((err_site ** 2).mean()), capacity_error_site_median_abs=err_site.abs().median(),
    capacity_error_generic_rmse=np.sqrt((err_gen ** 2).mean()),
    capacity_r2_site=1 - (err_site ** 2).sum() / ((t["capacity_own"] - t["capacity_own"].mean()) ** 2).sum(),
    generic_param_index=g_param, generic_dt=g_dt, generic_capacity=generic_capacity))
summary.to_csv(paths.RESULTS / "transfer_summary.csv", header=["value"], float_format="%.4f")
print(summary.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
cols = [("geh_own", "own day\n(refit)"), ("geh_site", "site\n(transferred)"),
        ("geh_triangle", "triangle\n(transferred)"), ("geh_generic", "one model\nfor all sites")]
grey = dict(color=INK_2, lw=1.2)
ax.boxplot([t[c].dropna() for c, _ in cols], tick_labels=[l for _, l in cols], widths=0.5, showfliers=False,
           medianprops=dict(color=ORANGE, lw=2), boxprops=grey, whiskerprops=grey, capprops=grey)
ax.axhline(5, color=INK_2, lw=1, ls="--")
ax.set_ylabel("GEH on congested intervals of the unseen day")
ax.set_title(f"(a) Predicting {len(t)} unseen station-days\n(25 and 29 Sep, never used in fitting)")
ax = axes[1]
ax.plot(t["capacity_site"], t["capacity_own"], "o", ms=3.5, color=BLUE, alpha=0.5, label="site calibration")
ax.axvline(generic_capacity, color=AQUA, lw=1.5, label="one model for all sites")
lim = [t[["capacity_site", "capacity_own"]].min().min() - 50, t[["capacity_site", "capacity_own"]].max().max() + 50]
ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
ax.set_xlabel("capacity predicted from the analysis days (veh/h)")
ax.set_ylabel("capacity calibrated on the unseen day (veh/h)")
ax.set_title(f"(b) Site capacity carries over to new days\nR2 = {summary['capacity_r2_site']:.2f}, "
             f"RMSE {summary['capacity_error_site_rmse']:.0f} vs {summary['capacity_error_generic_rmse']:.0f} veh/h without site info")
ax.legend(fontsize=9, loc="lower right")
save(fig, paths.FIGURES / "07_transfer.png")
print("saved figures/07_transfer.png")
