"""Does the simulator add anything over simpler ways of describing the same data?

Free-flow points are almost an identity here: density is computed as flow / speed, and PeMS's
free-flow speed is nearly constant. So model skill is judged on CONGESTED intervals only
(speed < 60 km/h) in the held-out hours (every 4th hour), never used for fitting. Four predictors,
all fitted on the training hours of the same day:

  CA                calibrated cellular automaton
  triangle          triangular FD, jam density free
  triangle, kj=133  triangular FD with the CA's jam density (one car per 7.5 m cell)
  empirical         no model: median training flow at a similar density (+/- 5 veh/km)

Capacities are compared under a decision rule fixed before this was run (temporaryRecords/WORKLOG R1):
the CA and triangle capacities "agree" if Spearman >= 0.9 and the median |relative difference| < 3%.

    python scripts/09_simple_baseline.py
        -> results/baseline_comparison.csv, results/baseline_summary.csv, figures/06_baseline.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.baseline import fit_triangle, predict_triangle  # noqa: E402
from nasch.calibration import (CELL_KM, DT_GRID, best_columns, geh, geh_matrix, holdout_masks,  # noqa: E402
                               load_library, model_flow, split_column)
from nasch.plotting import AQUA, BLUE, INK_2, ORANGE, plt, save, use_style  # noqa: E402

CONGESTED_KMH = 60.0
K_JAM_CELL = 1000 / 7.5
use_style()
lib = load_library(paths.CACHE / "fd_library.npz")
cal = pd.read_csv(paths.RESULTS / "calibration.csv")
kept = cal[cal["kept_for_av_gain"]]
data = pd.read_csv(paths.DATA_PROCESSED / "lane1_5min.csv.gz", dtype={"time": str})
groups = {key: g.sort_values("time") for key, g in data.groupby(["station", "date"])}


def empirical_predict(k_train, q_train, k_new, width=5.0):
    out = np.empty(len(k_new))
    for i, kk in enumerate(k_new):
        near = np.abs(k_train - kk) <= width
        out[i] = np.median(q_train[near]) if near.any() else q_train[np.argmin(np.abs(k_train - kk))]
    return out


def scores(pred, obs):
    return dict(geh=float(geh(pred, obs).mean()), mape=float(np.mean(np.abs(pred - obs) / obs)),
                bias=float(np.mean(pred - obs) / np.mean(obs)))


rows = []
for r in kept.to_dict("records"):
    d = groups[(r["station"], r["date"])]
    k, q, v = d["density_vpk"].to_numpy(), d["flow_vph"].to_numpy(), d["speed_kmh"].to_numpy()
    train, test = holdout_masks(d["time"].tolist())
    cong = v < CONGESTED_KMH
    row = dict(station=r["station"], date=r["date"], freeway=r["freeway"], direction=r["direction"],
               congested_test_intervals=int((test & cong).sum()), capacity_ca=r["capacity_vph"])
    # CA refitted on the training hours only
    J = model_flow(lib, k * CELL_KM)
    col, _ = best_columns(geh_matrix(J, q), train)
    i, dd = split_column(int(col[0]))
    ca_pred = 3600 * J[:, int(i)] / DT_GRID[dd]
    tri_free = fit_triangle(k[train], q[train])
    tri_fix = fit_triangle(k[train], q[train], k_jam=K_JAM_CELL)
    target = test & cong
    if target.sum() >= 3 and (train & cong).sum() >= 3:
        preds = {"ca": ca_pred[target],
                 "triangle": predict_triangle(tri_free, k[target]) if tri_free else None,
                 "triangle_kj133": predict_triangle(tri_fix, k[target]) if tri_fix else None,
                 "empirical": empirical_predict(k[train & cong], q[train & cong], k[target])}
        for name, p in preds.items():
            if p is not None:
                for m, val in scores(p, q[target]).items():
                    row[f"{m}_{name}"] = val
    all_free = fit_triangle(k, q)
    all_fix = fit_triangle(k, q, k_jam=K_JAM_CELL)
    if all_free:
        row.update(capacity_triangle=all_free["capacity"], jam_density_triangle=all_free["jam_density"])
    if all_fix:
        row["capacity_triangle_kj133"] = all_fix["capacity"]
    rows.append(row)

cmp_ = pd.DataFrame(rows)
cmp_.to_csv(paths.RESULTS / "baseline_comparison.csv", index=False, float_format="%.4f")


def agreement(col):
    ok = cmp_.dropna(subset=[col])
    rho = spearmanr(ok["capacity_ca"], ok[col])[0]
    rel = np.median(np.abs(ok[col] / ok["capacity_ca"] - 1))
    return rho, rel, bool(rho >= 0.9 and rel < 0.03)


def within_site_sd(col):
    ok = cmp_.dropna(subset=[col])
    multi = ok.groupby("station").filter(lambda g: len(g) >= 2)
    return np.sqrt(multi.groupby("station")[col].var(ddof=1).mean())


scored = cmp_.dropna(subset=["geh_ca", "geh_empirical"])
methods = ["ca", "triangle", "triangle_kj133", "empirical"]
summary = {"station_days": len(cmp_), "scored_station_days": len(scored)}
for m in methods:
    summary[f"congested_test_geh_median_{m}"] = scored[f"geh_{m}"].median()
    summary[f"congested_test_mape_median_{m}"] = scored[f"mape_{m}"].median()
    summary[f"congested_test_bias_median_{m}"] = scored[f"bias_{m}"].median()
for m in ["triangle", "triangle_kj133", "empirical"]:
    both = scored.dropna(subset=[f"geh_{m}"])
    summary[f"share_ca_better_than_{m}"] = (both["geh_ca"] < both[f"geh_{m}"]).mean()
for col in ["capacity_triangle", "capacity_triangle_kj133"]:
    rho, rel, agree = agreement(col)
    summary[f"{col}_spearman"] = rho
    summary[f"{col}_median_abs_rel_diff"] = rel
    summary[f"{col}_agrees_with_ca"] = agree
    summary[f"{col}_day_to_day_sd"] = within_site_sd(col)
summary["capacity_ca_day_to_day_sd"] = within_site_sd("capacity_ca")
summary["triangle_jam_density_median"] = cmp_["jam_density_triangle"].median()
summary = pd.Series(summary)
summary.to_csv(paths.RESULTS / "baseline_summary.csv", header=["value"])
print(summary.to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
ok = cmp_.dropna(subset=["capacity_triangle", "capacity_triangle_kj133"])
ax.plot(ok["capacity_ca"], ok["capacity_triangle"], "o", ms=3, color=ORANGE, alpha=0.35, label="triangle, jam density free")
ax.plot(ok["capacity_ca"], ok["capacity_triangle_kj133"], "o", ms=3, color=AQUA, alpha=0.35, label="triangle, jam density 133")
lim = [ok[["capacity_ca"]].min().min() - 100, ok[["capacity_ca"]].max().max() + 100]
ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
ax.set_xlabel("capacity from the calibrated CA (veh/h)")
ax.set_ylabel("capacity from a fitted triangle (veh/h)")
ax.set_title(f"(a) Same data, simpler model\nSpearman {summary['capacity_triangle_spearman']:.2f} (free) / "
             f"{summary['capacity_triangle_kj133_spearman']:.2f} (kj = 133)")
ax.legend(fontsize=9)
ax = axes[1]
labels = ["CA", "triangle", "triangle\nkj = 133", "no model\n(empirical)"]
grey = dict(color=INK_2, lw=1.2)
ax.boxplot([scored[f"geh_{m}"].dropna() for m in methods], tick_labels=labels, widths=0.5, showfliers=False,
           medianprops=dict(color=ORANGE, lw=2), boxprops=grey, whiskerprops=grey, capprops=grey)
ax.set_ylabel("GEH on congested held-out intervals")
ax.set_title(f"(b) Skill where it matters: congestion, unseen hours\n{len(scored)} station-days "
             f"(speed < 60 km/h)")
ax = axes[2]
vals = [summary["capacity_ca_day_to_day_sd"], summary["capacity_triangle_day_to_day_sd"],
        summary["capacity_triangle_kj133_day_to_day_sd"]]
bars = ax.bar(["CA", "triangle", "triangle kj=133"], vals, color=[BLUE, ORANGE, AQUA], width=0.5)
for b, val in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, val + 2, f"{val:.0f}", ha="center")
ax.grid(axis="x", visible=False)
ax.set_ylabel("SD of capacity across days at one site (veh/h)")
ax.set_title("(c) Repeatability of the capacity estimate\nfrom one weekday to the next")
save(fig, paths.FIGURES / "06_baseline.png")
print("saved figures/06_baseline.png")
