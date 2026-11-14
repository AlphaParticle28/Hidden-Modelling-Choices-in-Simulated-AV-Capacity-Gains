"""How far can each modelling choice move a reported AV capacity gain?

Two kinds of bar, which should not be ranked against each other:
  * "switch" bars: the median shift when one discrete choice is swapped for another (hatched),
    including calibrating lane 2 instead of lane 1 at the same detector (scripts/16);
  * "spread" bars: the 10th-90th percentile spread over a population (detectors, days, refits).

Panel (a) is in percentage points of "capacity gain from 0% to 100% AVs", for a reference AV
that never hesitates and keeps a 1.0 s time gap (Newell; discharge v / (v g + 7.5 m)). Panel (b)
shows the same quantities as % changes in the capacity ratio, 100 * log(1 + G). On that scale the
human-side bars do not depend on which AV is assumed. Error bars are 95% intervals from resampling
whole corridors (500 times).

    python scripts/12_error_budget.py   -> results/error_budget.csv, figures/09_error_budget.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nasch import paths  # noqa: E402
from nasch.calibration import DT_GRID, split_column  # noqa: E402
from nasch.plotting import AQUA, BLUE, INK, INK_2, ORANGE, plt, save, use_style  # noqa: E402

GAP_REF, L_EFF = 1.0, 7.5
use_style()


def av_cap(gap, v_kmh):
    v = np.asarray(v_kmh, float) / 3.6
    return 3600 * v / (v * gap + L_EFF)


cal = pd.read_csv(paths.RESULTS / "calibration.csv")
keep = cal["kept_for_av_gain"].to_numpy()
k = cal[keep].reset_index(drop=True)
k["corridor"] = k["freeway"].astype(str) + k["direction"]
het = pd.read_csv(paths.RESULTS / "heterogeneity_summary.csv", index_col=0)["value"]
gap = pd.read_csv(paths.RESULTS / "definition_gap.csv")
base = pd.read_csv(paths.RESULTS / "baseline_comparison.csv")
dis = pd.read_csv(paths.RESULTS / "human_discharge.csv")
J, vmax = dis["discharge"].to_numpy(), dis["vmax"].to_numpy()
B = np.load(paths.CACHE / "bootstrap_columns.npz")["columns"][keep]
pi, di = split_column(B)
dt = DT_GRID[di]
C_h_boot = 3600 * J[pi] / dt
C_av_cells_boot = 3600 * vmax[pi] / (vmax[pi] + 1) / dt
C_hat = k["capacity_vph"].to_numpy()

# per-station-day ingredients, in log units (% of the capacity ratio) and in points of gain
v_med = k["free_speed_kmh"].median()
C_ref = av_cap(GAP_REF, v_med)


def p10_90(x, axis=None):
    lo_, hi_ = np.percentile(x, [10, 90], axis=axis)
    return hi_ - lo_


k["est_log"] = p10_90(100 * np.log(C_h_boot), axis=1)
k["est_pts"] = p10_90(100 * C_ref / C_h_boot, axis=1)
k["cells_log"] = p10_90(100 * np.log(C_av_cells_boot / C_hat[:, None]), axis=1)
k["cells_pts"] = p10_90(100 * C_av_cells_boot / C_hat[:, None], axis=1)
gap["def_log"] = 100 * np.abs(np.log(gap["capacity_ring_vph"] / gap["capacity_vph"]))
gap["def_pts"] = 100 * np.abs(C_ref / gap["capacity_ring_vph"] - C_ref / gap["capacity_vph"])
base = base.dropna(subset=["capacity_triangle"])
base["model_log"] = 100 * np.abs(np.log(base["capacity_triangle"] / base["capacity_ca"]))
base["model_pts"] = 100 * np.abs(C_ref / base["capacity_triangle"] - C_ref / base["capacity_ca"])
lane2 = pd.read_csv(paths.RESULTS / "lane2_check.csv")
lane2["lane_log"] = 100 * np.abs(np.log(lane2["capacity_lane2"] / lane2["capacity_vph"]))
lane2["lane_pts"] = 100 * np.abs(C_ref / lane2["capacity_lane2"] - C_ref / lane2["capacity_vph"])

NAMES = ["Capacity definition|(ring peak vs queue discharge)", "Baseline model|(CA vs fitted triangle)",
         "Calibration lane|(lane 1 vs lane 2, same detector)", "Calibration detector|(which detector)",
         "Calibration day|(same detector, another weekday)", "Estimation error|(one detector, one day)",
         "AV encoded in cells|(inherits the human fit)", "AV time-gap assumption|(0.6 s vs 1.5 s)"]
NAMES = [n.replace("|", "\n") for n in NAMES]
KIND = ["switch", "switch", "switch", "spread", "spread", "spread", "spread", "switch"]


def budget(kk, gg, bb, ll):
    """Each bar as (log units, points of gain) for one (resampled) set of detector-days.
    Points are computed directly in gain space, not converted from the log values."""
    sites = kk.groupby("station")["capacity_vph"].mean()
    m = kk.groupby("station").filter(lambda g: len(g) >= 2)
    sd_day = np.sqrt(m.groupby("station")["capacity_vph"].var(ddof=1).mean())
    med = sites.median()
    logs = [gg["def_log"].median(), bb["model_log"].median(), ll["lane_log"].median(),
            100 * (np.log(sites.quantile(0.9)) - np.log(sites.quantile(0.1))),
            100 * 2.563 * sd_day / med, kk["est_log"].median(), kk["cells_log"].median(),
            100 * np.log(av_cap(0.6, v_med) / av_cap(1.5, v_med))]
    pts = [gg["def_pts"].median(), bb["model_pts"].median(), ll["lane_pts"].median(),
           100 * (C_ref / sites.quantile(0.1) - C_ref / sites.quantile(0.9)),
           100 * 2.563 * sd_day * C_ref / med ** 2, kk["est_pts"].median(), kk["cells_pts"].median(),
           100 * (av_cap(0.6, v_med) - av_cap(1.5, v_med)) / med]
    return logs + pts


point = budget(k, gap, base, lane2)
rng = np.random.default_rng(9)
corrs = k["corridor"].unique()
corridor_keys = k[["station", "date", "corridor"]].drop_duplicates()


def split_by_corridor(table):
    """The rows of `table` that belong to each corridor's kept station-days, so that a corridor drawn
    twice in a bootstrap replicate contributes its rows twice, as it does in the main table."""
    t = table.merge(corridor_keys, on=["station", "date"])
    return {c: t[t["corridor"] == c] for c in corrs}


gap_c, base_c, lane2_c = split_by_corridor(gap), split_by_corridor(base), split_by_corridor(lane2)
boots = []
for _ in range(500):
    pick = rng.choice(corrs, corrs.size, replace=True)
    kk = pd.concat([k[k["corridor"] == c].assign(station=lambda d, j=j: d["station"].astype(str) + f"#{j}")
                    for j, c in enumerate(pick)])
    gg = pd.concat([gap_c[c] for c in pick])
    bb = pd.concat([base_c[c] for c in pick])
    ll = pd.concat([lane2_c[c] for c in pick])
    boots.append(budget(kk, gg, bb, ll))
boots = np.array(boots)
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
n = len(NAMES)
G_ref = C_ref / k.groupby("station")["capacity_vph"].mean().median()          # ratio 1 + G, median detector
rows = []
for j, name in enumerate(NAMES):
    rows.append(dict(choice=name.replace("\n", " "), kind=KIND[j], log_pct=point[j], log_lo=lo[j], log_hi=hi[j],
                     points=point[n + j], points_lo=lo[n + j], points_hi=hi[n + j]))
res = pd.DataFrame(rows)
res.to_csv(paths.RESULTS / "error_budget.csv", index=False, float_format="%.1f")
print(res.round(1).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
y = np.arange(len(res))[::-1]
colors = [BLUE, BLUE, BLUE, BLUE, AQUA, AQUA, ORANGE, INK_2]
for ax, (col, lo_c, hi_c, xlabel, title) in zip(axes, [
        ("points", "points_lo", "points_hi", f"percentage points of AV gain (reference: {GAP_REF} s gap)",
         f"(a) In points of the reported gain (median gain {100 * (G_ref - 1):.0f}%)"),
        ("log_pct", "log_lo", "log_hi", "% change in the capacity ratio, 100 log(1+G)",
         "(b) As % of the ratio: human-side bars do not depend on the AV")]):
    for yy, (_, r), c in zip(y, res.iterrows(), colors):
        ax.barh(yy, r[col], color=c, height=0.6, hatch="//" if r["kind"] == "switch" else None,
                edgecolor="#fcfcfb" if r["kind"] == "spread" else INK_2, lw=0.5)
        ax.errorbar(r[col], yy, xerr=[[r[col] - r[lo_c]], [r[hi_c] - r[col]]], color=INK, capsize=3, lw=1)
        ax.text(r[hi_c] + 2, yy, f"{r[col]:.0f}", va="center", color=INK, fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y, res["choice"], fontsize=9.5)
fig.suptitle("What moves a simulated 'AVs add X% capacity' number?  Hatched = switching one choice "
             "(median shift); solid = 10th-90th percentile spread. 95% corridor-bootstrap intervals.",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, paths.FIGURES / "09_error_budget.png")
print("saved figures/09_error_budget.png")
